"""
CS336 Assignment 1 - Text Generation (§6)
===========================================
Generate text from a trained Transformer language model with support for
temperature scaling and top-p (nucleus) sampling.

Usage:
    
python -m cs336_basics.generate \
    --checkpoint /root/autodl-tmp/data/checkpoints/owt/checkpoint_step_1000.pt \
    --vocab-json bpe_outputs/openwebtext_vocab.json \
    --prompt-file /root/autodl-tmp/data/owt_val_encoded.bin \
    --prompt-offset 64 \
    --prompt-length 200 \
    --max-tokens 1024 \
    --temperature 0.8 \
    --top-p 0.9 \
    --eos-id 0 \
    --output /root/cs336-lab1/assignment1-basics-main/gen/tiny.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM


def load_vocab(vocab_path: str) -> dict[int, bytes]:
    """Load a BPE vocab JSON into a token_id -> bytes mapping.

    The JSON file stores bytes as latin-1 decoded strings.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        str_vocab = json.load(f)
    return {int(k): v.encode("latin-1") for k, v in str_vocab.items()}


def decode_tokens(token_ids: list[int], id_to_bytes: dict[int, bytes]) -> str:
    """Decode a list of token IDs into UTF-8 text."""
    raw = b"".join(id_to_bytes[tid] for tid in token_ids)
    return raw.decode("utf-8", errors="replace")


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    device: str = "cpu",
    on_token: callable = None,
    use_cache: bool = True,
    use_amp: bool = False,
) -> list[int]:
    """Generate text from a language model.

    Args:
        model: The TransformerLM model (in evaluation mode).
        prompt_ids: List of token IDs for the prompt.
        max_tokens: Maximum number of tokens to generate (including prompt).
        temperature: Temperature for softmax scaling.
        top_p: Nucleus sampling threshold (0.0 to 1.0). If None, sample from full distribution.
        eos_token_id: Token ID for <|endoftext|>. Generation stops when this is produced.
        device: Device to run generation on.
        on_token: Optional callback(token_id, decoded_text) called after each new token.
        use_cache: If True, cache the key/value tensors of every layer so each
            new token only runs a forward over the single newly added token
            (instead of re-processing the whole context every step). This turns
            generation from O(T^2) into O(T) and is much faster for long
            generations.
        use_amp: If True, run generation under bf16 autocast (faster on
            Ampere+ GPUs; requires ``device`` to be CUDA).

    Returns:
        List of generated token IDs (including the prompt).
    """
    model.eval()
    generated = list(prompt_ids)

    # Per-layer KV cache. Each layer keeps its keys/values so that on the next
    # step we only feed the single new token (the model prepends the cached
    # keys/values automatically).
    cache = (
        [{"k": None, "v": None} for _ in range(len(model.layers))]
        if use_cache
        else None
    )

    # Token positions used by the very first (prefill) forward pass. On later
    # incremental steps the model derives positions from the cache length.
    first_step = True

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
        if use_amp and device.startswith("cuda")
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )

    # Anything at or below this temperature is treated as greedy decoding.
    temp_is_greedy = temperature <= 1e-6

    while len(generated) < max_tokens:
        if first_step:
            # Prefill: run the whole (truncated) prompt through the model once.
            input_ids = torch.tensor(
                [generated[-model.context_length:]], dtype=torch.long, device=device
            )
            first_step = False
        elif use_cache and len(generated) <= model.context_length:
            # Incremental: only the newest token needs to be fed forward.
            input_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        else:
            # We reached (or exceeded) the context window: the KV cache can no
            # longer grow (RoPE cos/sin tables are sized to context_length), so
            # reset it and re-prefill from the most recent context_length tokens.
            if use_cache:
                for c in cache:
                    c["k"] = None
                    c["v"] = None
            context = generated[-model.context_length:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)

        with autocast_ctx:
            logits = model(input_ids, cache=cache)  # (1, new_seq_len, vocab_size)

        # Logits for the last (newest) position.
        next_logits = logits[0, -1, :].float()  # (vocab_size,)

        if temp_is_greedy:
            next_token = next_logits.argmax().item()
        else:
            scaled_logits = next_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)  # (vocab_size,)
            if top_p is not None and 0.0 < top_p < 1.0:
                next_token = _sample_top_p(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if on_token is not None:
            on_token(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated


def _sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    """Nucleus (top-p) sampling.

    Args:
        probs: Probability distribution over vocabulary (vocab_size,).
        top_p: Cumulative probability threshold.

    Returns:
        Sampled token ID.
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the smallest set of indices where cumulative prob >= top_p
    mask = cumulative_probs - sorted_probs > top_p
    # Alternatively: find cutoff where cumulative probability exceeds top_p
    # We keep all tokens before the first one that makes cumulative exceed top_p
    sorted_probs[mask] = 0.0

    # Renormalize
    if sorted_probs.sum() > 0:
        sorted_probs = sorted_probs / sorted_probs.sum()
    else:
        # Fallback: take the top token
        return sorted_indices[0].item()

    # Sample from the truncated distribution
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_indices[sampled_idx].item()


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    attn_impl: str = "manual",
) -> tuple[TransformerLM, dict[str, Any]]:
    """Load a model from a saved checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model onto.
        attn_impl: Attention implementation to use for generation ("manual" or
            "fused"). "fused" uses FlashAttention / memory-efficient attention
            which is faster and uses less memory.

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        attn_impl=attn_impl,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab-json", type=str, required=True,
                        help="Path to vocab.json for decoding tokens to text")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save generated text (if not set, only print to console)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to .bin file containing tokenized prompt data")
    parser.add_argument("--prompt-offset", type=int, default=0,
                        help="Starting offset in the prompt file (in tokens)")
    parser.add_argument("--prompt-length", type=int, default=None,
                        help="Number of tokens to use as prompt from file (default: context_length)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate (including prompt)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--eos-id", type=int, default=None,
                        help="EOS token ID (e.g. 0 for <|endoftext|>). Generation stops when produced.")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--attn-impl", type=str, default="fused", choices=["manual", "fused"],
                        help="Attention implementation for generation: 'fused' (FlashAttention, "
                             "faster, default) or 'manual' (reference)")
    parser.add_argument("--amp", action="store_true",
                        help="Run generation under bf16 autocast (faster on Ampere+ GPUs)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the KV cache (slower, but useful for debugging)")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load vocab
    print(f"Loading vocab from {args.vocab_json}...")
    id_to_bytes = load_vocab(args.vocab_json)
    print(f"  Vocabulary size: {len(id_to_bytes)}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint, device=device, attn_impl=args.attn_impl
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Context length: {model.context_length}")
    print(f"  Attention impl: {args.attn_impl}")

    # Load prompt from .bin file
    if args.prompt_file is None:
        print("Error: --prompt-file is required.")
        print("Example: --prompt-file ./encoded_ID/openwebtext_encoded.bin --prompt-offset 0 --prompt-length 64")
        return

    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {args.prompt_file}")
        return

    # Load tokenized data from .bin file
    file_size = prompt_path.stat().st_size
    dtype_size = np.dtype(np.uint16).itemsize
    assert file_size % dtype_size == 0, f"File size {file_size} not aligned to uint16"
    data = np.memmap(args.prompt_file, dtype=np.uint16, mode="r",
                     shape=(file_size // dtype_size,))

    # Determine prompt length
    if args.prompt_length is not None:
        prompt_len = args.prompt_length
    else:
        prompt_len = min(64, model.context_length)

    offset = args.prompt_offset
    if offset + prompt_len > len(data):
        print(f"Warning: offset ({offset}) + prompt_length ({prompt_len}) exceeds data length ({len(data)}). "
              f"Truncating prompt_length to {len(data) - offset}.")
        prompt_len = len(data) - offset

    prompt_ids = data[offset:offset + prompt_len].tolist()
    prompt_text = decode_tokens(prompt_ids, id_to_bytes)
    print(f"  Prompt: {prompt_len} tokens from offset {offset}")
    print(f"  Prompt text: {repr(prompt_text[:100])}{'...' if len(prompt_text) > 100 else ''}")

    # Prepare output file
    out_file = None
    if args.output:
        out_file = open(args.output, "w", encoding="utf-8")
        out_file.write("=" * 60 + "\n")
        out_file.write(f"Generation (temp={args.temperature}, top-p={args.top_p})\n")
        out_file.write("=" * 60 + "\n\n")

    # Generate with real-time decoding
    print()
    print("=" * 60)
    print("  Generating...")
    print("=" * 60)
    print()

    # Buffer for decoded text
    decoded_buffer = []
    token_count = 0
    start_time = time.time()

    def on_token(token_id: int):
        nonlocal token_count
        token_count += 1
        # Decode this single token
        token_bytes = id_to_bytes.get(token_id, b"")
        token_text = token_bytes.decode("utf-8", errors="replace")
        decoded_buffer.append(token_text)
        # Print to console in real-time
        sys.stdout.write(token_text)
        sys.stdout.flush()
        # Write to file in real-time
        if out_file:
            out_file.write(token_text)
            out_file.flush()

    # Print prompt first
    sys.stdout.write("[PROMPT] ")
    sys.stdout.write(prompt_text)
    sys.stdout.write("\n[GEN] ")
    sys.stdout.flush()
    if out_file:
        out_file.write("[PROMPT]\n" + prompt_text + "\n[GEN]\n")

    generated_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=args.eos_id,
        device=device,
        on_token=on_token,
        use_cache=not args.no_cache,
        use_amp=args.amp,
    )

    elapsed = time.time() - start_time
    new_tokens = len(generated_ids) - len(prompt_ids)
    print()
    print()
    print(f"  Generated {new_tokens} tokens in {elapsed:.1f}s ({new_tokens/elapsed:.1f} tok/s)")

    if out_file:
        out_file.write(f"\n\n--- {new_tokens} tokens in {elapsed:.1f}s ---\n")
        out_file.close()
        print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
