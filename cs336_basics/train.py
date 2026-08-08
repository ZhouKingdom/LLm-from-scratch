"""
CS336 Assignment 1 - Training Script (§5.3)
=============================================
Puts together all components: data loading, model, optimizer, loss,
checkpointing, and logging.

Usage:
python -m cs336_basics.train \
--train-data /root/autodl-tmp/data/owt_train_encoded.bin \
--val-data /root/autodl-tmp/data/owt_val_encoded.bin \
--vocab-size 32000 \
--d-model 768 \
--d-ff 2048 \
--num-layers 6 \
--num-heads 12 \
--batch-size 16 \
--gradient-accumulation-steps 4 \
--total-steps 10000 \
--max-lr 6e-4 \
--min-lr 6e-5 \
--warmup-iters 500 \
--gradient-clip 1.0 \
--save-dir /root/autodl-tmp/data/checkpoints/owt \
--device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.nn_utils import (
    cross_entropy,
    cross_entropy_chunked,
)
from cs336_basics.data import get_batch


def load_memmap(data_path: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    """Load a memory-mapped numpy array from a binary file.

    Supports both .npy files and raw binary files.
    """
    path = Path(data_path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    elif path.suffix == ".bin":
        # Raw binary file; infer length from file size
        file_size = path.stat().st_size
        dtype_size = np.dtype(dtype).itemsize
        assert file_size % dtype_size == 0, (
            f"File size {file_size} is not a multiple of dtype size {dtype_size}"
        )
        return np.memmap(path, dtype=dtype, mode="r", shape=(file_size // dtype_size,))
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def train_one_epoch(
    model: TransformerLM,
    optimizer: AdamW,
    train_data: np.ndarray,
    val_data: np.ndarray | None,
    batch_size: int,
    context_length: int,
    total_steps: int,
    warmup_iters: int,
    max_lr: float,
    min_lr: float,
    gradient_clip_norm: float | None,
    device: str,
    save_dir: str | None,
    gradient_accumulation_steps: int = 1,
    start_step: int = 0,
    log_interval: int = 10,
    val_interval: int = 100,
    save_interval: int = 1000,
    use_amp: bool = False,
    loss_chunks: int = 1,
    best_val_loss: float = float("inf"),
):
    """Run the full training loop with gradient accumulation support.

    The effective batch size is ``batch_size × gradient_accumulation_steps``.
    Gradients are accumulated over ``gradient_accumulation_steps`` micro-batches,
    then one optimizer step is taken.

    Args:
        model: The TransformerLM model.
        optimizer: AdamW optimizer.
        train_data: 1D numpy array of training token IDs.
        val_data: 1D numpy array of validation token IDs (optional).
        batch_size: Micro-batch size (per forward pass).
        context_length: Sequence length per example.
        total_steps: Total number of optimizer steps.
        warmup_iters: Number of linear warmup steps.
        max_lr: Maximum learning rate.
        min_lr: Minimum learning rate.
        gradient_clip_norm: Max L2 norm for gradient clipping (None = no clipping).
        device: PyTorch device string.
        save_dir: Directory to save checkpoints (None = no saving).
        gradient_accumulation_steps: Number of micro-batches to accumulate per optimizer step.
        start_step: Step to resume from (0 for fresh training).
        log_interval: Steps between logging.
        val_interval: Steps between validation evaluations.
        save_interval: Steps between checkpoints.
    """
    model.train()
    effective_batch_size = batch_size * gradient_accumulation_steps
    tokens_per_step = effective_batch_size * context_length
    total_tokens = total_steps * tokens_per_step

    print(f"{'='*60}")
    print(f"  Starting training")
    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Micro-batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Max LR: {max_lr}")
    print(f"  Min LR: {min_lr}")
    print(f"  Warmup steps: {warmup_iters}")
    print(f"{'='*60}")

    step = start_step
    best_val_loss = best_val_loss
    start_time = time.time()
    token_count = 0

    # Mixed-precision context (bf16 on Ampere+ GPUs).
    # Note: bf16 needs no gradient scaling (its exponent range matches fp32),
    # so we use plain backward/optimizer.step() below.
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", dtype=amp_dtype, enabled=False)
    )

    while step < total_steps:
        # Accumulate gradients over several micro-batches
        accumulated_loss = 0.0
        optimizer.zero_grad()

        for _ in range(gradient_accumulation_steps):
            # Sample a micro-batch
            inputs, targets = get_batch(train_data, batch_size, context_length, device)
            token_count += batch_size * context_length

            with autocast_ctx:
                if loss_chunks > 1:
                    # Memory-efficient path: keep hidden states (not full logits)
                    # and chunk the vocab projection for the cross-entropy.
                    hidden = model(inputs, return_hidden=True)
                    loss = cross_entropy_chunked(
                        hidden, targets, model.lm_head, num_chunks=loss_chunks
                    )
                else:
                    # Forward pass
                    logits = model(inputs)  # (batch, seq, vocab_size)
                    loss = cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                    )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass (accumulate gradients)
            loss.backward()
            accumulated_loss += loss.item()

        # Gradient clipping (applied to accumulated gradients)
        if gradient_clip_norm is not None:
            gradient_clipping(model.parameters(), gradient_clip_norm)

        # Update learning rate for this optimizer step
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_iters, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step
        optimizer.step()
        step += 1

        # Logging
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
            print(
                f"  Step {step:>6d}/{total_steps} | "
                f"loss {accumulated_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tokens_per_sec:.0f} | "
                f"elapsed {elapsed:.1f}s"
            )

        # Validation
        if val_data is not None and (step % val_interval == 0 or step == total_steps):
            val_loss = evaluate(
                model, val_data, batch_size, context_length, device,
                use_amp=use_amp, loss_chunks=loss_chunks,
            )
            model.train()
            print(f"  >>> Validation loss: {val_loss:.4f} | perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    save_checkpoint(model, optimizer, step, save_dir, best_val_loss, is_best=True)

        # Save checkpoint
        if save_dir and (step % save_interval == 0 or step == total_steps):
            save_checkpoint(model, optimizer, step, save_dir, best_val_loss)

    # Final stats
    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final loss: {accumulated_loss:.4f}")
    if val_data is not None:
        print(f"  Best validation loss: {best_val_loss:.4f} (perplexity: {math.exp(best_val_loss):.2f})")
    print(f"{'='*60}")


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 50,
    use_amp: bool = False,
    loss_chunks: int = 1,
) -> float:
    """Evaluate the model on validation data.

    Args:
        model: The model to evaluate.
        val_data: 1D numpy array of validation token IDs.
        batch_size: Batch size for evaluation.
        context_length: Context length for evaluation.
        device: PyTorch device string.
        num_batches: Number of batches to evaluate on.
        use_amp: Whether to run evaluation under bf16 autocast.
        loss_chunks: Number of vocab chunks for the memory-efficient loss.

    Returns:
        Average cross-entropy loss.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
        if device.startswith("cuda") and use_amp
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )

    with autocast_ctx:
        for _ in range(num_batches):
            inputs, targets = get_batch(val_data, batch_size, context_length, device)
            if loss_chunks > 1:
                hidden = model(inputs, return_hidden=True)
                loss = cross_entropy_chunked(
                    hidden, targets, model.lm_head, num_chunks=loss_chunks
                )
            else:
                logits = model(inputs)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
            total_loss += loss.item() * (batch_size * context_length)
            total_tokens += batch_size * context_length

    return total_loss / total_tokens


def save_checkpoint(
    model: TransformerLM,
    optimizer: AdamW,
    step: int,
    save_dir: str,
    best_val_loss: float = float("inf"),
    is_best: bool = False,
):
    """Save a model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    filename = "best_checkpoint.pt" if is_best else f"checkpoint_step_{step}.pt"
    path = os.path.join(save_dir, filename)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
            "d_model": model.d_model,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "rope_theta": model.rope_theta,
        },
    }
    torch.save(checkpoint, path)
    print(f"  >>> Checkpoint saved: {path}")


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a checkpoint and return its contents."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    # Data
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data (.bin or .npy)")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation data (.bin or .npy)")

    # Model architecture
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--d-model", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1344, help="Feed-forward inner dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Ablation flags
    parser.add_argument("--use-post-norm", action="store_true", help="Use post-norm instead of pre-norm")
    parser.add_argument("--remove-rmsnorm", action="store_true", help="Remove all RMSNorm layers")
    parser.add_argument("--remove-rope", action="store_true", help="Remove RoPE position embeddings")
    parser.add_argument("--ffn-type", type=str, default="swiglu", choices=["swiglu", "silu"],
                        help="Feed-forward network type")

    # Performance / memory
    parser.add_argument("--attn-impl", type=str, default="manual", choices=["manual", "fused"],
                        help="Attention implementation: 'manual' (reference, matches tests) or "
                             "'fused' (FlashAttention / memory-efficient, faster + less memory)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable bf16 mixed precision (autocast) to cut activation memory in half")
    parser.add_argument("--loss-chunks", type=int, default=1,
                        help="Compute the cross-entropy loss over N vocabulary chunks instead of "
                             "materializing the full logits tensor. Values >1 cut the largest "
                             "activation (the (batch*seq, vocab) logits) by a factor of N.")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch size (per forward pass)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of micro-batches to accumulate before each optimizer step")
    parser.add_argument("--total-steps", type=int, default=5000, help="Total optimizer steps")
    parser.add_argument("--max-lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-iters", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm (0 = no clip)")

    # AdamW hyperparameters
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam-eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight-decay", type=float, default=1e-1, help="Weight decay")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Automatically resume from the most recent checkpoint in --save-dir "
                             "(i.e. the one with the largest step number)")

    # I/O
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (steps)")
    parser.add_argument("--val-interval", type=int, default=100, help="Validation interval (steps)")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save interval (steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading training data...")
    train_data = load_memmap(args.train_data)
    print(f"  Training data: {len(train_data):,} tokens")

    val_data = None
    if args.val_data:
        print("Loading validation data...")
        val_data = load_memmap(args.val_data)
        print(f"  Validation data: {len(val_data):,} tokens")

    # Create model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_post_norm=args.use_post_norm,
        remove_rmsnorm=args.remove_rmsnorm,
        remove_rope=args.remove_rope,
        ffn_type=args.ffn_type,
        attn_impl=args.attn_impl,
    ).to(device)

    # Resume from checkpoint
    start_step = 0
    resume_path = args.resume
    if args.resume_latest:
        if not args.save_dir or not os.path.isdir(args.save_dir):
            raise SystemExit(
                f"--resume-latest requires an existing --save-dir, got {args.save_dir!r}"
            )
        ckpt_files = [
            f for f in os.listdir(args.save_dir)
            if f.startswith("checkpoint_step_") and f.endswith(".pt")
        ]
        if not ckpt_files:
            raise SystemExit(
                f"No checkpoint_step_*.pt found in {args.save_dir!r} -- nothing to resume from."
            )
        # Pick the checkpoint with the largest step number.
        resume_path = max(
            ckpt_files,
            key=lambda f: int(f[len("checkpoint_step_") : -len(".pt")]),
        )
        resume_path = os.path.join(args.save_dir, resume_path)
        print(f"  --resume-latest: found {resume_path}")

    best_val_loss = float("inf")
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed at step {start_step}")
        if best_val_loss != float("inf"):
            print(f"  Previous best validation loss: {best_val_loss:.4f}")

    # Count non-embedding parameters
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(p.numel() for p in model.token_embeddings.parameters())
    non_embedding_params = total_params - embedding_params
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-embedding parameters: {non_embedding_params:,}")

    # Create optimizer (load state if resuming)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if resume_path:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Train
    train_one_epoch(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        total_steps=args.total_steps,
        warmup_iters=args.warmup_iters,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        gradient_clip_norm=args.gradient_clip if args.gradient_clip > 0 else None,
        device=device,
        save_dir=args.save_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        start_step=start_step,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        use_amp=args.amp,
        loss_chunks=args.loss_chunks,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()
