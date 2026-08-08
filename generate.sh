#!/bin/bash
set -euo pipefail

# =============================================================================
#  Text generation script (mirrors train.sh)
#  Uses the project's uv environment (Python >=3.11, torch 2.6).
#
#  Edit the variables below, then run:
#      ./generate.sh
# =============================================================================

# --- Paths ----------------------------------------------------------------
# Model checkpoint (best_checkpoint.pt = lowest validation loss so far)
CHECKPOINT=/root/autodl-tmp/data/checkpoints/owt/best_checkpoint.pt

# BPE vocab (for decoding token ids back to text)
VOCAB_JSON=/root/cs336-lab1/assignment1-basics-main/data/bpe_outputs/openwebtext_vocab.json

# Tokenized prompt source (.bin, uint16 token ids) - the generator samples a
# slice of this file as the prompt.
PROMPT_FILE=/root/autodl-tmp/data/owt_val_encoded.bin

# Output file for the generated text
OUTPUT=/root/cs336-lab1/assignment1-basics-main/gen/out.txt

# --- Prompt ---------------------------------------------------------------
# Where in the prompt file to start reading (in tokens)
PROMPT_OFFSET=0
# How many tokens to use as the prompt (default: model context length)
PROMPT_LENGTH=64

# --- Generation -----------------------------------------------------------
# Total tokens to generate, INCLUDING the prompt (prompt + new = max-tokens)
MAX_TOKENS=256
# Sampling temperature (lower = more deterministic; 0 = greedy)
TEMPERATURE=0.8
# Nucleus (top-p) sampling threshold (0.0 - 1.0)
TOP_P=0.9
# EOS token id to stop at (0 = <|endoftext|>); leave empty to disable
EOS_ID=0

# --- Performance ----------------------------------------------------------
# Device: auto / cuda / cpu / mps
DEVICE=cuda
# 'fused' = FlashAttention (faster on Ampere+), 'manual' = reference
ATTN_IMPL=fused
# Enable bf16 autocast for generation (faster on Ampere+ GPUs)
USE_AMP=1
# Disable the KV cache (slower; only useful for debugging). 0 = use cache.
NO_CACHE=0

# --- Build the command ----------------------------------------------------
ARGS="--checkpoint $CHECKPOINT \
--vocab-json $VOCAB_JSON \
--prompt-file $PROMPT_FILE \
--prompt-offset $PROMPT_OFFSET \
--prompt-length $PROMPT_LENGTH \
--max-tokens $MAX_TOKENS \
--temperature $TEMPERATURE \
--top-p $TOP_P \
--device $DEVICE \
--attn-impl $ATTN_IMPL \
--output $OUTPUT"

if [ -n "$EOS_ID" ]; then
    ARGS="$ARGS --eos-id $EOS_ID"
fi
if [ "$USE_AMP" = "1" ]; then
    ARGS="$ARGS --amp"
fi
if [ "$NO_CACHE" = "1" ]; then
    ARGS="$ARGS --no-cache"
fi

echo "============================================================"
echo "  Generating text"
echo "  Checkpoint : $CHECKPOINT"
echo "  Prompt     : $PROMPT_LENGTH tokens @ offset $PROMPT_OFFSET"
echo "  Max tokens : $MAX_TOKENS (incl. prompt)"
echo "  Temp / top-p : $TEMPERATURE / $TOP_P"
echo "  Device     : $DEVICE ($ATTN_IMPL, amp=$USE_AMP)"
echo "============================================================"

mkdir -p "$(dirname "$OUTPUT")"

# shellcheck disable=SC2086
uv run python -m cs336_basics.generate $ARGS
