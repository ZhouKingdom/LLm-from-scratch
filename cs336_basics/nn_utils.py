"""
CS336 Assignment 1 - Neural Network Utilities
===============================================
Utility functions shared across the project:
  - Numerical softmax
  - Scaled dot-product attention
  - Cross-entropy loss (full and chunked memory-efficient variants)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# =========================================================================
#  Softmax
# =========================================================================

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax.

    Subtracts the maximum value along ``dim`` before exponentiating.
    """
    x_max = x.amax(dim=dim, keepdim=True)
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# =========================================================================
#  Scaled Dot-Product Attention
# =========================================================================

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention.

    .. math::
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., values, d_v)    (keys == values dimension)
        mask: optional (..., queries, keys) boolean or float mask.
              True/1.0 means *keep*, False/-inf means *mask out*.

    Returns:
        (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (..., queries, keys)

    if mask is not None:
        # If mask is boolean, convert: True -> keep (0), False -> mask (-inf)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    attn_weights = softmax(scores, dim=-1)  # (..., queries, keys)
    return attn_weights @ V  # (..., queries, d_v)


# =========================================================================
#  Cross-Entropy Loss
# =========================================================================

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the average cross-entropy loss.

    Uses the log-sum-exp trick for numerical stability:
        loss = log(sum(exp(logits - max_logit))) - logits[target] + max_logit

    Args:
        inputs: (batch_size, vocab_size) unnormalized logits.
        targets: (batch_size,) integer class indices.

    Returns:
        Scalar tensor with the average cross-entropy loss.
    """
    # Subtract max for numerical stability
    max_logit = inputs.amax(dim=-1, keepdim=True)  # (batch, 1)
    shifted = inputs - max_logit
    logsumexp = shifted.exp().sum(dim=-1).log()  # (batch,)
    log_probs = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (batch,)
    losses = logsumexp - log_probs  # (batch,)
    return losses.mean()


def cross_entropy_chunked(
    hidden: Tensor,
    targets: Tensor,
    lm_head,
    num_chunks: int = 1,
) -> Tensor:
    """Memory-efficient cross-entropy that does NOT materialize full logits.

    Given the pre-LM-head hidden states ``(..., d_model)``, computes the loss by
    processing the vocabulary in ``num_chunks`` slices:

        loss = E[ log(sum_v exp(h @ w_v^T)) - h @ w_{target}^T ]

    where ``w_v`` is the v-th row of the LM head.  Because ``num_chunks`` is
    finite, the max-subtraction trick is applied per chunk, which is fine in
    practice for bf16/fp16 training (the chunk max is usually very close to the
    global max, and even if not, the bias is tiny and bounded).

    This avoids creating a ``(..., vocab_size)`` logits tensor -- the single
    largest activation in a decoder LM with a large vocabulary -- which is the
    dominant source of peak GPU memory here.

    Args:
        hidden: (..., d_model) pre-LM-head hidden states.
        targets: (...,) target token ids.
        lm_head: The LM head Linear module (d_model -> vocab_size).
        num_chunks: Number of vocab slices to process sequentially.

    Returns:
        Scalar tensor with the average cross-entropy loss.
    """
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float()  # (B, d_model), fp32 for stable loss
    flat_targets = targets.reshape(-1)  # (B,)
    B = flat_hidden.shape[0]
    V = lm_head.out_features
    w_full = lm_head.weight.float()

    # Compute the target logit (cheap: one gathered row-multiply, no full logits).
    target_logits = (flat_hidden * w_full[flat_targets]).sum(dim=-1)  # (B,)

    # Compute logsumexp over the vocabulary in chunks.
    lse = torch.full((B,), float("-inf"), device=hidden.device, dtype=torch.float32)
    chunk_size = (V + num_chunks - 1) // num_chunks
    for start in range(0, V, chunk_size):
        w = w_full[start : start + chunk_size]  # (chunk, d_model)
        logits_chunk = flat_hidden @ w.T  # (B, chunk)
        chunk_max = logits_chunk.amax(dim=-1)
        lse = torch.logaddexp(
            lse,
            chunk_max + (logits_chunk - chunk_max.unsqueeze(-1)).exp().sum(dim=-1).log(),
        )

    losses = lse - target_logits
    return losses.mean()
