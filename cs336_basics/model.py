"""
CS336 Assignment 1 - Transformer Language Model
=================================================
This module implements the Transformer model components from scratch:
  - Linear, Embedding, RMSNorm
  - SwiGLU feed-forward network, Rotary Position Embeddings (RoPE)
  - Multi-head self-attention (with RoPE + optional KV cache)
  - Pre-norm Transformer block, full Transformer LM

Utility functions live in dedicated modules:
  - ``nn_utils.py``: softmax, scaled dot-product attention, cross-entropy loss
  - ``optimizer.py``: AdamW, LR schedule, gradient clipping
  - ``data.py``: data batching
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cs336_basics.nn_utils import scaled_dot_product_attention


# =========================================================================
# 3.4.2  Linear Module
# =========================================================================

class Linear(nn.Module):
    """A linear transformation without bias: y = x @ W.T  (row-vector convention).

    Stores weight of shape ``(out_features, in_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix: (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Truncated normal: N(0, 2/(d_in + d_out)), clipped to [-3σ, 3σ]
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) -> (..., out_features)
        return x @ self.weight.T

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# =========================================================================
# 3.4.3  Embedding Module
# =========================================================================

class Embedding(nn.Module):
    """A simple embedding lookup table.

    Stores weight of shape ``(num_embeddings, embedding_dim)``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Truncated normal: N(0, 1), clipped to [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (..., seq_len)  ->  (..., seq_len, embedding_dim)
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


# =========================================================================
# 3.5.1  RMSNorm
# =========================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    .. math::
        RMSNorm(a_i) = (a_i / RMS(a)) * g_i

    where  RMS(a) = sqrt(mean(a_i^2) + eps).
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # Upcast to float32 for numerical stability
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x * self.weight.to(torch.float32)

        return x.to(in_dtype)


# =========================================================================
# 3.5.2  Position-wise Feed-Forward Network (SwiGLU)
# =========================================================================

class SiLU(nn.Module):
    """SiLU (Swish) activation: x * sigmoid(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class PositionWiseFeedForward(nn.Module):
    """SwiGLU feed-forward network.

    .. math::
        FFN(x) = W_2 (SiLU(W_1 x) ⊙ W_3 x)

    where  d_ff ≈ (8/3) * d_model  (rounded to multiple of 64).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_model)
        # w1: (d_ff, d_model), w3: (d_ff, d_model), w2: (d_model, d_ff)
        gate = self.silu(self.w1(x))       # (..., d_ff)
        hidden = gate * self.w3(x)          # (..., d_ff)
        return self.w2(hidden)              # (..., d_model)


class SiLUFeedForward(nn.Module):
    """SiLU feed-forward network (without gating), for ablation studies.

    .. math::
        FFN_{SiLU}(x) = W_2 SiLU(W_1 x)

    Uses d_ff = 4 * d_model to match parameter count of SwiGLU.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.silu(self.w1(x)))


# =========================================================================
# 3.5.3  Rotary Position Embeddings (RoPE)
# =========================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Rotates pairs of embedding elements by angles that depend on position.
    No learnable parameters -- uses pre-computed cos/sin buffers.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Pre-compute cos and sin for all positions up to max_seq_len
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (max_seq_len,)
        # freqs: (d_k // 2,)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        # angles: (max_seq_len, d_k // 2)
        angles = pos[:, None] * freqs[None, :]  # outer product

        # Store as non-persistent buffers (not saved in state_dict)
        self.register_buffer("_cos", angles.cos(), persistent=False)
        self.register_buffer("_sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Tensor of shape ``(..., seq_len, d_k)``.
            token_positions: Long tensor of shape ``(..., seq_len)`` with positions.

        Returns:
            Tensor of same shape as ``x`` with RoPE applied.
        """
        # Gather cos/sin for the given positions
        # token_positions: (..., seq_len)  -> need to index into (max_seq_len, d_k//2)
        cos = self._cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self._sin[token_positions]  # (..., seq_len, d_k//2)

        # Reshape x into pairs
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., seq_len, d_k//2, 2)

        # Apply rotation
        x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]  # each (..., seq_len, d_k//2)
        x_rot0 = x0 * cos - x1 * sin
        x_rot1 = x0 * sin + x1 * cos

        out = torch.stack([x_rot0, x_rot1], dim=-1)  # (..., seq_len, d_k//2, 2)
        out = out.flatten(-2)                         # (..., seq_len, d_k)
        return out.to(x.dtype)


# =========================================================================
# 3.5.4  Multi-Head Self-Attention
# =========================================================================

class MultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE.

    Projects input to Q, K, V via single matrices, splits into heads,
    applies scaled dot-product attention (with causal masking), concatenates,
    and projects back.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        rope_theta: float | None = None,
        attn_impl: str = "manual",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert attn_impl in ("manual", "fused"), f"Unknown attn_impl: {attn_impl}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_impl = attn_impl
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Q, K, V projections -- each of shape (d_model, d_model)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Optional RoPE
        self.rope: RotaryPositionalEmbedding | None = None
        if rope_theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len, device=device)

        # Causal mask buffer (non-persistent)
        if max_seq_len is not None:
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
            self.register_buffer("_causal_mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        cache: dict[str, Tensor | None] | None = None,
    ) -> Tensor:
        """Forward pass of multi-head self-attention.

        Args:
            x: (batch_size, seq_len, d_model)
            token_positions: (batch_size, seq_len) or None. Required if RoPE is used.
            cache: Optional KV-cache dict with keys ``"k"``/``"v"`` (each
                ``(batch, num_heads, seq_so_far, d_k)`` or ``None``). When
                provided, the layer caches its key/value tensors so that only
                the newly added tokens need to be processed on the next call
                (incremental / autoregressive generation). When ``None``, the
                full sequence is processed normally (training path).

        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Reshape to (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # 3. Apply RoPE to Q and K if enabled
        if self.rope is not None:
            if token_positions is None:
                # Default: positions 0..seq_len-1 broadcasted
                token_positions = (
                    torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                )
            # Apply RoPE per head: (batch, num_heads, seq, d_k)
            Q = self.rope(Q, token_positions[:, None, :].expand(-1, self.num_heads, -1))
            K = self.rope(K, token_positions[:, None, :].expand(-1, self.num_heads, -1))

        # Incremental (KV-cache) path used during autoregressive generation.
        if cache is not None:
            past_len = cache["k"].shape[2] if cache["k"] is not None else 0
            if past_len > 0:
                K = torch.cat([cache["k"], K], dim=2)
                V = torch.cat([cache["v"], V], dim=2)
            cache["k"] = K
            cache["v"] = V
            if seq_len == 1:
                # Single new token: it attends to every cached key (all past
                # positions + itself), so no masking is needed.
                attn_output = scaled_dot_product_attention(Q, K, V, mask=None)
            else:
                # Prefill of several new tokens: causal masking is required,
                # where query i may attend to keys 0 .. past_len + i.
                total_len = K.shape[2]
                mask = torch.zeros(seq_len, total_len, dtype=torch.bool, device=K.device)
                for i in range(seq_len):
                    mask[i, : past_len + i + 1] = True
                attn_output = scaled_dot_product_attention(Q, K, V, mask=mask[None, None])
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.output_proj(attn_output)

        # 4. Causal mask
        causal_mask = self._causal_mask[:seq_len, :seq_len]  # (seq, seq)
        # Expand to (batch, num_heads, seq, seq) for broadcasting
        causal_mask = causal_mask[None, None, :, :]

        # 5. Scaled dot-product attention
        if self.attn_impl == "fused":
            # Fused FlashAttention / memory-efficient attention.
            # Avoids materializing the full (batch, heads, seq, seq) scores
            # matrix, cutting activation memory from O(n^2) to O(n).
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True
            )
        else:
            attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # attn_output: (batch, num_heads, seq, d_v)

        # 6. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 7. Output projection
        return self.output_proj(attn_output)


# =========================================================================
# 3.6  Pre-norm Transformer Block
# =========================================================================

class TransformerBlock(nn.Module):
    """A Transformer block with configurable normalization strategy.

    Pre-norm (default):
        y = x + MultiHeadSelfAttention(RMSNorm(x))
        z = y + FFN(RMSNorm(y))

    Post-norm:
        y = RMSNorm(x + MultiHeadSelfAttention(x))
        z = RMSNorm(y + FFN(y))

    No norm:
        y = x + MultiHeadSelfAttention(x)
        z = y + FFN(y)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        use_post_norm: bool = False,
        remove_rmsnorm: bool = False,
        ffn_type: str = "swiglu",
        attn_impl: str = "manual",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.use_post_norm = use_post_norm
        self.remove_rmsnorm = remove_rmsnorm

        # RMSNorm layers (only created if not removed)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if not remove_rmsnorm else nn.Identity()
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if not remove_rmsnorm else nn.Identity()

        self.attn = MultiheadSelfAttention(
            d_model, num_heads, max_seq_len=max_seq_len, rope_theta=rope_theta,
            attn_impl=attn_impl, device=device, dtype=dtype,
        )

        # FFN type: SwiGLU (default) or SiLU-only (for ablation)
        if ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = PositionWiseFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        cache: dict[str, Tensor | None] | None = None,
    ) -> Tensor:
        if self.use_post_norm:
            # Post-norm: RMSNorm(x + sublayer(x))
            residual = x
            x = self.attn(x, token_positions=token_positions, cache=cache)
            x = self.ln1(residual + x)

            residual = x
            x = self.ffn(x)
            x = self.ln2(residual + x)
        else:
            # Pre-norm (default): x + sublayer(RMSNorm(x))
            residual = x
            x = self.ln1(x)
            x = self.attn(x, token_positions=token_positions, cache=cache)
            x = residual + x

            residual = x
            x = self.ln2(x)
            x = self.ffn(x)
            x = residual + x

        return x


# =========================================================================
# 3.6  Full Transformer Language Model
# =========================================================================

class TransformerLM(nn.Module):
    """Full decoder-only Transformer language model.

    Architecture:
        token embeddings -> [num_layers TransformerBlock] -> final RMSNorm -> LM head

    Supports ablation configurations:
        - remove_rmsnorm: remove all RMSNorm layers
        - use_post_norm: use post-norm instead of pre-norm
        - remove_rope: disable Rotary Position Embeddings
        - ffn_type: "swiglu" (default) or "silu"
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_post_norm: bool = False,
        remove_rmsnorm: bool = False,
        remove_rope: bool = False,
        ffn_type: str = "swiglu",
        attn_impl: str = "manual",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta if not remove_rope else None

        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length,
                rope_theta=self.rope_theta,
                use_post_norm=use_post_norm,
                remove_rmsnorm=remove_rmsnorm,
                ffn_type=ffn_type,
                attn_impl=attn_impl,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final normalization (removed if remove_rmsnorm) and LM head
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if not remove_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: Tensor,
        return_hidden: bool = False,
        cache: list[dict[str, Tensor | None]] | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len)  with seq_len <= context_length.
            return_hidden: If True, return the pre-LM-head hidden states
                ``(batch_size, seq_len, d_model)`` instead of the logits.
                This lets the training loop compute the cross-entropy loss
                on chunks of the vocabulary, avoiding materializing the full
                ``(batch_size, seq_len, vocab_size)`` logits tensor (the single
                largest activation) and greatly reducing peak GPU memory.
            cache: Optional list of per-layer KV-cache dicts (one ``dict`` with
                ``"k"``/``"v"`` keys per layer), used for incremental
                autoregressive generation. When provided, each layer reuses its
                cached keys/values so only the new tokens are computed.

        Returns:
            (batch_size, seq_len, vocab_size) logits, or
            (batch_size, seq_len, d_model) hidden states if ``return_hidden``.
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        x = self.token_embeddings(token_ids)  # (batch, seq, d_model)

        # Token positions for RoPE. In the incremental (KV-cache) path, the
        # current step's absolute positions continue from wherever the cache
        # already ended, so RoPE angles stay correct for every new token.
        if cache is not None and cache[0]["k"] is not None:
            # cache[0]["k"] holds the keys from all PREVIOUS calls; the current
            # step's tokens come right after them.
            start_pos = cache[0]["k"].shape[2]
            token_positions = torch.arange(start_pos, start_pos + seq_len, device=token_ids.device)
        else:
            token_positions = torch.arange(seq_len, device=token_ids.device)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        # Pass through Transformer blocks
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, token_positions=token_positions, cache=layer_cache)

        # Final RMSNorm
        x = self.ln_final(x)

        if return_hidden:
            return x

        # LM head (output projection)
        logits = self.lm_head(x)  # (batch, seq, vocab_size)

        return logits

