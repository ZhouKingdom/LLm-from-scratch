"""
CS336 Assignment 1 - Optimizer & Training Helpers
===================================================
Training-related components:
  - AdamW optimizer (decoupled weight decay)
  - Cosine learning rate schedule with linear warmup
  - Global L2 gradient clipping
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


# =========================================================================
#  AdamW Optimizer
# =========================================================================

class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Follows Algorithm 1 from Loshchilov & Hutter (2019).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                state["step"] += 1
                t = state["step"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                # Update parameters (Adam step)
                denom = v_hat.sqrt().add_(eps)
                p.data.addcdiv_(m_hat, denom, value=-lr)

                # Decoupled weight decay
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


# =========================================================================
#  Cosine Learning Rate Schedule with Warmup
# =========================================================================

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing learning rate schedule with linear warmup.

    Args:
        it: Current iteration number.
        max_learning_rate: Maximum learning rate (alpha_max).
        min_learning_rate: Minimum / final learning rate (alpha_min).
        warmup_iters: Number of warm-up iterations (T_w).
        cosine_cycle_iters: Number of cosine annealing iterations (T_c).

    Returns:
        Learning rate at iteration ``it``.
    """
    if it < warmup_iters:
        # Linear warmup
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    else:
        # Post-annealing: constant min
        return min_learning_rate


# =========================================================================
#  Gradient Clipping
# =========================================================================

def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """Clip the combined L2 norm of all parameter gradients.

    Args:
        parameters: Iterable of trainable parameters.
        max_l2_norm: Maximum allowed L2 norm.
    """
    # Compute the total L2 norm of all gradients
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm) + 1e-6

    # Scale if needed
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in parameters:
            p.grad.data.mul_(scale)
