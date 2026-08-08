"""
CS336 Assignment 1 - Data Loading & Batching
=============================================
Data-related helpers:
  - get_batch: sample a random batch of (inputs, targets) sequences
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of input sequences and corresponding targets.

    Args:
        dataset: 1D numpy array of token IDs.
        batch_size: Number of sequences per batch.
        context_length: Length of each sequence.
        device: PyTorch device string.

    Returns:
        (inputs, targets) each of shape (batch_size, context_length) as LongTensors.
    """
    # Random starting indices
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start+1, size=(batch_size,))

    # Build input and target sequences
    inputs = np.stack([dataset[s:s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1:s + context_length + 1] for s in starts])

    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(targets).long().to(device),
    )
