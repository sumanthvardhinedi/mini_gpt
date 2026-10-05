"""
Phase 8: Language Model Dataset, Train/Val Split, and DataLoaders for MiniGPT.

Educational Notes:
------------------
1. Next-Token Prediction Shifting:
   A decoder-only GPT model is trained self-supervised on raw text by predicting the
   immediately following token at every position in a context window of length `T`.
   Given token stream `[10, 20, 30, 40, 50]` and `sequence_length = 3`:
   - Window 0: `input_ids = [10, 20, 30]`, `target_ids = [20, 30, 40]`
     (When seeing `[10]`, predict `20`; when seeing `[10, 20]`, predict `30`;
      when seeing `[10, 20, 30]`, predict `40`.)
   - Window 1: `input_ids = [20, 30, 40]`, `target_ids = [30, 40, 50]`

2. Contiguous Train/Validation Split:
   Unlike image classification where individual examples are shuffled before splitting,
   sliding-window text chunks overlap heavily. Randomly mixing sliding windows would leak
   identical sentences into both training and validation sets!
   Instead, we split the raw 1D token stream sequentially at `int(len(token_ids) * train_ratio)`:
   the first 90% of the text forms `train_dataset`, and the final 10% forms `val_dataset`.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset


class LanguageModelDataset(Dataset):
    """
    PyTorch Dataset that slices a 1D stream of token IDs into `(input_ids, target_ids)`
    windows of length `sequence_length`, where `target_ids` is shifted right by 1 token.
    """

    def __init__(
        self,
        token_ids: Union[List[int], torch.Tensor],
        sequence_length: int = 128,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}.")

        if isinstance(token_ids, list):
            if len(token_ids) == 0:
                raise ValueError("token_ids cannot be empty.")
            tensor_ids = torch.tensor(token_ids, dtype=torch.long)
        elif isinstance(token_ids, torch.Tensor):
            if token_ids.dim() != 1:
                raise ValueError(f"token_ids tensor must be 1D, got {token_ids.dim()}D.")
            tensor_ids = token_ids.long()
        else:
            raise TypeError("token_ids must be a list of ints or a 1D torch.Tensor.")

        if tensor_ids.numel() <= sequence_length:
            raise ValueError(
                f"token_ids length ({tensor_ids.numel()}) must be strictly greater than "
                f"sequence_length ({sequence_length}) to form at least one (input, target) pair."
            )

        self.token_ids: torch.Tensor = tensor_ids
        self.sequence_length: int = sequence_length

    def __len__(self) -> int:
        """Return the number of sliding-window examples available."""
        return self.token_ids.numel() - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the `(input_ids, target_ids)` pair starting at position `idx`.
        Both tensors have shape `(sequence_length,)`.
        """
        num_items = len(self)
        if idx < 0:
            idx += num_items
        if idx < 0 or idx >= num_items:
            raise IndexError(f"Dataset index {idx} out of range [0, {num_items - 1}].")

        input_ids = self.token_ids[idx : idx + self.sequence_length]
        target_ids = self.token_ids[idx + 1 : idx + self.sequence_length + 1]
        return input_ids, target_ids


def create_datasets(
    token_ids: Union[List[int], torch.Tensor],
    sequence_length: int = 128,
    train_ratio: float = 0.9,
) -> Tuple[LanguageModelDataset, LanguageModelDataset]:
    """
    Deterministically split a contiguous token stream into training and validation datasets.
    Validation tokens always come strictly after training tokens in the stream.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0.0 and 1.0 (exclusive), got {train_ratio}.")

    total_len = len(token_ids) if isinstance(token_ids, list) else token_ids.numel()
    split_idx = int(total_len * train_ratio)

    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_dataset = LanguageModelDataset(train_tokens, sequence_length=sequence_length)
    val_dataset = LanguageModelDataset(val_tokens, sequence_length=sequence_length)
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: LanguageModelDataset,
    val_dataset: Optional[LanguageModelDataset] = None,
    batch_size: int = 8,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training (with optional shuffling) and validation
    (always unshuffled for deterministic evaluation).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader
