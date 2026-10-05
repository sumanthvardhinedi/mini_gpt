"""
Phase 8: Cross-Entropy Loss, Training Loop, Evaluation, and Checkpointing for MiniGPT.

Educational Notes:
------------------
1. Next-Token Cross-Entropy Loss:
   The MiniGPT forward pass outputs `logits` of shape `(batch_size, sequence_length, vocab_size)`,
   while `targets` has shape `(batch_size, sequence_length)`.
   PyTorch's `nn.CrossEntropyLoss` expects 2D predictions `(N, vocab_size)` and 1D integer
   class labels `(N,)`, where `N = batch_size * sequence_length`.
   Flattening both tensors treats every position in every batch item as a next-token
   classification problem over `vocab_size` classes.

2. The 5-Step PyTorch Training Loop:
   For each batch `(input_ids, target_ids)`:
   a) `logits = model(input_ids)`               (Forward pass)
   b) `loss = compute_loss(logits, target_ids)` (Cross-entropy loss)
   c) `optimizer.zero_grad()`                   (Clear old gradients from previous step)
   d) `loss.backward()`                         (Backpropagate gradients through the network)
   e) `optimizer.step()`                        (Update weights using AdamW)

3. Evaluation Mode & `torch.no_grad()`:
   During validation, `model.eval()` disables stochastic layers like Dropout, and
   `torch.no_grad()` skips building the autograd computation graph to save memory and speed
   up evaluation. Afterward, `model.train()` is restored.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Deliberately small CPU-friendly configuration for training MiniGPT."""

    vocab_size: int = 100
    max_sequence_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    hidden_dim: int = 256
    dropout: float = 0.0
    sequence_length: int = 16
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 5
    device: str = "cpu"


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy next-token prediction loss.

    Args:
        logits: Model output tensor of shape `(batch_size, sequence_length, vocab_size)`
                or `(N, vocab_size)`.
        targets: Ground-truth next-token IDs of shape `(batch_size, sequence_length)`
                 or `(N,)`.

    Returns:
        Scalar loss tensor.
    """
    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.size()
        if targets.shape != torch.Size([batch_size, seq_len]):
            raise ValueError(
                f"Shape mismatch: logits (B={batch_size}, T={seq_len}, V={vocab_size}) "
                f"vs targets {targets.shape}."
            )
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len).long()
    elif logits.dim() == 2 and targets.dim() == 1:
        if logits.size(0) != targets.size(0):
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}."
            )
        logits_flat = logits
        targets_flat = targets.long()
    else:
        raise ValueError(
            f"Unsupported shapes for compute_loss: logits {logits.shape}, targets {targets.shape}."
        )

    criterion = nn.CrossEntropyLoss()
    return criterion(logits_flat, targets_flat)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> float:
    """
    Evaluate average cross-entropy loss over `dataloader` using `torch.no_grad()`.
    Switches `model` to `.eval()` mode during evaluation and restores its previous state.
    """
    was_training = getattr(model, "training", True)
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = compute_loss(logits, target_ids)
            total_loss += float(loss.item())
            num_batches += 1

    if was_training:
        model.train()

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
) -> None:
    """
    Save model weights, optimizer state, epoch index, and loss value to `path`.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "loss": float(loss),
    }
    torch.save(payload, str(checkpoint_path))


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load model weights and optional optimizer state from `path` and return metadata.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint: Dict[str, Any] = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
) -> float:
    """
    Perform a single forward -> loss -> zero_grad -> backward -> optimizer.step() update.
    """
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    logits = model(input_ids)
    loss = compute_loss(logits, target_ids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[float]]:
    """
    Run the complete training loop over `epochs` using AdamW, tracking `train_loss`
    and `val_loss` per epoch.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    model.to(device)
    model.train()

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in train_loader:
            batch_loss = train_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                device=device,
            )
            epoch_loss += batch_loss
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)
        history["train_loss"].append(avg_train_loss)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device=device)
            history["val_loss"].append(val_loss)

        if checkpoint_path is not None:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_train_loss,
                path=checkpoint_path,
            )

    return history
