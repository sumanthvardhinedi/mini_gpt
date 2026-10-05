"""
Phase 10: Reproducible Training Experiment Runner for MiniGPT.

Orchestrates:
1. Deterministic seeding across Python `random`, `numpy`, and `torch`
2. Loading experiment configuration from JSON (`configs/small_gpt.json`)
3. Training the BPE tokenizer on `data/training.txt`
4. Encoding corpus into token IDs and splitting into train/validation datasets
5. Initializing `MiniGPT` and `AdamW` (with weight decay)
6. Running multi-epoch training, validation evaluation, and perplexity calculation
7. Saving the best checkpoint (with model state, optimizer state, epoch, train/val loss,
   config, tokenizer vocabulary, and tokenizer merge rules) and final checkpoint
8. Persisting structured epoch metrics to `checkpoints/training_metrics.json`
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from model.gpt import MiniGPT
from tokenizer.bpe_tokenizer import BPETokenizer
from training.dataset import create_dataloaders, create_datasets
from training.metrics import MetricsTracker
from training.train import evaluate, train_step

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def set_seed(seed: int) -> None:
    """
    Deterministically seed Python `random`, `numpy` (if available), and `torch`.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}.")

    random.seed(seed)
    if np is not None and hasattr(np, "random") and hasattr(np.random, "seed"):
        np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    """Complete configuration for a reproducible MiniGPT training experiment."""

    vocab_size: int = 96
    sequence_length: int = 12
    max_sequence_length: int = 32
    embedding_dim: int = 16
    num_heads: int = 4
    num_layers: int = 1
    hidden_dim: int = 32
    dropout: float = 0.0
    batch_size: int = 16
    learning_rate: float = 0.04
    epochs: int = 5
    weight_decay: float = 0.01
    train_ratio: float = 0.85
    seed: int = 42
    checkpoint_path: str = "checkpoints/best_model.pt"
    best_checkpoint_path: str = "checkpoints/best_model.pt"
    final_checkpoint_path: str = "checkpoints/final_model.pt"
    metrics_path: str = "checkpoints/training_metrics.json"
    data_path: str = "data/training.txt"
    max_tokens: Optional[int] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}.")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}.")
        if self.max_sequence_length < self.sequence_length:
            self.max_sequence_length = self.sequence_length
        if self.embedding_dim <= 0 or self.num_heads <= 0 or self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be positive and divisible by num_heads ({self.num_heads})."
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}.")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0 and 1, got {self.train_ratio}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        normalized = dict(data)
        if "random_seed" in normalized and "seed" not in normalized:
            normalized["seed"] = normalized.pop("random_seed")
        if "train_val_ratio" in normalized and "train_ratio" not in normalized:
            normalized["train_ratio"] = normalized.pop("train_val_ratio")
        if "checkpoint_path" in normalized and "best_checkpoint_path" not in normalized:
            normalized["best_checkpoint_path"] = normalized["checkpoint_path"]
        if "sequence_length" in normalized and "max_sequence_length" not in normalized:
            normalized["max_sequence_length"] = max(32, int(normalized["sequence_length"]) * 2)

        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in normalized.items() if k in valid_keys}
        return cls(**filtered)


def resolve_project_path(base_dir: Path, raw_path: Union[str, Path]) -> Path:
    """Resolve `raw_path` relative to `base_dir` unless already absolute."""
    p = Path(raw_path)
    return p if p.is_absolute() else (base_dir / p)


def encode_corpus_with_limit(
    text: str,
    tokenizer: BPETokenizer,
    max_tokens: Optional[int] = None,
) -> List[int]:
    """Encode `text` with `tokenizer`, optionally capping at `max_tokens` for fast CPU runs."""
    if max_tokens is not None:
        char_limit = max(max_tokens * 8, 2000)
        return tokenizer.encode(text[:char_limit])[:max_tokens]
    return tokenizer.encode(text)


def build_model_from_config(
    config: ExperimentConfig,
    tokenizer: BPETokenizer,
    device: Union[str, torch.device] = "cpu",
) -> MiniGPT:
    """Instantiate a `MiniGPT` model matching `config` and `tokenizer` vocabulary size."""
    actual_vocab_size = max(tokenizer.current_vocab_size, config.vocab_size)
    return MiniGPT(
        vocab_size=actual_vocab_size,
        max_sequence_length=config.max_sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load and validate an `ExperimentConfig` from a JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration JSON root must be an object.")
    return ExperimentConfig.from_dict(raw)


def serialize_tokenizer_state(tokenizer: BPETokenizer) -> Dict[str, Any]:
    """Serialize `BPETokenizer` vocabulary and merge rules into JSON/checkpoint-friendly primitives."""
    return tokenizer.to_state_dict()


def restore_tokenizer_state(state: Dict[str, Any]) -> BPETokenizer:
    """Reconstruct a `BPETokenizer` from serialized vocabulary and merge rules."""
    return BPETokenizer.from_state_dict(state)


def save_experiment_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: ExperimentConfig,
    tokenizer: BPETokenizer,
) -> Path:
    """
    Save complete experiment checkpoint including model weights, optimizer state,
    epoch, training/validation losses, configuration, and BPE tokenizer state.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tok_state = serialize_tokenizer_state(tokenizer)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "loss": float(val_loss),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "config": config.to_dict(),
        "tokenizer_vocab": tok_state["str_to_int"],
        "tokenizer_merges": tok_state["merges"],
        "tokenizer_state": tok_state,
    }
    torch.save(payload, str(target))
    return target


def load_experiment_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Dict[str, Any], BPETokenizer]:
    """
    Load an experiment checkpoint and reconstruct the saved `BPETokenizer`.
    Optionally restores `model` and `optimizer` states if passed.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {target}")

    checkpoint: Dict[str, Any] = torch.load(str(target), map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "tokenizer_state" in checkpoint:
        tokenizer = restore_tokenizer_state(checkpoint["tokenizer_state"])
    else:
        vocab = {str(k): int(v) for k, v in checkpoint["tokenizer_vocab"].items()}
        tokenizer = restore_tokenizer_state(
            {
                "vocab_size": len(vocab),
                "unk_token": "<UNK>",
                "str_to_int": vocab,
                "merges": checkpoint.get("tokenizer_merges", []),
            }
        )
    return checkpoint, tokenizer


def run_experiment(
    config: Union[ExperimentConfig, str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete, reproducible MiniGPT training experiment and persist
    best checkpoint, final checkpoint, and `training_metrics.json`.
    """
    project_root = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )

    if isinstance(config, (str, Path)):
        cfg_path = resolve_project_path(project_root, config)
        exp_config = load_experiment_config(cfg_path)
    elif isinstance(config, ExperimentConfig):
        exp_config = config
    else:
        raise TypeError("config must be an ExperimentConfig or path to a JSON file.")

    start_time = time.time()

    # 1. Deterministic seeding
    set_seed(exp_config.seed)

    # 2. Load training corpus
    data_file = resolve_project_path(project_root, exp_config.data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Training corpus not found: {data_file}")
    text = data_file.read_text(encoding="utf-8")
    if len(text.strip()) == 0:
        raise ValueError("Training corpus cannot be empty.")

    # 3. Train BPE tokenizer & encode text into token IDs
    bpe_training_slice = text[:6000] if len(text) > 6000 else text
    tokenizer = BPETokenizer(vocab_size=exp_config.vocab_size, training_text=bpe_training_slice)
    token_ids = encode_corpus_with_limit(text, tokenizer, exp_config.max_tokens)

    # 4. Create train/validation datasets and DataLoaders
    train_dataset, val_dataset = create_datasets(
        token_ids=token_ids,
        sequence_length=exp_config.sequence_length,
        train_ratio=exp_config.train_ratio,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=exp_config.batch_size,
        shuffle_train=False,
    )
    assert val_loader is not None

    # 5. Create MiniGPT model & AdamW optimizer
    model = build_model_from_config(exp_config, tokenizer, device=exp_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_config.learning_rate,
        weight_decay=exp_config.weight_decay,
    )

    best_ckpt_path = resolve_project_path(project_root, exp_config.best_checkpoint_path)
    final_ckpt_path = resolve_project_path(project_root, exp_config.final_checkpoint_path)
    metrics_file_path = resolve_project_path(project_root, exp_config.metrics_path)

    tracker = MetricsTracker()
    initial_eval_train_loss = evaluate(model, train_loader, device=exp_config.device)
    initial_eval_val_loss = evaluate(model, val_loader, device=exp_config.device)

    # 6. Training loop across epochs
    for epoch in range(1, exp_config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in train_loader:
            step_loss = train_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                device=exp_config.device,
            )
            epoch_train_loss += step_loss
            num_batches += 1

        avg_train_loss = epoch_train_loss / max(1, num_batches)
        val_loss = evaluate(model, val_loader, device=exp_config.device)

        _, is_best = tracker.record_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
        )

        if is_best:
            save_experiment_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                config=exp_config,
                tokenizer=tokenizer,
            )

    # 7. Save final checkpoint & training_metrics.json
    last_epoch_entry = tracker.epochs[-1]
    save_experiment_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=last_epoch_entry.epoch,
        train_loss=last_epoch_entry.train_loss,
        val_loss=last_epoch_entry.val_loss,
        config=exp_config,
        tokenizer=tokenizer,
    )
    tracker.save(metrics_file_path)

    elapsed = time.time() - start_time
    first_entry = tracker.epochs[0]

    return {
        "config": exp_config.to_dict(),
        "metrics": tracker.to_dict(),
        "pre_train_loss": initial_eval_train_loss,
        "pre_val_loss": initial_eval_val_loss,
        "initial_train_loss": first_entry.train_loss,
        "final_train_loss": last_epoch_entry.train_loss,
        "initial_val_loss": first_entry.val_loss,
        "final_val_loss": last_epoch_entry.val_loss,
        "initial_train_perplexity": first_entry.train_perplexity,
        "final_train_perplexity": last_epoch_entry.train_perplexity,
        "initial_val_perplexity": first_entry.val_perplexity,
        "final_val_perplexity": last_epoch_entry.val_perplexity,
        "best_epoch": tracker.best_epoch,
        "best_val_loss": tracker.best_val_loss,
        "epochs": exp_config.epochs,
        "num_parameters": model.num_parameters(),
        "training_time_sec": elapsed,
        "best_checkpoint_path": str(best_ckpt_path),
        "final_checkpoint_path": str(final_ckpt_path),
        "metrics_path": str(metrics_file_path),
        "model": model,
        "tokenizer": tokenizer,
    }
