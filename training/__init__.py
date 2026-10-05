"""Training package for MiniGPT (Phases 8 & 10: Dataset, Training Loop, Metrics, and Experiment Runner)."""

from .dataset import LanguageModelDataset, create_dataloaders, create_datasets
from .experiment import (
    ExperimentConfig,
    load_experiment_checkpoint,
    load_experiment_config,
    restore_tokenizer_state,
    run_experiment,
    save_experiment_checkpoint,
    serialize_tokenizer_state,
    set_seed,
)
from .metrics import EpochMetrics, MetricsTracker, compute_perplexity
from .train import (
    TrainingConfig,
    compute_loss,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_model,
    train_step,
)

__all__ = [
    "LanguageModelDataset",
    "create_datasets",
    "create_dataloaders",
    "TrainingConfig",
    "compute_loss",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "train_step",
    "train_model",
    "EpochMetrics",
    "MetricsTracker",
    "compute_perplexity",
    "ExperimentConfig",
    "load_experiment_config",
    "set_seed",
    "serialize_tokenizer_state",
    "restore_tokenizer_state",
    "save_experiment_checkpoint",
    "load_experiment_checkpoint",
    "run_experiment",
]
