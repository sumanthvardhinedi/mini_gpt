"""
MiniGPT Phase 10: CLI Training Script.

Usage from the `mini-gpt` directory:
    python scripts/train.py
    python scripts/train.py --config configs/small_gpt.json
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.experiment import load_experiment_config, run_experiment  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the MiniGPT training experiment."""
    parser = argparse.ArgumentParser(
        description="Run a reproducible MiniGPT training experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/small_gpt.json",
        help="Path to the experiment JSON configuration file (default: configs/small_gpt.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load the CLI configuration, execute the training experiment, and print summary metrics."""
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = PROJECT_ROOT / config_path
        if candidate.exists():
            config_path = candidate

    config = load_experiment_config(config_path)
    summary = run_experiment(config=config, base_dir=PROJECT_ROOT)

    print("=== MiniGPT Phase 10: Training Experiment Complete ===")
    print(f"Epochs              : {summary['epochs']}")
    print(f"Parameters          : {summary['num_parameters']}")
    print(f"Initial Train Loss  : {summary['initial_train_loss']:.4f} (PPL: {summary['initial_train_perplexity']:.2f})")
    print(f"Final Train Loss    : {summary['final_train_loss']:.4f} (PPL: {summary['final_train_perplexity']:.2f})")
    print(f"Initial Val Loss    : {summary['initial_val_loss']:.4f} (PPL: {summary['initial_val_perplexity']:.2f})")
    print(f"Final Val Loss      : {summary['final_val_loss']:.4f} (PPL: {summary['final_val_perplexity']:.2f})")
    print(f"Best Val Epoch      : {summary['best_epoch']} (Val Loss: {summary['best_val_loss']:.4f})")
    print(f"Training Time (s)   : {summary['training_time_sec']:.2f}s")
    print(f"Best Checkpoint     : {summary['best_checkpoint_path']}")
    print(f"Metrics JSON        : {summary['metrics_path']}")
    return summary


if __name__ == "__main__":
    main()
