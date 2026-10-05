"""
MiniGPT Phase 11: Evaluation, Visualization, and Generation Demo CLI.

Usage from the `mini-gpt` directory:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --max-new-tokens 16 --temperature 0.8 --top-k 5 --seed 42
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import run_evaluation_pipeline  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for MiniGPT Phase 11 evaluation and generation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MiniGPT checkpoint, plot training curves, and generate sample text."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to the saved model checkpoint (default: checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="checkpoints/training_metrics.json",
        help="Path to training_metrics.json (default: checkpoints/training_metrics.json).",
    )
    parser.add_argument(
        "--curves-output",
        type=str,
        default="outputs/training_curves.png",
        help="Output path for training curves PNG (default: outputs/training_curves.png).",
    )
    parser.add_argument(
        "--generations-output",
        type=str,
        default="outputs/generations.txt",
        help="Output path for generated text report (default: outputs/generations.txt).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of new subword tokens to generate per prompt (default: 16).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (> 0) for top-k generation (default: 0.8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K sampling cutoff (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the Phase 11 evaluation, visualization, and generation pipeline from CLI arguments."""
    args = parse_args(argv)
    summary = run_evaluation_pipeline(
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        curves_output_path=args.curves_output,
        generations_output_path=args.generations_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        base_dir=PROJECT_ROOT,
    )

    print("=== MiniGPT Phase 11: Evaluation & Generation Summary ===")
    print(f"Checkpoint Epoch      : {summary['epoch']}")
    print(f"Trainable Parameters  : {summary['num_parameters']}")
    print(f"Validation Loss       : {summary['val_loss']:.4f}")
    print(f"Validation Perplexity : {summary['val_perplexity']:.2f}")
    print(f"Training Curves PNG   : {summary['curves_path']}")
    print(f"Generations Report    : {summary['generations_path']}")
    return summary


if __name__ == "__main__":
    main()
