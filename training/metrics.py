"""
Phase 10: Training Metrics and Perplexity Tracking for MiniGPT.

Educational Notes:
------------------
1. What is Perplexity?
   Perplexity is the exponential of the cross-entropy loss:
       `perplexity = math.exp(loss)`
   Intuitively, if a language model has a perplexity of `K`, it is as uncertain about
   the next token as if it were choosing uniformly at random among `K` equally likely
   tokens. For an untrained model over a vocabulary of size `V`, initial loss is near
   `ln(V)` and initial perplexity is near `V`. As the model learns, both loss and
   perplexity decrease toward `1.0`.

2. Numerical Overflow Protection:
   Early in training or under bad hyperparameters, `loss` can occasionally be large
   (e.g., `loss > 709` overflows 64-bit IEEE-754 floating point `math.exp`). Clamping
   `loss` to a safe ceiling (`max_loss = 20.0`, where `exp(20) ≈ 4.85e8`) prevents
   `OverflowError` while preserving exact values in the normal training range.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


MAX_SAFE_EXP_LOSS: float = 20.0


def compute_perplexity(loss: float, max_loss: float = MAX_SAFE_EXP_LOSS) -> float:
    """
    Calculate perplexity `math.exp(loss)` with safe numerical overflow protection.

    Args:
        loss: Non-negative cross-entropy loss value.
        max_loss: Maximum exponent value before clamping to prevent overflow.

    Returns:
        Perplexity as a float.
    """
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise TypeError(f"loss must be a numeric value, got {type(loss).__name__}.")
    loss_val = float(loss)
    if math.isnan(loss_val):
        raise ValueError("loss cannot be NaN.")
    if loss_val < 0.0:
        raise ValueError(f"loss must be non-negative, got {loss_val}.")

    clamped_loss = min(loss_val, float(max_loss))
    try:
        return float(math.exp(clamped_loss))
    except OverflowError:
        return float("inf")


@dataclass
class EpochMetrics:
    """Metrics recorded at the end of a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_perplexity: float
    val_perplexity: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


class MetricsTracker:
    """
    Tracks per-epoch training loss, validation loss, and perplexity, identifies the
    best validation epoch, and persists metrics to JSON (`checkpoints/training_metrics.json`).
    """

    def __init__(self) -> None:
        self.epochs: List[EpochMetrics] = []
        self.best_val_loss: float = float("inf")
        self.best_epoch: Optional[int] = None

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> Tuple[EpochMetrics, bool]:
        """
        Record metrics for `epoch`, compute train/validation perplexities, and return
        `(epoch_metrics, is_new_best)` indicating whether `val_loss` improved.
        """
        if epoch <= 0:
            raise ValueError(f"epoch must be positive, got {epoch}.")

        train_ppl = compute_perplexity(train_loss)
        val_ppl = compute_perplexity(val_loss)

        entry = EpochMetrics(
            epoch=int(epoch),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            train_perplexity=float(train_ppl),
            val_perplexity=float(val_ppl),
        )
        self.epochs.append(entry)

        is_new_best = float(val_loss) < self.best_val_loss
        if is_new_best:
            self.best_val_loss = float(val_loss)
            self.best_epoch = int(epoch)

        return entry, is_new_best

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics in the canonical `{"epochs": [...]}` JSON structure."""
        return {"epochs": [e.to_dict() for e in self.epochs]}

    def save(self, path: Union[str, Path]) -> Path:
        """Persist tracked epoch metrics to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MetricsTracker":
        """Load a `MetricsTracker` instance from a saved JSON file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Metrics file not found: {target}")

        raw = json.loads(target.read_text(encoding="utf-8"))
        tracker = cls()
        for item in raw.get("epochs", []):
            entry = EpochMetrics(
                epoch=int(item["epoch"]),
                train_loss=float(item["train_loss"]),
                val_loss=float(item["val_loss"]),
                train_perplexity=float(item["train_perplexity"]),
                val_perplexity=float(item["val_perplexity"]),
            )
            tracker.epochs.append(entry)
            if entry.val_loss < tracker.best_val_loss:
                tracker.best_val_loss = entry.val_loss
                tracker.best_epoch = entry.epoch
        return tracker
