"""
Phase 11: Training Curve Visualization for MiniGPT.

Creates a clean two-panel figure saved to `outputs/training_curves.png` showing:
- Left Panel : Training Loss vs. Epoch and Validation Loss vs. Epoch
- Right Panel: Training Perplexity vs. Epoch and Validation Perplexity vs. Epoch

Uses `matplotlib` when available, and includes a built-in pure-Python PNG renderer
(using standard-library `zlib` and `struct`) so publication-ready PNG charts are
generated reliably in any Python environment without external binary dependencies.
"""

import json
import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Tuple, Union
import zlib


RGB = Tuple[int, int, int]


class _SimpleCanvas:
    """Lightweight anti-aliased RGB pixel canvas with pure-Python PNG encoding."""

    def __init__(self, width: int, height: int, bg: RGB = (250, 248, 245)) -> None:
        self.width = width
        self.height = height
        self.pixels: bytearray = bytearray(width * height * 3)
        r, g, b = bg
        for i in range(0, len(self.pixels), 3):
            self.pixels[i] = r
            self.pixels[i + 1] = g
            self.pixels[i + 2] = b

    def set_pixel(self, x: int, y: int, color: RGB, alpha: float = 1.0) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            if alpha >= 0.999:
                self.pixels[idx] = color[0]
                self.pixels[idx + 1] = color[1]
                self.pixels[idx + 2] = color[2]
            elif alpha > 0.0:
                inv = 1.0 - alpha
                self.pixels[idx] = int(self.pixels[idx] * inv + color[0] * alpha)
                self.pixels[idx + 1] = int(self.pixels[idx + 1] * inv + color[1] * alpha)
                self.pixels[idx + 2] = int(self.pixels[idx + 2] * inv + color[2] * alpha)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        xa, xb = max(0, min(x0, x1)), min(self.width - 1, max(x0, x1))
        ya, yb = max(0, min(y0, y1)), min(self.height - 1, max(y0, y1))
        for y in range(ya, yb + 1):
            row_start = (y * self.width + xa) * 3
            for x in range(xa, xb + 1):
                i = row_start + (x - xa) * 3
                self.pixels[i] = color[0]
                self.pixels[i + 1] = color[1]
                self.pixels[i + 2] = color[2]

    def stroke_rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB, thickness: int = 1) -> None:
        for t in range(thickness):
            self.draw_line(x0 + t, y0 + t, x1 - t, y0 + t, color, 1)
            self.draw_line(x0 + t, y1 - t, x1 - t, y1 - t, color, 1)
            self.draw_line(x0 + t, y0 + t, x0 + t, y1 - t, color, 1)
            self.draw_line(x1 - t, y0 + t, x1 - t, y1 - t, color, 1)

    def draw_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: RGB,
        thickness: int = 2,
    ) -> None:
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy)) * 2) + 1
        radius = max(0, thickness - 1)
        for s in range(steps + 1):
            t = s / max(1, steps)
            cx = int(round(x0 + dx * t))
            cy = int(round(y0 + dy * t))
            for ox in range(-radius, radius + 1):
                for oy in range(-radius, radius + 1):
                    if ox * ox + oy * oy <= radius * radius + 1:
                        self.set_pixel(cx + ox, cy + oy, color, 0.95)

    def draw_circle(self, cx: int, cy: int, radius: int, color: RGB, fill: bool = True) -> None:
        r2 = radius * radius
        for ox in range(-radius - 1, radius + 2):
            for oy in range(-radius - 1, radius + 2):
                d2 = ox * ox + oy * oy
                if fill and d2 <= r2:
                    self.set_pixel(cx + ox, cy + oy, color)
                elif not fill and abs(d2 - r2) <= radius:
                    self.set_pixel(cx + ox, cy + oy, color)

    def to_png_bytes(self) -> bytes:
        raw_rows = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw_rows.append(0)  # Filter type 0 (None)
            start = y * stride
            raw_rows.extend(self.pixels[start : start + stride])

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        idat = zlib.compress(bytes(raw_rows), level=6)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", idat)
            + _chunk(b"IEND", b"")
        )


def _render_panel(
    canvas: _SimpleCanvas,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    epochs: List[int],
    train_vals: List[float],
    val_vals: List[float],
    train_color: RGB = (37, 99, 235),
    val_color: RGB = (5, 150, 105),
) -> None:
    """Render a single curve panel (axes, gridlines, train & validation series)."""
    canvas.fill_rect(x0, y0, x1, y1, (255, 255, 255))
    canvas.stroke_rect(x0, y0, x1, y1, (214, 210, 202), thickness=1)

    pad_l, pad_r, pad_t, pad_b = 36, 24, 28, 32
    px0, py0 = x0 + pad_l, y0 + pad_t
    px1, py1 = x1 - pad_r, y1 - pad_b

    # Horizontal gridlines
    for g in range(5):
        gy = int(py0 + (py1 - py0) * (g / 4.0))
        canvas.draw_line(px0, gy, px1, gy, (238, 235, 229), thickness=1)

    # Vertical gridlines per epoch
    n_pts = len(epochs)
    for i in range(n_pts):
        gx = int(px0 + (px1 - px0) * (i / max(1, n_pts - 1)))
        canvas.draw_line(gx, py0, gx, py1, (242, 240, 235), thickness=1)

    all_vals = train_vals + val_vals
    min_v = min(all_vals)
    max_v = max(all_vals)
    span = max(max_v - min_v, 1e-5)
    v_low = min_v - 0.12 * span
    v_high = max_v + 0.12 * span

    def _to_xy(idx: int, val: float) -> Tuple[int, int]:
        x_frac = idx / max(1, n_pts - 1) if n_pts > 1 else 0.5
        y_frac = (val - v_low) / (v_high - v_low)
        cx = int(round(px0 + x_frac * (px1 - px0)))
        cy = int(round(py1 - y_frac * (py1 - py0)))
        return cx, cy

    train_pts = [_to_xy(i, v) for i, v in enumerate(train_vals)]
    val_pts = [_to_xy(i, v) for i, v in enumerate(val_vals)]

    for i in range(len(train_pts) - 1):
        canvas.draw_line(
            train_pts[i][0],
            train_pts[i][1],
            train_pts[i + 1][0],
            train_pts[i + 1][1],
            train_color,
            thickness=2,
        )
    for i in range(len(val_pts) - 1):
        canvas.draw_line(
            val_pts[i][0],
            val_pts[i][1],
            val_pts[i + 1][0],
            val_pts[i + 1][1],
            val_color,
            thickness=2,
        )

    for cx, cy in train_pts:
        canvas.draw_circle(cx, cy, radius=4, color=(255, 255, 255), fill=True)
        canvas.draw_circle(cx, cy, radius=3, color=train_color, fill=True)

    for cx, cy in val_pts:
        canvas.draw_circle(cx, cy, radius=4, color=(255, 255, 255), fill=True)
        canvas.draw_circle(cx, cy, radius=3, color=val_color, fill=True)

    # Legend indicators in top-right of panel
    canvas.draw_line(px1 - 90, py0 + 8, px1 - 68, py0 + 8, train_color, thickness=2)
    canvas.draw_circle(px1 - 79, py0 + 8, radius=3, color=train_color, fill=True)
    canvas.draw_line(px1 - 42, py0 + 8, px1 - 20, py0 + 8, val_color, thickness=2)
    canvas.draw_circle(px1 - 31, py0 + 8, radius=3, color=val_color, fill=True)

    # Axes lines
    canvas.draw_line(px0, py0, px0, py1, (107, 114, 128), thickness=1)
    canvas.draw_line(px0, py1, px1, py1, (107, 114, 128), thickness=1)


def plot_training_curves(
    metrics_source: Union[str, Path, Dict[str, Any]],
    output_path: Union[str, Path] = "outputs/training_curves.png",
) -> Path:
    """
    Generate the Training & Validation Loss and Perplexity vs. Epoch curves and save
    to `output_path` (`outputs/training_curves.png`).

    Args:
        metrics_source: Path to `training_metrics.json` or a parsed metrics dictionary
                        containing `"epochs"`.
        output_path: Destination PNG file path.

    Returns:
        Resolved `Path` to the generated PNG image.
    """
    if isinstance(metrics_source, (str, Path)):
        metrics_path = Path(metrics_source)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        metrics_data: Dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    elif isinstance(metrics_source, dict):
        metrics_data = metrics_source
    else:
        raise TypeError("metrics_source must be a file path or a metrics dict.")

    epoch_rows = metrics_data.get("epochs", [])
    if not epoch_rows:
        raise ValueError("metrics_source contains no epoch records under 'epochs'.")

    epochs = [int(r["epoch"]) for r in epoch_rows]
    train_losses = [float(r["train_loss"]) for r in epoch_rows]
    val_losses = [float(r["val_loss"]) for r in epoch_rows]
    train_ppls = [
        float(r.get("train_perplexity", math.exp(min(float(r["train_loss"]), 20.0))))
        for r in epoch_rows
    ]
    val_ppls = [
        float(r.get("val_perplexity", math.exp(min(float(r["val_loss"]), 20.0))))
        for r in epoch_rows
    ]

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Try matplotlib first if installed; otherwise use the built-in PNG canvas renderer
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140)
        fig.patch.set_facecolor("#FAF8F5")

        ax_loss.plot(epochs, train_losses, marker="o", color="#2563EB", linewidth=2, label="Train Loss")
        ax_loss.plot(epochs, val_losses, marker="s", color="#059669", linewidth=2, label="Val Loss")
        ax_loss.set_title("Cross-Entropy Loss vs. Epoch")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()

        ax_ppl.plot(epochs, train_ppls, marker="o", color="#2563EB", linewidth=2, label="Train Perplexity")
        ax_ppl.plot(epochs, val_ppls, marker="s", color="#059669", linewidth=2, label="Val Perplexity")
        ax_ppl.set_title("Perplexity vs. Epoch")
        ax_ppl.set_xlabel("Epoch")
        ax_ppl.set_ylabel("Perplexity")
        ax_ppl.grid(True, alpha=0.3)
        ax_ppl.legend()

        fig.tight_layout()
        fig.savefig(str(target), dpi=140)
        plt.close(fig)
        return target
    except ImportError:
        pass

    canvas = _SimpleCanvas(width=920, height=380, bg=(250, 248, 245))
    # Panel 1 (left): Training Loss & Validation Loss vs. Epoch
    _render_panel(
        canvas,
        x0=24,
        y0=24,
        x1=448,
        y1=356,
        epochs=epochs,
        train_vals=train_losses,
        val_vals=val_losses,
        train_color=(37, 99, 235),
        val_color=(5, 150, 105),
    )
    # Panel 2 (right): Training Perplexity & Validation Perplexity vs. Epoch
    _render_panel(
        canvas,
        x0=472,
        y0=24,
        x1=896,
        y1=356,
        epochs=epochs,
        train_vals=train_ppls,
        val_vals=val_ppls,
        train_color=(37, 99, 235),
        val_color=(217, 119, 6),
    )

    target.write_bytes(canvas.to_png_bytes())
    return target
