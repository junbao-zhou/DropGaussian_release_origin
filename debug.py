import logging
import os
import torch
from torchvision.io import write_png
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def save_image(
    image: torch.Tensor,
    filename: str | Path,
):
    filename = Path(filename)
    print(f"Debug: Saving image to {filename = }")
    filename.parent.mkdir(parents=True, exist_ok=True)
    # os.mkdir(filename.parent)
    print(f"Debug: Created directory {filename.parent = }")

    assert (
        image.ndim == 3 and image.shape[0] == 3
    ), "Image tensor must have shape (3, H, W)"

    image = image.detach().cpu()

    if image.dtype != torch.uint8:
        image_to_save = (image.clamp(0.0, 1.0) * 255).to(torch.uint8)
    else:
        image_to_save = image
    write_png(image_to_save, str(filename))


# New: save a scatter plot while handling NaN/Inf robustly
def save_scatter(
    x,
    y,
    filename: str | Path,
    title: str | None = None,
    xlabel: str = "gaussian_scores",
    ylabel: str = "probability",
):
    matplotlib.use("Agg")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    x_np = torch.as_tensor(x).detach().float().cpu().numpy()
    y_np = torch.as_tensor(y).detach().float().cpu().numpy()

    # Masks
    x_posinf = np.isposinf(x_np)
    x_neginf = np.isneginf(x_np)
    x_nan = np.isnan(x_np)
    y_invalid = ~np.isfinite(y_np)

    finite_x_mask = np.isfinite(x_np)
    finite_x = x_np[finite_x_mask]
    has_finite = finite_x.size > 0
    max_finite = finite_x.max() if has_finite else 0.0
    min_finite = finite_x.min() if has_finite else 0.0

    # Prepare plot x (clamp infs)
    x_plot = x_np.copy()
    if x_posinf.any() and has_finite:
        x_plot[x_posinf] = max_finite
    if x_neginf.any() and has_finite:
        x_plot[x_neginf] = min_finite

    # Valid finite (excluding original +/-inf & NaN) and valid y
    finite_valid = (
        finite_x_mask & (~x_posinf) & (~x_neginf) & (~x_nan) & np.isfinite(y_np)
    )
    posinf_valid = x_posinf & np.isfinite(y_np)
    neginf_valid = x_neginf & np.isfinite(y_np)

    plt.figure(figsize=(6, 4), dpi=150)

    any_points = False
    if finite_valid.any():
        plt.scatter(
            x_plot[finite_valid],
            y_np[finite_valid],
            s=4,
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
            color="#666666",
        )
        any_points = True
    if posinf_valid.any():
        plt.scatter(
            x_plot[posinf_valid],
            y_np[posinf_valid],
            s=18,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
            color="#d62728",
        )
        any_points = True
    if neginf_valid.any():
        plt.scatter(
            x_plot[neginf_valid],
            y_np[neginf_valid],
            s=18,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
            color="#1f77b4",
        )
        any_points = True

    if not any_points:
        plt.text(
            0.5,
            0.5,
            "No finite points to plot",
            ha="center",
            va="center",
            fontsize=10,
            transform=plt.gca().transAxes,
        )

    # Title: exclude +/-inf counts per requirement; keep NaN / non‑finite y diagnostics
    t = title or "score vs prob"
    extras = []
    c_nan = int(x_nan.sum())
    c_ybad = int(y_invalid.sum())
    if c_nan:
        extras.append(f"nan x: {c_nan}")
    if c_ybad:
        extras.append(f"non-finite y: {c_ybad}")
    if extras:
        t = f"{t} ({', '.join(extras)})"
    plt.title(t)

    # X label with +/-inf counts
    c_pos = int(x_posinf.sum())
    c_neg = int(x_neginf.sum())
    if c_pos or c_neg:
        xlabel = f"{xlabel} (+inf={c_pos}, -inf={c_neg})"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(filename))
    plt.close()


def _calc_hist(
    data: torch.Tensor,
    bins: int = 100,
    include_stats: bool = False,
):
    """
    Compute histogram on finite values only.
    Optionally return stats for NaN/+Inf/-Inf counts plus mean/std of finite values.

    Returns:
      counts, bin_edges  (both torch.Tensor)
      if include_stats=True, also returns a dict with:
        {
          "total": int,
          "finite": int,
          "posinf": int,
          "neginf": int,
          "nan": int,
          "has_finite": bool,
          "mean": float | None,
          "std": float | None
        }
    """
    arr = data.detach().float().view(-1).cpu()

    is_finite = torch.isfinite(arr)
    is_inf = torch.isinf(arr)
    is_posinf = is_inf & (arr > 0)
    is_neginf = is_inf & (arr < 0)
    is_nan = torch.isnan(arr)

    n_total = arr.numel()
    n_finite = int(is_finite.sum().item())
    n_posinf = int(is_posinf.sum().item())
    n_neginf = int(is_neginf.sum().item())
    n_nan = int(is_nan.sum().item())
    has_finite = n_finite > 0

    if has_finite:
        finite = arr[is_finite]
        # mean/std on finite values
        mean_val = finite.mean().item()
        std_val = finite.std(unbiased=False).item()
        if hasattr(torch, "histogram"):
            counts, bin_edges = torch.histogram(finite, bins=bins)
        else:
            vmin = finite.min().item()
            vmax = finite.max().item()
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            counts = torch.histc(finite, bins=bins, min=vmin, max=vmax)
            bin_edges = torch.linspace(vmin, vmax, steps=bins + 1)
    else:
        counts = torch.zeros(bins, dtype=torch.int64)
        bin_edges = torch.linspace(0.0, 1.0, steps=bins + 1)
        mean_val = float("nan")
        std_val = float("nan")

    if include_stats:
        stats = {
            "total": int(n_total),
            "finite": int(n_finite),
            "posinf": int(n_posinf),
            "neginf": int(n_neginf),
            "nan": int(n_nan),
            "has_finite": bool(has_finite),
            "mean": mean_val,
            "std": std_val,
        }
        return counts, bin_edges, stats
    else:
        return counts, bin_edges


class SimpleLogger:
    def __init__(
        self,
        filename: str | None = None,
        level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(f"SimpleLogger:{id(self)}")
        self.logger.setLevel(level)
        self.logger.propagate = (
            False  # Avoid duplicate messages if root configured
        )

        handler: logging.Handler
        if filename:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        else:
            handler = logging.StreamHandler()
        self.filename = filename
        self.log_dir = Path(filename).parent if filename else None

        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(fmt)
        self.logger.addHandler(handler)

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def debug_save_image(
        self,
        image: torch.Tensor,
        filename: str | Path,
        level: int = logging.DEBUG,
    ):
        # Only save when the logger is enabled for the given level
        if self.logger.isEnabledFor(level):
            self.logger.log(level, f"Saving image to {Path(filename)}")
            # Call the module-level helper (also gated by IS_DEBUG)
            save_image(image, filename)

    def debug_save_hist(
        self,
        name: str,
        data: torch.Tensor,
        hist_dir: str | None,
        tag: str = "",
        level: int = logging.DEBUG,
        bins: int = 100,
    ):
        self.logger.debug(f"Saving histogram for {name} to {Path(hist_dir)}")

        if not self.logger.isEnabledFor(level):
            return

        if hist_dir is None:
            return

        os.makedirs(hist_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        base = os.path.join(hist_dir, f"{name}{suffix}")
        png_path = base + ".png"

        arr = data.detach().float().view(-1).cpu()

        counts, bin_edges, stats = _calc_hist(
            arr,
            bins=bins,
            include_stats=True,
        )

        matplotlib.use("Agg")
        plt.figure(figsize=(6, 4), dpi=150)

        if stats["has_finite"]:
            finite_np = arr[torch.isfinite(arr)].numpy()
            plt.hist(finite_np, bins=bin_edges.numpy())
        else:
            plt.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                fontsize=11,
                transform=plt.gca().transAxes,
            )

        title = f"{name}{suffix}"
        annot = f"+inf={stats['posinf']}, -inf={stats['neginf']}, nan={stats['nan']}"
        plt.title(f"{title} ({annot})")

        # x label with mean/std on a second line
        mean_val = stats["mean"]
        std_val = stats["std"]
        if stats["has_finite"]:
            extra = f"mean={mean_val:.4g}, std={std_val:.4g}"
        else:
            extra = "mean=nan, std=nan"
        plt.xlabel(f"value\n{extra}")

        plt.ylabel("count")
        plt.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        return {
            "name": name,
            "tag": tag,
            "path": png_path,
            "bins": int(bins),
            "counts": counts,
            "bin_edges": bin_edges,
            "stats": stats,
        }

    # New: scatter saver gated by log level
    def debug_save_scatter(
        self,
        name: str,
        x,
        y,
        scatter_dir: str | Path | None,
        tag: str = "",
        level: int = logging.DEBUG,
    ):
        if not self.logger.isEnabledFor(level):
            return
        if scatter_dir is None:
            return
        os.makedirs(scatter_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        base = os.path.join(str(scatter_dir), f"{name}{suffix}")
        self.logger.debug(f"Saving scatter {name} to {base}.png")
        save_scatter(x, y, base + ".png", title=f"{name}{suffix}")
