"""Measured silhouettes: rasterise projected triangles, extract reference masks, compare.

The bridge returns orthographic triangle projections of the evaluated mesh; rasterising them
here gives an exact silhouette without any GPU or renderer. Reference images are turned into
masks by separating the object from a uniform background. Comparison normalises both masks to
their bounding box (keeping aspect ratio) and computes IoU, so the check measures *shape*,
independent of absolute scale or framing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS = 160


def rasterize(triangles: Sequence[float], size: int = CANVAS, margin: int = 4) -> tuple[np.ndarray, dict[str, float]]:
    """Flat [x0,y0,x1,y1,x2,y2,...] world-space triangles -> boolean mask fitted to the canvas."""
    pts = np.asarray(triangles, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return np.zeros((size, size), dtype=bool), {"width": 0.0, "height": 0.0}
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    scale = (size - 2 * margin) / max(float(span.max()), 1e-9)
    offset = (size - span * scale) / 2
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    screen = (pts - lo) * scale + offset
    screen[:, 1] = size - screen[:, 1]  # world up is image up
    for tri in screen.reshape(-1, 3, 2):
        draw.polygon([tuple(p) for p in tri], fill=255)
    return np.asarray(image) > 127, {"width": float(span[0]), "height": float(span[1])}


def mask_from_image(image: Image.Image, size: int = CANVAS, margin: int = 4) -> np.ndarray:
    """Foreground mask of a reference image (object on a roughly uniform background)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    threshold = max(25.0, otsu_threshold(distance))
    mask = Image.fromarray(((distance > threshold) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return fit_mask(np.asarray(mask) > 127, size, margin)


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold over a 1-D sample of values."""
    hist, edges = np.histogram(values, bins=64)
    total = hist.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    mean_total = cum_mean[-1]
    best, threshold = -1.0, 0.0
    for i in range(len(hist) - 1):
        w0, w1 = cum[i], total - cum[i]
        if w0 == 0 or w1 == 0:
            continue
        m0 = cum_mean[i] / w0
        m1 = (mean_total - cum_mean[i]) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, threshold = between, edges[i + 1]
    return float(threshold)


def fit_mask(mask: np.ndarray, size: int = CANVAS, margin: int = 4) -> np.ndarray:
    """Crop to the bounding box and centre in a square canvas, preserving aspect ratio."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=bool)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = (size - 2 * margin) / max(h, w)
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(new, Image.Resampling.NEAREST)) > 127
    canvas = np.zeros((size, size), dtype=bool)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def aspect_ratio(mask: np.ndarray) -> float | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1))


def width_profile(mask: np.ndarray, bins: int = 20) -> list[float]:
    """Width of the silhouette in horizontal bands from bottom (0) to top (bins-1), normalised."""
    ys, _xs = np.nonzero(mask)
    if len(ys) == 0:
        return [0.0] * bins
    top, bottom = ys.min(), ys.max()
    rows = mask[top:bottom + 1]
    widths = rows.sum(axis=1)[::-1].astype(np.float64)  # bottom first
    peak = widths.max() or 1.0
    edges = np.linspace(0, len(widths), bins + 1).astype(int)
    return [float(widths[a:max(a + 1, b)].mean() / peak) for a, b in zip(edges, edges[1:])]


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("RGB")
