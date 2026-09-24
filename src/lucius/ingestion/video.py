"""Video temporal analysis with intelligent sampling (10C, 10E).

1. Coarse pass: decode at a low rate, keep tiny grayscale thumbnails and a per-cell change grid.
2. Change events: samples whose change exceeds an adaptive threshold (median + k*MAD) are grouped.
3. Dense pass: around each event, sample densely to locate the last stable frame before and the
   first stable frame after the change.
4. Representative frames: the middle of every static span.

Only these selected frames ever reach a vision model; every frame keeps its media timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from lucius.errors import MediaError

THUMB = (96, 54)
GRID = 8


HEADER_BAND = 0.07       # top share of the frame holding Blender's header (mode, tools)
HEADER_THUMB = (256, 10)


@dataclass
class Sample:
    t: float
    thumb: np.ndarray
    header: np.ndarray | None = None
    change: float = 0.0
    header_change: float = 0.0
    grid: np.ndarray | None = None


@dataclass
class ChangeEvent:
    t_start: float
    t_end: float
    peak: float
    t_before: float = 0.0     # last stable time before the change
    t_after: float = 0.0      # first stable time after the change
    cells: list[tuple[int, int]] = field(default_factory=list)


def _thumb(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = frame_bgr[:, :, ::-1]
    return np.asarray(Image.fromarray(rgb).convert("L").resize(THUMB, Image.Resampling.BILINEAR), dtype=np.int16)


def _header(frame_bgr: np.ndarray) -> np.ndarray:
    """The header band at a higher horizontal resolution: mode/tool labels are small text."""
    band = frame_bgr[: max(1, int(frame_bgr.shape[0] * HEADER_BAND)), :, ::-1]
    return np.asarray(Image.fromarray(np.ascontiguousarray(band)).convert("L").resize(HEADER_THUMB,
                                                                                     Image.Resampling.BILINEAR),
                      dtype=np.int16)


def _header_change(a: np.ndarray | None, b: np.ndarray | None, cells: int = 16) -> float:
    """Strongest change among horizontal cells of the header band (a label is a small part of it)."""
    if a is None or b is None:
        return 0.0
    diff = np.abs(a - b).astype(np.float32) / 255.0
    width = diff.shape[1]
    return float(max(diff[:, i * width // cells:(i + 1) * width // cells].mean() for i in range(cells)))


def _grid_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a - b).astype(np.float32) / 255.0
    h, w = diff.shape
    return np.array([[diff[i * h // GRID:(i + 1) * h // GRID, j * w // GRID:(j + 1) * w // GRID].mean()
                      for j in range(GRID)] for i in range(GRID)])


class VideoReader:
    def __init__(self, path: Path) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise MediaError("video support requires opencv-python-headless") from exc
        self._cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise MediaError(f"cannot open video {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = count / self.fps if self.fps else 0.0

    def frame_at(self, t: float) -> np.ndarray | None:
        self.capture.set(self._cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = self.capture.read()
        return frame if ok else None

    def image_at(self, t: float) -> Image.Image | None:
        frame = self.frame_at(t)
        return None if frame is None else Image.fromarray(frame[:, :, ::-1])

    def iter_samples(self, rate: float) -> list[Sample]:
        """Sequential decode keeping one frame per 1/rate seconds (grab() skips decoding the rest)."""
        cv2 = self._cv2
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        step = max(1, round(self.fps / rate)) if self.fps else 1
        samples, index = [], 0
        while True:
            if not self.capture.grab():
                break
            if index % step == 0:
                ok, frame = self.capture.retrieve()
                if ok:
                    samples.append(Sample(t=index / self.fps if self.fps else float(index), thumb=_thumb(frame),
                                          header=_header(frame)))
            index += 1
        return samples

    def close(self) -> None:
        self.capture.release()


class VideoAnalyzer:
    def __init__(self, coarse_fps: float = 2.0, dense_fps: float = 8.0, k: float = 4.0, floor: float = 0.015,
                 merge_gap_s: float = 0.75) -> None:
        self.coarse_fps = coarse_fps
        self.dense_fps = dense_fps
        self.k = k
        self.floor = floor
        self.merge_gap_s = merge_gap_s

    def coarse(self, reader: VideoReader) -> tuple[list[Sample], float]:
        samples = reader.iter_samples(self.coarse_fps)
        for prev, cur in zip(samples, samples[1:]):
            cur.grid = _grid_change(prev.thumb, cur.thumb)
            # The strongest cell, not the frame mean: a narrowed object or a changed header label is a
            # small area of the frame and would vanish in a whole-frame average.
            cur.header_change = _header_change(prev.header, cur.header)
            cur.change = max(float(cur.grid.max()), cur.header_change)
        changes = np.array([s.change for s in samples[1:]]) if len(samples) > 1 else np.zeros(1)
        median = float(np.median(changes))
        mad = float(np.median(np.abs(changes - median)))
        threshold = max(self.floor, median + self.k * 1.4826 * mad)
        return samples, threshold

    def events(self, samples: list[Sample], threshold: float) -> list[ChangeEvent]:
        events: list[ChangeEvent] = []
        for i, sample in enumerate(samples[1:], start=1):
            if sample.change <= threshold:
                continue
            t0 = samples[i - 1].t
            cells = [] if sample.grid is None else [tuple(c) for c in np.argwhere(sample.grid > threshold)]
            if events and t0 - events[-1].t_end <= self.merge_gap_s:
                last = events[-1]
                last.t_end = sample.t
                last.peak = max(last.peak, sample.change)
                last.cells = sorted(set(last.cells) | set(cells))
            else:
                events.append(ChangeEvent(t_start=t0, t_end=sample.t, peak=sample.change, cells=cells))
        return events

    def refine(self, reader: VideoReader, event: ChangeEvent, threshold: float) -> ChangeEvent:
        """Dense sampling to locate stable frames on both sides of the change."""
        step = 1.0 / self.dense_fps
        times = np.arange(max(0.0, event.t_start - step), min(reader.duration, event.t_end + 2 * step) + 1e-9, step)
        thumbs = []
        for t in times:
            frame = reader.frame_at(float(t))
            if frame is not None:
                thumbs.append((float(t), _thumb(frame), _header(frame)))
        event.t_before, event.t_after = event.t_start, event.t_end
        if len(thumbs) < 2:
            return event
        diffs = [max(float(_grid_change(a[1], b[1]).max()), _header_change(a[2], b[2]))
                 for a, b in zip(thumbs, thumbs[1:])]
        changing = [i for i, d in enumerate(diffs) if d > threshold]
        if changing:
            event.t_before = thumbs[changing[0]][0]
            event.t_after = thumbs[min(len(thumbs) - 1, changing[-1] + 1)][0]
        return event

    @staticmethod
    def representative_times(samples: list[Sample], events: list[ChangeEvent], duration: float) -> list[float]:
        bounds = [0.0] + [t for e in events for t in (e.t_before, e.t_after)] + [duration]
        times = []
        for a, b in zip(bounds[::2], bounds[1::2]):
            if b - a >= 0.3:
                times.append(round((a + b) / 2, 3))
        return times
