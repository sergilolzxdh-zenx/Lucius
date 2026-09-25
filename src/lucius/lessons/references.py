"""Frames of the tutorial itself, to put next to what Lucius built.

The video file often cannot be downloaded (platforms block anonymous downloads), but its
storyboard -- the small preview frames the player shows when you hover the timeline, one every
few seconds, in sprite sheets -- is listed in the saved metadata and served like thumbnails.
They are small (about 160x90) but show the tutor's result at a given time. The video's thumbnail
usually shows the finished project.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lucius.logging_setup import get_logger

log = get_logger("lessons.references")


def _storyboard(info: dict[str, Any]) -> dict[str, Any] | None:
    boards = [f for f in info.get("formats") or [] if str(f.get("format_id", "")).startswith("sb")
              and f.get("fragments") and f.get("rows") and f.get("columns")]
    return max(boards, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0), default=None)


def _fetch(url: str, dest: Path) -> Path | None:
    if dest.exists():
        return dest
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        log.warning("reference image not fetched (%s): %s", url[:80], exc)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def storyboard_frame(info: dict[str, Any], t: float, cache_dir: Path, dest: Path) -> Path | None:
    """The storyboard frame shown at ``t`` seconds, saved as ``dest`` (PNG), or None if unavailable."""
    from PIL import Image

    board = _storyboard(info)
    if board is None:
        return None
    rows, columns = int(board["rows"]), int(board["columns"])
    per_sheet = rows * columns
    fps = float(board.get("fps") or 0.0)
    fragments = board["fragments"]
    if fps <= 0:
        duration = float(info.get("duration") or 0.0)
        fps = per_sheet * len(fragments) / duration if duration else 0.0
    if fps <= 0:
        return None
    index = max(0, int(t * fps))
    sheet_no, tile = divmod(index, per_sheet)
    if sheet_no >= len(fragments):
        sheet_no, tile = len(fragments) - 1, per_sheet - 1
    url = fragments[sheet_no].get("url")
    if not url:
        return None
    sheet_path = _fetch(url, cache_dir / f"{info.get('id', 'video')}_{board['format_id']}_{sheet_no}.jpg")
    if sheet_path is None:
        return None
    sheet = Image.open(sheet_path)
    w, h = sheet.width // columns, sheet.height // rows
    row, col = divmod(tile, columns)
    if row >= rows:
        return None
    frame = sheet.crop((col * w, row * h, (col + 1) * w, (row + 1) * h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.save(dest)
    return dest


def thumbnail(info: dict[str, Any], cache_dir: Path) -> Path | None:
    """The video's thumbnail (often the finished project)."""
    url = info.get("thumbnail")
    if not url:
        return None
    return _fetch(url, cache_dir / f"{info.get('id', 'video')}_thumbnail.jpg")
