"""Content-addressed frame store.

Frames are image files outside SQLite (``frames/ab/abcdef....jpg``). Identical images are
stored once. Each stored frame gets a 64-bit difference hash so visual change can be
measured cheaply without decoding images again or calling any model.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lucius.errors import StorageError


@dataclass(frozen=True)
class StoredImage:
    rel_path: str
    sha256: str
    width: int
    height: int
    dhash: str


def dhash(image: Image.Image, size: int = 8) -> str:
    """Difference hash: robust to compression noise, sensitive to structural change."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


THUMB_SIZE = (32, 24)


def thumbnail(image: Image.Image) -> np.ndarray:
    """Small grayscale signature used to measure visual change between frames."""
    return np.asarray(image.convert("L").resize(THUMB_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)


def visual_change(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    """Mean absolute thumbnail difference in [0, 1] (catches flat-colour UI changes dhash misses)."""
    if previous is None or previous.shape != current.shape:
        return None
    return float(np.abs(previous.astype(np.int16) - current.astype(np.int16)).mean() / 255.0)


def hash_distance(a: str | None, b: str | None) -> float:
    """Normalised Hamming distance between two dhashes (0 identical .. 1 opposite)."""
    if not a or not b or len(a) != len(b):
        return 1.0
    diff = int(a, 16) ^ int(b, 16)
    return bin(diff).count("1") / (len(a) * 4)


class FrameStore:
    def __init__(self, root: str | Path, *, image_format: str = "jpeg", jpeg_quality: int = 82,
                 max_width: int = 1600) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width

    def _encode(self, image: Image.Image) -> tuple[bytes, str]:
        if image.width > self.max_width:
            ratio = self.max_width / image.width
            image = image.resize((self.max_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        if self.image_format == "png":
            image.save(buffer, format="PNG", optimize=False)
            return buffer.getvalue(), "png"
        image.convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality)
        return buffer.getvalue(), "jpg"

    def put_image(self, image: Image.Image) -> StoredImage:
        data, ext = self._encode(image)
        digest = hashlib.sha256(data).hexdigest()
        rel = f"{digest[:2]}/{digest}.{ext}"
        path = self.root / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic: a crash never leaves a half-written frame
        with Image.open(io.BytesIO(data)) as stored:
            width, height = stored.size
            stored_hash = dhash(stored)
        return StoredImage(rel, digest, width, height, stored_hash)

    def put_bytes(self, data: bytes) -> StoredImage:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:  # PIL raises many exception types for bad data
            raise StorageError("frame bytes are not a decodable image", cause=str(exc)) from exc
        return self.put_image(image)

    def path(self, rel_path: str) -> Path:
        resolved = (self.root / rel_path).resolve()
        if self.root.resolve() not in resolved.parents:
            raise StorageError("frame path escapes the frame store", rel_path=rel_path)
        return resolved

    def open(self, rel_path: str) -> Image.Image:
        return Image.open(self.path(rel_path))

    def verify(self, rel_path: str, sha256: str) -> bool:
        path = self.path(rel_path)
        if not path.exists():
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == sha256
