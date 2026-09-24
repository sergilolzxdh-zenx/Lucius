"""Media assets: validation, content-addressed storage and provenance (10B, 10P, 10Q).

Every uploaded file is validated (type, size, decodability), hashed, stored once, and given its
own data policy. Nothing about an upload makes it training data: external media default to
learning/reference use only.
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from lucius.errors import MediaError, NotFoundError
from lucius.ids import new_id
from lucius.provenance import DataPolicy, SourceClass
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
TEXT_EXT = {".txt", ".md"}
MAX_BYTES = {"video": 4 * 1024 ** 3, "image": 64 * 1024 ** 2, "blender_project": 2 * 1024 ** 3,
             "text": 4 * 1024 ** 2, "image_sequence": 64 * 1024 ** 2, "screenshot": 64 * 1024 ** 2}


class MediaKind(StrEnum):
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    BLENDER_PROJECT = "blender_project"
    TEXT = "text"
    SCREENSHOT = "screenshot"


class MediaRole(StrEnum):
    DEMONSTRATION = "demonstration"   # shows HOW something is done
    REFERENCE = "reference"           # shows WHAT the result should look like
    TARGET = "target"
    BEFORE = "before"
    AFTER = "after"
    INTERMEDIATE = "intermediate"
    INSTRUCTION = "instruction"
    PROJECT = "project"


SOURCE_FOR_KIND = {
    MediaKind.VIDEO: SourceClass.EXTERNAL_VIDEO, MediaKind.IMAGE: SourceClass.EXTERNAL_IMAGE,
    MediaKind.IMAGE_SEQUENCE: SourceClass.EXTERNAL_IMAGE, MediaKind.SCREENSHOT: SourceClass.EXTERNAL_IMAGE,
    MediaKind.BLENDER_PROJECT: SourceClass.EXTERNAL_PROJECT, MediaKind.TEXT: SourceClass.EXTERNAL_DOCUMENTATION,
}


class MediaAsset(BaseModel):
    id: str
    demonstration_id: str | None = None
    kind: MediaKind
    role: MediaRole
    filename: str
    path: str
    sha256: str
    size_bytes: int
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    policy: DataPolicy
    analysis: dict[str, Any] = Field(default_factory=dict)
    created_at: float
    duplicate_of: str | None = None


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"   # Blender 5 compresses .blend files with zstd by default
GZIP_MAGIC = b"\x1f\x8b"           # older Blender versions used gzip for compressed files


def detect_kind(filename: str, data_head: bytes) -> MediaKind:
    ext = Path(filename).suffix.lower()
    if data_head.startswith(b"BLENDER") or (ext == ".blend" and data_head.startswith((ZSTD_MAGIC, GZIP_MAGIC))):
        return MediaKind.BLENDER_PROJECT
    if ext in VIDEO_EXT:
        return MediaKind.VIDEO
    if ext in IMAGE_EXT:
        return MediaKind.IMAGE
    if ext in TEXT_EXT:
        return MediaKind.TEXT
    raise MediaError(f"unsupported media type: {filename}", filename=filename)


def _decode(row: Any) -> MediaAsset:
    analysis = loads(row["analysis"], {})
    return MediaAsset(
        id=row["id"], demonstration_id=row["demonstration_id"], kind=MediaKind(row["kind"]), role=MediaRole(row["role"]),
        filename=row["filename"], path=row["path"], sha256=row["sha256"], size_bytes=row["size_bytes"],
        mime=row["mime"], width=row["width"], height=row["height"], duration_s=row["duration_s"], fps=row["fps"],
        frame_count=row["frame_count"], policy=DataPolicy.model_validate(loads(row["policy"], {})), analysis=analysis,
        created_at=row["created_at"], duplicate_of=analysis.get("duplicate_of"))


class MediaStore:
    def __init__(self, db: Database, root: Path) -> None:
        self.db = db
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, asset: MediaAsset) -> Path:
        resolved = (self.root / asset.path).resolve()
        if self.root.resolve() not in resolved.parents:
            raise MediaError("media path escapes the media store", path=asset.path)
        return resolved

    def ingest(self, source: Path, *, filename: str | None = None, role: MediaRole, demonstration_id: str | None = None,
               policy: DataPolicy | None = None, kind: MediaKind | None = None) -> MediaAsset:
        source = Path(source)
        if not source.is_file():
            raise MediaError(f"{source} is not a file")
        filename = filename or source.name
        with source.open("rb") as handle:
            head = handle.read(16)
        kind = kind or detect_kind(filename, head)
        size = source.stat().st_size
        if size == 0:
            raise MediaError("empty file", filename=filename)
        if size > MAX_BYTES[kind.value]:
            raise MediaError(f"{kind.value} exceeds the size limit", filename=filename, size=size)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        meta = self._probe(source, kind)
        rel = f"{sha[:2]}/{sha}{Path(filename).suffix.lower()}"
        target = self.root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target.with_suffix(target.suffix + ".tmp"))
            target.with_suffix(target.suffix + ".tmp").replace(target)
        duplicate = self.db.scalar("SELECT id FROM media_assets WHERE sha256 = ? ORDER BY created_at LIMIT 1", (sha,))
        policy = policy or DataPolicy.for_external(SOURCE_FOR_KIND[kind], reference_only=role != MediaRole.DEMONSTRATION)
        asset = MediaAsset(id=new_id("media"), demonstration_id=demonstration_id, kind=kind, role=role,
                           filename=filename, path=rel, sha256=sha, size_bytes=size,
                           mime=mimetypes.guess_type(filename)[0], policy=policy, created_at=now(),
                           analysis={"duplicate_of": duplicate} if duplicate else {}, duplicate_of=duplicate, **meta)
        self.db.insert("media_assets", {
            "id": asset.id, "demonstration_id": demonstration_id, "kind": kind.value, "role": role.value,
            "filename": filename, "path": rel, "sha256": sha, "size_bytes": size, "mime": asset.mime,
            "width": asset.width, "height": asset.height, "duration_s": asset.duration_s, "fps": asset.fps,
            "frame_count": asset.frame_count, "policy": dumps(policy), "analysis": dumps(asset.analysis),
            "created_at": asset.created_at})
        return asset

    @staticmethod
    def _probe(path: Path, kind: MediaKind) -> dict[str, Any]:
        if kind in (MediaKind.IMAGE, MediaKind.SCREENSHOT):
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    return {"width": img.width, "height": img.height}
            except Exception as exc:
                raise MediaError(f"image is not decodable: {exc}") from exc
        if kind == MediaKind.VIDEO:
            try:
                import cv2
            except ImportError as exc:
                raise MediaError("video support requires opencv-python-headless (pip install lucius[media])") from exc
            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
                if not capture.isOpened() or not ok:
                    raise MediaError("video is not decodable")
                fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps <= 0 or fps > 480:
                    raise MediaError("video reports an invalid frame rate", fps=fps)
                return {"width": frame.shape[1], "height": frame.shape[0], "fps": float(fps), "frame_count": count,
                        "duration_s": round(count / fps, 3) if count else None}
            finally:
                capture.release()
        if kind == MediaKind.TEXT:
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise MediaError("text must be UTF-8") from exc
        return {}

    def get(self, asset_id: str) -> MediaAsset:
        row = self.db.query_one("SELECT * FROM media_assets WHERE id = ?", (asset_id,))
        if row is None:
            raise NotFoundError(f"media {asset_id} not found", media_id=asset_id)
        return _decode(row)

    def list(self, *, demonstration_id: str | None = None, role: MediaRole | None = None) -> list[MediaAsset]:
        clauses, params = [], []
        if demonstration_id:
            clauses.append("demonstration_id = ?")
            params.append(demonstration_id)
        if role:
            clauses.append("role = ?")
            params.append(role.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return [_decode(r) for r in self.db.query(f"SELECT * FROM media_assets {where} ORDER BY created_at", params)]

    def update_analysis(self, asset_id: str, analysis: dict[str, Any]) -> None:
        current = self.get(asset_id)
        merged = {**current.analysis, **analysis}
        self.db.execute("UPDATE media_assets SET analysis = ? WHERE id = ?", (dumps(merged), asset_id))

    def set_policy(self, asset_id: str, policy: DataPolicy) -> None:
        self.db.execute("UPDATE media_assets SET policy = ? WHERE id = ?", (dumps(policy), asset_id))
