"""Reference-image understanding (10G, 10I).

A reference says *what the result should look like*, not *how it was made*. Measured constraints
(silhouette mask, aspect ratio, symmetry, width profile, taper, dominant colours) become
planning parameters and evaluation targets; model-inferred descriptions (object class, style,
parts) are stored separately with their provenance. Nothing here becomes a procedural skill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel

from lucius.evaluation.checks import ReferenceSilhouette
from lucius.evaluation.silhouette import CANVAS, aspect_ratio, fit_mask, iou, mask_from_image, width_profile
from lucius.ids import new_id
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

VIEWS = ("front", "side", "top", "back", "left", "right")


class VisualConstraint(BaseModel):
    id: str
    media_asset_id: str
    constraint_type: str      # silhouette, aspect_ratio, symmetry, width_profile, taper, dominant_colors, object_class...
    target: str               # a view, or a parameter/part name
    value: dict[str, Any]
    source: str               # measured, model_inferred, human_confirmed
    confidence: float
    created_at: float


def _raw_mask(image: Image.Image) -> np.ndarray:
    """Foreground mask at the image's own resolution (aspect preserved)."""
    small = image.convert("RGB")
    if max(small.size) > 512:
        small.thumbnail((512, 512))
    rgb = np.asarray(small, dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    distance = np.linalg.norm(rgb - np.median(border, axis=0), axis=2)
    return distance > max(25.0, float(np.percentile(distance, 50)) + 10.0)


def analyze_reference(image: Image.Image) -> dict[str, Any]:
    raw = _raw_mask(image)
    mask = mask_from_image(image)
    profile = width_profile(mask, bins=20)
    symmetry = iou(mask, np.fliplr(mask))
    lower = [w for i, w in enumerate(profile) if i < 12]
    tip = float(np.mean(profile[16:])) / (max(lower) or 1.0)
    quantized = image.convert("RGB").resize((96, 96)).quantize(colors=5)
    palette = quantized.getpalette() or []
    counts = sorted(quantized.getcolors() or [], reverse=True)
    colours = [f"#{palette[i * 3]:02x}{palette[i * 3 + 1]:02x}{palette[i * 3 + 2]:02x}" for _c, i in counts[:5]]
    coverage = float(raw.mean())
    return {"mask": mask, "aspect_ratio": aspect_ratio(raw), "width_profile": profile, "symmetry": symmetry,
            "tip_width_ratio": tip, "dominant_colors": colours, "foreground_coverage": coverage}


class ReferenceStore:
    def __init__(self, db: Database, media_root: Path) -> None:
        self.db = db
        self.masks_dir = Path(media_root) / "derived"
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def add(self, media_asset_id: str, constraint_type: str, target: str, value: dict[str, Any], *, source: str,
            confidence: float) -> VisualConstraint:
        c = VisualConstraint(id=new_id("constraint"), media_asset_id=media_asset_id, constraint_type=constraint_type,
                             target=target, value=value, source=source, confidence=confidence, created_at=now())
        self.db.insert("reference_constraints", {
            "id": c.id, "media_asset_id": media_asset_id, "constraint_type": constraint_type, "target": target,
            "value": dumps(value), "source": source, "confidence": confidence, "created_at": c.created_at})
        return c

    def measure(self, media_asset_id: str, image: Image.Image, view: str = "front",
                target: str | None = None) -> list[VisualConstraint]:
        """Deterministic measurements of a reference image (replaces earlier measurements of it)."""
        self.db.execute("DELETE FROM reference_constraints WHERE media_asset_id = ? AND source = 'measured'",
                        (media_asset_id,))
        analysis = analyze_reference(image)
        mask_path = self.masks_dir / f"{media_asset_id}_{view}.png"
        Image.fromarray((analysis["mask"] * 255).astype(np.uint8)).save(mask_path)
        coverage = analysis["foreground_coverage"]
        reliable = 0.02 < coverage < 0.9  # a mask covering (almost) nothing or everything is not a silhouette
        base = 0.85 if reliable else 0.3
        out = [
            self.add(media_asset_id, "silhouette", view, {"mask_file": mask_path.name, "canvas": CANVAS,
                                                          "coverage": round(coverage, 4), "target": target},
                     source="measured", confidence=base),
            self.add(media_asset_id, "aspect_ratio", view, {"height_over_width": round(analysis["aspect_ratio"] or 0, 4)},
                     source="measured", confidence=base),
            self.add(media_asset_id, "symmetry", view, {"mirror_iou": round(analysis["symmetry"], 4),
                                                        "symmetric": analysis["symmetry"] > 0.9},
                     source="measured", confidence=base),
            self.add(media_asset_id, "width_profile", view, {"bins": [round(w, 4) for w in analysis["width_profile"]]},
                     source="measured", confidence=base),
            self.add(media_asset_id, "taper", view, {"tip_width_ratio": round(analysis["tip_width_ratio"], 4),
                                                     "tapered": analysis["tip_width_ratio"] < 0.7},
                     source="measured", confidence=base),
            self.add(media_asset_id, "dominant_colors", "appearance", {"colors": analysis["dominant_colors"],
                                                                       "note": "aesthetic, not procedural"},
                     source="measured", confidence=0.6),
        ]
        return out

    def for_media(self, media_asset_ids: list[str]) -> list[VisualConstraint]:
        if not media_asset_ids:
            return []
        marks = ",".join("?" for _ in media_asset_ids)
        rows = self.db.query(f"SELECT * FROM reference_constraints WHERE media_asset_id IN ({marks}) ORDER BY created_at",
                             media_asset_ids)
        return [VisualConstraint(id=r["id"], media_asset_id=r["media_asset_id"], constraint_type=r["constraint_type"],
                                 target=r["target"], value=loads(r["value"], {}), source=r["source"],
                                 confidence=r["confidence"], created_at=r["created_at"]) for r in rows]

    def silhouettes(self, media_asset_ids: list[str], min_iou: float = 0.75) -> list[ReferenceSilhouette]:
        """Reference silhouettes usable by the evaluator's measured visual checks."""
        out = []
        for c in self.for_media(media_asset_ids):
            if c.constraint_type != "silhouette" or c.confidence < 0.5:
                continue
            path = self.masks_dir / c.value["mask_file"]
            if path.exists():
                mask = np.asarray(Image.open(path)) > 127
                out.append(ReferenceSilhouette(view=c.target, mask=fit_mask(mask), media_asset_id=c.media_asset_id,
                                               min_iou=min_iou, target=c.value.get("target")))
        return out

    def parameter_hints(self, media_asset_ids: list[str]) -> dict[str, float]:
        """Numbers a planner can use: e.g. the front-view height/width ratio of the target."""
        hints: dict[str, float] = {}
        for c in self.for_media(media_asset_ids):
            if c.constraint_type == "aspect_ratio" and c.confidence >= 0.5 and c.value.get("height_over_width"):
                hints[f"aspect_{c.target}"] = float(c.value["height_over_width"])
            if c.constraint_type == "project_dimensions":
                for axis, value in (c.value.get("dimensions") or {}).items():
                    hints[f"{c.target}_{axis}"] = float(value)
        return hints
