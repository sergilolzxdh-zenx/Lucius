"""Before/after state-transition analysis (10F, 10H).

Deterministic image evidence first: *where* the image changed, *how much*, and how the
foreground object's silhouette changed. This yields a change category (camera, ui, geometry,
selection, material/lighting) and several candidate operations with confidences -- never a
single forced answer. A vision model can add candidates later (recorded as model_inferred).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

from lucius.evaluation.silhouette import otsu_threshold
from lucius.trajectory.model import CandidateAction

WORK = 320
UI_TOP = 0.08          # Blender header band
UI_RIGHT = 0.78        # properties/outliner column starts about here in the default layout
UI_LEFT = 0.05         # toolbar


class Transition(BaseModel):
    t_before: float
    t_after: float
    magnitude: float                       # mean absolute difference (0..1)
    changed_fraction: float                # share of pixels that changed noticeably
    bbox: list[float] | None = None        # normalised x0, y0, x1, y1 of the changed region
    kind: str                              # camera, ui, geometry, selection, material_or_lighting, none
    candidates: list[CandidateAction] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    shape: dict[str, Any] = Field(default_factory=dict)


def _prep(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    w, h = image.size
    size = (WORK, max(1, round(h * WORK / w)))
    rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.int16)
    gray = np.asarray(image.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.int16)
    return rgb, gray


def _edges(gray: np.ndarray) -> np.ndarray:
    img = Image.fromarray(gray.clip(0, 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
    return np.asarray(img) > 40


def _foreground(gray: np.ndarray, region: tuple[slice, slice]) -> np.ndarray:
    """Object pixels in the viewport: the minority side of an Otsu luminance split.

    Robust to smooth background gradients (a Blender viewport), unlike a distance-to-border test.
    """
    area = gray[region].astype(np.float64)
    threshold = otsu_threshold(area.ravel())
    bright = area > threshold
    return bright if bright.mean() <= 0.5 else ~bright


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    return {k: round(v / total, 3) for k, v in scores.items()} if total > 0 else scores


def analyze_transition(before: Image.Image, after: Image.Image, *, t_before: float = 0.0,
                       t_after: float = 0.0) -> Transition:
    rgb_a, gray_a = _prep(before)
    rgb_b, gray_b = _prep(after.resize(before.size))
    diff = np.abs(gray_a - gray_b)
    changed = diff > 25
    h, w = changed.shape
    fraction = float(changed.mean())
    magnitude = float(diff.mean() / 255.0)
    box = _bbox(changed)
    evidence = [f"{fraction:.1%} of pixels changed", f"mean difference {magnitude:.3f}"]
    if box is None or fraction < 0.001:
        return Transition(t_before=t_before, t_after=t_after, magnitude=magnitude, changed_fraction=fraction,
                          kind="none", evidence=evidence + ["no significant change"])
    nb = [round(box[0] / w, 3), round(box[1] / h, 3), round(box[2] / w, 3), round(box[3] / h, 3)]
    evidence.append(f"change region x {nb[0]:.2f}-{nb[2]:.2f}, y {nb[1]:.2f}-{nb[3]:.2f}")
    edges_a, edges_b = _edges(gray_a), _edges(gray_b)
    union = np.logical_or(edges_a, edges_b).sum()
    edge_iou = float(np.logical_and(edges_a, edges_b).sum() / union) if union else 1.0
    colour_shift = float(np.abs(rgb_a.mean(axis=(0, 1)) - rgb_b.mean(axis=(0, 1))).mean() / 255.0)

    if fraction > 0.45:
        if edge_iou > 0.6 and colour_shift > 0.04:
            kind = "material_or_lighting"
            scores = {"material_change": 0.5, "lighting_change": 0.4, "unknown_action": 0.1}
            evidence.append(f"edges unchanged (IoU {edge_iou:.2f}) while colours shifted {colour_shift:.2f}")
        else:
            kind = "camera"
            scores = {"viewport_orbit": 0.5, "viewport_pan": 0.2, "viewport_zoom": 0.2, "view_preset": 0.1}
            evidence.append(f"most of the image moved (edge IoU {edge_iou:.2f}): viewport/camera change")
        return Transition(t_before=t_before, t_after=t_after, magnitude=magnitude, changed_fraction=fraction, bbox=nb,
                          kind=kind, candidates=_candidates(scores, evidence), evidence=evidence)

    in_header = nb[3] <= UI_TOP + 0.02
    in_side = nb[0] >= UI_RIGHT or nb[2] <= UI_LEFT + 0.02
    if in_header or in_side:
        kind = "ui"
        if in_header:
            scores = {"mode_change": 0.35, "tool_change": 0.25, "ui_click": 0.4}
            evidence.append("change confined to the header band (mode/tool indicators)")
        else:
            scores = {"add_modifier": 0.3, "ui_click": 0.5, "tool_change": 0.2}
            evidence.append("change confined to a side panel (properties/toolbar)")
        return Transition(t_before=t_before, t_after=t_after, magnitude=magnitude, changed_fraction=fraction, bbox=nb,
                          kind=kind, candidates=_candidates(scores, evidence), evidence=evidence)

    # Viewport change: compare the foreground object's silhouette before and after.
    region = (slice(int(UI_TOP * h), h), slice(int(UI_LEFT * w), int(UI_RIGHT * w)))
    fg_a, fg_b = _foreground(gray_a, region), _foreground(gray_b, region)
    ba, bb = _bbox(fg_a), _bbox(fg_b)
    shape: dict[str, Any] = {"area_before": int(fg_a.sum()), "area_after": int(fg_b.sum())}
    viewport_fraction = float(changed[region].mean())
    if ba is None or bb is None:
        kind, scores = "geometry", {"add_primitive": 0.4, "delete": 0.3, "unknown_action": 0.3}
        evidence.append("an object appeared or disappeared in the viewport")
        shape["appeared"] = ba is None and bb is not None
    else:
        wa, ha = ba[2] - ba[0], ba[3] - ba[1]
        wb, hb = bb[2] - bb[0], bb[3] - bb[1]
        aspect_a, aspect_b = ha / max(1, wa), hb / max(1, wb)
        area_ratio = shape["area_after"] / max(1, shape["area_before"])
        shift = (abs((bb[0] + bb[2]) - (ba[0] + ba[2])) + abs((bb[1] + bb[3]) - (ba[1] + ba[3]))) / 2
        shape.update(bbox_before=list(ba), bbox_after=list(bb), aspect_before=round(aspect_a, 3),
                     aspect_after=round(aspect_b, 3), area_ratio=round(area_ratio, 3), centre_shift_px=round(shift, 1))
        if viewport_fraction < 0.01 and abs(area_ratio - 1) < 0.05:
            kind, scores = "selection", {"select_click": 0.6, "select_box": 0.2, "unknown_action": 0.2}
            evidence.append("tiny change without silhouette change: likely selection highlight")
        else:
            kind = "geometry"
            aspect_change = abs(np.log(max(aspect_b, 1e-3) / max(aspect_a, 1e-3)))
            scores = {"scale": 0.1, "extrude": 0.1, "translate": 0.1, "inset": 0.05, "bevel": 0.05,
                      "loop_cut": 0.05, "unknown_action": 0.1}
            if aspect_change > 0.15:
                grew_one_side = (abs(bb[0] - ba[0]) < 3) != (abs(bb[2] - ba[2]) < 3) or \
                                (abs(bb[1] - ba[1]) < 3) != (abs(bb[3] - ba[3]) < 3)
                if grew_one_side and area_ratio > 1.05:
                    scores["extrude"] += 0.5
                    evidence.append("silhouette grew on one side: extrusion-like")
                else:
                    scores["scale"] += 0.55
                    evidence.append(f"silhouette aspect changed {aspect_a:.2f} -> {aspect_b:.2f}: scaling-like")
            elif shift > 4 and abs(area_ratio - 1) < 0.1:
                scores["translate"] += 0.5
                evidence.append(f"silhouette moved {shift:.0f}px with constant size")
            elif abs(area_ratio - 1) < 0.05:
                scores["bevel"] += 0.15
                scores["loop_cut"] += 0.15
                scores["inset"] += 0.1
                evidence.append("interior change without silhouette change: detail/topology-like")
            else:
                scores["scale"] += 0.3
                evidence.append(f"silhouette area changed x{area_ratio:.2f}")
    return Transition(t_before=t_before, t_after=t_after, magnitude=magnitude, changed_fraction=fraction, bbox=nb,
                      kind=kind, candidates=_candidates(scores, evidence), evidence=evidence, shape=shape)


def _candidates(scores: dict[str, float], evidence: list[str]) -> list[CandidateAction]:
    normalised = _normalise(scores)
    return sorted((CandidateAction(action_type=k, confidence=v, evidence=evidence[-1:])
                   for k, v in normalised.items() if v > 0), key=lambda c: c.confidence, reverse=True)
