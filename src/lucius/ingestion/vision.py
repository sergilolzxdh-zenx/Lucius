"""Vision-model analysis of media (optional; model output is always ``model_inferred``).

Called only on selected frames: before/after pairs around detected transitions (batched) and
reference images. Operation names are constrained to Lucius' action vocabulary so the model
cannot invent new operations silently; unknown is always an allowed answer.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from lucius.errors import ProviderError
from lucius.logging_setup import get_logger
from lucius.providers.base import ImageInput, Providers
from lucius.trajectory import vocabulary as vocab

log = get_logger("ingestion.vision")

CATEGORIES = ["geometry", "proportion", "detail", "material", "camera", "lighting", "ui", "selection", "none"]


def _operations() -> list[str]:
    return sorted(k for k in vocab.ACTION_TYPES if not k.startswith(("op.", "api."))) + ["material_change",
                                                                                       "lighting_change"]


def transition_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"transitions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "change_category": {"type": "string", "enum": CATEGORIES},
                "description": {"type": "string"},
                "visible_mode": {"type": ["string", "null"]},
                "visible_tool": {"type": ["string", "null"]},
                "visible_keystrokes": {"type": "array", "items": {"type": "string"}},
                "candidate_operations": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"operation": {"type": "string", "enum": _operations()},
                                   "confidence": {"type": "number"},
                                   "evidence": {"type": "string"}},
                    "required": ["operation", "confidence", "evidence"], "additionalProperties": False}},
            },
            "required": ["index", "change_category", "description", "visible_mode", "visible_tool",
                         "visible_keystrokes", "candidate_operations"],
            "additionalProperties": False}}},
        "required": ["transitions"], "additionalProperties": False,
    }


REFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object_class": {"type": "string"},
        "style": {"type": "string"},
        "view": {"type": "string", "enum": ["front", "side", "top", "perspective", "unknown"]},
        "parts": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "fraction_of_total_length": {"type": "number"}},
            "required": ["name", "fraction_of_total_length"], "additionalProperties": False}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["object_class", "style", "view", "parts", "notes"],
    "additionalProperties": False,
}

TRANSITION_SYSTEM = (
    "You analyse frames from Blender tutorials. For each numbered before/after pair, describe what changed and "
    "which Blender operation could explain it, using only the allowed operation names. Give several candidates "
    "with confidences when unsure, and include 'unknown_action' if the evidence is weak. Report the interaction "
    "mode and tool only if they are readable in the header, and keystroke overlays only if visible. When the "
    "narrator's words around a pair are given (automatic captions, possibly in another language and with "
    "recognition errors), use them as a hint about what is being done, but judge from the images: a narrator "
    "also explains things they do not do."
)


class VisionAnalyzer:
    def __init__(self, providers: Providers, max_images: int = 6) -> None:
        self.providers = providers
        self.max_images = max(2, max_images - max_images % 2)

    @property
    def available(self) -> bool:
        return self.providers.has("vlm")

    def describe_transitions(self, pairs: list[tuple[Image.Image, Image.Image]], context: str,
                             narration: list[str] | None = None,
                             indices: list[int] | None = None) -> dict[int, dict[str, Any]]:
        """Analyse before/after pairs in batches. ``indices`` names the pairs (default 0..n-1) so a
        selected subset keeps its original numbering; ``narration`` holds the words spoken around each."""
        if not self.available or not pairs:
            return {}
        indices = indices if indices is not None else list(range(len(pairs)))
        out: dict[int, dict[str, Any]] = {}
        per_call = self.max_images // 2
        for start in range(0, len(pairs), per_call):
            images = []
            spoken = []
            batch = list(zip(indices[start:start + per_call], pairs[start:start + per_call],
                             (narration or [""] * len(pairs))[start:start + per_call]))
            for i, (before, after), words in batch:
                images.append(ImageInput.from_image(before, label=f"Pair {i} BEFORE:", max_side=960))
                images.append(ImageInput.from_image(after, label=f"Pair {i} AFTER:", max_side=960))
                if words:
                    spoken.append(f"Narration around pair {i}: \"{words[:600]}\"")
            try:
                result = self.providers.vlm.complete_json(
                    purpose="media_transition_analysis", system=TRANSITION_SYSTEM,
                    prompt="\n".join([f"Context: {context}", *spoken,
                                       "Analyse pairs " + ", ".join(str(b[0]) for b in batch) + "."]),
                    schema=transition_schema(), images=images, max_tokens=6000)
            except ProviderError as exc:
                log.warning("transition analysis unavailable: %s", exc.message)
                return out
            for item in result.data.get("transitions", []):
                item["model"] = result.model
                out[int(item.get("index", -1))] = item
        return out

    def describe_reference(self, image: Image.Image) -> dict[str, Any] | None:
        if not self.available:
            return None
        try:
            result = self.providers.vlm.complete_json(
                purpose="reference_analysis",
                system="Describe the object in this reference image for 3D modelling: class, style, the view it "
                       "is drawn from, and its main parts as fractions of total length. Only state what is visible.",
                prompt="Return the JSON description.", schema=REFERENCE_SCHEMA,
                images=[ImageInput.from_image(image, max_side=1024)], max_tokens=3000)
        except ProviderError as exc:
            log.warning("reference analysis unavailable: %s", exc.message)
            return None
        return {**result.data, "model": result.model}
