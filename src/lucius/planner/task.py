"""Natural-language task analysis (deterministic; a model can refine it but is not required)."""

from __future__ import annotations

import re

from lucius.planner.model import TaskSpec
from lucius.taxonomy import classify_task

UNIT = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "mm": 0.001}
# phrase -> parameter name (object-class specific names first)
PARAM_PHRASES = [
    (r"blade\s+length", "blade_length"), (r"blade\s+width", "blade_width"), (r"blade\s+thickness", "thickness"),
    (r"guard\s+width", "guard_size_x"), (r"guard\s+height", "guard_size_z"),
    (r"length", "length"), (r"width", "width"), (r"height", "height"), (r"thickness", "thickness"),
    (r"(?:number\s+of\s+)?loop\s*cuts?", "loop_cuts"),
]
QUALIFIERS = {
    "long": ("length", 1.3), "longer": ("length", 1.3), "short": ("length", 0.75), "shorter": ("length", 0.75),
    "wide": ("width", 1.3), "broad": ("width", 1.3), "wider": ("width", 1.3), "narrow": ("width", 0.75),
    "thin": ("width", 0.75), "slim": ("width", 0.75), "tall": ("height", 1.3), "flat": ("height", 0.6),
    "big": ("scale", 1.3), "large": ("scale", 1.3), "small": ("scale", 0.75), "tiny": ("scale", 0.5),
}
NUMBER = r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meters?)?"


def parse_task(text: str, reference_ids: list[str] | None = None) -> TaskSpec:
    object_class, categories = classify_task(text)
    lowered = text.lower()
    params: dict[str, float] = {}
    sources: dict[str, str] = {}
    for phrase, name in PARAM_PHRASES:
        if name in params:
            continue
        for pattern in (rf"{phrase}\s*(?:of|=|:|is)?\s*{NUMBER}", rf"{NUMBER}\s*(?:long\s+)?{phrase}"):
            match = re.search(pattern, lowered)
            if match:
                value = float(match.group(1)) * UNIT.get(match.group(2) or "m", 1.0)
                params[name] = int(value) if name == "loop_cuts" else round(value, 4)
                sources[name] = "task"
                break
    triple = re.search(rf"{NUMBER}\s*[x×]\s*{NUMBER}\s*[x×]\s*{NUMBER}", lowered)
    if triple:
        values = [float(triple.group(i)) * UNIT.get(triple.group(i + 1) or "m", 1.0) for i in (1, 3, 5)]
        params["dimensions"] = [round(v, 4) for v in values]
        sources["dimensions"] = "task"
    named = re.search(r"\b(?:named|called)\s+[\"']?([A-Za-z][A-Za-z0-9_]{0,40})", text)
    if named:
        params["object_name"] = named.group(1)
        sources["object_name"] = "task"
    primitive = re.search(r"\b(cube|cylinder|plane|cone|torus|uv sphere|sphere|ico sphere)\b", lowered)
    if primitive:
        params["kind"] = {"sphere": "uv_sphere", "uv sphere": "uv_sphere", "ico sphere": "ico_sphere"}.get(
            primitive.group(1), primitive.group(1))
        sources["kind"] = "task"
    words = set(re.findall(r"[a-z]+", lowered))
    qualifiers = {}
    for word, (dimension, factor) in QUALIFIERS.items():
        if word in words:
            qualifiers[dimension] = factor
    return TaskSpec(text=text, object_class=object_class, categories=list(categories), params=params,
                    qualifiers=qualifiers, reference_ids=reference_ids or [], param_sources=sources)
