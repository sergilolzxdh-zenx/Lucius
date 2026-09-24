"""Curricula: staged practice structures (sections 39-41).

Curricula are *system-defined structure* (what to practise, how success is measured, when a
stage counts as mastered) -- not learned knowledge. Each practice task samples parameters from
ranges so that repeated practice exercises generalisation rather than one memorised instance.
Stages whose tasks need capabilities the agent has not been shown report
``needs_demonstration`` instead of pretending to practise.
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from lucius.skills.schema import Checkpoint


class MasteryGate(BaseModel):
    min_attempts: int = 5
    completion_rate: float = 0.8
    checkpoint_pass_rate: float = 0.85
    max_takeover_rate: float = 0.2
    max_false_success_rate: float = 0.1
    min_consistency: float = 0.7
    min_generalization: float = 0.5      # share of successes on distinct parameter instances
    min_human_rating: float | None = None


class PracticeTaskTemplate(BaseModel):
    name: str
    task: str                                     # text with {param} placeholders
    params: dict[str, Any] = Field(default_factory=dict)   # name -> [min, max] | list of choices | constant
    criteria: list[dict[str, Any]] = Field(default_factory=list)  # checkpoint dicts with {param} references
    task_params: dict[str, Any] = Field(default_factory=dict)     # skill parameter values, may reference params
    reference: str | None = None                  # synthetic reference generator key
    # Fixed references for other parts of the result: {"generator", "values", "target"}.
    part_references: list[dict[str, Any]] = Field(default_factory=list)
    requires_gui: bool = False
    difficulty: int = 1

    def sample(self, rng: random.Random) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, spec in self.params.items():
            if isinstance(spec, list) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
                lo, hi = spec
                values[name] = rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else round(rng.uniform(lo, hi), 3)
            elif isinstance(spec, list):
                values[name] = rng.choice(spec)
            else:
                values[name] = spec
        return values

    def instantiate(self, values: dict[str, Any]) -> tuple[str, list[Checkpoint], dict[str, Any]]:
        derived = derive(values)
        text = self.task.format(**derived)
        params = _fill(self.task_params, derived)
        checkpoints = []
        for i, c in enumerate(self.criteria):
            check = _fill(c["check"], derived)
            checkpoints.append(Checkpoint(id=f"task_{self.name}_{i}", description=c["description"].format(**derived),
                                          level=c.get("level", 2), method=c.get("method", "structural"),
                                          check=check, required=c.get("required", True)))
        return text, checkpoints, params


def derive(values: dict[str, Any]) -> dict[str, Any]:
    """Derived quantities used in task texts/criteria (e.g. the expected height after an extrusion)."""
    out = dict(values)
    if "extrude" in values:
        out["extruded_height"] = round(2.0 + float(values["extrude"]), 4)
    if "cuts" in values:
        out["cut_verts"] = 8 + 4 * int(values["cuts"])
    return out


def _fill(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("{") and value.endswith("}") and value[1:-1] in params:
        return params[value[1:-1]]
    if isinstance(value, list):
        return [_fill(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, params) for k, v in value.items()}
    return value


class Stage(BaseModel):
    index: int
    name: str
    description: str
    tasks: list[PracticeTaskTemplate] = Field(default_factory=list)
    gate: MasteryGate = Field(default_factory=MasteryGate)
    requires_gui: bool = False


class Curriculum(BaseModel):
    name: str
    title: str
    description: str
    stages: list[Stage]


# -- synthetic reference generators ---------------------------------------------------------------

def blade_reference(values: dict[str, Any], size: int = 512) -> Image.Image:
    """Front-view silhouette of a blade: full width up to ``tip_start`` of the length (default 2/3),
    then tapering linearly to ``tip_width`` (default 20%) of the width."""
    length, width = float(values["length"]), float(values["width"])
    scale = (size * 0.9) / max(length, width)
    w, h = width * scale, length * scale
    cx, bottom = size / 2, size * 0.95
    top = bottom - h
    shoulder = bottom - h * float(values.get("tip_start", 2 / 3))
    tip = w * float(values.get("tip_width", 0.2))
    image = Image.new("RGB", (size, size), (235, 235, 235))
    ImageDraw.Draw(image).polygon([(cx - w / 2, bottom), (cx + w / 2, bottom), (cx + w / 2, shoulder),
                                   (cx + tip / 2, top), (cx - tip / 2, top), (cx - w / 2, shoulder)], fill=(40, 40, 40))
    return image


def box_reference(values: dict[str, Any], size: int = 512) -> Image.Image:
    """Front-view silhouette of a box ``width`` x ``height``."""
    width, height = float(values["width"]), float(values["height"])
    scale = (size * 0.9) / max(width, height)
    w, h = width * scale, height * scale
    image = Image.new("RGB", (size, size), (235, 235, 235))
    ImageDraw.Draw(image).rectangle([(size - w) / 2, (size - h) / 2, (size + w) / 2, (size + h) / 2], fill=(40, 40, 40))
    return image


# generator key -> (image function, object role the reference depicts)
REFERENCE_GENERATORS = {"blade": (blade_reference, "blade"), "box": (box_reference, None)}

# The cross-guard of the sword tasks, as a user would supply it on a reference sheet.
GUARD_REFERENCE = {"generator": "box", "values": {"width": 1.6, "height": 0.2}, "target": "guard"}

DIMENSION_CHECK = {"description": "{name} has dimensions {x}x{y}x{z}",
                   "check": {"type": "dimensions_match", "object": "{name}", "vector": ["{x}", "{y}", "{z}"],
                             "tolerance": 0.05}}


def _curricula() -> dict[str, Curriculum]:
    navigation = Stage(index=1, name="Navigation", requires_gui=True,
                       description="Orbit, pan, zoom and orthographic views in the interactive viewport",
                       tasks=[PracticeTaskTemplate(
                           name="ortho_views", task="Show the front and then the right orthographic view",
                           requires_gui=True, criteria=[{"description": "front and right views were shown", "level": 4,
                                                         "method": "human", "check": {"type": "human"}}])])
    primitives = Stage(index=2, name="Primitive manipulation", description="Create and size primitives precisely",
                       tasks=[PracticeTaskTemplate(
                           name="sized_primitive", task="Add a cube named {name} with dimensions {x}x{y}x{z}",
                           params={"name": "Prop", "x": [0.5, 3.0], "y": [0.5, 3.0], "z": [0.5, 3.0]},
                           task_params={"object_name": "{name}", "kind": "cube", "dimensions": ["{x}", "{y}", "{z}"]},
                           criteria=[{"description": "{name} exists",
                                      "check": {"type": "object_exists", "object": "{name}"}}, DIMENSION_CHECK])])
    edit_basics = Stage(index=3, name="Edit Mode basics", description="Extrude, inset and loop cuts on a primitive",
                        tasks=[PracticeTaskTemplate(
                            name="extrude_top", task="Add a cube named {name} and extrude its top by {extrude}",
                            params={"name": "Block", "extrude": [0.2, 1.5]},
                            task_params={"object_name": "{name}", "kind": "cube", "axis": "z", "distance": "{extrude}"},
                            criteria=[{"description": "{name} is {extruded_height} tall after extrusion",
                                       "check": {"type": "dimensions_match", "object": "{name}",
                                                 "axes": {"z": "{extruded_height}"}, "tolerance": 0.05}}])])
    hard_surface = Stage(index=4, name="Structured hard-surface",
                         description="Block out hard-surface parts from parameters (requires a demonstrated skill)",
                         tasks=[PracticeTaskTemplate(
                             name="blade_blockout", task="Block out a sword blade with blade length {length} "
                                                         "and blade width {width}",
                             params={"length": [3.0, 7.0], "width": [0.2, 0.45]}, reference="blade",
                             part_references=[GUARD_REFERENCE],
                             task_params={"blade_length": "{length}", "blade_width": "{width}"},
                             criteria=[{"description": "blade is {length} long and {width} wide",
                                        "check": {"type": "dimensions_match", "object": "Blade",
                                                  "axes": {"z": "{length}", "x": "{width}"}, "tolerance": 0.08}}])])
    topology = Stage(index=5, name="Topology awareness", description="Controlled edge loops and vertex counts",
                     tasks=[PracticeTaskTemplate(
                         name="loop_cuts", task="Add a cube named {name} with {cuts} loop cuts along z",
                         params={"name": "Grid", "cuts": [1, 4]},
                         task_params={"object_name": "{name}", "kind": "cube", "axis": "z", "cuts": "{cuts}"},
                         criteria=[{"description": "{name} has {cut_verts} vertices",
                                    "check": {"type": "vertex_count", "object": "{name}", "min": "{cut_verts}",
                                              "max": "{cut_verts}"}}])])
    reference = Stage(index=6, name="Reference matching", description="Match a reference silhouette",
                      tasks=[PracticeTaskTemplate(
                          name="blade_reference", task="Model the sword blade shown in the reference "
                                                       "(blade length {length}, blade width {width})",
                          params={"length": [3.0, 7.0], "width": [0.2, 0.45]}, reference="blade",
                          part_references=[GUARD_REFERENCE],
                          task_params={"blade_length": "{length}", "blade_width": "{width}"},
                          criteria=[{"description": "silhouette matches the reference from the front", "level": 3,
                                     "method": "visual_measured",
                                     "check": {"type": "silhouette", "object": "Blade", "views": ["front"]}}])])
    organic = Stage(index=7, name="Organic forms", description="Soft, curved forms (needs demonstrations)",
                    tasks=[PracticeTaskTemplate(name="organic_blob", task="Model a smooth organic pebble",
                                                criteria=[{"description": "a smooth pebble exists", "level": 4,
                                                           "method": "human", "check": {"type": "human"}}])])
    sculpt = Stage(index=8, name="Sculpting basics", requires_gui=True, description="Brush-based sculpting",
                   tasks=[PracticeTaskTemplate(name="sculpt_basics", task="Sculpt a simple rock", requires_gui=True,
                                               criteria=[{"description": "a sculpted rock exists", "level": 4,
                                                          "method": "human", "check": {"type": "human"}}])])
    anatomy = Stage(index=9, name="Anatomy modules", requires_gui=True, description="Head and body proportions",
                    tasks=[PracticeTaskTemplate(name="head_proportions", task="Sculpt a head with basic proportions",
                                                requires_gui=True,
                                                criteria=[{"description": "head proportions are plausible", "level": 4,
                                                           "method": "human", "check": {"type": "human"}}])])
    integrated = Stage(index=10, name="Integrated asset production", description="Complete multi-part assets",
                       tasks=[PracticeTaskTemplate(
                           name="sword_asset", task="Model a sword with blade length {length} and blade width {width}",
                           params={"length": [3.0, 7.0], "width": [0.2, 0.45]}, reference="blade",
                           task_params={"blade_length": "{length}", "blade_width": "{width}"},
                           criteria=[{"description": "blade and guard exist",
                                      "check": {"type": "object_count", "min": 2, "max": 50}},
                                     {"description": "blade is {length} long",
                                      "check": {"type": "dimensions_match", "object": "Blade",
                                                "axes": {"z": "{length}"}, "tolerance": 0.08}}])])
    return {
        "sculpting": Curriculum(name="sculpting", title="Train sculpting",
                                description="From navigation to anatomy and integrated assets",
                                stages=[navigation, primitives, edit_basics, hard_surface, topology, reference, organic,
                                        sculpt, anatomy, integrated]),
        "hard_surface": Curriculum(name="hard_surface", title="Train hard surface",
                                   description="Precise primitives to multi-part hard-surface assets",
                                   stages=[s.model_copy(update={"index": i + 1}) for i, s in
                                           enumerate([primitives, edit_basics, hard_surface, topology, integrated])]),
        "topology": Curriculum(name="topology", title="Train topology",
                               description="Edge loops, insets and controlled vertex counts",
                               stages=[s.model_copy(update={"index": i + 1}) for i, s in
                                       enumerate([edit_basics, topology])]),
        "reference_matching": Curriculum(name="reference_matching", title="Train reference matching",
                                         description="Match target silhouettes measured from references",
                                         stages=[s.model_copy(update={"index": i + 1}) for i, s in
                                                 enumerate([hard_surface, reference])]),
    }


CURRICULA = _curricula()
ALIASES = {"sculpt": "sculpting", "hard surface": "hard_surface", "hard-surface": "hard_surface",
           "reference": "reference_matching", "reference matching": "reference_matching"}


def resolve_curriculum(name: str) -> Curriculum | None:
    key = name.strip().lower().removeprefix("train").strip()
    key = ALIASES.get(key, key.replace(" ", "_"))
    return CURRICULA.get(key)
