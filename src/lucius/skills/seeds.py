"""Minimal system-seeded skills (section 70).

These describe basic Blender capabilities the agent can execute through the bridge. They are
marked ``system_seeded`` and carry no demonstrations, so they start at neutral confidence and
are never presented as something the user taught. They exist so compound plans have
building blocks before any demonstration covers them.
"""

from __future__ import annotations

from lucius.skills.library import SkillLibrary
from lucius.skills.schema import ActionTemplate, Checkpoint, ParamSpec, SkillDefinition, SkillPhase, SkillStatus

SEED = "system_seeded"


def _single(skill_id: str, name: str, purpose: str, action: ActionTemplate, *, params: list[ParamSpec] | None = None,
            categories: list[str] | None = None, triggers: list[str] | None = None,
            checkpoints: list[Checkpoint] | None = None, phase: str = "primary_form") -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id, name=name, purpose=purpose, categories=categories or ["fundamentals"],
        triggers=triggers or [], parameters=params or [], checkpoints=checkpoints or [],
        phases=[SkillPhase(name=phase, actions=[action], checkpoints=[c.id for c in checkpoints or []])],
        source_class=SEED, notes=["system seeded capability; not learned from the user"],
    )


def seed_definitions() -> list[SkillDefinition]:
    obj = ParamSpec(name="object_name", kind="str", description="target object", source="task")
    views_cp = Checkpoint(id="views_inspected", description="model viewed from front and side", level=4,
                          method="human", required=False, check={"type": "human", "prompt": "Were both views shown?"})
    return [
        _single("seed_viewport_orbit", "Viewport orbit", "Rotate the view around the scene",
                ActionTemplate(action_type="viewport_orbit", args={"direction": "{direction}", "angle": "{angle}"},
                               object_ref=None),
                params=[ParamSpec(name="direction", kind="enum", default="ORBITLEFT",
                                  choices=["ORBITLEFT", "ORBITRIGHT", "ORBITUP", "ORBITDOWN"]),
                        ParamSpec(name="angle", default=0.2618, unit="rad")],
                categories=["navigation"], triggers=["orbit", "rotate view", "navigation"], phase="inspection"),
        _single("seed_view_preset", "Orthographic view", "Switch to a front/side/top orthographic view",
                ActionTemplate(action_type="view_preset", args={"view": "{view}", "ortho": True}, object_ref=None,
                               gui_hint=["NUMPAD_1"]),
                params=[ParamSpec(name="view", kind="enum", default="FRONT",
                                  choices=["FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"])],
                categories=["navigation"], triggers=["front view", "side view", "top view", "orthographic"],
                phase="inspection"),
        _single("seed_reference_inspection", "Reference inspection",
                "Look at the model from front and side orthographic views",
                ActionTemplate(action_type="view_preset", args={"views": ["FRONT", "RIGHT"], "ortho": True},
                               object_ref=None),
                categories=["navigation", "inspection"], triggers=["inspect", "check silhouette", "reference"],
                checkpoints=[views_cp], phase="inspection"),
        _single("seed_add_primitive", "Add primitive", "Add a mesh primitive to the scene",
                ActionTemplate(action_type="add_primitive", args={"kind": "{kind}", "name": "{object_name}",
                                                                  "size": "{size}"}, requires_mode="OBJECT",
                               gui_hint=["SHIFT+A"]),
                params=[ParamSpec(name="kind", kind="enum", default="cube",
                                  choices=["cube", "plane", "cylinder", "cone", "uv_sphere", "ico_sphere", "torus"]),
                        ParamSpec(name="size", default=2.0, unit="m"), obj],
                triggers=["add", "create", "primitive", "cube", "cylinder"],
                checkpoints=[Checkpoint(id="object_exists", description="object exists",
                                        check={"type": "object_exists", "object": "{object_name}"})]),
        _single("seed_select_object", "Select object", "Make an object selected and active",
                ActionTemplate(action_type="select_objects", args={"names": ["{object_name}"]}, requires_mode="OBJECT"),
                params=[obj], triggers=["select"]),
        _single("seed_move_object", "Move object", "Translate an object to a location",
                ActionTemplate(action_type="translate", args={"location": "{location}", "space": "object"},
                               requires_mode="OBJECT", gui_hint=["G"]),
                params=[obj, ParamSpec(name="location", kind="vec3", default=[0.0, 0.0, 0.0], unit="m")],
                triggers=["move", "translate", "position"]),
        _single("seed_rotate_object", "Rotate object", "Rotate an object (Euler, radians)",
                ActionTemplate(action_type="rotate", args={"rotation": "{rotation}", "space": "object"},
                               requires_mode="OBJECT", gui_hint=["R"]),
                params=[obj, ParamSpec(name="rotation", kind="vec3", default=[0.0, 0.0, 0.0], unit="rad")],
                triggers=["rotate", "orientation"]),
        _single("seed_scale_object", "Resize object", "Set an object's dimensions",
                ActionTemplate(action_type="set_dimensions", args={"dimensions": "{dimensions}"},
                               requires_mode="OBJECT", gui_hint=["S"]),
                params=[obj, ParamSpec(name="dimensions", kind="vec3", default=[1.0, 1.0, 1.0], unit="m")],
                triggers=["scale", "resize", "dimensions", "size"],
                checkpoints=[Checkpoint(id="dimensions", description="object has the requested dimensions",
                                        check={"type": "dimensions_match", "object": "{object_name}",
                                               "vector": "{dimensions}", "tolerance": 0.05})]),
        _single("seed_rename", "Rename object", "Give an object a meaningful name",
                ActionTemplate(action_type="rename", args={"new_name": "{new_name}"}, gui_hint=["F2"]),
                params=[obj, ParamSpec(name="new_name", kind="str", default="Object")], triggers=["rename", "name"]),
        _single("seed_save", "Save file", "Save the .blend file to an allowed location",
                ActionTemplate(action_type="save", args={"path": "{path}"}, object_ref=None, gui_hint=["CTRL+S"]),
                params=[ParamSpec(name="path", kind="str", default=None)], triggers=["save"], phase="verification"),
        _single("seed_object_mode", "Object mode", "Switch to object mode",
                ActionTemplate(action_type="mode_change", args={"mode": "OBJECT"}, gui_hint=["TAB"]),
                params=[obj], triggers=["object mode"]),
        _single("seed_edit_mode", "Edit mode", "Switch to edit mode",
                ActionTemplate(action_type="mode_change", args={"mode": "EDIT"}, gui_hint=["TAB"]),
                params=[obj], triggers=["edit mode"]),
        _single("seed_extrude", "Extrude", "Extrude the selected faces along an axis",
                ActionTemplate(action_type="extrude", args={"axis": "{axis}", "distance": "{distance}"},
                               requires_mode="EDIT", gui_hint=["E"]),
                params=[obj, ParamSpec(name="axis", kind="enum", default="z", choices=["x", "y", "z"]),
                        ParamSpec(name="distance", default=0.5, unit="m")], triggers=["extrude"]),
        _single("seed_inset", "Inset faces", "Inset the selected faces",
                ActionTemplate(action_type="inset", args={"thickness": "{thickness}"}, requires_mode="EDIT",
                               gui_hint=["I"]),
                params=[obj, ParamSpec(name="thickness", default=0.05, unit="m")], triggers=["inset"]),
        _single("seed_loop_cut", "Loop cut", "Add edge loops across an axis",
                ActionTemplate(action_type="loop_cut", args={"axis": "{axis}", "cuts": "{cuts}"}, requires_mode="EDIT",
                               gui_hint=["CTRL+R"]),
                params=[obj, ParamSpec(name="axis", kind="enum", default="z", choices=["x", "y", "z"]),
                        ParamSpec(name="cuts", kind="int", default=1, unit="count")], triggers=["loop cut", "edge loop"]),
        _single("seed_bevel", "Bevel edges", "Bevel the selected edges",
                ActionTemplate(action_type="bevel", args={"offset": "{offset}", "segments": "{segments}"},
                               requires_mode="EDIT", gui_hint=["CTRL+B"]),
                params=[obj, ParamSpec(name="offset", default=0.02, unit="m"),
                        ParamSpec(name="segments", kind="int", default=2, unit="count")],
                triggers=["bevel", "round edges"], phase="detail"),
        _single("seed_mirror_symmetry", "Mirror symmetry", "Add a mirror modifier for symmetric modelling",
                ActionTemplate(action_type="add_modifier", args={"type": "MIRROR"}),
                params=[obj], triggers=["mirror", "symmetry", "symmetric"], phase="setup",
                checkpoints=[Checkpoint(id="mirror_present", description="mirror modifier present",
                                        check={"type": "modifier_present", "object": "{object_name}",
                                               "modifier": "MIRROR"})]),
    ]


def seed_system_skills(library: SkillLibrary) -> list[str]:
    created = []
    for definition in seed_definitions():
        if library.exists(definition.skill_id):
            continue
        library.create(definition, created_by="system", change_note="system seeded",
                       status=SkillStatus.CANDIDATE_SKILL)
        library.rescore(definition.skill_id)
        created.append(definition.skill_id)
    return created
