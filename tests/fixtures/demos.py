"""Development fixture: scripted demonstrations.

Produces exactly the raw event stream the recorder would capture (input events, add-on
operator reports, state pushes, undo handlers) for a scripted Blender session. Used only by
tests to exercise the processing pipeline deterministically -- never by production code.
"""

from __future__ import annotations

import copy
from typing import Any

from lucius.sessions import Actor, CapturedEvent, EventKind


class DemoScript:
    def __init__(self, t0: float = 1_700_000_000.0) -> None:
        self.t = t0
        self.seq = 0
        self.events: list[CapturedEvent] = []
        self.values: dict[str, Any] = {
            "blender_version": "4.5.14 LTS", "mode": "OBJECT", "workspace": "Layout",
            "active_tool": "builtin.select_box", "active_object": "Cube", "selected_objects": ["Cube"],
            "object_count": 3, "viewport": {"named_view": "user", "perspective": "PERSP"},
            "active_object_summary": {"name": "Cube", "dimensions": [2.0, 2.0, 2.0], "modifiers": [],
                                      "mesh": {"verts": 8, "faces": 6}},
        }
        self.window = {"allowed": True, "is_blender": True, "title": "untitled.blend - Blender 4.5",
                       "bounds": {"x": 0, "y": 0, "width": 1600, "height": 900}}
        self._emit(EventKind.WINDOW_CONTEXT, self.window, actor=Actor.SYSTEM)
        self.state(reason="initial")

    # -- primitives -------------------------------------------------------------------------------
    def _emit(self, kind: EventKind, payload: dict[str, Any], actor: Actor = Actor.HUMAN,
              blender_active: bool | None = True) -> None:
        self.events.append(CapturedEvent(seq=self.seq, ts=self.t, kind=kind, actor=actor,
                                         blender_active=blender_active, payload=copy.deepcopy(payload)))
        self.seq += 1

    def wait(self, seconds: float) -> DemoScript:
        self.t += seconds
        return self

    def state(self, reason: str | None = None, geometry_changed: list[str] | None = None, **changes: Any) -> DemoScript:
        for key, value in changes.items():
            if key == "view":
                self.values["viewport"] = {**self.values["viewport"], "named_view": value}
            elif key == "perspective":
                self.values["viewport"] = {**self.values["viewport"], "perspective": value}
            elif key in ("dimensions", "modifiers", "mesh", "name"):
                summary = self.values["active_object_summary"]
                summary[key] = [{"type": m, "name": m.title()} for m in value] if key == "modifiers" else value
            else:
                self.values[key] = value
        payload = {"ts": self.t, "values": self.values, "unavailable": {}, "geometry_changed": geometry_changed or [],
                   "reason": reason}
        self._emit(EventKind.BLENDER_STATE, payload, actor=Actor.SYSTEM)
        self.wait(0.05)
        return self

    def key(self, key: str, char: str | None = None, modifiers: list[str] | None = None,
            actor: Actor = Actor.HUMAN) -> DemoScript:
        payload = {"key": key, "char": char, "modifiers": modifiers or []}
        self._emit(EventKind.KEY_DOWN, payload, actor=actor)
        self.wait(0.08)
        self._emit(EventKind.KEY_UP, payload, actor=actor)
        self.wait(0.12)
        return self

    def type_number(self, text: str) -> DemoScript:
        names = {".": "PERIOD", "-": "MINUS"}
        for ch in text:
            self.key(names.get(ch, {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE",
                                    "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE"}.get(ch, ch)), char=ch)
        return self

    def operator(self, idname: str, properties: dict[str, Any] | None = None, macros: list[dict] | None = None,
                 geometry_changed: list[str] | None = None, adjusted: bool = False) -> DemoScript:
        op: dict[str, Any] = {"idname": idname, "name": idname, "properties": properties or {}}
        if macros:
            op["macros"] = macros
        self._emit(EventKind.BLENDER_OPERATOR, {"operator": op, "adjusted": adjusted,
                                                "geometry_changed": geometry_changed or []})
        self.wait(0.05)
        return self

    def click(self, button: str = "LEFT", x: float = 800, y: float = 450) -> DemoScript:
        self._emit(EventKind.MOUSE_DOWN, {"x": x, "y": y, "wx": x, "wy": y, "button": button})
        self.wait(0.1)
        self._emit(EventKind.MOUSE_UP, {"x": x, "y": y, "wx": x, "wy": y, "button": button})
        self.wait(0.1)
        return self

    def drag(self, button: str = "MIDDLE", dx: float = 120, dy: float = 40, modifiers: list[str] | None = None) -> DemoScript:
        for m in modifiers or []:
            self._emit(EventKind.KEY_DOWN, {"key": m, "char": None, "modifiers": []})
        x, y = 800.0, 450.0
        self._emit(EventKind.MOUSE_DOWN, {"x": x, "y": y, "wx": x, "wy": y, "button": button})
        for i in range(1, 5):
            self.wait(0.1)
            self._emit(EventKind.MOUSE_MOVE, {"x": x + dx * i / 4, "y": y + dy * i / 4})
        self._emit(EventKind.MOUSE_UP, {"x": x + dx, "y": y + dy, "button": button})
        for m in modifiers or []:
            self._emit(EventKind.KEY_UP, {"key": m, "char": None, "modifiers": []})
        self.wait(0.1)
        return self

    def undo(self) -> DemoScript:
        self.key("Z", modifiers=["CTRL"])
        self._emit(EventKind.UNDO, {})
        self.wait(0.05)
        return self

    def annotate(self, text: str) -> DemoScript:
        self._emit(EventKind.ANNOTATION, {"text": text, "label": None}, actor=Actor.HUMAN, blender_active=None)
        return self

    # -- composite Blender interactions -------------------------------------------------------------
    def add_cube(self, name: str = "Blade") -> DemoScript:
        self.key("A", modifiers=["SHIFT"]).click(x=820, y=300)
        self.operator("MESH_OT_primitive_cube_add", {"size": 2.0, "location": [0, 0, 0]}, geometry_changed=[name])
        self.state(active_object=name, selected_objects=[name], object_count=self.values["object_count"] + 1,
                   dimensions=[2.0, 2.0, 2.0], mesh={"verts": 8, "faces": 6}, name=name)
        return self

    def tab(self, to_mode: str) -> DemoScript:
        self.key("TAB")
        self.operator("OBJECT_OT_editmode_toggle")
        self.state(mode=to_mode)
        return self

    def scale(self, axis: str, value: float, dims: list[float] | None = None) -> DemoScript:
        self.key("S").key(axis.upper()).type_number(str(value)).key("RET")
        vec = [value if a == axis.lower() else 1.0 for a in "xyz"]
        self.operator("TRANSFORM_OT_resize", {"value": vec, "constraint_axis": [a == axis.lower() for a in "xyz"]},
                      geometry_changed=[self.values["active_object"]])
        if dims is not None:
            self.state(dimensions=dims, geometry_changed=[self.values["active_object"]])
        return self

    def extrude(self, axis: str, value: float, dims: list[float] | None = None) -> DemoScript:
        self.key("E").key(axis.upper()).type_number(str(value)).key("RET")
        vec = [value if a == axis.lower() else 0.0 for a in "xyz"]
        self.operator("MESH_OT_extrude_region_move", {},
                      macros=[{"idname": "MESH_OT_extrude_region", "properties": {}},
                              {"idname": "TRANSFORM_OT_translate", "properties": {"value": vec}}],
                      geometry_changed=[self.values["active_object"]])
        if dims is not None:
            self.state(dimensions=dims, geometry_changed=[self.values["active_object"]])
        return self

    def loop_cut(self, cuts: int = 1) -> DemoScript:
        self.key("R", modifiers=["CTRL"]).click()
        self.operator("MESH_OT_loopcut_slide", {}, macros=[{"idname": "MESH_OT_loopcut", "properties": {"number_cuts": cuts}},
                                                            {"idname": "TRANSFORM_OT_edge_slide", "properties": {}}],
                      geometry_changed=[self.values["active_object"]])
        return self

    def bevel(self, offset: float = 0.02, segments: int = 2) -> DemoScript:
        self.key("B", modifiers=["CTRL"]).type_number(str(offset)).key("RET")
        self.operator("MESH_OT_bevel", {"offset": offset, "segments": segments, "affect": "EDGES"},
                      geometry_changed=[self.values["active_object"]])
        return self

    def add_modifier(self, mtype: str) -> DemoScript:
        self.click(x=1500, y=400)
        self.operator("OBJECT_OT_modifier_add", {"type": mtype})
        self.state(modifiers=[m["type"] for m in self.values["active_object_summary"]["modifiers"]] + [mtype])
        return self

    def inspect(self, views: list[str]) -> DemoScript:
        keys = {"front": ("NUMPAD_1", []), "right": ("NUMPAD_3", []), "top": ("NUMPAD_7", []),
                "back": ("NUMPAD_1", ["CTRL"]), "left": ("NUMPAD_3", ["CTRL"])}
        for view in views:
            key, mods = keys[view]
            self.key(key, modifiers=mods).state(view=view, perspective="ORTHO").wait(0.8)
        self.drag("MIDDLE").state(view="user", perspective="PERSP")
        return self


def sword_blockout_demo(*, with_mistake: bool = True, blade_length: float = 6.0, blade_width: float = 0.3,
                        variant: str = "a", t0: float = 1_700_000_000.0) -> DemoScript:
    """A human blocking out a simple sword: setup, blade blockout, inspection, a premature bevel
    that gets undone (failure + correction), taper, verification and the guard."""
    d = DemoScript(t0)
    d.wait(1.0).key("X").operator("OBJECT_OT_delete", geometry_changed=[]).state(object_count=2, active_object=None,
                                                                                 selected_objects=[])
    d.wait(1.5).add_cube("Blade")
    d.add_modifier("MIRROR")
    d.wait(5.0)  # pause: thinking about proportions
    d.tab("EDIT_MESH")
    d.scale("x", round(blade_width / 2, 3), dims=[blade_width, 2.0, 2.0])
    d.scale("y", 0.05, dims=[blade_width, 0.1, 2.0])
    d.scale("z", round(blade_length / 2, 3), dims=[blade_width, 0.1, blade_length])
    d.loop_cut(2 if variant == "a" else 3)
    d.wait(4.5)
    d.inspect(["front", "right"])
    d.wait(4.5)
    if with_mistake:
        d.bevel(0.05, 3)
        d.wait(1.0).annotate("bevel too early - the tip is too broad").undo()
        d.state(geometry_changed=["Blade"])
        d.wait(0.6)
    d.key("S").key("X").type_number("0.2").key("RET")
    d.operator("TRANSFORM_OT_resize", {"value": [0.2, 1.0, 1.0], "constraint_axis": [True, False, False]},
               geometry_changed=["Blade"])
    d.state(geometry_changed=["Blade"])
    d.wait(4.5)
    d.inspect(["front", "right", "top"])
    d.wait(4.5)
    d.tab("OBJECT")
    d.wait(4.5).add_cube("Guard")
    d.tab("EDIT_MESH")
    d.scale("x", 0.8, dims=[1.6, 2.0, 2.0])
    d.scale("z", 0.1, dims=[1.6, 2.0, 0.2])
    d.tab("OBJECT")
    d.wait(4.5)
    d.inspect(["front", "right"])
    d.key("S", modifiers=["CTRL"]).operator("WM_OT_save_mainfile")
    d.state(reason="final")
    return d
