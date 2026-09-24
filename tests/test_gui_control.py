"""Keyboard and mouse control of Blender (the ``gui`` execution layer).

Unit tests drive a scripted Blender; ``test_live_keyboard_and_mouse_control`` drives a real windowed Blender on
a virtual X display when a Blender executable is available (``LUCIUS_BLENDER`` or ``blender`` on PATH).
"""

from __future__ import annotations

import math
import os
import shutil
import time
from types import SimpleNamespace

import pytest

from lucius.config import SafetyConfig
from lucius.errors import ActionRejected
from lucius.executor.gui import (
    GuiActuator,
    GuiBackend,
    GuiError,
    GuiInterference,
    Screen,
    key_sequence,
    number,
    translate,
)
from lucius.executor.safety import ActionValidator
from lucius.planner.model import PlanAction
from lucius.recorder.ledger import AgentActionLedger
from lucius.recorder.window import Bounds


def act(bridge_action, /, layer="blender_api", **args):
    return PlanAction(id=f"act_{bridge_action}_{time.monotonic_ns()}", layer=layer, name=bridge_action, args=args,
                      action_type=bridge_action)


LAYOUT = {"pid": 1, "mode": "EDIT_MESH", "active_object": "Box", "pivot_point": "MEDIAN_POINT",
          "auto_perspective": True, "orbit_step_deg": 15.0,
          "windows": [{"width": 1600, "height": 900, "modal_operators": [], "areas": [
              {"type": "VIEW_3D", "x": 2, "y": 87, "width": 1311, "height": 786,
               "regions": [{"type": "WINDOW", "x": 2, "y": 87, "width": 1311, "height": 786}]},
              {"type": "PROPERTIES", "x": 1316, "y": 24, "width": 282, "height": 691, "regions": []}]}]}


def keys_of(plan):
    return [(e["key"], tuple(e["modifiers"])) if e["kind"] == "key" else e["text"] for e in plan.sequence]


# -- exact keyboard forms --------------------------------------------------------------------------------

def test_translations_follow_blenders_default_keymap():
    extrude_z = translate(act("extrude", object="Box", offset=[0, 0, 0.5]), LAYOUT)
    assert keys_of(extrude_z) == [("E", ()), ("Z", ()), ("Z", ()), "0.5", ("RET", ())]  # Z twice: global Z
    assert extrude_z.modal_after == 1 and extrude_z.mode == "EDIT_MESH"
    assert ("TRANSFORM_OT_translate", "value", [0.0, 0.0, 0.5]) in extrude_z.expect.checks
    assert keys_of(translate(act("extrude", object="Box", offset=[-0.25, 0, 0]), LAYOUT))[:2] == [("E", ()), ("X", ())]
    assert translate(act("extrude", object="Box", offset=[0.1, 0, 0.5]), LAYOUT) is None       # two axes: no exact form
    assert keys_of(translate(act("scale_selection", object="Box", factor=[2, 2, 2]), LAYOUT)) == [("S", ()), "2",
                                                                                                    ("RET", ())]
    assert keys_of(translate(act("scale_selection", object="Box", factor=[1, 0.5, 1]), LAYOUT))[:2] == [("S", ()),
                                                                                                          ("Y", ())]
    assert translate(act("scale_selection", object="Box", factor=[2, 2, 2], pivot="bbox_center"), LAYOUT) is None
    inset = translate(act("inset", object="Box", thickness=0.1), LAYOUT)
    assert keys_of(inset) == [("I", ()), "0.1", ("RET", ())] and (None, "use_individual", False) in inset.expect.checks
    assert translate(act("inset", object="Box", thickness=0.1, depth=0.2), LAYOUT) is None
    bevel = translate(act("bevel", object="Box", offset=0.02, segments=1), LAYOUT)
    assert keys_of(bevel) == [("B", ("CTRL",)), "0.02", ("S", ()), "1", ("RET", ())]  # segments always typed
    assert keys_of(translate(act("set_mode", object="Box", mode="OBJECT"), LAYOUT)) == [("TAB", ())]
    assert translate(act("set_mode", object="Box", mode="EDIT"), LAYOUT).sequence == []  # already there
    assert keys_of(translate(act("select_all", object="Box", action="DESELECT"), LAYOUT)) == [("A", ("ALT",))]
    assert keys_of(translate(act("set_view", view="LEFT"), LAYOUT)) == [("NUMPAD_3", ("CTRL",))]
    assert translate(act("set_view", view="FRONT", ortho=False), LAYOUT) is None  # auto-perspective would differ
    assert keys_of(translate(act("orbit_view", direction="ORBITUP", angle=math.radians(30)), LAYOUT)) == [
        ("NUMPAD_8", ()), ("NUMPAD_8", ())]
    assert translate(act("orbit_view", direction="ORBITUP", angle=0.3), LAYOUT) is None  # not a multiple of 15°
    assert translate(act("add_modifier", object="Box", type="MIRROR"), LAYOUT) is None     # stays with the add-on


def test_object_mode_actions_click_the_object_first():
    layout = {**LAYOUT, "mode": "OBJECT", "active_object": "Ball"}
    move = translate(act("transform_object", object="Box", location=[1.0, 0, 0], relative=True), layout)
    assert move.select_first == "Box" and keys_of(move) == [("G", ()), ("X", ()), "1", ("RET", ())]
    rotate = translate(act("transform_object", object="Box", rotation=[0, 0, math.pi / 2], relative=True), layout)
    assert keys_of(rotate)[2] == "90" and (None, "value", math.pi / 2) in rotate.expect.checks
    assert translate(act("transform_object", object="Box", location=[1, 0, 0]), layout) is None  # absolute: no
    assert translate(act("extrude", object="Box", offset=[0, 0, 1]), {**LAYOUT, "active_object": "Ball"}) is None
    click = translate(act("select_objects", names=["Box"]), layout)
    assert click.select_first == "Box" and click.sequence == [] and click.expect.state == {"active_object": "Box"}
    assert number(0.1) == "0.1" and number(-2.0) == "-2" and number(1e-7) == "0"


# -- safety ------------------------------------------------------------------------------------------------

def test_safety_allowlists_keys_and_bounds_the_mouse():
    validator = ActionValidator(SafetyConfig(), bridge_actions=set(), gui_only=set(), background=False)

    def gui(*events):
        validator.validate(act("key_sequence", layer="gui", sequence=list(events)))

    gui(*key_sequence(["E", "Z"], "0.5"), {"kind": "pointer", "x": 0.2, "y": 0.8}, {"kind": "click", "button": "left"},
        {"kind": "drag", "button": "middle", "dx": 0.3, "dy": 0.0}, {"kind": "scroll", "clicks": -3})
    for bad in ({"kind": "key", "key": "S", "modifiers": ["CTRL"]},          # saving goes through the add-on
                {"kind": "key", "key": "F4", "modifiers": ["SHIFT"]},        # Python console
                {"kind": "key", "key": "P", "modifiers": ["ALT"]},           # run script
                {"kind": "key", "key": "Q", "modifiers": ["CTRL"]},          # quit
                {"kind": "key", "key": "SUPER_L", "modifiers": []},          # not allowlisted
                {"kind": "key", "key": "A", "modifiers": ["OSKEY"]},
                {"kind": "text", "text": "import os"},
                {"kind": "pointer", "x": 1.5, "y": 0.5},
                {"kind": "drag", "dx": 3.0},
                {"kind": "click", "button": "back"},
                {"kind": "scroll", "clicks": 100},
                {"kind": "type_anything"}):
        with pytest.raises(ActionRejected):
            gui(bad)
    with pytest.raises(ActionRejected, match="disabled"):
        ActionValidator(SafetyConfig(allow_gui_actions=False), bridge_actions=set(), gui_only=set(),
                        background=False).validate(act("key_sequence", layer="gui", sequence=[]))


# -- a scripted Blender ------------------------------------------------------------------------------------

class FakeInjector:
    def __init__(self):
        self.events = []
        self.pos = (0, 0)
        self.drift = None      # (after_n_events, new_position): a person grabs the mouse

    def _tick(self):
        if self.drift and len(self.events) >= self.drift[0]:
            self.pos, self.drift = self.drift[1], None

    def _record(self, *event):
        self.events.append(event)
        self._tick()

    def press(self, key):
        self._record("press", key)

    def release(self, key):
        self._record("release", key)

    def move(self, x, y):
        self.pos = (x, y)
        self._record("move", x, y)

    def position(self):
        return self.pos

    def button(self, button, down):
        self._record("button", button, down)

    def scroll(self, clicks):
        self.events.append(("scroll", clicks))

    def typed(self):
        return [e[1] for e in self.events if e[0] == "press"]


class FakeLocator:
    def __init__(self, focused=True):
        self._focused = focused

    def bounds(self):
        return Bounds(x=0, y=0, width=1600, height=900)

    def focused(self):
        return self._focused

    def activate(self):
        raise GuiError("no")


class FakeBlender:
    """Answers the add-on protocol; ``on_keys`` decides what Blender 'does' with the typed keys."""

    background = False
    connected = True
    info = {"actions": {n: {} for n in ("extrude", "scale_selection", "set_mode", "select_all", "snapshot", "restore",
                                        "add_modifier", "inset", "bevel", "transform_object", "select_objects")},
            "gui_only_actions": []}

    def __init__(self, injector, layout=None):
        self.injector = injector
        self.layout = {**LAYOUT, **(layout or {})}
        self.log = []
        self.executed = []
        self.modal_after_key = {"E": "MESH_OT_extrude_region_move", "S": "TRANSFORM_OT_resize", "G": "TRANSFORM_OT_translate",
                                "I": "MESH_OT_inset"}
        self.result = None     # operator appended once RET is typed

    def request(self, cmd, args=None, timeout=None):
        typed = self.injector.typed()
        if cmd == "gui_layout":
            modal = [self.modal_after_key[typed[-1]]] if typed and typed[-1] in self.modal_after_key else []
            if "RET" in typed[-1:] or not typed:
                modal = []
            windows = [{**self.layout["windows"][0], "modal_operators": modal}]
            return {**self.layout, "windows": windows}
        if cmd == "operator_log":
            if typed and typed[-1] == "RET" and self.result is not None and self.result not in self.log:
                self.log.append(self.result)
            return {"operators": list(self.log)}
        if cmd == "project":
            return {"object": args["object"], "x": 600.0, "y": 400.0, "inside_region": True}
        raise AssertionError(cmd)

    def execute(self, action, args=None, timeout=None):
        self.executed.append((action, args))
        return {"action": action}

    def get_state(self, include_objects=True):
        return SimpleNamespace(get=lambda key: None)


def backend_for(layout=None, focused=True, fallback=True):
    injector = FakeInjector()
    blender = FakeBlender(injector, layout)
    ledger = AgentActionLedger()
    actuator = GuiActuator(injector, FakeLocator(focused), ledger, event_delay_s=0.0)
    backend = GuiBackend(blender, actuator, fallback_to_bridge=fallback, verify_timeout_s=0.3)
    backend._prepared = True
    return backend, blender, injector, ledger


def test_verified_keyboard_action_and_ledger_attribution():
    backend, blender, injector, ledger = backend_for()
    blender.result = {"id": "1", "idname": "MESH_OT_extrude_region_move", "properties": {}, "macros": [
        {"idname": "TRANSFORM_OT_translate", "properties": {"value": [0.0, 0.0, 0.5], "orient_type": "GLOBAL"}}]}
    outcome = backend.execute(act("extrude", object="Box", offset=[0, 0, 0.5]))
    assert outcome.ok and outcome.result["via"] == "gui" and outcome.result["verified"] is True
    assert injector.typed() == ["E", "Z", "Z", "ZERO", "PERIOD", "FIVE", "RET"]
    assert injector.events[0][0] == "move" and Screen(LAYOUT, FakeLocator().bounds()).inside_view3d(*injector.events[0][1:])
    assert [name for name, _ in blender.executed] == ["snapshot"]   # scene saved before a mutating attempt
    assert ledger.attribute(time.time(), "key_down", {"key": "E"}) == "agent"   # recordings credit the agent
    assert backend.stats == {"gui": 1, "bridge": 0, "fallbacks": 0}


def test_mismatch_is_rolled_back_and_done_by_the_add_on():
    backend, blender, injector, _ = backend_for()
    blender.result = {"id": "1", "idname": "TRANSFORM_OT_resize", "properties": {"value": [0.5, 1.0, 1.0]}}
    outcome = backend.execute(act("scale_selection", object="Box", factor=[1, 1, 0.5]))
    assert outcome.ok and outcome.result["via"] == "bridge_fallback"
    assert "value" in outcome.result["gui_attempt"]["problems"][0]
    names = [name for name, _ in blender.executed]
    assert names == ["snapshot", "restore", "set_mode", "scale_selection"]  # exact rollback, back to edit mode
    strict, blender2, _, _ = backend_for(fallback=False)
    blender2.result = blender.result
    refused = strict.execute(act("scale_selection", object="Box", factor=[1, 1, 0.5]))
    assert not refused.ok and refused.error["code"] == "gui_unverified"


def test_axis_keys_wait_for_the_modal_operator():
    """If E starts nothing (nothing selected, a popup open), 'Z Z 0.5 RET' must not be typed: in object mode
    Z opens a pie menu and X deletes (found live: a failed sequence deleted the object)."""
    backend, blender, injector, _ = backend_for()
    blender.modal_after_key = {}
    outcome = backend.execute(act("extrude", object="Box", offset=[0, 0, 0.5]))
    assert injector.typed() == ["E"] and outcome.result["via"] == "bridge_fallback"
    assert "did not start" in outcome.result["gui_attempt"]["error"]["message"]


def test_actions_without_keyboard_form_go_to_the_add_on():
    backend, blender, injector, _ = backend_for()
    outcome = backend.execute(act("add_modifier", object="Box", type="MIRROR"))
    assert outcome.ok and outcome.result["via"] == "bridge" and injector.events == []


def test_unfocused_blender_gets_no_input_and_interference_stops_the_agent():
    backend, blender, injector, _ = backend_for(focused=False)
    outcome = backend.execute(act("set_mode", object="Box", mode="OBJECT"))
    assert injector.events == [] and outcome.result["via"] == "bridge_fallback"
    assert "not the focused window" in outcome.result["gui_attempt"]["error"]["message"]

    backend, blender, injector, _ = backend_for()
    injector.drift = (2, (5, 5))  # a person moves the mouse after the first key
    outcome = backend.execute(act("extrude", object="Box", offset=[0, 0, 0.5]))
    assert not outcome.ok and outcome.error["code"] == "human_interference"
    assert injector.typed()[-1] == "ESC"                       # any modal operator cancelled
    assert [n for n, _ in blender.executed] == ["snapshot"]    # and no fallback behind the person's back


def test_actuator_refuses_points_outside_the_viewport():
    actuator = GuiActuator(FakeInjector(), FakeLocator(), event_delay_s=0.0)
    screen = Screen(LAYOUT, FakeLocator().bounds())
    with pytest.raises(GuiError, match="outside the 3D viewport"):
        actuator.perform([{"kind": "pointer", "space": "window", "x": 1400, "y": 300}], screen, "a")  # properties editor
    actuator.stop()
    with pytest.raises(GuiInterference, match="stopped"):
        actuator.perform([{"kind": "key", "key": "E", "modifiers": []}], screen, "b")


# -- live: a real windowed Blender on a virtual display ------------------------------------------------------

def _blender_gui_binary():
    from lucius.blender.interactive import blender_binary

    return blender_binary()


@pytest.mark.xvfb
def test_live_keyboard_and_mouse_control(xvfb, tmp_path, monkeypatch):
    binary = _blender_gui_binary()
    if binary is None or shutil.which("Xvfb") is None:
        pytest.skip("needs a Blender executable (LUCIUS_BLENDER) and Xvfb")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path).chmod(0o700)
    from lucius.blender.interactive import InteractiveBlender
    from lucius.executor.gui import X11BlenderWindow, XTestInjector

    with InteractiveBlender(binary, display=os.environ["DISPLAY"]) as blender:
        bridge = blender.bridge
        injector = XTestInjector()
        backend = GuiBackend(bridge, GuiActuator(injector, X11BlenderWindow(blender.pid), AgentActionLedger(),
                                                 event_delay_s=0.06))
        run = backend.execute
        assert run(act("reset_scene", keep_camera_light=True)).ok
        assert run(act("add_primitive", kind="cube", name="Box")).ok
        assert run(act("add_primitive", kind="uv_sphere", name="Ball", location=[4.0, 0.0, 0.0])).ok
        clicked = run(act("select_objects", names=["Box"]))                         # mouse: click the cube
        assert clicked.ok and clicked.result["via"] == "gui", clicked.result
        for action, via in [(act("set_mode", object="Box", mode="EDIT"), "gui"),
                            (act("select_all", object="Box", action="DESELECT"), "gui"),
                            (act("select_faces_by_normal", object="Box", direction=[0, 0, 1]), "bridge"),
                            (act("extrude", object="Box", offset=[0, 0, 0.5]), "gui"),
                            (act("scale_selection", object="Box", factor=[0.5, 1, 1]), "gui"),
                            (act("inset", object="Box", thickness=0.1), "gui"),
                            (act("set_mode", object="Box", mode="OBJECT"), "gui"),
                            (act("set_view", view="FRONT"), "gui"),
                            (act("orbit_view", direction="ORBITLEFT", angle=math.radians(15)), "gui"),
                            (act("transform_object", object="Box", location=[1.0, 0, 0], relative=True), "gui")]:
            outcome = run(action)
            assert outcome.ok and outcome.result["via"] == via, (action.name, outcome.result)
        box = next(o for o in bridge.inspect_structure(names=["Box"], views=())["objects"] if o["name"] == "Box")
        assert box["dimensions"][2] == pytest.approx(2.5, abs=1e-4)     # extruded 0.5 up
        assert box["location"][0] == pytest.approx(1.0, abs=1e-4)       # moved 1 along X
        assert box["evaluated"]["verts"] == 16                          # cube + extrusion ring + inset ring
        before = bridge.get_state(include_objects=False).get("viewport")["rotation"]
        drag = run(act("key_sequence", layer="gui", sequence=[{"kind": "pointer", "x": 0.5, "y": 0.5},
                                                              {"kind": "drag", "button": "middle", "dx": 0.15}]))
        assert drag.ok and bridge.get_state(include_objects=False).get("viewport")["rotation"] != before  # mouse orbit


@pytest.mark.xvfb
def test_live_agent_run_by_keyboard(xvfb, tmp_path, monkeypatch):
    """The whole loop -- learned skill, plan, execution, checkpoints -- with actions typed into a real Blender."""
    binary = _blender_gui_binary()
    if binary is None:
        pytest.skip("needs a Blender executable (LUCIUS_BLENDER)")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    tmp_path.chmod(0o700)
    from lucius.app import Lucius
    from lucius.blender.interactive import InteractiveBlender
    from lucius.executor.gui import X11BlenderWindow, XTestInjector
    from tests.fixtures.demos import sword_blockout_demo
    from tests.test_e2e_learning import record, references

    app = Lucius(data_dir=tmp_path / "data", background_processing=False)
    try:
        record(app, sword_blockout_demo(blade_length=6.0, blade_width=0.3, variant="a", t0=1_700_000_000.0))
        record(app, sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False,
                                        t0=1_700_100_000.0))
        with InteractiveBlender(binary, display=os.environ["DISPLAY"]) as blender:
            backend = GuiBackend(blender.bridge, GuiActuator(XTestInjector(), X11BlenderWindow(blender.pid), app.ledger,
                                                             event_delay_s=0.06))
            refs, ref_ids = references(app, 5.0, 0.28)
            run = app.engine.run("Make another sword with blade length 5 and blade width 0.28, use what I taught you",
                                 backend, references=refs, reference_ids=ref_ids, reset_scene=True)
            assert run.verdict == "success", run.final_report.summary
            dims = {o["name"]: o["dimensions"] for o in backend.structure()["objects"]}
            assert dims["Blade"][2] == pytest.approx(5.0, rel=0.01) and dims["Blade"][0] == pytest.approx(0.28, rel=0.01)
            assert backend.stats["gui"] >= 10 and backend.stats["fallbacks"] == 0, backend.stats
    finally:
        app.close()
