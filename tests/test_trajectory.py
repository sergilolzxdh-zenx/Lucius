from __future__ import annotations

from lucius.provenance import ActionSource, EvidenceKind
from lucius.trajectory import TrajectoryBuilder, compress
from lucius.trajectory.compress import render_spans
from tests.fixtures.demos import DemoScript, sword_blockout_demo


def build(demo: DemoScript, **kw):
    return TrajectoryBuilder().build("ses_test", demo.events, [], **kw)


def test_hotkey_modal_is_confirmed_by_operator():
    d = DemoScript()
    d.add_cube("Blade").tab("EDIT_MESH").scale("x", 0.15)
    steps = build(d, operator_log_available=True).steps
    scale = [s for s in steps if s.action_type == "scale"]
    assert len(scale) == 1
    s = scale[0]
    assert s.action_source == ActionSource.OBSERVED and s.evidence_kind == EvidenceKind.DIRECT_BLENDER_EVENT
    assert s.params["axis"] == "x"
    assert s.params["value"] == [0.15, 1.0, 1.0]  # observed operator value wins over typed text
    assert any("typed value 0.15" in e for e in s.evidence)
    assert s.mode_label == "EDIT_MESH"
    modes = [x for x in steps if x.action_type == "mode_change"]
    assert modes and modes[0].params.get("mode") == "EDIT_MESH"


def test_hotkeys_without_operator_log_are_inferred_not_observed():
    d = DemoScript()
    d.key("TAB").state(mode="EDIT_MESH").key("S").key("Z").type_number("2").key("RET")
    steps = build(d, operator_log_available=False).steps
    scale = next(s for s in steps if s.action_type == "scale")
    assert scale.action_source == ActionSource.INFERRED
    assert scale.params == {"axis": "z", "value": 2.0}
    assert 0.5 < scale.action_confidence < 0.9


def test_cancelled_modal_is_marked():
    d = DemoScript()
    d.tab("EDIT_MESH").key("G").key("X").key("ESC")
    steps = build(d).steps
    grab = next(s for s in steps if s.action_type == "translate")
    assert grab.meta["cancelled"] and grab.action_confidence <= 0.4


def test_navigation_and_state_inference():
    d = DemoScript()
    d.drag("MIDDLE").drag("MIDDLE", modifiers=["SHIFT"]).key("NUMPAD_1").state(view="front", perspective="ORTHO")
    d.state(active_tool="builtin.extrude_region")
    steps = build(d).steps
    types = [s.action_type for s in steps]
    assert types[:3] == ["viewport_orbit", "viewport_pan", "view_preset"]
    assert steps[2].params["view"] == "front"
    tool = next(s for s in steps if s.action_type == "tool_change")
    assert tool.action_source == ActionSource.INFERRED and tool.params["active_tool"] == "builtin.extrude_region"


def test_undo_links_action_and_alternative():
    steps = build(sword_blockout_demo()).steps
    bevel = next(s for s in steps if s.action_type == "bevel")
    undo = next(s for s in steps if s.action_type == "undo")
    assert undo.meta["undoes"] == bevel.idx
    assert bevel.meta["undone_by"] == undo.idx
    alternative = steps[bevel.meta["replaced_by"]]
    assert alternative.action_type == "scale" and alternative.meta["alternative_to"] == [bevel.idx]
    assert undo.undo_redo_flag == "undo" and undo.action_source == ActionSource.OBSERVED


def test_build_is_deterministic_and_ordered():
    demo = sword_blockout_demo()
    a = build(demo).steps
    b = build(demo).steps
    assert [(s.action_type, s.t_start) for s in a] == [(s.action_type, s.t_start) for s in b]
    assert all(x.t_start <= y.t_start for x, y in zip(a, a[1:]))
    assert [s.idx for s in a] == list(range(len(a)))


def test_compression_merges_inspection_spans():
    demo = sword_blockout_demo()
    steps = build(demo).steps
    spans = compress(steps)
    inspections = [s for s in spans if s.kind == "viewport_inspection"]
    assert inspections
    assert any({"front", "right"} <= set(s.params["views"]) for s in inspections)
    assert len(spans) < len(steps)
    text = render_spans(spans, demo.events[0].ts)
    assert "viewport inspection" in text and "bevel" in text
