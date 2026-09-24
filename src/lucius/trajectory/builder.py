"""Raw events -> semantic trajectory steps (V1: replayable episodes).

Evidence is layered rather than guessed:

1. Direct input gives *inferred* actions via Blender's default keymap: ``S X 0.5 RET`` is a
   scale along X by 0.5. Keymaps can be customised, so these carry moderate confidence.
2. Operators observed by the add-on confirm (and supersede) input-derived drafts: the step
   becomes ``observed`` with the operator's exact parameters.
3. State diffs (mode, tool, workspace, view, modifiers, geometry) explain changes no event
   accounted for, as ``inferred`` steps.
4. Undo/redo are linked to the exact step they reverted and to the alternative that followed
   -- the raw material for failure and correction learning.

Raw events are never modified; the builder is deterministic, so re-running it is safe.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

from lucius.ids import new_id
from lucius.provenance import ActionSource, EvidenceKind
from lucius.recorder.keys import MODIFIERS, hotkey_label
from lucius.sessions.models import CapturedEvent, FrameRecord
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep

AXIS_KEYS = {"X": "x", "Y": "y", "Z": "z"}
CONFIRM_KEYS = {"RET", "NUMPAD_ENTER", "SPACE"}
NUMERIC_KEYS = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6",
                "SEVEN": "7", "EIGHT": "8", "NINE": "9", "PERIOD": ".", "MINUS": "-", "NUMPAD_PERIOD": ".",
                "NUMPAD_MINUS": "-", **{f"NUMPAD_{i}": str(i) for i in range(10)}}
SELECT_MODE_KEYS = {"ONE": "VERT", "TWO": "EDGE", "THREE": "FACE"}
UI_PATH_ACTIONS = {"add_menu", "search_menu", "context_menu", "ui_click", "select_click", "text_entry"}


@dataclass
class _Draft:
    t_start: float
    t_end: float
    action_type: str
    source: ActionSource
    evidence_kind: EvidenceKind
    confidence: float
    actor: str
    seq_start: int
    seq_end: int
    params: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    undo_redo_flag: str | None = None
    operator: dict[str, Any] | None = None
    confirmed: bool = False

    def touch(self, event: CapturedEvent) -> None:
        self.t_end = max(self.t_end, event.ts)
        self.seq_end = max(self.seq_end, event.seq)


@dataclass
class BuildResult:
    steps: list[TrajectoryStep]
    annotations: list[dict[str, Any]]
    takeovers: list[dict[str, Any]]
    stats: dict[str, int]


class TrajectoryBuilder:
    def __init__(self, *, operator_merge_window_s: float = 4.0, scroll_merge_s: float = 0.6,
                 drag_threshold_px: float = 5.0, text_merge_s: float = 1.5) -> None:
        self.operator_merge_window_s = operator_merge_window_s
        self.scroll_merge_s = scroll_merge_s
        self.drag_threshold_px = drag_threshold_px
        self.text_merge_s = text_merge_s

    def build(self, session_id: str, events: list[CapturedEvent], frames: list[FrameRecord], *,
              operator_log_available: bool = False) -> BuildResult:
        run = _Run(self, operator_log_available)
        for event in sorted(events, key=lambda e: e.seq):
            run.handle(event)
        run.finish()
        steps = run.to_steps(session_id, frames)
        return BuildResult(steps=steps, annotations=run.annotations, takeovers=run.takeovers, stats=run.stats)


class _Run:
    def __init__(self, cfg: TrajectoryBuilder, operator_log_available: bool) -> None:
        self.cfg = cfg
        self.operator_log = operator_log_available
        self.drafts: list[_Draft] = []
        self.modal: _Draft | None = None
        self.typed = ""
        self.held: set[str] = set()
        self.buttons: dict[str, dict[str, Any]] = {}
        self.scroll: _Draft | None = None
        self.text: _Draft | None = None
        self.state: dict[str, Any] = {}
        self.state_timeline: list[tuple[float, dict[str, Any]]] = []
        self.window: dict[str, Any] = {}
        self.window_timeline: list[tuple[float, dict[str, Any]]] = []
        self.annotations: list[dict[str, Any]] = []
        self.takeovers: list[dict[str, Any]] = []
        self._takeover_open: dict[str, Any] | None = None
        self.stats = {"events": 0, "operators": 0, "operators_folded": 0, "state_inferred": 0, "cancelled": 0}

    # -- helpers ------------------------------------------------------------------------------------
    def _new(self, event: CapturedEvent, action_type: str, *, source: ActionSource, evidence_kind: EvidenceKind,
             confidence: float, params: dict[str, Any] | None = None, evidence: list[str] | None = None,
             actor: str | None = None, t_start: float | None = None) -> _Draft:
        draft = _Draft(t_start=t_start if t_start is not None else event.ts, t_end=event.ts, action_type=action_type,
                       source=source, evidence_kind=evidence_kind, confidence=confidence,
                       actor=actor or event.actor.value, seq_start=event.seq, seq_end=event.seq,
                       params=dict(params or {}), evidence=list(evidence or []))
        if self._takeover_open is not None:
            draft.meta["during_takeover"] = True
        self.drafts.append(draft)
        return draft

    def _recent(self, ts: float, window: float, predicate) -> _Draft | None:
        for draft in reversed(self.drafts):
            if ts - draft.t_end > window:
                break
            if predicate(draft):
                return draft
        return None

    def _mode(self) -> str | None:
        return self.state.get("mode")

    def _close_modal(self, event: CapturedEvent, *, cancelled: bool, reason: str) -> None:
        modal = self.modal
        if modal is None:
            return
        modal.touch(event)
        if self.typed:
            try:
                modal.params["value"] = float(self.typed)
                modal.evidence.append(f"typed value {self.typed}")
            except ValueError:
                modal.evidence.append(f"unparsed typed input {self.typed!r}")
        if cancelled:
            modal.meta["cancelled"] = True
            modal.confidence = min(modal.confidence, 0.4)
            modal.evidence.append(f"cancelled via {reason}")
            self.stats["cancelled"] += 1
        else:
            modal.evidence.append(f"confirmed via {reason}")
        self.modal = None
        self.typed = ""

    def _input_confidence(self, action_type: str) -> float:
        # With the operator log available, a mutating hotkey that never gets confirmed by an
        # operator is suspicious; without it, direct input is the best evidence we have.
        return 0.75 if not self.operator_log or not vocab.mutates(action_type) else 0.55

    # -- dispatch -----------------------------------------------------------------------------------
    def handle(self, event: CapturedEvent) -> None:
        self.stats["events"] += 1
        handler = getattr(self, f"_on_{event.kind.value}", None)
        if handler is not None:
            handler(event)

    def _on_key_down(self, event: CapturedEvent) -> None:
        key = event.payload.get("key", "UNKNOWN")
        mods = set(event.payload.get("modifiers") or []) | {m for m in self.held if m != key}
        if key in MODIFIERS:
            self.held.add(key)
            return
        if self.modal is not None:
            self._modal_key(event, key, mods)
            return
        if key in {"ESC"}:
            return
        action = vocab.map_hotkey(key, frozenset(mods), self._mode())
        label = hotkey_label(key, mods)
        if action is None:
            char = event.payload.get("char")
            if char and not (mods - {"SHIFT"}):
                self._text(event, char)
                return
            self._new(event, "unknown_action", source=ActionSource.OBSERVED, evidence_kind=EvidenceKind.DIRECT_INPUT,
                      confidence=0.3, params={"hotkey": label}, evidence=[f"unmapped hotkey {label}"])
            return
        self.text = None
        params: dict[str, Any] = {}
        if action == "view_preset":
            params["view"] = vocab.VIEW_KEYS.get((key, "CTRL" in mods), "user")
        elif action == "select_mode":
            params["type"] = SELECT_MODE_KEYS.get(key)
        elif action in ("viewport_zoom",):
            params["direction"] = "in" if key == "NUMPAD_PLUS" else "out"
        draft = self._new(event, action, source=ActionSource.INFERRED, evidence_kind=EvidenceKind.DIRECT_INPUT,
                          confidence=self._input_confidence(action), params=params,
                          evidence=[f"hotkey {label} (default keymap)"])
        draft.inputs.append({"hotkey": label})
        if action in ("undo", "redo"):
            draft.undo_redo_flag = action
        if action in vocab.MODAL_ACTIONS:
            self.modal = draft
            self.typed = ""

    def _modal_key(self, event: CapturedEvent, key: str, mods: set[str]) -> None:
        modal = self.modal
        assert modal is not None
        modal.touch(event)
        if key in AXIS_KEYS:
            axis = AXIS_KEYS[key]
            modal.params["axis"] = "".join(a for a in "xyz" if a != axis) if "SHIFT" in mods else axis
            modal.inputs.append({"axis_key": key, "exclude": "SHIFT" in mods})
        elif key in NUMERIC_KEYS or (event.payload.get("char") or "?") in "0123456789.-":
            self.typed += NUMERIC_KEYS.get(key) or event.payload["char"]
        elif key == "BACK_SPACE":
            self.typed = self.typed[:-1]
        elif key in CONFIRM_KEYS:
            self._close_modal(event, cancelled=False, reason=key)
        elif key == "ESC":
            self._close_modal(event, cancelled=True, reason="ESC")
        elif vocab.map_hotkey(key, frozenset(mods), self._mode()) == modal.action_type:
            modal.params["variant"] = "repeat"  # e.g. G G = edge slide
        else:
            modal.inputs.append({"key": key})

    def _on_key_up(self, event: CapturedEvent) -> None:
        key = event.payload.get("key")
        if key in MODIFIERS:
            self.held.discard(key)

    def _text(self, event: CapturedEvent, char: str) -> None:
        if self.text is not None and event.ts - self.text.t_end <= self.cfg.text_merge_s:
            self.text.params["text"] += char
            self.text.touch(event)
            return
        self.text = self._new(event, "text_entry", source=ActionSource.OBSERVED, evidence_kind=EvidenceKind.DIRECT_INPUT,
                              confidence=0.9, params={"text": char})

    def _on_mouse_down(self, event: CapturedEvent) -> None:
        button = event.payload.get("button")
        if self.modal is not None:
            if button == "LEFT":
                self._close_modal(event, cancelled=False, reason="LEFT click")
                return
            if button == "RIGHT":
                self._close_modal(event, cancelled=True, reason="RIGHT click")
                return
        self.buttons[button] = {"event": event, "x": event.payload.get("x", 0.0), "y": event.payload.get("y", 0.0),
                                "path": 0.0, "last": (event.payload.get("x", 0.0), event.payload.get("y", 0.0)),
                                "mods": set(self.held)}

    def _on_mouse_move(self, event: CapturedEvent) -> None:
        x, y = event.payload.get("x", 0.0), event.payload.get("y", 0.0)
        for info in self.buttons.values():
            lx, ly = info["last"]
            info["path"] += math.hypot(x - lx, y - ly)
            info["last"] = (x, y)
        if self.modal is not None:
            self.modal.touch(event)
            self.modal.meta["pointer_adjusted"] = True

    def _on_mouse_up(self, event: CapturedEvent) -> None:
        button = event.payload.get("button")
        info = self.buttons.pop(button, None)
        if info is None:
            return
        start: CapturedEvent = info["event"]
        x, y = event.payload.get("x", 0.0), event.payload.get("y", 0.0)
        displacement = math.hypot(x - info["x"], y - info["y"])
        dragged = max(displacement, info["path"]) >= self.cfg.drag_threshold_px
        mods = info["mods"]
        raw = {"button": button, "from": [start.payload.get("wx"), start.payload.get("wy")],
               "to": [event.payload.get("wx"), event.payload.get("wy")], "path_px": round(info["path"], 1)}
        if button == "MIDDLE":
            if not dragged:
                return
            action = "viewport_pan" if "SHIFT" in mods else "viewport_zoom" if "CTRL" in mods else "viewport_orbit"
            draft = self._new(start, action, source=ActionSource.INFERRED, evidence_kind=EvidenceKind.DIRECT_INPUT,
                              confidence=0.8, evidence=[f"middle-mouse drag {round(displacement)}px"])
        elif button == "LEFT":
            action = "select_box" if dragged else "select_click"
            draft = self._new(start, action, source=ActionSource.INFERRED, evidence_kind=EvidenceKind.DIRECT_INPUT,
                              confidence=0.5, params={"extend": "SHIFT" in mods},
                              evidence=["left drag" if dragged else "left click"])
        elif button == "RIGHT":
            draft = self._new(start, "context_menu", source=ActionSource.INFERRED,
                              evidence_kind=EvidenceKind.DIRECT_INPUT, confidence=0.6, evidence=["right click"])
        else:
            return
        draft.touch(event)
        draft.inputs.append(raw)

    def _on_scroll(self, event: CapturedEvent) -> None:
        dy = event.payload.get("dy", 0)
        if self.modal is not None:
            self.modal.params["wheel"] = self.modal.params.get("wheel", 0) + (1 if dy > 0 else -1)
            self.modal.touch(event)
            return
        if self.scroll is not None and event.ts - self.scroll.t_end <= self.cfg.scroll_merge_s:
            self.scroll.params["ticks"] += 1
            self.scroll.touch(event)
            return
        self.scroll = self._new(event, "viewport_zoom", source=ActionSource.INFERRED,
                                evidence_kind=EvidenceKind.DIRECT_INPUT, confidence=0.8,
                                params={"direction": "in" if dy > 0 else "out", "ticks": 1},
                                evidence=["mouse wheel"])

    def _on_blender_operator(self, event: CapturedEvent) -> None:
        self.stats["operators"] += 1
        op = event.payload.get("operator") or {}
        idname = op.get("idname", "UNKNOWN")
        mapped = vocab.map_operator(op)
        if mapped is None:
            action_type = f"op.{idname.lower()}"
            if action_type not in vocab.ACTION_TYPES:
                vocab.register_action_type(action_type, "operator", bool(event.payload.get("geometry_changed")), 1)
            params = {}
        else:
            action_type, params = mapped
        if event.payload.get("adjusted"):
            target = self._recent(event.ts, 120.0, lambda d: d.operator is not None and d.operator.get("idname") == idname)
            if target is not None:
                target.params.update(params)
                target.operator = op
                target.evidence.append("parameters adjusted in redo panel")
                target.touch(event)
                return
        target = self._recent(event.ts, self.cfg.operator_merge_window_s,
                              lambda d: d.action_type == action_type and not d.confirmed
                              and d.source != ActionSource.OBSERVED and not d.meta.get("cancelled"))
        if target is None and self.modal is not None and self.modal.action_type == action_type:
            target = self.modal
            self._close_modal(event, cancelled=False, reason="operator finished")
        if target is not None:
            self.stats["operators_folded"] += 1
            target.params.update(params)
            target.source = ActionSource.OBSERVED
            target.evidence_kind = EvidenceKind.DIRECT_BLENDER_EVENT
            target.confidence = 0.98
            target.confirmed = True
            target.operator = op
            target.evidence.append(f"operator {idname}")
            target.touch(event)
            if event.actor.value == "agent":
                target.actor = "agent"
            return
        ui_path = self._absorb_ui_path(event.ts)
        draft = self._new(event, action_type, source=ActionSource.OBSERVED,
                          evidence_kind=EvidenceKind.DIRECT_BLENDER_EVENT, confidence=0.98, params=params,
                          evidence=[f"operator {idname}"],
                          t_start=ui_path[0].t_start if ui_path else None)
        if ui_path:
            draft.seq_start = ui_path[0].seq_start
            draft.inputs.append({"ui_path": [p.inputs[0] if p.inputs else {"action": p.action_type} for p in ui_path]})
            draft.evidence.append("invoked via " + " -> ".join(p.action_type for p in ui_path))
        draft.operator = op
        draft.confirmed = True
        if event.payload.get("geometry_changed"):
            draft.meta["geometry_changed"] = event.payload["geometry_changed"]

    def _absorb_ui_path(self, ts: float) -> list[_Draft]:
        """Menu/click drafts that directly led to an operator become that operator's GUI path."""
        path: list[_Draft] = []
        while self.drafts:
            last = self.drafts[-1]
            if last.action_type not in UI_PATH_ACTIONS or last.confirmed or ts - last.t_end > 3.0:
                break
            path.insert(0, self.drafts.pop())
        if path:
            self.text = None
        return path

    def _history(self, event: CapturedEvent, flag: str) -> None:
        target = self._recent(event.ts, 1.5, lambda d: d.action_type == flag and not d.confirmed)
        if target is not None:
            target.source = ActionSource.OBSERVED
            target.evidence_kind = EvidenceKind.DIRECT_BLENDER_EVENT
            target.confidence = 0.99
            target.confirmed = True
            target.evidence.append(f"{flag}_post handler")
            target.touch(event)
            return
        draft = self._new(event, flag, source=ActionSource.OBSERVED, evidence_kind=EvidenceKind.DIRECT_BLENDER_EVENT,
                          confidence=0.99, evidence=[f"{flag}_post handler"])
        draft.undo_redo_flag = flag
        draft.confirmed = True

    def _on_undo(self, event: CapturedEvent) -> None:
        self._history(event, "undo")

    def _on_redo(self, event: CapturedEvent) -> None:
        self._history(event, "redo")

    def _on_blender_state(self, event: CapturedEvent) -> None:
        values = event.payload.get("values") or {}
        compact = _compact_state(event.payload)
        previous = self.state
        self.state = compact
        self.state_timeline.append((event.ts, compact))
        if not previous or event.payload.get("reason") == "initial":
            return
        checks = [
            ("mode", "mode_change", lambda d: d.action_type == "mode_change"),
            ("workspace", "workspace_change", lambda d: d.action_type == "workspace_change"),
            ("active_tool", "tool_change", lambda d: d.action_type == "tool_change"),
            ("view", "view_preset", lambda d: vocab.is_navigation(d.action_type)),
        ]
        for key, action, explains in checks:
            before, after = previous.get(key), compact.get(key)
            if before == after or after is None:
                continue
            explained = self._recent(event.ts, 1.5, explains)
            if explained is not None:
                explained.params.setdefault(key, after)
                explained.evidence.append(f"state {key}: {before} -> {after}")
                continue
            self.stats["state_inferred"] += 1
            self._new(event, action, source=ActionSource.INFERRED, evidence_kind=EvidenceKind.DIRECT_BLENDER_EVENT,
                      confidence=0.85, params={key: after, "from": before}, actor="human",
                      evidence=[f"state {key}: {before} -> {after}"])
        added = [m for m in compact.get("modifiers", []) if m not in previous.get("modifiers", [])]
        for modifier in added:
            if self._recent(event.ts, 3.0, lambda d: d.action_type == "add_modifier") is None:
                self.stats["state_inferred"] += 1
                self._new(event, "add_modifier", source=ActionSource.INFERRED,
                          evidence_kind=EvidenceKind.DIRECT_BLENDER_EVENT, confidence=0.85, params={"type": modifier},
                          evidence=[f"modifier {modifier} appeared"])
        changed = values.get("geometry_changed") or event.payload.get("geometry_changed") or []
        if changed and self._recent(event.ts, 2.5, lambda d: vocab.mutates(d.action_type)) is None:
            self.stats["state_inferred"] += 1
            self._new(event, "geometry_edit", source=ActionSource.INFERRED,
                      evidence_kind=EvidenceKind.DIRECT_BLENDER_EVENT, confidence=0.5, params={"objects": changed},
                      evidence=["geometry changed without an observed operator"])

    def _on_window_context(self, event: CapturedEvent) -> None:
        if event.payload.get("allowed"):
            self.window = event.payload
            self.window_timeline.append((event.ts, event.payload))

    def _on_agent_action(self, event: CapturedEvent) -> None:
        action = event.payload.get("action") or {}
        if action.get("layer") not in ("blender_api", "internal"):
            return  # GUI actions surface through their (agent-attributed) input events
        name = action.get("name", "")
        result = event.payload.get("result") or {}
        ok = not result.get("error")
        actor = event.actor.value  # agent executions, or human actions issued through the bridge console
        draft = self._new(event, action.get("action_type") or vocab.BRIDGE_ACTION_MAP.get(name, f"api.{name}"),
                          source=ActionSource.OBSERVED,
                          evidence_kind=EvidenceKind.AGENT_EXECUTION if actor == "agent" else EvidenceKind.DIRECT_BLENDER_EVENT,
                          confidence=1.0 if ok else 0.3, params=dict(action.get("args") or {}), actor=actor,
                          evidence=[f"bridge action {name}" + ("" if ok else " (failed)")])
        draft.confirmed = True
        draft.meta["bridge_action"] = name
        if not ok:
            draft.meta["error"] = result.get("error")

    def _on_annotation(self, event: CapturedEvent) -> None:
        self.annotations.append({"ts": event.ts, "seq": event.seq, **event.payload})

    def _on_takeover_start(self, event: CapturedEvent) -> None:
        self._takeover_open = {"start_ts": event.ts, "start_seq": event.seq, **event.payload}

    def _on_takeover_end(self, event: CapturedEvent) -> None:
        if self._takeover_open is None:
            return
        self._takeover_open.update({"end_ts": event.ts, "end_seq": event.seq, "after_state": event.payload.get("state"),
                                    "before_state": self._takeover_open.pop("state", None)})
        self.takeovers.append(self._takeover_open)
        self._takeover_open = None

    def _boundary(self, event: CapturedEvent) -> None:
        if self.modal is not None:
            self._close_modal(event, cancelled=True, reason=event.kind.value)
        self.buttons.clear()
        self.scroll = None
        self.text = None

    _on_focus_lost = _boundary
    _on_recording_paused = _boundary

    # -- finalisation -------------------------------------------------------------------------------
    def finish(self) -> None:
        if self.modal is not None:
            self.modal.meta["unterminated"] = True
            self.modal = None
        if self._takeover_open is not None:
            self._takeover_open["unterminated"] = True
            self.takeovers.append(self._takeover_open)

    def to_steps(self, session_id: str, frames: list[FrameRecord]) -> list[TrajectoryStep]:
        drafts = sorted(self.drafts, key=lambda d: (d.t_start, d.seq_start))
        frame_ts = [f.ts for f in frames]
        state_ts = [t for t, _ in self.state_timeline]
        window_ts = [t for t, _ in self.window_timeline]
        steps: list[TrajectoryStep] = []
        for idx, d in enumerate(drafts):
            next_start = drafts[idx + 1].t_start if idx + 1 < len(drafts) else math.inf
            before = _at_or_before(self.state_timeline, state_ts, d.t_start)
            after = _first_after(self.state_timeline, state_ts, d.t_end, limit=min(next_start, d.t_end + 3.0)) \
                or _at_or_before(self.state_timeline, state_ts, min(next_start, d.t_end + 3.0))
            window = _at_or_before(self.window_timeline, window_ts, d.t_start) or {}
            frame_before = _frame_at_or_before(frames, frame_ts, d.t_start)
            frame_after = _frame_after(frames, frame_ts, d.t_end, limit=min(next_start + 0.5, d.t_end + 3.0)) \
                or _frame_at_or_before(frames, frame_ts, min(next_start, d.t_end + 3.0))
            payload: dict[str, Any] = {"params": d.params}
            if d.inputs:
                payload["input"] = d.inputs
            if d.operator is not None:
                payload["operator"] = d.operator
            steps.append(TrajectoryStep(
                id=new_id("step"), session_id=session_id, idx=idx, t_start=d.t_start, t_end=d.t_end,
                frame_before_id=frame_before.id if frame_before else None,
                frame_after_id=frame_after.id if frame_after else None, action_type=d.action_type,
                action_payload=payload, action_source=d.source, evidence_kind=d.evidence_kind,
                action_confidence=round(d.confidence, 3), evidence=d.evidence,
                window_title=window.get("title"), window_bounds=window.get("bounds"),
                mode_label=(before or {}).get("mode"), tool_label=(before or {}).get("active_tool"),
                selection_hint=_selection_hint(before), undo_redo_flag=d.undo_redo_flag, actor=d.actor,
                event_seq_start=d.seq_start, event_seq_end=d.seq_end, state_before=before, state_after=after,
                meta=d.meta,
            ))
        link_undo_relations(steps)
        return steps


def link_undo_relations(steps: list[TrajectoryStep]) -> None:
    """Annotate ACTION -> UNDO -> ALTERNATIVE relationships in place."""
    stack: list[int] = []
    undone: list[int] = []
    awaiting_alternative: list[int] = []
    for step in steps:
        if step.meta.get("cancelled"):
            continue
        if step.action_type == "undo":
            if stack:
                target = stack.pop()
                undone.append(target)
                awaiting_alternative.append(target)
                step.meta["undoes"] = target
                steps[target].meta["undone_by"] = step.idx
        elif step.action_type == "redo":
            if undone:
                target = undone.pop()
                stack.append(target)
                if target in awaiting_alternative:
                    awaiting_alternative.remove(target)
                step.meta["redoes"] = target
                steps[target].meta["redone_by"] = step.idx
        elif vocab.mutates(step.action_type):
            if awaiting_alternative:
                step.meta["alternative_to"] = list(awaiting_alternative)
                for target in awaiting_alternative:
                    steps[target].meta.setdefault("replaced_by", step.idx)
                awaiting_alternative = []
                undone = []  # a new action clears Blender's redo stack
            stack.append(step.idx)


def _compact_state(payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("values") or {}
    out: dict[str, Any] = {}
    for key in ("mode", "workspace", "active_tool", "active_object", "object_count"):
        if key in values:
            out[key] = values[key]
    if values.get("selected_objects") is not None:
        out["selected"] = values["selected_objects"][:20]
    viewport = values.get("viewport") or {}
    if viewport:
        out["view"] = viewport.get("named_view")
        out["perspective"] = viewport.get("perspective")
    summary = values.get("active_object_summary") or {}
    if summary:
        out["dimensions"] = summary.get("dimensions")
        out["modifiers"] = [m.get("type") for m in summary.get("modifiers", [])]
        if summary.get("mesh"):
            out["mesh"] = {k: summary["mesh"].get(k) for k in ("verts", "faces")}
    if isinstance(values.get("edit_selection"), dict):
        out["edit_selection"] = values["edit_selection"]
    return out


def _selection_hint(state: dict[str, Any] | None) -> str | None:
    if not state:
        return None
    parts = []
    if state.get("active_object"):
        parts.append(state["active_object"])
    sel = state.get("edit_selection")
    if isinstance(sel, dict):
        parts.append(f"{sel.get('verts', 0)}v/{sel.get('edges', 0)}e/{sel.get('faces', 0)}f")
    elif state.get("selected"):
        parts.append(f"{len(state['selected'])} selected")
    return "; ".join(parts) or None


def _at_or_before(timeline: list[tuple[float, Any]], keys: list[float], ts: float) -> Any:
    i = bisect.bisect_right(keys, ts) - 1
    return timeline[i][1] if i >= 0 else None


def _first_after(timeline: list[tuple[float, Any]], keys: list[float], ts: float, limit: float) -> Any:
    i = bisect.bisect_left(keys, ts)
    if i < len(timeline) and timeline[i][0] <= limit:
        return timeline[i][1]
    return None


def _frame_at_or_before(frames: list[FrameRecord], keys: list[float], ts: float) -> FrameRecord | None:
    i = bisect.bisect_right(keys, ts) - 1
    if i >= 0:
        return frames[i]
    return frames[0] if frames else None


def _frame_after(frames: list[FrameRecord], keys: list[float], ts: float, limit: float) -> FrameRecord | None:
    i = bisect.bisect_left(keys, ts)
    if i < len(frames) and frames[i].ts <= limit:
        return frames[i]
    return None
