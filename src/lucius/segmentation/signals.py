"""Stage 1: deterministic boundary signals between consecutive trajectory steps."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from lucius.segmentation.model import BoundaryReason
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep


@dataclass(frozen=True)
class SignalConfig:
    pause_threshold_s: float = 4.0
    navigation_min_steps: int = 2
    visual_change_threshold: float = 0.18
    boundary_threshold: float = 0.6
    strong_boundary: float = 0.9
    min_segment_steps: int = 2
    min_segment_seconds: float = 1.5


CREATES_OBJECT = {"add_primitive", "duplicate"}
GLUE_FAMILIES = {"mode", "file", "selection", "ui"}


def _nav(step: TrajectoryStep) -> bool:
    return vocab.is_navigation(step.action_type)


def _active_object(step: TrajectoryStep) -> str | None:
    return (step.state_before or {}).get("active_object")


def boundary_signals(steps: list[TrajectoryStep], cfg: SignalConfig,
                     frame_change: dict[str, float] | None = None) -> dict[int, list[BoundaryReason]]:
    """Boundary reasons keyed by the index of the step that *starts* a new segment."""
    reasons: dict[int, list[BoundaryReason]] = {}
    if len(steps) < 2:
        return reasons
    gaps = [b.t_start - a.t_end for a, b in zip(steps, steps[1:])]
    typical = statistics.median(gaps) if gaps else 0.0
    pause = max(cfg.pause_threshold_s, 3.0 * typical)
    frame_change = frame_change or {}

    def add(i: int, code: str, strength: float, detail: str = "") -> None:
        reasons.setdefault(i, []).append(BoundaryReason(code=code, strength=strength, detail=detail))

    nav_run = [False] * len(steps)
    i = 0
    while i < len(steps):
        if _nav(steps[i]):
            j = i
            while j < len(steps) and _nav(steps[j]):
                j += 1
            if j - i >= cfg.navigation_min_steps:
                for k in range(i, j):
                    nav_run[k] = True
            i = j
        else:
            i += 1

    for i in range(1, len(steps)):
        prev, cur = steps[i - 1], steps[i]
        gap = cur.t_start - prev.t_end
        if gap >= pause:
            add(i, "pause", 0.7, f"{gap:.1f}s without action")
        if cur.action_type == "mode_change" and prev.action_type not in CREATES_OBJECT:
            # (entering edit mode right after creating an object is part of shaping it)
            add(i, "mode_change", 0.75, f"{cur.mode_label} -> {cur.params.get('mode')}")
        elif (cur.mode_label and prev.mode_label and cur.mode_label != prev.mode_label
              and prev.action_type != "mode_change"):
            add(i, "mode_change", 0.75, f"{prev.mode_label} -> {cur.mode_label}")
        if cur.action_type == "workspace_change":
            add(i, "workspace_change", 0.9, str(cur.params.get("workspace")))
        if cur.action_type == "tool_change":
            add(i, "tool_change", 0.5, str(cur.params.get("active_tool")))
        if cur.action_type == "undo" and prev.action_type != "undo":
            add(i, "undo", 0.8, "undo sequence begins")
        if nav_run[i] and not nav_run[i - 1]:
            add(i, "navigation_start", 0.7, "sustained viewport navigation begins")
        if nav_run[i - 1] and not nav_run[i]:
            add(i, "navigation_end", 0.7, "navigation ends")
        if cur.actor != prev.actor:
            add(i, "actor_change", 1.0, f"{prev.actor} -> {cur.actor}")
        if bool(cur.meta.get("during_takeover")) != bool(prev.meta.get("during_takeover")):
            add(i, "takeover", 1.0, "human takeover boundary")
        a, b = _active_object(prev), _active_object(cur)
        if a and b and a != b and prev.action_type not in CREATES_OBJECT:  # creation auto-activates the new object
            add(i, "object_change", 0.8, f"{a} -> {b}")
        if cur.action_type == "add_primitive":
            add(i, "new_object", 0.7, str(cur.params.get("kind")))
        lp = vocab.detail_level(prev.action_type, prev.params)
        lc = vocab.detail_level(cur.action_type, cur.params)
        if lp >= 0 and lc >= 0 and abs(lc - lp) >= 2:
            add(i, "detail_level_jump", 0.5, f"detail level {lp} -> {lc}")
        change = frame_change.get(cur.frame_before_id or "", 0.0)
        if change >= cfg.visual_change_threshold:
            add(i, "visual_transition", 0.4, f"frame change {change:.2f}")
    return reasons


def select_boundaries(steps: list[TrajectoryStep], reasons: dict[int, list[BoundaryReason]],
                      cfg: SignalConfig) -> list[int]:
    """Choose segment start indices: strong signals always cut; weaker ones respect minimum size."""
    starts = [0]
    for i in sorted(reasons):
        strength = min(1.0, sum(r.strength for r in reasons[i]))
        if strength < cfg.boundary_threshold:
            continue
        last = starts[-1]
        long_enough = (i - last >= cfg.min_segment_steps
                       and steps[i - 1].t_end - steps[last].t_start >= cfg.min_segment_seconds)
        if max(r.strength for r in reasons[i]) >= cfg.strong_boundary or long_enough:
            starts.append(i)
    return _absorb_glue(steps, starts, reasons, cfg)


def _is_glue(step: TrajectoryStep) -> bool:
    return vocab.spec(step.action_type).family in GLUE_FAMILIES


def _absorb_glue(steps: list[TrajectoryStep], starts: list[int], reasons: dict[int, list[BoundaryReason]],
                 cfg: SignalConfig) -> list[int]:
    """Segments made only of glue steps (mode switch, save, selection) join a neighbour:
    forwards (a Tab usually begins the next piece of work), or backwards at the end."""
    def strong(i: int) -> bool:
        return any(r.strength >= 1.0 for r in reasons.get(i, []))

    changed = True
    while changed and len(starts) > 1:
        changed = False
        bounds = starts + [len(steps)]
        for k in range(len(starts)):
            a, b = bounds[k], bounds[k + 1]
            if not all(_is_glue(s) for s in steps[a:b]):
                continue
            if k + 1 < len(starts) and not strong(starts[k + 1]):
                del starts[k + 1]
                changed = True
                break
            if k > 0 and not strong(starts[k]):
                del starts[k]
                changed = True
                break
    return starts
