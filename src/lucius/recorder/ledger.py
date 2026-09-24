"""Attribution of input events to the agent or the human.

On X11, synthetic input from the agent is indistinguishable from human input at the OS level.
The executor therefore registers every GUI action it is about to inject; captured events that
match a registered action within its time window are attributed to the agent. Anything else
arriving while the agent is acting is human interference -- the trigger for a takeover.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from lucius.timeutil import now


@dataclass
class ExpectedInput:
    kind: str                     # key_down, key_up, mouse_down, mouse_up, mouse_move, scroll
    key: str | None = None
    button: str | None = None
    x: float | None = None
    y: float | None = None
    tolerance_px: float = 4.0

    def matches(self, kind: str, payload: dict) -> bool:
        if kind != self.kind:
            return False
        if self.key is not None and payload.get("key") != self.key:
            return False
        if self.button is not None and payload.get("button") != self.button:
            return False
        if self.x is not None and self.y is not None:
            px, py = payload.get("x"), payload.get("y")
            if px is None or py is None:
                return False
            if abs(px - self.x) > self.tolerance_px or abs(py - self.y) > self.tolerance_px:
                return False
        return True


@dataclass
class _Window:
    action_id: str
    start: float
    end: float
    expected: list[ExpectedInput] = field(default_factory=list)


class AgentActionLedger:
    def __init__(self, grace_s: float = 0.6) -> None:
        self.grace_s = grace_s
        self._windows: list[_Window] = []
        self._lock = threading.Lock()

    def begin(self, action_id: str, expected: list[ExpectedInput], duration_s: float = 1.0) -> None:
        t = now()
        with self._lock:
            self._prune(t)
            self._windows.append(_Window(action_id, t, t + duration_s, list(expected)))

    def end(self, action_id: str) -> None:
        t = now()
        with self._lock:
            for window in self._windows:
                if window.action_id == action_id:
                    window.end = min(window.end, t)

    def agent_active(self, ts: float | None = None) -> bool:
        t = now() if ts is None else ts
        with self._lock:
            return any(w.start <= t <= w.end + self.grace_s for w in self._windows)

    def attribute(self, ts: float, kind: str, payload: dict) -> str:
        """Return 'agent' if the event was an injected agent input, otherwise 'human'."""
        with self._lock:
            for window in self._windows:
                if not window.start - 0.05 <= ts <= window.end + self.grace_s:
                    continue
                for i, expected in enumerate(window.expected):
                    if expected.matches(kind, payload):
                        del window.expected[i]
                        return "agent"
                if kind == "mouse_move" and window.expected:
                    return "agent"  # pointer travel between injected clicks
        return "human"

    def _prune(self, t: float) -> None:
        self._windows = [w for w in self._windows if w.end + self.grace_s * 4 > t]
