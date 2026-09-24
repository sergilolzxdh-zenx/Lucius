"""Capture-time compression of meaningless cursor motion.

Raw pointer streams are hundreds of events per second. We keep a move when enough time *or*
distance has passed, when the direction turns sharply, and always the last position before a
button event -- enough to reconstruct drags and hovers without flooding storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _Move:
    ts: float
    x: float
    y: float


class MouseMoveCompressor:
    def __init__(self, max_hz: float = 12.0, min_distance: float = 6.0, turn_degrees: float = 45.0) -> None:
        self.min_interval = 1.0 / max_hz
        self.min_distance = min_distance
        self.turn_cos = math.cos(math.radians(turn_degrees))
        self._last_kept: _Move | None = None
        self._last_seen: _Move | None = None
        self._direction: tuple[float, float] | None = None
        self.dropped = 0

    def offer(self, ts: float, x: float, y: float) -> bool:
        """Return True if this move should be stored."""
        move = _Move(ts, x, y)
        prev_seen = self._last_seen
        self._last_seen = move
        kept = self._last_kept
        if kept is None:
            self._keep(move, None)
            return True
        dx, dy = x - kept.x, y - kept.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            self.dropped += 1
            return False
        turned = False
        if prev_seen is not None and self._direction is not None:
            sx, sy = x - prev_seen.x, y - prev_seen.y
            step = math.hypot(sx, sy)
            if step > 0:
                cos = (sx * self._direction[0] + sy * self._direction[1]) / step
                turned = cos < self.turn_cos
        if (ts - kept.ts >= self.min_interval and dist >= self.min_distance) or (turned and dist >= self.min_distance):
            self._keep(move, (dx / dist, dy / dist))
            return True
        self.dropped += 1
        return False

    def flush_pending(self) -> tuple[float, float, float] | None:
        """Last observed-but-dropped position (emitted before a button event)."""
        seen, kept = self._last_seen, self._last_kept
        if seen is None or kept is None or (seen.x == kept.x and seen.y == kept.y):
            return None
        self._keep(seen, None)
        return seen.ts, seen.x, seen.y

    def _keep(self, move: _Move, direction: tuple[float, float] | None) -> None:
        self._last_kept = move
        if direction is not None:
            self._direction = direction
