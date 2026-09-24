from __future__ import annotations

import time
from datetime import datetime, timezone


def now() -> float:
    """Wall-clock seconds since the epoch (the single time base used in storage)."""
    return time.time()


def iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(now() if ts is None else ts, tz=timezone.utc).isoformat()


def fmt_offset(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"
