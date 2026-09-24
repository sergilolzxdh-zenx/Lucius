"""Sortable, prefixed identifiers (``ses_0192f3c1a7b4e9d2c1f0``).

The first 12 hex digits encode milliseconds since the epoch so identifiers sort by creation
time, which keeps SQLite index locality good and makes ids human-orderable in exports.
"""

from __future__ import annotations

import secrets
import time

PREFIXES = {
    "session": "ses",
    "frame": "frm",
    "event": "evt",
    "step": "stp",
    "segment": "seg",
    "intent": "int",
    "skill_version": "skv",
    "example": "sex",
    "episode": "epi",
    "semantic": "sem",
    "failure": "fail",
    "correction": "cor",
    "retrieval": "ret",
    "evaluation": "evl",
    "run": "run",
    "plan": "pln",
    "benchmark": "bmk",
    "benchmark_result": "bmr",
    "experiment": "exp",
    "practice_task": "ptk",
    "mastery": "mst",
    "dataset": "dst",
    "sample": "smp",
    "media": "med",
    "demonstration": "dem",
    "job": "job",
    "edge": "edg",
    "constraint": "vcn",
    "model_call": "mcl",
    "edit": "edt",
}


def new_id(kind: str) -> str:
    prefix = PREFIXES.get(kind, kind)
    return f"{prefix}_{int(time.time() * 1000):012x}{secrets.token_hex(5)}"
