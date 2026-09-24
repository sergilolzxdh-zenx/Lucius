"""User workflow preferences (section 33).

Kept separate from skills: a skill describes *what works*, preferences describe *how this user
likes to work*. They are learned only from the user's own demonstrations and corrections (not
from tutorials), and they bias retrieval, planning and variant choice without being baked into
any skill. Explicit human settings always win over learned values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from lucius.memory.episodic import Episode
from lucius.provenance import SourceClass
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

USER_SOURCES = {SourceClass.USER_DEMO.value, SourceClass.HUMAN_CORRECTION.value}
ORTHO = {"front", "back", "left", "right", "top", "bottom"}


class Preference(BaseModel):
    key: str
    value: Any
    confidence: float
    source: str                     # learned | human
    support: int = 0
    evidence: list[str] = Field(default_factory=list)
    updated_at: float = 0.0


def _phases(ep: Episode) -> list[dict[str, Any]]:
    return ep.structured.get("phases", [])


def _index(phases: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int | None:
    return next((i for i, p in enumerate(phases) if predicate(p)), None)


def _blockout_first(ep: Episode) -> bool | None:
    phases = _phases(ep)
    detail = _index(phases, lambda p: p["label"] == "detail_pass")
    blockout = _index(phases, lambda p: p["label"] == "primary_blockout")
    if detail is None or blockout is None:
        return None
    return blockout < detail


def _symmetry_early(ep: Episode) -> bool | None:
    phases = _phases(ep)
    mirror = _index(phases, lambda p: "add_modifier" in p.get("actions", {}) and p["label"] in
                    ("scene_setup", "primary_blockout"))
    if mirror is None:
        return None
    return mirror <= max(1, len(phases) // 4)


def _bevel_late(ep: Episode) -> bool | None:
    phases = [p for p in _phases(ep) if p.get("outcome") != "failure"]
    bevel = [i for i, p in enumerate(phases) if "bevel" in p.get("actions", {}) and p["label"] != "corrective_pass"]
    if not bevel:
        undone_bevel = any(c.get("action") == "bevel" for c in ep.structured.get("corrections", []))
        return True if undone_bevel else None  # an early bevel the user undid is evidence for "late"
    return bevel[0] >= len(phases) * 0.6


def _frequent_inspection(ep: Episode) -> bool | None:
    phases = _phases(ep)
    if len(phases) < 3:
        return None
    return sum(1 for p in phases if p["label"] in ("inspection", "verification")) >= 2


def _orthographic_checks(ep: Episode) -> bool | None:
    inspections = [p for p in _phases(ep) if p["label"] in ("inspection", "verification")]
    if not inspections:
        return None
    return any(set(p.get("views", [])) & ORTHO for p in inspections)


LEARNERS: dict[str, Callable[[Episode], bool | None]] = {
    "blockout_first": _blockout_first,
    "symmetry_early": _symmetry_early,
    "bevel_late": _bevel_late,
    "frequent_viewport_inspection": _frequent_inspection,
    "orthographic_checks": _orthographic_checks,
}


class WorkflowPreferences:
    def __init__(self, db: Database, user_id: str = "local") -> None:
        self.db = db
        self.user_id = user_id

    def learn(self, episodes: list[Episode]) -> dict[str, Preference]:
        mine = [ep for ep in episodes if ep.source_class in USER_SOURCES]
        current = self.get_all()
        for key, learner in LEARNERS.items():
            if current.get(key) and current[key].source == "human":
                continue
            votes = [(ep.session_id, v) for ep in mine if (v := learner(ep)) is not None]
            if not votes:
                continue
            share = sum(1 for _s, v in votes if v) / len(votes)
            value = share >= 0.5
            confidence = round(abs(share - 0.5) * 2 * (1 - 1 / (len(votes) + 1)), 3)
            self._put(Preference(key=key, value=value, confidence=confidence, source="learned", support=len(votes),
                                 evidence=[s for s, v in votes if v == value][:50], updated_at=now()))
        return self.get_all()

    def set(self, key: str, value: Any) -> Preference:
        pref = Preference(key=key, value=value, confidence=1.0, source="human", support=0, evidence=["set by user"],
                          updated_at=now())
        self._put(pref)
        return pref

    def clear(self, key: str) -> None:
        self.db.execute("DELETE FROM preferences WHERE user_id = ? AND key = ?", (self.user_id, key))

    def _put(self, pref: Preference) -> None:
        self.db.execute("INSERT INTO users(id, name, created_at) VALUES (?, ?, ?) ON CONFLICT(id) DO NOTHING",
                        (self.user_id, self.user_id, now()))
        self.db.insert("preferences", {
            "user_id": self.user_id, "key": pref.key, "value": dumps(pref.value), "confidence": pref.confidence,
            "source": pref.source, "evidence": dumps({"support": pref.support, "sessions": pref.evidence}),
            "updated_at": pref.updated_at}, or_replace=True)

    def get_all(self) -> dict[str, Preference]:
        rows = self.db.query("SELECT * FROM preferences WHERE user_id = ?", (self.user_id,))
        out = {}
        for r in rows:
            evidence = loads(r["evidence"], {})
            out[r["key"]] = Preference(key=r["key"], value=loads(r["value"]), confidence=r["confidence"],
                                       source=r["source"], support=evidence.get("support", 0),
                                       evidence=evidence.get("sessions", []), updated_at=r["updated_at"])
        return out

    def active(self, min_confidence: float = 0.3) -> dict[str, Any]:
        """Preferences confident enough to influence planning."""
        return {k: p.value for k, p in self.get_all().items() if p.source == "human" or p.confidence >= min_confidence}
