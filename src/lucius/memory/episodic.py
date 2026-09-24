"""Episodic memory: one structured record per processed session (section 24).

Episodes keep the full shape of an experience -- phases with intents and outcomes, failures,
corrections, skills involved -- so they serve provenance, debugging, retrieval, skill
extraction and failure analysis. The summary text is generated deterministically from the
structure (it can be embedded and read), never the other way round.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.intent import Intent
from lucius.segmentation.model import Segment
from lucius.sessions.models import Session
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import fmt_offset, now
from lucius.trajectory.model import TrajectoryStep


class Episode(BaseModel):
    id: str
    session_id: str
    task_text: str | None
    task_class: str | None
    source_class: str
    outcome: str | None
    summary: str
    structured: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    created_at: float

    def text(self) -> str:
        return f"{self.task_text or ''}\n{self.summary}"


def _decode(row: Any) -> Episode:
    return Episode(id=row["id"], session_id=row["session_id"], task_text=row["task_text"], task_class=row["task_class"],
                   source_class=row["source_class"], outcome=row["outcome"], summary=row["summary"],
                   structured=loads(row["structured"], {}), confidence=row["confidence"], created_at=row["created_at"])


class EpisodicMemory:
    def __init__(self, db: Database, bus: EventBus | None = None) -> None:
        self.db = db
        self.bus = bus

    def build(self, session: Session, steps: list[TrajectoryStep], segments: list[Segment],
              intents: dict[str, Intent | None], *, failure_ids: list[str], skill_ids: list[str],
              annotations: list[dict[str, Any]], takeovers: list[dict[str, Any]] | None = None) -> Episode:
        t0 = steps[0].t_start if steps else session.start_time
        phases = []
        for seg in segments:
            intent = intents.get(seg.id)
            phases.append({
                "segment_id": seg.id, "label": seg.label, "title": seg.title, "outcome": seg.outcome,
                "t_start": round(seg.t_start - t0, 2), "t_end": round(seg.t_end - t0, 2),
                "confidence": seg.label_confidence, "origin": seg.origin,
                "intent": None if intent is None else {"category": intent.category, "target": intent.target,
                                                       "confidence": intent.confidence},
                "actions": seg.meta.get("action_counts", {}), "views": seg.meta.get("views", []),
                "objects": seg.meta.get("objects", []), "annotations": seg.meta.get("annotations", []),
            })
        undone = [s for s in steps if "undone_by" in s.meta]
        corrections = [{"undone_step": s.id, "action": s.action_type, "alternative_steps":
                        [steps[s.meta["replaced_by"]].id] if "replaced_by" in s.meta else []} for s in undone]
        outcome = session.outcome.value if session.outcome else (
            "corrected" if any(p["outcome"] == "corrected" for p in phases) else
            "failure" if any(p["outcome"] == "failure" for p in phases) else "unknown")
        structured = {
            "phases": phases, "failures": failure_ids, "corrections": corrections, "skills": skill_ids,
            "annotations": [a.get("text") for a in annotations], "takeovers": len(takeovers or []),
            "duration_s": round((steps[-1].t_end - t0) if steps else 0.0, 2), "step_count": len(steps),
            "objects": sorted({o for p in phases for o in p["objects"]}),
            "blender_version": session.environment.blender_version,
        }
        flow = " -> ".join(
            f"{p['label'].replace('_', ' ')}" + (f" ({', '.join(p['objects'])})" if p["objects"] else "")
            for p in phases)
        notes = "; ".join(f"'{a.get('text')}'" for a in annotations[:3])
        summary = (f"{session.kind.value} from {session.source.value}"
                   f" for '{session.task_text or 'unspecified task'}' ({fmt_offset(structured['duration_s'])}):"
                   f" {flow}.")
        if corrections:
            summary += " Corrections: " + ", ".join(f"{c['action']} undone" for c in corrections) + "."
        if notes:
            summary += f" Notes: {notes}."
        if skill_ids:
            summary += " Skills: " + ", ".join(skill_ids) + "."
        confidence = round(sum(p["confidence"] for p in phases) / len(phases), 3) if phases else 0.0
        existing = self.db.query_one("SELECT id, created_at FROM memory_episodes WHERE session_id = ?", (session.id,))
        episode = Episode(id=existing["id"] if existing else new_id("episode"), session_id=session.id,
                          task_text=session.task_text, task_class=session.task_class, source_class=session.source.value,
                          outcome=outcome, summary=summary, structured=structured, confidence=confidence,
                          created_at=existing["created_at"] if existing else now())
        self.db.insert("memory_episodes", {
            "id": episode.id, "session_id": session.id, "task_text": episode.task_text,
            "task_class": episode.task_class, "source_class": episode.source_class, "outcome": outcome,
            "summary": summary, "structured": dumps(structured), "confidence": confidence,
            "created_at": episode.created_at}, or_replace=True)
        if self.bus is not None:
            self.bus.publish(EventType.MEMORY_CREATED, episode.id, kind="episodic", session_id=session.id)
        return episode

    def get(self, session_id: str) -> Episode | None:
        row = self.db.query_one("SELECT * FROM memory_episodes WHERE session_id = ?", (session_id,))
        return None if row is None else _decode(row)

    def by_id(self, episode_id: str) -> Episode | None:
        row = self.db.query_one("SELECT * FROM memory_episodes WHERE id = ?", (episode_id,))
        return None if row is None else _decode(row)

    def list(self, *, task_class: str | None = None, limit: int = 500) -> list[Episode]:
        if task_class:
            rows = self.db.query("SELECT * FROM memory_episodes WHERE task_class = ? ORDER BY created_at DESC LIMIT ?",
                                 (task_class, limit))
        else:
            rows = self.db.query("SELECT * FROM memory_episodes ORDER BY created_at DESC LIMIT ?", (limit,))
        return [_decode(r) for r in rows]
