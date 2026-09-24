"""Dataset quality validation (section 46).

Every session considered for a dataset is checked for: missing or corrupted frames, invalid
timestamps, broken event ordering, unknown action types, missing outcome, corrupted references,
duplicate sessions and (dataset-wide) inconsistent Blender versions. Issues carry a severity
and the quality score is derived from them -- never silently ignored.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lucius.ingestion.media import MediaStore
from lucius.sessions.models import Session
from lucius.sessions.store import SessionStore
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.store import TrajectoryStore

SEVERITY_WEIGHT = {"error": 0.35, "warning": 0.1, "info": 0.02}
TS_TOLERANCE_S = 0.5


class Issue(BaseModel):
    code: str
    severity: str
    detail: str
    count: int = 1


class SessionQuality(BaseModel):
    session_id: str
    score: float
    issues: list[Issue] = Field(default_factory=list)


class QualityReport(BaseModel):
    sessions: list[SessionQuality] = Field(default_factory=list)
    dataset_issues: list[Issue] = Field(default_factory=list)
    mean_score: float | None = None
    blender_versions: dict[str, int] = Field(default_factory=dict)


def _score(issues: list[Issue]) -> float:
    penalty = sum(SEVERITY_WEIGHT[i.severity] * min(3, i.count) for i in issues)
    return round(max(0.0, 1.0 - penalty), 4)


class DatasetValidator:
    def __init__(self, sessions: SessionStore, trajectories: TrajectoryStore, media: MediaStore) -> None:
        self.sessions = sessions
        self.trajectories = trajectories
        self.media = media

    def validate_session(self, session: Session, *, seen_hashes: dict[str, str] | None = None) -> SessionQuality:
        issues: list[Issue] = []
        frames = self.sessions.frames_for(session.id)
        missing = [f.id for f in frames if not self.sessions.frames.verify(f.path, f.sha256)]
        if missing:
            issues.append(Issue(code="missing_or_corrupted_frames", severity="error", count=len(missing),
                                detail=f"{len(missing)} frame file(s) missing or not matching their hash"))
        events = self.sessions.events(session.id)
        end = session.end_time or (events[-1].ts if events else session.start_time)
        bad_ts = [e.seq for e in events if e.ts < session.start_time - TS_TOLERANCE_S or e.ts > end + TS_TOLERANCE_S]
        if bad_ts:
            issues.append(Issue(code="invalid_timestamps", severity="error", count=len(bad_ts),
                                detail=f"{len(bad_ts)} event(s) outside the session time range"))
        regressions = sum(1 for a, b in zip(events, events[1:]) if b.ts + TS_TOLERANCE_S < a.ts)
        if regressions:
            issues.append(Issue(code="broken_event_ordering", severity="error", count=regressions,
                                detail=f"{regressions} event(s) whose time goes backwards in sequence order"))
        steps = self.trajectories.for_session(session.id)
        unknown = [s.action_type for s in steps if s.action_type not in vocab.ACTION_TYPES]
        if unknown:
            issues.append(Issue(code="unknown_action_types", severity="warning", count=len(unknown),
                                detail="unmapped action types: " + ", ".join(sorted(set(unknown))[:8])))
        unresolved = sum(1 for s in steps if s.action_type == "unknown_action")
        if unresolved:
            issues.append(Issue(code="unresolved_actions", severity="info", count=unresolved,
                                detail=f"{unresolved} step(s) whose operation could not be determined"))
        if not steps:
            issues.append(Issue(code="no_trajectory", severity="error", detail="session has not been processed"))
        if session.outcome is None:
            issues.append(Issue(code="missing_outcome", severity="warning", detail="session outcome is not recorded"))
        for media_id in session.reference_ids + session.meta.get("media", []):
            try:
                asset = self.media.get(media_id)
                if asset.analysis.get("remote"):
                    continue  # fetched by the provider from its URL; there is no local copy to verify
                if self.media.digest(asset) != asset.sha256:
                    issues.append(Issue(code="corrupted_reference", severity="error", detail=f"media {media_id}"))
            except Exception:
                issues.append(Issue(code="corrupted_reference", severity="error", detail=f"media {media_id} missing"))
        if seen_hashes is not None and session.content_hash:
            other = seen_hashes.get(session.content_hash)
            if other and other != session.id:
                issues.append(Issue(code="duplicate_session", severity="error", detail=f"same content as {other}"))
            seen_hashes.setdefault(session.content_hash, session.id)
        return SessionQuality(session_id=session.id, score=_score(issues), issues=issues)

    def validate(self, sessions: list[Session]) -> QualityReport:
        seen: dict[str, str] = {}
        results = [self.validate_session(s, seen_hashes=seen) for s in sessions]
        versions: dict[str, int] = {}
        for s in sessions:
            version = (s.environment.blender_version or "unknown").split(" ")[0]
            versions[version] = versions.get(version, 0) + 1
        dataset_issues = []
        majors = {v.rsplit(".", 1)[0] for v in versions if v != "unknown"}
        if len(majors) > 1:
            dataset_issues.append(Issue(code="inconsistent_blender_versions", severity="warning",
                                        detail=f"sessions span Blender {sorted(majors)}"))
        mean = round(sum(r.score for r in results) / len(results), 4) if results else None
        return QualityReport(sessions=results, dataset_issues=dataset_issues, mean_score=mean, blender_versions=versions)

    @staticmethod
    def summary(report: QualityReport) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for s in report.sessions:
            for issue in s.issues:
                counts[issue.code] = counts.get(issue.code, 0) + issue.count
        return {"mean_score": report.mean_score, "issue_counts": counts,
                "dataset_issues": [i.model_dump() for i in report.dataset_issues],
                "blender_versions": report.blender_versions}
