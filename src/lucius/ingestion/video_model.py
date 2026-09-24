"""A video model watches a tutorial by URL, chunk by chunk (no download).

Some providers fetch public videos themselves (Gemini reads YouTube URLs and clips them with
start/end offsets). That is the only way to analyse a tutorial when the video platform blocks
downloads from the machine running Lucius, and it sees motion that before/after frame pairs miss.

Each chunk (<=5 minutes) is sent with the narration spoken in it ("[MM:SS] words"), and the model
lists the Blender operations it sees, restricted to Lucius' action vocabulary, with times on the
video's own timeline. Everything it reports is ``model_inferred``; the captions give an
independent timeline to check its times against (see ``captions.estimate_lag``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucius.errors import ProviderError, ProviderUnavailable
from lucius.ingestion.captions import CaptionTrack
from lucius.ingestion.download import parse_timestamp
from lucius.ingestion.vision import _operations
from lucius.logging_setup import get_logger
from lucius.providers.base import Providers, VideoInput

log = get_logger("ingestion.video_model")

EVIDENCE = ["visible_keystroke_overlay", "visible_menu_or_panel", "visible_change", "narration_only"]
MODIFIER_ACTIONS = {"add_modifier", "apply_modifier", "remove_modifier", "set_symmetry"}

WATCH_SYSTEM = (
    "You watch a clip of a Blender tutorial and list every modelling operation the person performs, in order. "
    "Use only the allowed operation names; use unknown_action when you see a change but cannot tell the "
    "operation. Times are MM:SS (or H:MM:SS) on the video's own timeline, as shown by the player. Report an "
    "operation only if it is performed in the clip, not merely explained; evidence says how you know it. The "
    "narration (automatic captions, possibly in another language, with recognition errors) tells you what "
    "the narrator says at each time. Parameters: axis (x, y, z) if constrained, value if a number is typed or "
    "stated, kind for added mesh primitives (cube, cylinder, plane, circle, uv_sphere, ico_sphere, cone, torus, "
    "monkey); lights, cameras, curves and text are not add_primitive: use unknown_action for them, "
    "modifier_type for modifiers (e.g. MIRROR, SUBSURF, SOLIDIFY, ARRAY, BEVEL, BOOLEAN). Leave a parameter null "
    "when it is not visible or stated. confidence is 0..1."
)


def watch_schema() -> dict[str, Any]:
    nullable_str = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "operations": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"}, "end": {"type": "string"},
                    "operation": {"type": "string", "enum": _operations()},
                    "object": nullable_str, "mode": nullable_str,
                    "axis": {"type": ["string", "null"], "enum": ["x", "y", "z", None]},
                    "value": {"type": ["number", "null"]}, "kind": nullable_str, "modifier_type": nullable_str,
                    "evidence": {"type": "string", "enum": EVIDENCE},
                    "confidence": {"type": "number"}, "description": {"type": "string"},
                },
                "required": ["start", "end", "operation", "object", "mode", "axis", "value", "kind", "modifier_type",
                             "evidence", "confidence", "description"],
                "additionalProperties": False}},
            "summary": {"type": "string"},
        },
        "required": ["operations", "summary"], "additionalProperties": False,
    }


@dataclass
class WatchedOperation:
    start: float
    end: float
    action: str
    params: dict[str, Any]
    confidence: float
    evidence: str
    description: str
    object: str | None = None
    mode: str | None = None


@dataclass
class WatchReport:
    operations: list[WatchedOperation] = field(default_factory=list)
    chunks: int = 0
    answered: int = 0
    summaries: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    model: str | None = None
    relative_times: int = 0           # chunks whose times the model gave from the clip start

    def stats(self) -> dict[str, Any]:
        return {"chunks": self.chunks, "answered": self.answered, "operations": len(self.operations),
                "model": self.model, "errors": self.errors[:5], "relative_time_chunks": self.relative_times}


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class VideoWatcher:
    def __init__(self, providers: Providers, *, fps: float = 1.0, resolution: str = "low", chunk_s: float = 300.0,
                 narration_chars: int = 6000) -> None:
        self.providers = providers
        self.fps = fps
        self.resolution = resolution
        self.chunk_s = chunk_s
        self.narration_chars = narration_chars

    @property
    def available(self) -> bool:
        return self.providers.has("vlm") and bool(getattr(self.providers.vlm, "supports_video", False))

    def chunks(self, start: float, end: float) -> list[tuple[float, float]]:
        count = max(1, round((end - start) / self.chunk_s + 0.49))
        size = (end - start) / count
        return [(start + i * size, start + (i + 1) * size) for i in range(count)]

    def watch(self, url: str, start: float, end: float, *, context: str,
              narration: CaptionTrack | None = None) -> WatchReport:
        if not self.available:
            raise ProviderUnavailable("no configured vision model accepts video input")
        report = WatchReport()
        for a, b in self.chunks(start, end):
            report.chunks += 1
            lines = [f"Task: {context}", f"Clip: {_clock(a)} to {_clock(b)} of the video."]
            if narration is not None:
                spoken = [f"[{_clock(cue.start)}] {cue.text}" for cue in narration.window(a, b).cues]
                text = "\n".join(spoken)[: self.narration_chars]
                if text:
                    lines += [f"Narration ({narration.language or 'unknown language'}, "
                              f"{narration.source} captions):", text]
            lines.append("List the operations performed in this clip.")
            try:
                result = self.providers.vlm.complete_json(
                    purpose="video_watch", system=WATCH_SYSTEM, prompt="\n".join(lines), schema=watch_schema(),
                    videos=[VideoInput(url, a, b, self.fps, self.resolution)], max_tokens=16000)
            except ProviderError as exc:
                report.errors.append(f"{_clock(a)}-{_clock(b)}: {exc.code}: {exc.message[:200]}")
                log.warning("video chunk %s-%s not analysed: %s", _clock(a), _clock(b), exc.message)
                if isinstance(exc, ProviderUnavailable) or not exc.details.get("transient", True):
                    break  # quota exhausted or configuration problem: later chunks would fail the same way
                continue
            report.answered += 1
            report.model = result.model
            if result.data.get("summary"):
                report.summaries.append(f"[{_clock(a)}] {result.data['summary']}")
            operations, relative = self._parse(result.data.get("operations", []), a, b)
            report.relative_times += int(relative)
            report.operations += operations
        report.operations.sort(key=lambda o: o.start)
        return report

    @staticmethod
    def _parse(items: list[dict[str, Any]], a: float, b: float) -> tuple[list[WatchedOperation], bool]:
        times: list[tuple[float, float]] = []
        kept: list[dict[str, Any]] = []
        for item in items:
            try:
                t0 = parse_timestamp(str(item.get("start", "")))
                t1 = parse_timestamp(str(item.get("end") or item.get("start", "")))
            except ValueError:
                continue
            times.append((t0, max(t0, t1)))
            kept.append(item)
        # Models usually give the player's time; some count from the clip start. Decide per chunk by which
        # reading puts more of the reported times inside the clip (a misread time must not flip it).
        as_absolute = sum(1 for t0, _ in times if a - 5.0 <= t0 <= b + 5.0)
        as_relative = sum(1 for t0, _ in times if -1.0 <= t0 <= (b - a) + 5.0) if a > 60.0 else 0
        relative = as_relative > as_absolute
        out = []
        for (t0, t1), item in zip(times, kept):
            if relative:
                t0, t1 = t0 + a, t1 + a
            if not (a - 5.0 <= t0 <= b + 5.0):
                continue  # outside the clip: a misread time
            action = str(item.get("operation") or "unknown_action")
            params = {k: item[k] for k in ("axis", "value") if item.get(k) is not None}
            if action == "add_primitive" and item.get("kind"):
                params["kind"] = str(item["kind"]).lower().replace(" ", "_")
            modifier = str(item.get("modifier_type") or "").upper()
            if action in MODIFIER_ACTIONS and modifier and modifier not in ("NONE", "NULL"):
                params["type"] = modifier
            out.append(WatchedOperation(
                start=round(min(max(t0, a), b), 3), end=round(min(max(t1, t0, a), b + 1.0), 3),
                action=action, params=params,
                confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                evidence=str(item.get("evidence") or "visible_change"), description=str(item.get("description") or ""),
                object=item.get("object"), mode=item.get("mode")))
        return out, relative
