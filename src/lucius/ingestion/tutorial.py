"""Tutorials: a long video with narration, split into chapters, each taught as one demonstration.

A 10-hour course covers dozens of unrelated topics; one demonstration per chapter keeps each
session about one thing, gives it a task text (the chapter title) and lets a long course be
processed, resumed and inspected piece by piece. Without chapters, the video is cut into fixed
windows; ``start``/``end`` select one window (a test slice or an important part).

Chapter titles are translated to English (one model call per video) when the video is in another
language, so an English task finds skills learned from a Spanish tutorial. The original title is
kept next to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucius.errors import MediaError, ProviderError, ProviderUnavailable
from lucius.ingestion.download import Chapter, DownloadedVideo, chapters_of, download_video, select_chapters
from lucius.ingestion.media import MediaKind, MediaRole
from lucius.logging_setup import get_logger
from lucius.provenance import DataPolicy, SourceClass

if TYPE_CHECKING:
    from lucius.app import Lucius
    from lucius.ingestion.service import Demonstration

log = get_logger("ingestion.tutorial")

# Chapters that rarely show modelling work: introductions, installation, promotions.
DEFAULT_SKIP = (r"(?i)^(intro(duction|duccion|ducción)?|outro|bienvenida|welcome|final|credits|créditos)\b|install|"
                r"instalaci|descuento|discount|sponsor|patreon|resultado final|final result")

TRANSLATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"titles": {"type": "array", "items": {
        "type": "object", "properties": {"index": {"type": "integer"}, "english": {"type": "string"}},
        "required": ["index", "english"], "additionalProperties": False}}},
    "required": ["titles"], "additionalProperties": False,
}


@dataclass
class TutorialPart:
    title: str
    start: float
    end: float
    task_text: str
    chapter: Chapter | None = None
    skipped: str | None = None
    demonstration_id: str | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "start": round(self.start, 3), "end": round(self.end, 3),
                "task_text": self.task_text, "skipped": self.skipped, "demonstration_id": self.demonstration_id,
                **({"result": self.result} if self.result else {})}


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _quota_exhausted(result: dict[str, Any]) -> bool:
    text = str(result.get("error") or "") + " ".join((result.get("watched_by_model") or {}).get("errors", []))
    return "daily quota" in text or "no quota" in text


class TutorialImporter:
    def __init__(self, app: Lucius) -> None:
        self.app = app

    # -- sources -----------------------------------------------------------------------------------------
    def fetch(self, url: str, *, language: str | None = None, cookies: str | None = None,
              download_dir: str | Path | None = None, video: bool = True,
              fallback_to_remote: bool = True) -> DownloadedVideo:
        """Download metadata, captions and the video. If the platform blocks the video download and a
        video-capable model is configured, continue without it: the model watches the URL instead."""
        dest = Path(download_dir) if download_dir else self.app.config.data_dir / "downloads"
        try:
            return download_video(url, dest, language=language, cookies=cookies, video=video)
        except MediaError as exc:
            if not (video and fallback_to_remote and self.app.ingestion.watcher.available):
                raise
            log.warning("video download failed (%s); the video model will watch %s by URL instead", exc.message, url)
            return download_video(url, dest, language=language, cookies=cookies, video=False)

    @staticmethod
    def local(video: str | Path, *, captions: str | Path | None = None, info: str | Path | None = None,
              language: str | None = None) -> DownloadedVideo:
        """A video already on disk, with optional captions and a yt-dlp ``.info.json`` for chapters."""
        import json

        from lucius.ingestion.video import VideoReader

        video = Path(video)
        reader = VideoReader(video)
        duration = reader.duration
        reader.close()
        meta: dict[str, Any] = json.loads(Path(info).read_text()) if info else {}
        return DownloadedVideo(video_id=meta.get("id") or video.stem, url=meta.get("webpage_url") or str(video),
                               title=meta.get("title") or video.stem, duration=duration,
                               language=language or meta.get("language"), video_path=video,
                               captions_path=Path(captions) if captions else None,
                               captions_source="unknown" if captions else None,
                               info_path=Path(info) if info else video, chapters=chapters_of(meta) if meta else [],
                               license=meta.get("license"), channel=meta.get("channel"))

    @staticmethod
    def from_info(url: str, info: str | Path, *, captions: str | Path | None = None,
                  language: str | None = None) -> DownloadedVideo:
        """A video known only by URL and saved metadata (a yt-dlp ``.info.json``): nothing is fetched from
        the platform; a video model watches the URL."""
        import json

        meta: dict[str, Any] = json.loads(Path(info).read_text())
        return DownloadedVideo(video_id=meta.get("id") or url, url=url, title=meta.get("title") or url,
                               duration=float(meta.get("duration") or 0.0), language=language or meta.get("language"),
                               video_path=None, captions_path=Path(captions) if captions else None,
                               captions_source="unknown" if captions else None, info_path=Path(info),
                               chapters=chapters_of(meta), license=meta.get("license"), channel=meta.get("channel"))

    # -- planning ----------------------------------------------------------------------------------------
    def plan(self, video: DownloadedVideo, *, chapters: str | None = None, start: float | None = None,
             end: float | None = None, window_s: float = 900.0, skip: str | None = DEFAULT_SKIP,
             translate: bool = True) -> list[TutorialPart]:
        duration = video.duration or (max((c.end for c in video.chapters), default=0.0))
        if start is not None or end is not None:
            a, b = max(0.0, start or 0.0), min(duration, end if end is not None else duration)
            covering = max(video.chapters, key=lambda c: max(0.0, min(b, c.end) - max(a, c.start)), default=None)
            topic = covering.title if covering else video.title
            part = TutorialPart(title=f"{video.title} [{_clock(a)}-{_clock(b)}]", start=a, end=b, task_text=topic,
                                chapter=covering)
            self._translate(video, [part], enabled=translate)
            return [part]
        if video.chapters:
            parts = [TutorialPart(title=f"{video.title} — {c.index + 1}. {c.title}", start=c.start, end=c.end,
                                  task_text=c.title, chapter=c) for c in select_chapters(video.chapters, chapters)]
        else:
            parts, t = [], 0.0
            while t < duration - 1.0:
                b = min(duration, t + window_s)
                parts.append(TutorialPart(title=f"{video.title} [{_clock(t)}-{_clock(b)}]", start=t, end=b,
                                          task_text=video.title))
                t = b
        for part in parts:
            if skip and part.chapter is not None and re.search(skip, part.chapter.title):
                part.skipped = "chapter title matches the skip pattern (intro, installation, promotion...)"
            elif part.end - part.start < 20.0:
                part.skipped = "shorter than 20 s"
        self._translate(video, [p for p in parts if not p.skipped], enabled=translate)
        return parts

    def _translate(self, video: DownloadedVideo, parts: list[TutorialPart], *, enabled: bool) -> None:
        language = (video.language or "").split("-")[0].lower()
        if not enabled or not parts or language in ("", "en") or not self.app.providers.has("llm"):
            return
        titles = "\n".join(f"{i}: {p.task_text}" for i, p in enumerate(parts))
        try:
            result = self.app.providers.llm.complete_json(
                purpose="chapter_title_translation",
                system="Translate Blender tutorial chapter titles to short English task descriptions. Keep "
                       "Blender terms (modifier, extrude, shader...) in their English form.",
                prompt=f"Language: {language}\nVideo: {video.title}\nTitles:\n{titles}", schema=TRANSLATE_SCHEMA,
                max_tokens=2000)
        except ProviderError as exc:
            log.warning("chapter titles not translated: %s", exc.message)
            return
        for item in result.data.get("titles", []):
            index = item.get("index")
            english = str(item.get("english") or "").strip()
            if isinstance(index, int) and 0 <= index < len(parts) and english:
                parts[index].task_text = f"{english} ({parts[index].task_text})"

    # -- running -----------------------------------------------------------------------------------------
    def existing(self, part: TutorialPart) -> Demonstration | None:
        row = self.app.db.query_one("SELECT id FROM demonstrations WHERE title = ? AND status = 'READY' "
                                    "ORDER BY created_at DESC LIMIT 1", (part.title,))
        return self.app.ingestion.get(row["id"]) if row else None

    def run(self, video: DownloadedVideo, parts: list[TutorialPart], *, validate: bool = True,
            skip_existing: bool = True, on_part: Any = None) -> list[TutorialPart]:
        from lucius.ingestion.service import MediaInput

        remote = video.video_path is None
        if remote and not self.app.ingestion.watcher.available:
            raise ProviderUnavailable("the video was not downloaded and no configured model can watch it by URL "
                                      "(set providers.vlm to gemini)")
        policy = DataPolicy.for_external(SourceClass.EXTERNAL_VIDEO, license=video.license or "unknown")
        policy.notes = f"{video.url} ({video.channel or 'unknown channel'})"
        for part in parts:
            if part.skipped:
                continue
            if skip_existing and (done := self.existing(part)) is not None:
                part.demonstration_id = done.id
                part.skipped = "already processed"
                continue
            inputs = [MediaInput(url=video.url, kind=MediaKind.VIDEO, role=MediaRole.DEMONSTRATION,
                                 clip_start=part.start, clip_end=part.end, source_url=video.url) if remote else
                      MediaInput(path=str(video.video_path), role=MediaRole.DEMONSTRATION, clip_start=part.start,
                                 clip_end=part.end, source_url=video.url)]
            if video.captions_path is not None:
                inputs.append(MediaInput(path=str(video.captions_path), role=MediaRole.NARRATION,
                                         kind=MediaKind.CAPTIONS, language=video.language, source_url=video.url))
            demo = self.app.ingestion.create(title=part.title, task_text=part.task_text, inputs=inputs,
                                             policy=policy.model_copy())
            part.demonstration_id = demo.id
            result = self.app.ingestion.process(demo.id, validate=validate)
            part.result = self.summary(result.id)
            if on_part is not None:
                on_part(part)
            if _quota_exhausted(part.result):
                # Every later chapter would fail the same way; re-running later resumes here (done parts are skipped).
                for rest in parts[parts.index(part) + 1:]:
                    if not rest.skipped:
                        rest.skipped = "model quota exhausted; run the command again later to continue"
                break
        return parts

    def summary(self, demo_id: str) -> dict[str, Any]:
        demo = self.app.ingestion.get(demo_id)
        out: dict[str, Any] = {"status": demo.status.value}
        if demo.error:
            out["error"] = demo.error
        history = {h["status"]: h for h in demo.status_history}
        out["vision"] = history.get("INFERRING_ACTIONS", {}).get("vision")
        if demo.session_id:
            steps = self.app.trajectories.for_session(demo.session_id)
            known = [s for s in steps if s.action_type != "unknown_action"]
            out["steps"] = len(steps)
            out["steps_identified"] = len(known)
            out["actions"] = sorted({s.action_type for s in known})
            out["steps_with_narration"] = sum(1 for s in steps if s.meta.get("narration"))
            segments = self.app.db.query("SELECT label FROM segments WHERE session_id = ? ORDER BY idx",
                                         (demo.session_id,))
            out["segments"] = [r["label"] for r in segments]
        for asset in self.app.ingestion.media.list(demonstration_id=demo_id):
            if asset.kind == MediaKind.VIDEO:
                if "narration" in asset.analysis:
                    out["narration"] = asset.analysis["narration"]
                if "watch" in asset.analysis:
                    out["watched_by_model"] = asset.analysis["watch"]
        ready = history.get("VALIDATING", {})
        out["skills"] = ready.get("skills", [])
        out["validation"] = history.get("READY", {}).get("validation")
        return out
