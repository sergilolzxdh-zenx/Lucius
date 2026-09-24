"""Download a tutorial video with its captions and chapters (yt-dlp; ``pip install lucius[youtube]``).

Only what analysis needs is fetched: the video stream without audio (the narration comes from the
captions) at <=720p, H.264 preferred because OpenCV decodes it everywhere, captions in the video's
own language (manual if published, else the platform's speech recognition, json3 when available
for per-word times) and the metadata (title, duration, chapters, licence).

Platforms rate-limit anonymous downloads from some networks ("confirm you're not a bot"). The
downloader spaces its requests, tries other player clients and backs off; if the network stays
blocked, a cookies file from a signed-in browser (``--cookies``) is the documented remedy.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucius.errors import MediaError
from lucius.logging_setup import get_logger

log = get_logger("ingestion.download")

VIDEO_FORMAT = ("bv*[height<=720][vcodec^=avc1]/bv*[height<=720][vcodec^=av01]/bv*[height<=720]"
                "/b[height<=720]/18/b")
CLIENT_FALLBACKS = (None, "android_vr", "tv", "mweb")   # None: yt-dlp's default clients
# Signs of anti-bot blocking (other player clients may still work): the bot check, rate limits, and 403 on
# video data when the platform wants a proof-of-origin token from this client.
BLOCK_MARKERS = ("confirm you", "not a bot", "429", "too many requests", "sign in to confirm", "403", "forbidden",
                 "page needs to be reloaded")


@dataclass
class Chapter:
    index: int
    title: str
    start: float
    end: float


@dataclass
class DownloadedVideo:
    video_id: str
    url: str
    title: str
    duration: float
    language: str | None
    video_path: Path | None
    captions_path: Path | None
    captions_source: str | None          # "manual" or "auto"
    info_path: Path
    chapters: list[Chapter] = field(default_factory=list)
    license: str | None = None
    channel: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "url": self.url, "title": self.title, "duration": self.duration,
                "language": self.language, "video_path": str(self.video_path) if self.video_path else None,
                "captions_path": str(self.captions_path) if self.captions_path else None,
                "captions_source": self.captions_source, "info_path": str(self.info_path),
                "license": self.license, "channel": self.channel,
                "chapters": [c.__dict__ for c in self.chapters]}


def _yt_dlp() -> Any:
    try:
        import yt_dlp
    except ImportError as exc:
        raise MediaError("downloading needs yt-dlp (pip install lucius[youtube])") from exc
    return yt_dlp


def _base_options(dest: Path, cookies: Path | None, client: str | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "outtmpl": str(dest / "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True, "noprogress": True,
        "sleep_interval_requests": 1.5, "sleep_interval_subtitles": 3, "retries": 5, "fragment_retries": 10,
        "overwrites": False,
    }
    # yt-dlp needs a JavaScript runtime for YouTube; the `deno` wheel installs one next to Python.
    deno = shutil.which("deno") or (Path(sys.prefix) / "bin" / "deno")
    if Path(deno).exists():
        options["js_runtimes"] = {"deno": {"path": str(deno)}}
    if cookies is not None:
        options["cookiefile"] = str(cookies)
    if client:
        options["extractor_args"] = {"youtube": {"player_client": [client]}}
    return options


def _blocked(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in BLOCK_MARKERS)


def _run(url: str, options: dict[str, Any], *, download: bool, attempts_per_client: int = 2,
         backoff_s: float = 20.0) -> dict[str, Any]:
    """Extract (and optionally download), trying other player clients when the platform blocks us."""
    yt_dlp = _yt_dlp()
    last: Exception | None = None
    for client in CLIENT_FALLBACKS:
        opts = {**options}
        if client and "extractor_args" not in options:
            opts["extractor_args"] = {"youtube": {"player_client": [client]}}
        for attempt in range(attempts_per_client):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=download)
                    return ydl.sanitize_info(info)
            except yt_dlp.utils.DownloadError as exc:
                last = exc
                if not _blocked(exc):
                    raise MediaError(f"download failed: {exc}", url=url) from exc
                log.warning("blocked by the platform (client %s, attempt %d): %s", client or "default", attempt + 1,
                            str(exc)[:160])
                time.sleep(backoff_s * (attempt + 1))
    raise MediaError("the video platform is blocking downloads from this network; retry later or pass a cookies "
                     "file exported from a signed-in browser (--cookies)", url=url, error=str(last)[:300])


def _pick_captions(info: dict[str, Any], language: str | None) -> tuple[list[str], str | None]:
    """Caption languages to request, best first: manual in the video's language, then the original ASR track."""
    lang = language or info.get("language")
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    if lang:
        for key in (lang, *[k for k in manual if k.split("-")[0] == lang]):
            if key in manual:
                return [key], "manual"
        for key in (f"{lang}-orig", lang):
            if key in auto:
                return [key], "auto"
    originals = [k for k in auto if k.endswith("-orig")]
    if originals:
        return originals[:1], "auto"
    if manual:
        return [next(iter(manual))], "manual"
    return [], None


def _caption_format(info: dict[str, Any], key: str, source: str) -> str:
    tracks = (info.get("subtitles") if source == "manual" else info.get("automatic_captions")) or {}
    available = {t.get("ext") for t in tracks.get(key, [])}
    return next((ext for ext in ("json3", "vtt", "srt") if ext in available), "best")


def chapters_of(info: dict[str, Any]) -> list[Chapter]:
    duration = float(info.get("duration") or 0.0)
    out = []
    for index, chapter in enumerate(info.get("chapters") or []):
        start = float(chapter.get("start_time") or 0.0)
        end = float(chapter.get("end_time") or duration)
        out.append(Chapter(index=index, title=str(chapter.get("title") or f"chapter {index + 1}"), start=start, end=end))
    return out


def download_video(url: str, dest: str | Path, *, language: str | None = None, cookies: str | Path | None = None,
                   video: bool = True, captions: bool = True) -> DownloadedVideo:
    """Fetch metadata, captions and (optionally) the video. Files already present are reused."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cookie_path = Path(cookies) if cookies else None
    options = _base_options(dest, cookie_path, None)
    info = _run(url, {**options, "skip_download": True}, download=False)
    video_id = info["id"]
    info_path = dest / f"{video_id}.info.json"
    info_path.write_text(json.dumps(info))
    caption_path: Path | None = None
    caption_source: str | None = None
    if captions:
        keys, caption_source = _pick_captions(info, language)
        if keys:
            existing = sorted(dest.glob(f"{video_id}.{keys[0]}.*"))
            if existing:
                caption_path = existing[0]
            else:
                fmt = _caption_format(info, keys[0], caption_source or "auto")
                _run(url, {**options, "skip_download": True, "writesubtitles": caption_source == "manual",
                           "writeautomaticsub": caption_source == "auto", "subtitleslangs": keys,
                           "subtitlesformat": fmt}, download=True)
                found = sorted(dest.glob(f"{video_id}.{keys[0]}.*"))
                caption_path = found[0] if found else None
            if caption_path is None:
                log.warning("captions %s for %s could not be downloaded", keys, video_id)
    video_path: Path | None = None
    if video:
        existing = [p for p in dest.glob(f"{video_id}.*")
                    if p.suffix.lower() in (".mp4", ".webm", ".mkv") and not p.name.endswith(".part")]
        if existing:
            video_path = existing[0]
        else:
            done = _run(url, {**options, "format": VIDEO_FORMAT}, download=True)
            candidates = [Path(d["filepath"]) for d in done.get("requested_downloads") or [] if d.get("filepath")]
            video_path = next((p for p in candidates if p.exists()), None)
            if video_path is None:
                raise MediaError("the video download finished without a file", url=url)
    return DownloadedVideo(
        video_id=video_id, url=url, title=str(info.get("title") or video_id), duration=float(info.get("duration") or 0),
        language=language or info.get("language"), video_path=video_path, captions_path=caption_path,
        captions_source=caption_source if caption_path else None, info_path=info_path, chapters=chapters_of(info),
        license=info.get("license"), channel=info.get("channel"))


def parse_timestamp(value: str) -> float:
    """'1:02:03', '12:30', '95' or '95.5' -> seconds."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"not a timestamp: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def select_chapters(chapters: list[Chapter], spec: str | None) -> list[Chapter]:
    """'all' / None, or 1-based indices and ranges: '3,5-7'."""
    if not spec or spec.strip().lower() == "all":
        return list(chapters)
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        elif part:
            wanted.add(int(part))
    return [c for c in chapters if c.index + 1 in wanted]
