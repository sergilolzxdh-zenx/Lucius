"""``lucius`` command line.

Everything the control center does is available here for scripting. Commands operate on one
data directory (``--data-dir`` or ``LUCIUS_DATA_DIR``, default ``.lucius``).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

from lucius.errors import LuciusError

ADDON_DIR = Path(__file__).parent / "blender" / "addon" / "lucius_bridge"


def _app(args: argparse.Namespace, *, background: bool = False):  # type: ignore[no-untyped-def]
    from lucius.app import Lucius

    return Lucius(data_dir=args.data_dir, background_processing=background)


def _print(value: Any) -> None:
    from pydantic import BaseModel

    def encode(v: Any) -> Any:
        if isinstance(v, BaseModel):
            return v.model_dump(mode="json")
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)

    print(json.dumps(value, indent=2, default=encode))


def _backend(app, name: str):  # type: ignore[no-untyped-def]
    if name == "gui":
        return app.gui_backend()
    return app.live_backend() if name == "live" else app.headless_backend()


# -- commands ------------------------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    from lucius.api import load_token, serve

    app = _app(args, background=True)
    load_token(app.config)
    print(f"Lucius control center: http://{args.host}:{args.port}/  (API token in {app.config.data_dir / 'api_token'})")
    try:
        serve(app, host=args.host, port=args.port)
    finally:
        app.close()
    return 0


def cmd_addon(args: argparse.Namespace) -> int:
    """Package the Blender add-on as a zip for Edit > Preferences > Add-ons > Install."""
    out = Path(args.output)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ADDON_DIR.rglob("*.py")):
            z.write(f, Path("lucius_bridge") / f.relative_to(ADDON_DIR))
    print(f"wrote {out} -- install it in Blender (Preferences > Add-ons > Install), enable 'Lucius Bridge'")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        db = app.db
        _print({
            "data_dir": str(app.config.data_dir),
            "sessions": db.scalar("SELECT COUNT(*) FROM sessions"),
            "skills": {r["status"]: r["n"] for r in db.query("SELECT status, COUNT(*) AS n FROM skills GROUP BY status")},
            "failures": db.scalar("SELECT COUNT(*) FROM failure_records"),
            "runs": {r["status"]: r["n"] for r in db.query("SELECT status, COUNT(*) AS n FROM runs GROUP BY status")},
            "providers": app.providers.available(), "provider_notes": app.provider_notes,
        })
    finally:
        app.close()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List the models the provider's key can use (the key is read from the environment)."""
    if args.provider == "gemini":
        from lucius.providers.gemini_provider import list_gemini_models, probe_gemini_models

        models = list_gemini_models()
        if args.check:
            # Listed is not callable: retired models, zero-quota models and audio/image models are listed too.
            probes = {p["id"]: p for p in probe_gemini_models([m["id"] for m in models])}
            models = [{**m, **probes[m["id"]]} for m in models]
            models.sort(key=lambda m: (not m["usable"], m["id"]))
        _print(models)
        return 0
    print("listing is only implemented for gemini; set providers.anthropic_model in Settings", file=sys.stderr)
    return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show the config, or set ``section.field`` values (validated; environment overrides are not persisted)."""
    import os

    from pydantic import ValidationError as PydanticValidationError

    from lucius.config import LuciusConfig

    base = Path(args.data_dir or os.environ.get("LUCIUS_DATA_DIR", ".lucius"))
    path = base / "config.json"
    raw = json.loads(path.read_text()) if path.exists() else {}
    for assignment in args.set or []:
        key, sep, value = assignment.partition("=")
        section, _, field = key.strip().partition(".")
        if not sep or not field:
            print(f"expected section.field=value, got {assignment!r}", file=sys.stderr)
            return 2
        model = LuciusConfig.model_fields.get(section)
        sub_fields = getattr(model.annotation, "model_fields", {}) if model else {}
        if field not in sub_fields:
            print(f"unknown setting {section}.{field}", file=sys.stderr)  # a typo would otherwise be ignored
            return 2
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value  # bare strings need no quotes: providers.gemini_model=gemini-3.5-flash-lite
        raw.setdefault(section, {})[field] = parsed
    try:
        config = LuciusConfig.model_validate({**raw, "data_dir": base})
    except PydanticValidationError as exc:  # names the invalid field
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    if args.set:
        print(f"saved {config.save()}")
    shown = config.model_dump(mode="json", exclude={"data_dir"})
    shown["blender"]["bridge_token"] = "set" if config.blender.bridge_token else None
    _print(shown if not args.section else shown.get(args.section))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from lucius.provenance import DataPolicy
    from lucius.sessions import Outcome

    app = _app(args)
    try:
        recorder = app.recorder()
        status = recorder.status()
        print("capture sources:", json.dumps(status["sources"]))
        session = recorder.start(task_text=args.task, policy=DataPolicy.for_live_demo(training_consent=args.training_consent))
        print(f"RECORDING {session.id} -- demonstrate in Blender, press Ctrl+C to stop")
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        while not stop.wait(1.0):
            pass
        session = recorder.stop(outcome=Outcome(args.outcome))
        print(f"saved {session.id}; processing...")
        result = app.pipeline.process(session.id)
        _print({"session_id": session.id, "processing": result})
    finally:
        app.close()
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        _print(app.pipeline.process(args.session_id, force=args.force))
    finally:
        app.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        backend = _backend(app, args.backend)
        refs = app.ingestion.references.silhouettes(args.reference) if args.reference else []
        result = app.engine.run(args.task, backend, mode="execute", references=refs, reference_ids=args.reference or [],
                                human=None)
        _print({"run_id": result.run_id, "verdict": result.verdict, "metrics": result.metrics,
                "reason_codes": result.reason_codes,
                "unresolved": result.plan.unresolved if result.plan else []})
        return 0 if result.verdict in ("success", "subjective_pass") else 2
    finally:
        app.close()


def cmd_skills(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        if args.action == "remove":
            skills = [s for s in app.library.list(include_inactive=True) if s.source_class != "system_seeded"]
            ids = [s.id for s in skills] if args.learned else list(args.skill_ids or [])
            if not ids:
                print("give skill ids, or --learned for every learned skill", file=sys.stderr)
                return 2
            for skill_id in ids:
                app.library.remove(skill_id)
            forget_lessons(app.config.data_dir / "lessons", set(ids))
            _print({"removed": len(ids), "skills": ids})
            return 0
        rows = [{"id": s.id, "status": s.status.value, "confidence": round(s.confidence, 3), "version": s.current_version,
                 "uses": f"{s.success_count}/{s.usage_count}", "source": s.source_class}
                for s in app.library.list(include_inactive=True)
                if args.all or s.source_class != "system_seeded"]
        _print(rows)
    finally:
        app.close()
    return 0


def forget_lessons(lessons_dir: Path, removed: set[str]) -> None:
    """Chapters whose skill was removed are no longer learned: a later `learn` or `teach course` builds them."""
    for state_path in lessons_dir.glob("*/state.json"):
        state = json.loads(state_path.read_text())
        chapters = state.get("chapters", {})
        kept = {key: entry for key, entry in chapters.items() if entry.get("skill_id") not in removed}
        if len(kept) != len(chapters):
            state["chapters"] = kept
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def cmd_practice(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        report = app.practice.train(args.curriculum, stage=args.stage, attempts=args.attempts,
                                    backend=_backend(app, args.backend), seed=args.seed)
        _print(report)
    finally:
        app.close()
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from lucius.ingestion import MediaInput
    from lucius.ingestion.media import MediaRole

    app = _app(args)
    try:
        inputs = [MediaInput(path=str(Path(p).resolve()), filename=Path(p).name, role=MediaRole(args.role)) for p in args.files]
        instructions = Path(args.instructions).read_text().splitlines() if args.instructions else None
        demo = app.ingestion.create(title=args.title, task_text=args.task, inputs=inputs, instructions=instructions)
        demo = app.ingestion.process(demo.id)
        _print(demo)
    finally:
        app.close()
    return 0


def cmd_tutorial(args: argparse.Namespace) -> int:
    """Learn from a narrated tutorial video: download (or use a local file), split into chapters, process."""
    from lucius.ingestion.download import parse_timestamp
    from lucius.ingestion.tutorial import DEFAULT_SKIP, TutorialImporter

    app = _app(args)
    try:
        if args.max_vision_pairs is not None:
            app.config.processing.max_vision_pairs = args.max_vision_pairs
        if args.video_fps is not None:
            app.ingestion.watcher.fps = args.video_fps
        if args.video_resolution is not None:
            app.ingestion.watcher.resolution = args.video_resolution
        importer = TutorialImporter(app)
        video = _tutorial_video(importer, args, video=not args.remote)
        print(f"{video.title} ({video.duration / 60:.1f} min, language {video.language}, "
              f"{len(video.chapters)} chapters, captions: {video.captions_path or 'none'}, video: "
              f"{video.video_path or 'watched by URL (' + str(app.providers.available()['vlm']) + ')'})",
              file=sys.stderr)
        parts = importer.plan(video, chapters=args.chapters,
                              start=parse_timestamp(args.start) if args.start else None,
                              end=parse_timestamp(args.end) if args.end else None,
                              window_s=args.window_min * 60.0, skip=None if args.no_skip else DEFAULT_SKIP,
                              translate=not args.no_translate)
        if args.plan_only:
            _print({"video": video.to_dict(), "parts": [p.to_dict() for p in parts]})
            return 0

        def progress(part) -> None:  # type: ignore[no-untyped-def]
            result = part.result
            print(f"  {part.title}: {result.get('status')} -- {result.get('steps_identified', 0)}/"
                  f"{result.get('steps', 0)} steps identified, {len(result.get('skills', []))} skills",
                  file=sys.stderr)

        importer.run(video, parts, validate=not args.no_validate, skip_existing=not args.reprocess, on_part=progress)
        _print({"video": video.to_dict(), "parts": [p.to_dict() for p in parts]})
        return 0 if all(p.skipped or p.result.get("status") == "READY" for p in parts) else 2
    finally:
        app.close()


def _tutorial_video(importer, args: argparse.Namespace, *, video: bool):  # type: ignore[no-untyped-def]
    if Path(args.source).exists():
        return importer.local(args.source, captions=args.captions, info=args.info, language=args.language)
    if args.info:
        # Saved metadata: nothing is fetched from the platform (a video model watches the URL).
        return importer.from_info(args.source, args.info, captions=args.captions, language=args.language)
    downloaded = importer.fetch(args.source, language=args.language, cookies=args.cookies,
                                download_dir=args.download_dir, video=video)
    if args.captions:
        downloaded.captions_path, downloaded.captions_source = Path(args.captions), "unknown"
    return downloaded


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cmd_learn(args: argparse.Namespace) -> int:
    """Learn a tutorial as recipes: watch each chapter, rebuild it in Blender, compare with the video, practise."""
    from lucius.ingestion.download import parse_timestamp
    from lucius.ingestion.tutorial import DEFAULT_SKIP, TutorialImporter
    from lucius.lessons import LessonLearner

    app = _app(args)
    try:
        importer = TutorialImporter(app)
        video = _tutorial_video(importer, args, video=False)
        parts = importer.plan(video, chapters=args.chapters,
                              start=parse_timestamp(args.start) if args.start else None,
                              end=parse_timestamp(args.end) if args.end else None,
                              skip=None if args.no_skip else DEFAULT_SKIP, translate=not args.no_translate)
        _say(f"{video.title}: {len([p for p in parts if not p.skipped])} chapters to learn "
             f"(projects in {app.config.projects_dir})")
        learner = LessonLearner(app, practice_rounds=args.practice, target_score=args.target)
        learner.rewrite = args.rewrite
        results = learner.learn(video, parts, redo=args.redo, on_progress=_say)
        _print({"video": video.url, "chapters": [r.to_dict() for r in results]})
        return 0 if results and all(r.status in ("learned", "nothing_to_learn") for r in results) else 2
    finally:
        app.close()


def cmd_make(args: argparse.Namespace) -> int:
    """Make something with what Lucius has learned (optionally from reference images)."""
    from lucius.lessons import Maker

    app = _app(args)
    try:
        maker = Maker(app, iterations=args.iterations)
        backend = _backend(app, args.backend) if args.backend != "headless" else None
        result = maker.make(args.task, references=args.reference or [], allow_unlearned=args.allow_unlearned,
                            backend=backend, on_progress=_say, offline=args.offline)
        _print(result.to_dict())
        return 0 if result.status in ("made", "rebuilt") else 2
    finally:
        app.close()


def cmd_teach(args: argparse.Namespace) -> int:
    """Teach without a model API: build a teacher's recipe, show the tutorial next to it, keep it when scored."""
    from lucius.ingestion.download import parse_timestamp
    from lucius.lessons.teacher import Teacher, load_recipe

    app = _app(args)
    try:
        teacher = Teacher(app, name=args.teacher)
        if args.what == "frames":
            video = teacher.video(args.video)
            out = Path(args.out or app.config.data_dir / "lessons" / args.video / "frames.png")
            _print({"sheet": str(teacher.frames(video, parse_timestamp(args.start), parse_timestamp(args.end), out,
                                                every=args.every))})
            return 0
        if args.what == "narration":
            from lucius.ingestion.captions import CaptionTrack

            video = teacher.video(args.video)
            if video.captions_path is None:
                print("this video has no saved captions", file=sys.stderr)
                return 2
            track = CaptionTrack.from_file(video.captions_path, language=video.language)
            line, start = [], None
            for cue in track.window(parse_timestamp(args.start), parse_timestamp(args.end)).cues:
                start = cue.start if start is None else start
                line.append(cue.text)
                if cue.start - start > 20:
                    print(f"[{int(start // 60)}:{int(start % 60):02d}] {' '.join(line)}")
                    line, start = [], None
            if line and start is not None:
                print(f"[{int(start // 60)}:{int(start % 60):02d}] {' '.join(line)}")
            return 0
        if args.what == "pack":
            if not args.video or not args.out:
                print("give --video and --out (the course folder to write)", file=sys.stderr)
                return 2
            _print({"course": str(teacher.export_course(args.video, args.out))})
            return 0
        if args.what == "objects":
            if not args.out:
                print("give --out (the objects pack folder to write)", file=sys.stderr)
                return 2
            from lucius.lessons import Maker

            ids = [k["skill_id"] for k in Maker(app).known_objects()]
            _print({"pack": str(teacher.export_objects(ids, args.out)), "objects": ids})
            return 0
        if args.what == "course":
            if not args.recipe:
                print("give the course folder (e.g. courses/lrlpwIumFnE)", file=sys.stderr)
                return 2
            results = teacher.install_course(args.recipe, redo=args.redo, on_progress=_say)
            _print(results)
            return 0 if results and all(r["status"] != "failed" for r in results) else 2
        recipe = load_recipe(args.recipe)
        if args.what == "chapter":
            result = teacher.teach_chapter(recipe, args.video, args.chapter, score=args.score, frame_at=args.frame_at,
                                           note=args.note or "")
        else:
            result = teacher.teach_task(recipe, args.task, references=args.reference or [], score=args.score,
                                        note=args.note or "")
        _print(result.to_dict())
        return 0 if result.ok else 2
    finally:
        app.close()


def cmd_projects(args: argparse.Namespace) -> int:
    from lucius.lessons import ProjectStore, rate_project

    app = _app(args)
    try:
        store = ProjectStore(app.config.projects_dir)
        if args.action != "list" and not args.project_id:
            print("give the project id (see `lucius projects list`)", file=sys.stderr)
            return 2
        if args.action == "list":
            _print(store.list(kind=args.kind))
        elif args.action == "show":
            _print(store.get(args.project_id).data)
        else:
            _print(rate_project(app, args.project_id, good=args.action == "good", note=args.note or ""))
    finally:
        app.close()
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from lucius.dataset.service import DatasetFilters

    app = _app(args)
    try:
        result = app.datasets.build(args.name, purpose=args.purpose, filters=DatasetFilters(min_quality=args.min_quality))
        _print(result)
    finally:
        app.close()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        _print(app.benchmarks.experiment(args.name, arms=args.arms, benchmarks=args.benchmarks,
                                         backend=_backend(app, args.backend), repeats=args.repeats))
    finally:
        app.close()
    return 0


def cmd_export_session(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        print(app.datasets.export_session(args.session_id, include_frames=not args.no_frames))
    finally:
        app.close()
    return 0


def cmd_delete_session(args: argparse.Namespace) -> int:
    app = _app(args)
    try:
        _print(app.sessions.delete(args.session_id))
    finally:
        app.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lucius", description="Continual-learning Blender agent")
    p.add_argument("--data-dir", default=None, help="data directory (default: $LUCIUS_DATA_DIR or .lucius)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="start the local API and control center")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("addon", help="package the Blender add-on as an installable zip")
    s.add_argument("--output", default="lucius_bridge.zip")
    s.set_defaults(func=cmd_addon)

    sub.add_parser("status", help="summary of the data directory").set_defaults(func=cmd_status)

    s = sub.add_parser("models", help="list the models your provider key can use")
    s.add_argument("--provider", choices=["gemini"], default="gemini")
    s.add_argument("--check", action="store_true",
                   help="send each model one small JSON+image request and report whether Lucius can use it")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("config", help="show or set configuration (e.g. --set providers.gemini_model=MODEL)")
    s.add_argument("section", nargs="?", help="only show this section (providers, recording, ...)")
    s.add_argument("--set", action="append", metavar="SECTION.FIELD=VALUE", help="repeatable; values are JSON or bare strings")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("record", help="WATCH ME from the terminal (Ctrl+C to stop)")
    s.add_argument("--task", default=None)
    s.add_argument("--outcome", default="success", choices=["success", "failure", "partial", "unknown"])
    s.add_argument("--training-consent", action="store_true", help="allow this recording in training datasets")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("process", help="(re)process a recorded session")
    s.add_argument("session_id")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_process)

    s = sub.add_parser("run", help="ask the agent to perform a task")
    s.add_argument("task")
    s.add_argument("--backend", choices=["live", "gui", "headless"], default="live",
                   help="live: the add-on performs actions; gui: keyboard and mouse (verified by the add-on)")
    s.add_argument("--reference", action="append", help="reference media id (repeatable)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("skills", help="list skills, or remove them (e.g. half-learned ones, to learn again)")
    s.add_argument("action", nargs="?", choices=["list", "remove"], default="list")
    s.add_argument("skill_ids", nargs="*", help="remove: the skills to forget")
    s.add_argument("--learned", action="store_true", help="remove: every learned skill (not the built-in ones)")
    s.add_argument("--all", action="store_true", help="list: include system-seeded capabilities")
    s.set_defaults(func=cmd_skills)

    s = sub.add_parser("practice", help="practise a curriculum stage")
    s.add_argument("curriculum")
    s.add_argument("--stage", type=int, default=None)
    s.add_argument("--attempts", type=int, default=3)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--backend", choices=["live", "headless"], default="headless")
    s.set_defaults(func=cmd_practice)

    s = sub.add_parser("import", help="teach from media files (video, images, .blend, text)")
    s.add_argument("files", nargs="*")
    s.add_argument("--title", required=True)
    s.add_argument("--task", default=None)
    s.add_argument("--role", default="demonstration",
                   choices=["demonstration", "reference", "before", "after", "intermediate", "target", "project"])
    s.add_argument("--instructions", default=None, help="text file with one instruction per line")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("tutorial", help="learn from a narrated tutorial video (URL or local file), chapter by chapter")
    s.add_argument("source", help="video URL (downloaded with yt-dlp) or a local video file")
    s.add_argument("--language", help="caption language (default: the video's language)")
    s.add_argument("--captions", help="captions file (json3, VTT or SRT); downloaded automatically for URLs")
    s.add_argument("--info", help="yt-dlp .info.json with chapters: for a local video, or for a URL whose "
                                  "metadata was saved earlier (the platform is then not contacted)")
    s.add_argument("--chapters", help="which chapters: 'all' (default) or 1-based '3,5-7'")
    s.add_argument("--start", help="only this part: start time (1:02:03, 12:30 or seconds)")
    s.add_argument("--end", help="only this part: end time")
    s.add_argument("--window-min", type=float, default=15.0, help="window length for videos without chapters")
    s.add_argument("--max-vision-pairs", type=int, help="vision-model pairs per chapter (quota control)")
    s.add_argument("--video-fps", type=float, help="frames per second a video model samples (default 1)")
    s.add_argument("--video-resolution", choices=["low", "medium", "high"], help="video model resolution")
    s.add_argument("--cookies", help="cookies.txt from a signed-in browser, if the platform blocks downloads")
    s.add_argument("--remote", action="store_true",
                   help="do not download the video: a video-capable model (Gemini) watches it by URL")
    s.add_argument("--download-dir", help="where downloads go (default <data-dir>/downloads)")
    s.add_argument("--no-skip", action="store_true", help="also process intro/installation/promotion chapters")
    s.add_argument("--no-translate", action="store_true", help="keep non-English chapter titles untranslated")
    s.add_argument("--no-validate", action="store_true", help="skip validation by reproduction in headless Blender")
    s.add_argument("--reprocess", action="store_true", help="process chapters again even if already done")
    s.add_argument("--plan-only", action="store_true", help="download and show the chapter plan without processing")
    s.set_defaults(func=cmd_tutorial)

    s = sub.add_parser("learn", help="learn a tutorial as recipes: watch each chapter, rebuild it in Blender, "
                                     "compare with the video and practise (renders saved as projects)")
    s.add_argument("source", help="video URL (watched by a video model), or a local video file")
    s.add_argument("--info", help="saved yt-dlp .info.json (chapters, captions; the platform is not contacted)")
    s.add_argument("--captions", help="captions file (json3, VTT or SRT)")
    s.add_argument("--language", help="caption language (default: the video's language)")
    s.add_argument("--cookies", help="cookies.txt from a signed-in browser, if the platform blocks metadata")
    s.add_argument("--download-dir", help="where metadata and captions go (default <data-dir>/downloads)")
    s.add_argument("--chapters", help="which chapters: 'all' (default) or 1-based '4,5-7'")
    s.add_argument("--start", help="only this part: start time (1:02:03, 12:30 or seconds)")
    s.add_argument("--end", help="only this part: end time")
    s.add_argument("--practice", type=int, default=2, help="practice rounds per chapter after the first attempt")
    s.add_argument("--target", type=float, default=8.0, help="stop practising a chapter at this score (0-10)")
    s.add_argument("--redo", action="store_true", help="learn chapters again even if already learned (practice "
                                                       "continues from the best earlier recipe)")
    s.add_argument("--rewrite", action="store_true", help="with --redo: write new recipes from the notes instead")
    s.add_argument("--no-skip", action="store_true", help="also learn intro/installation/promotion chapters")
    s.add_argument("--no-translate", action="store_true", help="keep non-English chapter titles untranslated")
    s.set_defaults(func=cmd_learn)

    s = sub.add_parser("make", help="make something with the techniques Lucius has learned")
    s.add_argument("task", help="what to make, e.g. \"a sword with a long blade\"")
    s.add_argument("--reference", action="append", help="reference image (PNG/JPEG/WebP); repeat for several")
    s.add_argument("--iterations", type=int, default=3, help="build-judge-revise rounds")
    s.add_argument("--allow-unlearned", action="store_true",
                   help="also use Blender actions no lesson has taught (the project records it)")
    s.add_argument("--backend", choices=["headless", "live", "gui"], default="headless",
                   help="headless Blender (default), your Blender through the add-on, or keyboard and mouse")
    s.add_argument("--offline", action="store_true",
                   help="no model: rebuild the learned object the task names (automatic when no model is set up)")
    s.set_defaults(func=cmd_make)

    s = sub.add_parser("teach", help="teach without a model API: build a recipe you wrote (for a tutorial chapter "
                                     "or a task), see it next to the tutorial, keep it with --score")
    s.add_argument("what", choices=["chapter", "task", "frames", "narration", "course", "pack", "objects"],
                   help="chapter/task: build RECIPE; frames: the tutorial's preview frames; narration: what the "
                        "tutor says (captions) between --start and --end; course: replay a course pack (a folder "
                        "with course.json) so Lucius learns every chapter on this machine; pack: write the chapters "
                        "kept for --video as a course pack in --out; objects: write the taught objects as a pack")
    s.add_argument("recipe", nargs="?", help="recipe JSON file (chapter, task) or course folder (course)")
    s.add_argument("--video", help="video id of a saved tutorial (e.g. lrlpwIumFnE)")
    s.add_argument("--chapter", type=int, help="1-based chapter number")
    s.add_argument("--task", help="what the recipe makes (task)")
    s.add_argument("--reference", action="append", help="reference image (task); repeat for several")
    s.add_argument("--score", type=float, help="your judgement 0-10: keeps the recipe as the skill (without it, a "
                                               "trial run)")
    s.add_argument("--frame-at", help="tutorial time to show next to the renders (default: 15 s before the end)")
    s.add_argument("--start", help="frames/narration: from (12:30)")
    s.add_argument("--end", help="frames/narration: to")
    s.add_argument("--every", type=float, default=10.0, help="frames: seconds between frames (the preview has one "
                                                           "every ~10 s)")
    s.add_argument("--out", help="frames: where the picture goes; pack: the course folder")
    s.add_argument("--teacher", default="teacher", help="who wrote and judged it (recorded with the skill)")
    s.add_argument("--note", help="what you changed or noticed")
    s.add_argument("--redo", action="store_true", help="course: build chapters already learned again")
    s.set_defaults(func=cmd_teach)

    s = sub.add_parser("projects", help="what Lucius built: list, show, or rate a project good/bad")
    s.add_argument("action", choices=["list", "show", "good", "bad"], nargs="?", default="list")
    s.add_argument("project_id", nargs="?")
    s.add_argument("--kind", choices=["lesson", "task"])
    s.add_argument("--note", help="what was good or wrong (kept with the rating)")
    s.set_defaults(func=cmd_projects)

    s = sub.add_parser("dataset", help="build a dataset (training datasets include consented sessions only)")
    s.add_argument("name")
    s.add_argument("--purpose", choices=["training", "export"], default="training")
    s.add_argument("--min-quality", type=float, default=0.5)
    s.set_defaults(func=cmd_dataset)

    s = sub.add_parser("benchmark", help="run an A/B experiment")
    s.add_argument("name")
    s.add_argument("--arms", nargs="+", default=["baseline", "memory_enhanced"])
    s.add_argument("--benchmarks", nargs="+", default=["sized_box", "blade_blockout"])
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--backend", choices=["live", "headless"], default="headless")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("export-session", help="export a session bundle (zip)")
    s.add_argument("session_id")
    s.add_argument("--no-frames", action="store_true")
    s.set_defaults(func=cmd_export_session)

    s = sub.add_parser("delete-session", help="delete a session and its unreferenced frames")
    s.add_argument("session_id")
    s.set_defaults(func=cmd_delete_session)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except LuciusError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        if exc.details:
            print(json.dumps(exc.details, default=str), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
