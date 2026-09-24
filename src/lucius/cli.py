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
        from lucius.providers.gemini_provider import list_gemini_models

        _print(list_gemini_models())
        return 0
    print("listing is only implemented for gemini; set providers.anthropic_model in Settings", file=sys.stderr)
    return 1


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
        rows = [{"id": s.id, "status": s.status.value, "confidence": round(s.confidence, 3), "version": s.current_version,
                 "uses": f"{s.success_count}/{s.usage_count}", "source": s.source_class}
                for s in app.library.list(include_inactive=True)
                if args.all or s.source_class != "system_seeded"]
        _print(rows)
    finally:
        app.close()
    return 0


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
    s.set_defaults(func=cmd_models)

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
    s.add_argument("--backend", choices=["live", "headless"], default="live")
    s.add_argument("--reference", action="append", help="reference media id (repeatable)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("skills", help="list skills")
    s.add_argument("--all", action="store_true", help="include system-seeded capabilities")
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
