"""Execute a recipe in Blender, step by step, and render what it built.

Every step passes the same safety validator as any planned action, then goes to the backend (headless
Blender, the user's Blender through the add-on, or keyboard and mouse). A run stops at the first
failing step: later steps act on a selection or object the failed step should have produced, so
their errors would only be noise. The failure is reported with the scene state for the model that
revises the recipe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucius.errors import ActionRejected, BlenderBridgeError
from lucius.executor.backend import ExecutionBackend
from lucius.executor.safety import ActionValidator
from lucius.ids import new_id
from lucius.lessons.catalogue import normalize_args
from lucius.lessons.recipe import Recipe
from lucius.logging_setup import get_logger
from lucius.planner.model import PlanAction
from lucius.trajectory import vocabulary as vocab

log = get_logger("lessons.runner")


@dataclass
class StepOutcome:
    index: int
    action: str
    ok: bool
    error: str | None = None
    result: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "action": self.action, "ok": self.ok, "error": self.error,
                **({"notes": self.notes} if self.notes else {})}


@dataclass
class RecipeRun:
    ok: bool
    outcomes: list[StepOutcome] = field(default_factory=list)
    failed: StepOutcome | None = None
    scene: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def steps_ok(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    def error_text(self) -> str:
        if self.failed is None:
            return ""
        return f"step {self.failed.index} ({self.failed.action}) failed: {self.failed.error}"

    def scene_text(self) -> str:
        """The scene after the run, one object per line (for prompts)."""
        lines = []
        for obj in self.scene.get("objects", []):
            dims = ", ".join(f"{v:.3g}" for v in obj.get("dimensions", []))
            loc = ", ".join(f"{v:.3g}" for v in obj.get("location", []))
            extra = []
            if obj.get("modifiers"):
                extra.append("modifiers " + "/".join(m["type"] for m in obj["modifiers"]))
            if obj.get("materials"):
                extra.append("materials " + "/".join(obj["materials"]))
            if obj.get("mesh"):
                extra.append(f"{obj['mesh']['verts']} verts")
            lines.append(f"- {obj['name']} ({obj['type']}): size [{dims}] m at [{loc}]"
                         + (f"; {'; '.join(extra)}" if extra else ""))
        return "\n".join(lines) or "(empty scene)"

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "steps_ok": self.steps_ok, "steps": len(self.outcomes), "error": self.error_text() or None,
                "duration_s": round(self.duration_s, 2), "scene": self.scene.get("objects", []),
                "outcomes": [o.to_dict() for o in self.outcomes]}


def record_run(db: Any, *, task_text: str, mode: str, status: str, environment: str, metrics: dict[str, Any],
               arm: str, backend: str = "bridge") -> str:
    """A finished recipe run (lesson attempt, made task or a person's review) in the runs table, so skill
    evidence can point at it."""
    from lucius.storage.db import dumps
    from lucius.timeutil import now

    run_id = new_id("run")
    t = now()
    db.insert("runs", {"id": run_id, "session_id": None, "task_text": task_text[:500], "task_class": None,
                       "mode": mode, "state": status.upper(), "status": status, "backend": backend,
                       "environment": environment, "arm": arm, "params": dumps({}), "metrics": dumps(metrics),
                       "started_at": t, "ended_at": t})
    return run_id


class RecipeRunner:
    def __init__(self, backend: ExecutionBackend, validator_factory: Any) -> None:
        self.backend = backend
        self._validator_factory = validator_factory   # () -> ActionValidator (a fresh action budget per run)

    # -- scene -----------------------------------------------------------------------------------------
    def _bridge(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        outcome = self.backend.execute(PlanAction(id=new_id("step"), layer="blender_api", name=name, args=args,
                                                  action_type=vocab.BRIDGE_ACTION_TYPES.get(name, name),
                                                  description=name))
        if not outcome.ok:
            error = outcome.error or {}
            raise BlenderBridgeError(error.get("message", f"{name} failed"), code=error.get("code", "bridge_error"))
        return outcome.result

    def prepare(self, start_from: Path | None = None) -> None:
        """Start from Blender's default scene without the cube (camera and light kept), or from a saved scene."""
        if start_from is not None:
            self._bridge("import_blend", {"path": str(start_from)})
        else:
            self._bridge("reset_scene", {"keep_camera_light": True})

    def scene(self) -> dict[str, Any]:
        bridge = getattr(self.backend, "bridge", None)
        if bridge is None:
            return {}
        try:
            return bridge.request("scene_summary")
        except BlenderBridgeError as exc:
            log.warning("scene summary unavailable: %s", exc.message)
            return {}

    # -- execution -------------------------------------------------------------------------------------
    def run(self, recipe: Recipe, *, start_from: Path | None = None, prepare: bool = True) -> RecipeRun:
        started = time.monotonic()
        if prepare:
            self.prepare(start_from)
        validator: ActionValidator = self._validator_factory()
        run = RecipeRun(ok=True)
        for index, step in enumerate(recipe.steps):
            args, notes = normalize_args(step.action, step.args)
            action = PlanAction(id=new_id("step"), layer="blender_api", name=step.action, args=args,
                                action_type=vocab.BRIDGE_ACTION_TYPES.get(step.action, step.action),
                                description=step.note, source=f"recipe:{index}")
            outcome = StepOutcome(index=index, action=step.action, ok=False, notes=notes)
            try:
                validator.validate(action)
                result = self.backend.execute(action)
            except ActionRejected as exc:
                outcome.error = f"rejected: {exc.message}"
            else:
                outcome.ok = result.ok
                outcome.result = result.result if result.ok else None
                if not result.ok:
                    error = result.error or {}
                    outcome.error = f"{error.get('code', 'error')}: {error.get('message', '')}"[:400]
            run.outcomes.append(outcome)
            if not outcome.ok:
                run.ok = False
                run.failed = outcome
                break
        run.scene = self.scene()
        run.duration_s = time.monotonic() - started
        return run

    def render(self, path: Path, *, camera: str = "auto", view: str = "three_quarter", frame: list[str] | None = None,
               samples: int = 24, width: int = 800, height: int = 600) -> Path:
        args: dict[str, Any] = {"path": str(path), "camera": camera, "view": view, "samples": samples, "width": width,
                                "height": height}
        if frame:
            args["frame"] = frame
        self._bridge("render_image", args)
        return path

    def save_blend(self, path: Path) -> Path:
        self._bridge("save_file", {"path": str(path), "copy": True})
        return path
