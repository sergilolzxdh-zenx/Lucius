"""Learn a tutorial chapter by chapter: watch it, write it down, rebuild it, compare, practise.

For each chapter:

1. **Watch** -- a video model watches the chapter (by URL, in <=5-minute pieces, with the narration)
   and writes lesson notes: every operation, what was selected, the values read from the screen or
   heard, and what the objects look like at the end. Notes are cached: re-running costs nothing.
2. **Write the recipe** -- the notes become a recipe of Blender actions with concrete values.
3. **Rebuild** -- the recipe runs in headless Blender, continuing from the previous chapter's scene
   (a course builds one project across chapters). A failing step goes back to the model with the
   error and the scene state, and the recipe is corrected (at most ``max_fixes`` times).
4. **Compare** -- the result is rendered and the model compares the renders with the end of the
   chapter in the video: a score from 0 to 10, and the differences as fixes to the recipe.
5. **Practise** -- the recipe is revised from those differences and rebuilt (``practice_rounds``
   times, or until the score reaches ``target_score``); the best attempt is kept.
6. **Keep** -- the best recipe becomes a skill (validated when it rebuilt without errors and was
   judged close to the tutorial), and the chapter's project folder holds every attempt's renders,
   the tutorial frame and a side-by-side sheet for a person to judge.

The comparison is a model's judgement, not a measurement; the project folder is there so a person
can check it (``lucius projects rate``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucius.errors import ProviderError, ProviderUnavailable
from lucius.ingestion.captions import CaptionTrack
from lucius.ingestion.download import DownloadedVideo, parse_timestamp
from lucius.lessons.catalogue import RECIPE_ACTIONS, catalogue
from lucius.lessons.projects import Project, ProjectStore, contact_sheet
from lucius.lessons.recipe import Recipe, apply_patch, parse_recipe, patch_schema, recipe_schema
from lucius.lessons.references import storyboard_frame, thumbnail
from lucius.lessons.runner import RecipeRun, RecipeRunner, record_run
from lucius.logging_setup import get_logger
from lucius.providers.base import ImageInput, VideoInput

if TYPE_CHECKING:
    from lucius.app import Lucius
    from lucius.ingestion.tutorial import TutorialPart

log = get_logger("lessons.learner")

NOTES_SYSTEM = (
    "You watch part of a Blender tutorial and write down exactly how to reproduce what the tutor builds. For every "
    "operation that changes the scene (adding or deleting objects, edit-mode operations, transforms, modifiers, "
    "shading, materials, lights, camera, particles, render settings) write: the time (MM:SS or H:MM:SS as the "
    "player shows it), the object, the operation with Blender's English name (Extrude, Inset Faces, Bevel, Loop "
    "Cut, Scale, Rotate, Bridge Edge Loops, Fill, Delete Faces, Subdivision Surface modifier, Shade Smooth, "
    "Principled BSDF...), what was selected, precisely and geometrically (which faces, edges or vertices, where "
    "on the object: 'the top face', 'the outer edge loop of the rim', '2x2 faces on the +X side between 60% and "
    "80% of the height'), the values (read them from the screen -- the header while transforming, the operator "
    "panel at the bottom left, the N panel, modifier/material/light panels -- else from the narration, else "
    "estimate them from the picture and say so) and the effect, with sizes in metres (the default cube is 2 m). "
    "Skip interface explanations, navigation (orbit, zoom, pan, views), operations that are undone (Ctrl+Z) and "
    "anything only talked about. The narration is automatic captions, possibly in another language, with "
    "recognition errors; write in English. objects_at_end describes every object in the scene at the end of the "
    "clip (shape, size in metres, position, material)."
)

RECIPE_RULES = (
    "Conventions: metres; Z is up; angles in degrees (rotation_deg, angle_deg); colours \"#RRGGBB\". Selections: "
    "use select_box with space \"local\" (object coordinates in metres, unaffected by moving the object) and "
    "bounds that contain only the intended elements -- compute them from the sizes you gave the geometry earlier "
    "(e.g. a cylinder of depth 0.1 at the origin spans z -0.05..0.05 locally until its top is extruded), and "
    "make the selection explicitly right before every bevel, inset, extrude, delete_elements, bridge_edge_loops "
    "or fill that does not simply continue the region the previous extrude left selected. Edit "
    "operations (extrude, inset, bevel, scale/rotate/translate_selection, delete_elements, bridge_edge_loops, fill) "
    "act on the current selection and enter edit mode themselves; after extrude the new region stays selected, so "
    "a chain like extrude, rotate_selection, extrude works like E, R, E in Blender. Alt+click on an edge loop is "
    "select_box element EDGE with a thin box around the loop (sharp_deg or boundary help). Ctrl+1/2/3 on an "
    "object is add_modifier SUBSURF with levels 1/2/3. Shade auto smooth is shade smooth. Sculpting, proportional "
    "editing, curves, texture images and node setups are not available: approximate them with the actions you "
    "have (e.g. a DISPLACE modifier with a CLOUDS texture for hand-deformed or sculpted irregularity, a SMOOTH "
    "modifier, a flat colour) and say so in the step's note. Use the object names "
    "the tutor uses, in English (Mug, Plate, Donut...). Every step needs \"object\" (except add_primitive, "
    "add_light, add_camera, set_world, delete_objects)."
)

RECIPE_SYSTEM = (
    "You turn notes taken while watching a Blender tutorial chapter into a recipe that Lucius executes in Blender "
    "with the actions listed. Reproduce what the tutor did, in the same order, with the values from the notes "
    "(estimate realistic ones where the notes do not give them). Do not add work the tutor did not do, except "
    "what is needed to express an operation with these actions. video_time is where the tutor does the step."
)

PATCH_RULES = (
    "Answer with edits to the recipe, by step index as numbered above: replace a step, insert a new step before "
    "one (insert_before with index = number of steps appends at the end), or delete one. Indices always refer "
    "to the recipe as shown, not after earlier edits. Change only what the problem needs; every step you do not "
    "edit is kept."
)

FIX_SYSTEM = (
    "A recipe for Blender failed while executing. Correct it so that it runs and still builds the same thing. "
    "Change only what is needed (usually the failing step or the steps that prepared its selection): read the "
    "error, the local bounds of the object and what the previous steps selected, and fix the coordinates or "
    "arguments."
)

COMPARE_SYSTEM = (
    "The video clip is the end of a chapter of a Blender tutorial; the images after it are renders of what Lucius "
    "built by following its recipe for that chapter. Judge how closely Lucius reproduced what the tutor built or "
    "changed in this chapter: shapes, proportions, details (bevels, holes, handles, thickness, lips), smoothness, "
    "and -- when the chapter is about them -- materials, lights, camera and render. Ignore differences of viewing "
    "(viewport vs render, background, camera angle, lighting) unless the chapter is about them. score 0-10: 10 "
    "the same object; 7 clearly the same with small differences; 4 recognisable but wrong in important ways; 1 "
    "unrelated. List the differences with a fix stated in terms of the recipe (which step, which value). "
    "reference_time is the MM:SS (player time) in the clip where the finished result is shown best."
)

REVISE_SYSTEM = (
    "Improve a Blender recipe so that its result looks more like the tutor's. Apply the listed fixes (and any "
    "other change the differences call for) and keep what already matches. The recipe must still run: keep "
    "selections consistent with the geometry each step creates."
)


def notes_schema() -> dict[str, Any]:
    nullable = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "operations": {"type": "array", "items": {
                "type": "object",
                "properties": {"time": {"type": "string"}, "object": nullable, "operation": {"type": "string"},
                               "selection": nullable, "values": nullable,
                               "value_source": {"type": "string", "enum": ["screen", "narration", "estimated", "none"]},
                               "effect": {"type": "string"}},
                "required": ["time", "object", "operation", "selection", "values", "value_source", "effect"],
                "additionalProperties": False}},
            "objects_at_end": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "shape": {"type": "string"}, "size": {"type": "string"},
                               "location": {"type": "string"}, "material": nullable},
                "required": ["name", "shape", "size", "location", "material"], "additionalProperties": False}},
            "summary": {"type": "string"},
        },
        "required": ["operations", "objects_at_end", "summary"], "additionalProperties": False,
    }


def compare_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "verdict": {"type": "string", "enum": ["good", "close", "poor"]},
            "matches": {"type": "array", "items": {"type": "string"}},
            "differences": {"type": "array", "items": {
                "type": "object",
                "properties": {"object": {"type": "string"}, "problem": {"type": "string"}, "fix": {"type": "string"}},
                "required": ["object", "problem", "fix"], "additionalProperties": False}},
            "reference_time": {"type": "string"},
        },
        "required": ["score", "verdict", "matches", "differences", "reference_time"], "additionalProperties": False,
    }


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _quota(exc: ProviderError) -> bool:
    return isinstance(exc, ProviderUnavailable) or "quota" in exc.message


class QuotaExhausted(Exception):
    """The model cannot be called any more today; the lesson resumes where it stopped when run again."""


class ModelUnavailable(QuotaExhausted):
    """The models are overloaded right now; the lesson resumes where it stopped when run again."""


@dataclass
class Attempt:
    number: int
    recipe: Recipe
    run: RecipeRun
    renders: list[Path] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    fixes: int = 0

    @property
    def score(self) -> float:
        if not self.run.ok:
            return -1.0 + self.run.steps_ok / max(1, len(self.recipe.steps))
        return float(self.comparison.get("score") or 0.0)


@dataclass
class ChapterResult:
    chapter: str
    project_id: str
    status: str                       # learned, partial, failed, quota
    score: float | None = None
    skill_id: str | None = None
    sheet: Path | None = None
    final_render: Path | None = None
    blend: Path | None = None
    attempts: int = 0
    notes: list[str] = field(default_factory=list)
    out_of_quota: bool = False        # the model ran out while practising: the lesson stops after this chapter

    def to_dict(self) -> dict[str, Any]:
        return {"chapter": self.chapter, "project_id": self.project_id, "status": self.status, "score": self.score,
                "skill_id": self.skill_id, "sheet": str(self.sheet) if self.sheet else None,
                "final_render": str(self.final_render) if self.final_render else None, "attempts": self.attempts,
                "notes": self.notes}


class LessonLearner:
    def __init__(self, app: Lucius, *, practice_rounds: int = 2, max_fixes: int = 3, target_score: float = 8.0,
                 pass_score: float = 6.0, notes_fps: float = 1.0, notes_resolution: str = "high",
                 compare_window_s: float = 75.0, render_samples: int = 24) -> None:
        self.app = app
        self.practice_rounds = practice_rounds
        self.max_fixes = max_fixes
        self.target_score = target_score
        self.pass_score = pass_score
        self.notes_fps = notes_fps
        self.notes_resolution = notes_resolution
        self.compare_window_s = compare_window_s
        self.render_samples = render_samples
        self.projects = ProjectStore(app.config.projects_dir)
        self.chunk_s = app.config.processing.video_model_chunk_s
        self.max_skips = 3               # steps an attempt may leave out when corrections cannot make them work
        self.rewrite = False             # True: write a new recipe even when an earlier one was built and judged
        self.chunk_attempts = 3          # per piece of video, when every model is overloaded
        self.retry_wait_s = 90.0

    # -- state (resume) --------------------------------------------------------------------------------
    def _lesson_dir(self, video: DownloadedVideo) -> Path:
        path = self.app.config.data_dir / "lessons" / video.video_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state(self, video: DownloadedVideo) -> dict[str, Any]:
        path = self._lesson_dir(video) / "state.json"
        return json.loads(path.read_text()) if path.exists() else {"chapters": {}}

    def _save_state(self, video: DownloadedVideo, state: dict[str, Any]) -> None:
        (self._lesson_dir(video) / "state.json").write_text(json.dumps(state, indent=1, ensure_ascii=False))

    # -- model calls -----------------------------------------------------------------------------------
    def _call(self, purpose: str, *, system: str, prompt: str, schema: dict[str, Any],
              images: list[ImageInput] | None = None, videos: list[VideoInput] | None = None,
              max_tokens: int = 24000) -> dict[str, Any]:
        provider = self.app.providers.vlm if (images or videos) else self.app.providers.llm
        try:
            result = provider.complete_json(purpose=purpose, system=system, prompt=prompt, schema=schema,
                                            images=images or [], max_tokens=max_tokens,
                                            **({"videos": videos} if videos else {}))
        except ProviderError as exc:
            if _quota(exc):
                raise QuotaExhausted(exc.message) from exc
            raise
        data = dict(result.data)
        data["_model"] = result.model
        return data

    # -- 1. watch ----------------------------------------------------------------------------------------
    def watch_notes(self, video: DownloadedVideo, part: TutorialPart, *, context: str = "") -> dict[str, Any]:
        cache = self._lesson_dir(video) / f"notes_{int(part.start)}_{int(part.end)}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        narration = None
        if video.captions_path is not None:
            try:
                narration = CaptionTrack.from_file(video.captions_path, language=video.language)
            except Exception as exc:  # captions are optional context
                log.warning("captions unreadable: %s", exc)
        chunks = self.app.ingestion.watcher.chunks(part.start, part.end)
        notes: dict[str, Any] = {"operations": [], "objects_at_end": [], "summaries": [], "models": [], "errors": []}
        objects_so_far = context
        for a, b in chunks:
            lines = [f"Chapter: {part.task_text}", f"Clip: {_clock(a)} to {_clock(b)} of the video."]
            if objects_so_far:
                lines.append(f"Scene before this clip:\n{objects_so_far}")
            if narration is not None:
                spoken = "\n".join(f"[{_clock(c.start)}] {c.text}" for c in narration.window(a, b).cues)[:9000]
                if spoken:
                    lines += [f"Narration ({narration.language or 'unknown language'}):", spoken]
            lines.append("Write the lesson notes for this clip.")
            piece = cache.with_name(f"{cache.stem}.piece_{int(a)}_{int(b)}.json")
            data = json.loads(piece.read_text()) if piece.exists() else None
            for attempt in range(0 if data is not None else self.chunk_attempts):
                try:
                    data = self._call("lesson_notes", system=NOTES_SYSTEM, prompt="\n".join(lines),
                                      schema=notes_schema(),
                                      videos=[VideoInput(video.url, a, b, self.notes_fps, self.notes_resolution)])
                    break
                except ProviderError as exc:
                    notes["errors"].append(f"{_clock(a)}-{_clock(b)}: {exc.message[:200]}")
                    log.warning("lesson notes for %s-%s failed (%d): %s", _clock(a), _clock(b), attempt + 1,
                                exc.message[:200])
                    if attempt + 1 < self.chunk_attempts:
                        time.sleep(self.retry_wait_s)
            if data is None:
                # A missing piece would leave a hole in the recipe: stop, and resume here later.
                raise ModelUnavailable(f"no model could watch {_clock(a)}-{_clock(b)}: {notes['errors'][-1]}")
            piece.write_text(json.dumps(data, indent=1, ensure_ascii=False))   # a restart resumes after this piece
            notes["operations"] += data.get("operations", [])
            notes["objects_at_end"] = data.get("objects_at_end", []) or notes["objects_at_end"]
            notes["summaries"].append(f"[{_clock(a)}] {data.get('summary', '')}")
            notes["models"].append(data.get("_model"))
            objects_so_far = _objects_text(notes["objects_at_end"])
        if notes["operations"] or notes["objects_at_end"]:
            cache.write_text(json.dumps(notes, indent=1, ensure_ascii=False))
        return notes

    # -- 2. recipe ---------------------------------------------------------------------------------------
    def _recipe_call(self, purpose: str, system: str, prompt: str, source: dict[str, Any]) -> tuple[Recipe, list[str]]:
        data = self._call(purpose, system=system + "\n\n" + RECIPE_RULES, prompt=prompt,
                          schema=recipe_schema(RECIPE_ACTIONS), max_tokens=32000)
        recipe, problems = parse_recipe(data, source={**source, "model": data.get("_model")})
        return recipe, problems

    def write_recipe(self, part: TutorialPart, notes: dict[str, Any], scene_before: str,
                     source: dict[str, Any]) -> tuple[Recipe, list[str]]:
        prompt = "\n\n".join([
            f"Chapter: {part.task_text}",
            f"Actions:\n{catalogue()}",
            f"Scene at the start of the chapter:\n{scene_before}",
            "Notes (in order):\n" + _notes_text(notes),
            "Objects at the end of the chapter:\n" + _objects_text(notes.get("objects_at_end", [])),
            "Write the recipe for this chapter."])
        return self._recipe_call("lesson_recipe", RECIPE_SYSTEM, prompt, source)

    def _patch_call(self, purpose: str, system: str, prompt: str, recipe: Recipe) -> tuple[Recipe, list[str]]:
        data = self._call(purpose, system=system + "\n\n" + PATCH_RULES + "\n\n" + RECIPE_RULES, prompt=prompt,
                          schema=patch_schema(RECIPE_ACTIONS), max_tokens=16000)
        return apply_patch(recipe, data, allowed=RECIPE_ACTIONS)

    def fix_recipe(self, recipe: Recipe, run: RecipeRun, problems: list[str]) -> tuple[Recipe, list[str]]:
        prompt = "\n\n".join([
            f"Actions:\n{catalogue()}",
            f"Recipe ({recipe.title}):\n{recipe.compact()}",
            f"Failure: {run.error_text()}" + (f"\nOther problems: {'; '.join(problems)}" if problems else ""),
            f"The steps before it did:\n{run.context_text()}",
            f"Scene when it failed (object sizes, and local bounds -- the coordinates select_box space \"local\" "
            f"uses):\n{run.scene_text()}",
            "Return the edits that correct the recipe."])
        return self._patch_call("lesson_recipe_fix", FIX_SYSTEM, prompt, recipe)

    def revise_recipe(self, recipe: Recipe, attempt: Attempt, part: TutorialPart) -> tuple[Recipe, list[str]]:
        diffs = "\n".join(f"- {d.get('object')}: {d.get('problem')} -> fix: {d.get('fix')}"
                          for d in attempt.comparison.get("differences", []))
        prompt = "\n\n".join([
            f"Chapter: {part.task_text}",
            f"Actions:\n{catalogue()}",
            f"Recipe ({recipe.title}):\n{recipe.compact()}",
            f"Scene it built:\n{attempt.run.scene_text()}",
            f"Judged {attempt.comparison.get('score')}/10 against the tutorial. Differences:\n{diffs or '(none listed)'}",
            "Return the edits that fix these differences."])
        return self._patch_call("lesson_recipe_revise", REVISE_SYSTEM, prompt, recipe)

    # -- 4. compare ----------------------------------------------------------------------------------------
    def compare(self, video: DownloadedVideo, part: TutorialPart, recipe: Recipe, renders: list[Path]) -> dict[str, Any]:
        start = max(part.start, part.end - self.compare_window_s)
        images = [ImageInput(path.read_bytes(), "image/png", label=f"Lucius render {i + 1} ({path.stem})")
                  for i, path in enumerate(renders)]
        prompt = "\n".join([
            f"Chapter: {part.task_text}",
            f"The clip runs from {_clock(start)} to {_clock(part.end)}.",
            f"What the recipe was meant to build: {recipe.expected_result or recipe.summary}",
            "Compare the renders with the tutor's result."])
        try:
            data = self._call("lesson_compare", system=COMPARE_SYSTEM, prompt=prompt, schema=compare_schema(),
                              images=images, videos=[VideoInput(video.url, start, part.end, 1.0, "high")],
                              max_tokens=8000)
        except ProviderError as exc:
            return {"score": None, "verdict": "unknown", "matches": [], "differences": [],
                    "error": exc.message[:300]}
        data["score"] = max(0.0, min(10.0, float(data.get("score") or 0.0)))
        return data

    # -- rendering ------------------------------------------------------------------------------------------
    def _render(self, runner: RecipeRunner, project: Project, recipe: Recipe, label: str,
                scene_camera: bool) -> list[Path]:
        frame = [n for n in recipe.object_names() if n in {o["name"] for o in runner.scene().get("objects", [])
                                                         if o.get("type") == "MESH"}]
        renders = []
        views = [("scene", "three_quarter")] if scene_camera else []
        views += [("auto", "three_quarter"), ("auto", "front")]
        for camera, view in views[:2]:
            name = f"{label}_{'camera' if camera == 'scene' else view}.png"
            try:
                renders.append(runner.render(project.path(name), camera=camera, view=view, frame=frame or None,
                                             samples=self.render_samples))
            except Exception as exc:
                log.warning("render %s failed: %s", name, exc)
        return renders

    # -- the chapter -----------------------------------------------------------------------------------------
    def learn_part(self, video: DownloadedVideo, part: TutorialPart, runner: RecipeRunner, *,
                   start_from: Path | None, scene_before: str, uses_camera: bool, index: int,
                   info: dict[str, Any] | None, on_progress: Callable[[str], None] | None = None) -> ChapterResult:
        say = on_progress or (lambda message: log.info(message))
        source = {"kind": "tutorial", "video": video.url, "video_id": video.video_id, "video_title": video.title,
                  "chapter": part.chapter.title if part.chapter else part.title, "chapter_index": index,
                  "start": part.start, "end": part.end}
        project = self.projects.create("lesson", f"{video.title[:50]} — {part.task_text}", source=source,
                                       start_from=str(start_from) if start_from else None)
        result = ChapterResult(chapter=part.task_text, project_id=project.id, status="failed")
        say(f"[{part.task_text}] watching {_clock(part.start)}-{_clock(part.end)}")
        notes = self.watch_notes(video, part, context=scene_before)
        (project.path("notes.json")).write_text(json.dumps(notes, indent=1, ensure_ascii=False))
        if not notes.get("operations"):
            result.notes.append("the model saw no operations in this chapter" +
                                (f" ({'; '.join(notes.get('errors', [])[:2])})" if notes.get("errors") else ""))
            project.data.update(status="nothing_to_learn", notes=result.notes)
            project.save()
            result.status = "nothing_to_learn"
            return result
        previous = None if self.rewrite else self._previous_best(video, part)
        if previous is not None:
            # Learning accumulates: practise from the best recipe so far rather than writing a new one (a new
            # recipe from the same notes is a fresh gamble, and often worse).
            recipe, problems = previous[0], []
            project.data["continued_from"] = {"project": previous[2], "score": previous[1]}
            say(f"[{part.task_text}] continuing from the best earlier recipe ({previous[1]}/10, "
                f"{len(recipe.steps)} steps)")
        else:
            say(f"[{part.task_text}] {len(notes['operations'])} operations noted; writing the recipe")
            try:
                recipe, problems = self.write_recipe(part, notes, scene_before, source)
            except ProviderError as exc:
                result.notes.append(f"the recipe could not be written: {exc.message[:200]}")
                project.data.update(status="failed", notes=result.notes)
                project.save()
                return result
        attempts: list[Attempt] = []
        out_of_quota: QuotaExhausted | None = None
        try:
            for number in range(1, self.practice_rounds + 2):
                run = runner.run(recipe, start_from=start_from, default_cube=start_from is None)
                fixes = 0
                while not run.ok and fixes < self.max_fixes:
                    fixes += 1
                    say(f"[{part.task_text}] attempt {number}: {run.error_text()} -> correcting the recipe ({fixes})")
                    try:
                        recipe, problems = self.fix_recipe(recipe, run, problems)
                    except ProviderError as exc:
                        result.notes.append(f"correction failed: {exc.message[:160]}")
                        break
                    run = runner.run(recipe, start_from=start_from, default_cube=start_from is None)
                skipped: list[str] = []
                while not run.ok and run.failed is not None and len(skipped) < self.max_skips:
                    # Corrections did not make this step work: build the rest without it, and say so.
                    step = recipe.steps[run.failed.index]
                    skipped.append(f"step {run.failed.index} {step.action} {step.args} ({run.failed.error})")
                    say(f"[{part.task_text}] attempt {number}: skipping step {run.failed.index} ({step.action})")
                    recipe = recipe.without_step(run.failed.index)
                    run = runner.run(recipe, start_from=start_from, default_cube=start_from is None)
                if skipped:
                    result.notes.append(f"attempt {number} skipped {len(skipped)} step(s) it could not make work")
                attempt = Attempt(number=number, recipe=recipe, run=run, fixes=fixes)
                attempt.renders = self._render(runner, project, recipe, f"attempt{number}",
                                               uses_camera or "add_camera" in recipe.actions_used())
                if run.ok and attempt.renders:
                    attempt.comparison = self.compare(video, part, recipe, attempt.renders)
                attempts.append(attempt)
                (project.path(f"attempt{number}_recipe.json")).write_text(recipe.model_dump_json(indent=1))
                project.add_attempt({"number": number, "score": attempt.comparison.get("score"), "run": run.to_dict(),
                                     "fixes": fixes, "skipped_steps": skipped, "renders": [p.name for p in attempt.renders],
                                     "comparison": attempt.comparison, "problems": problems})
                say(f"[{part.task_text}] attempt {number}: {'built' if run.ok else 'failed'} "
                    f"({run.steps_ok}/{len(recipe.steps)} steps), score {attempt.comparison.get('score')}")
                if not run.ok or attempt.score >= self.target_score or number > self.practice_rounds:
                    break
                if not attempt.comparison.get("differences"):
                    break  # nothing to practise on (the comparison failed or found no difference)
                try:
                    recipe, problems = self.revise_recipe(recipe, attempt, part)
                except ProviderError as exc:
                    result.notes.append(f"revision failed: {exc.message[:160]}")
                    break
        except QuotaExhausted as exc:
            # Keep what was built and judged so far; the lesson stops after saving this chapter.
            out_of_quota = exc
            result.notes.append(f"stopped practising: {exc}")
        if not attempts:
            raise out_of_quota or QuotaExhausted("no attempt could be made")
        best = max(attempts, key=lambda a: a.score)
        # Rebuild the best attempt so the saved scene (the next chapter's start) is exactly that recipe's result.
        final_run = best.run if best is attempts[-1] else runner.run(best.recipe, start_from=start_from,
                                                                     default_cube=start_from is None)
        result.attempts = len(attempts)
        result.score = best.comparison.get("score")
        (project.path("recipe.json")).write_text(best.recipe.model_dump_json(indent=1))
        if final_run.ok:
            result.blend = runner.save_blend(project.path("scene.blend"))
            result.final_render = best.renders[0] if best.renders else None
        reference = self._reference_frame(video, part, best, project, info)
        tiles = [(reference, f"Tutorial at {best.comparison.get('reference_time') or _clock(part.end)}")]
        tiles += [(a.renders[0] if a.renders else None,
                   f"Lucius attempt {a.number}: " + (f"{a.comparison.get('score')}/10" if a.run.ok else "failed"))
                  for a in attempts]
        if best.renders[1:]:
            tiles.append((best.renders[1], "best attempt, front"))
        result.sheet = contact_sheet(tiles, project.path("sheet.png"), title=part.task_text)
        passed = final_run.ok and (result.score or 0.0) >= self.pass_score
        result.status = "learned" if passed else ("partial" if final_run.ok else "failed")
        result.skill_id = self.store_skill(best.recipe, part, video, final_run.ok, result.score, index, project)
        project.data.update(status=result.status, score=result.score, skill_id=result.skill_id,
                            sheet="sheet.png", final_render=result.final_render.name if result.final_render else None,
                            blend="scene.blend" if result.blend else None, best_attempt=best.number,
                            recipe_steps=len(best.recipe.steps), actions=sorted(best.recipe.actions_used()))
        project.save()
        result.out_of_quota = out_of_quota is not None
        return result

    def _previous_best(self, video: DownloadedVideo, part: TutorialPart) -> tuple[Recipe, float, str] | None:
        """The best recipe an earlier run of this chapter built and had judged."""
        best: tuple[Recipe, float, str] | None = None
        for summary in self.projects.list(kind="lesson", limit=1000):
            project = self.projects.get(summary["id"])
            src = project.data.get("source") or {}
            if (src.get("video_id") != video.video_id or int(src.get("start", -1)) != int(part.start)
                    or int(src.get("end", -1)) != int(part.end)):
                continue
            # Every attempt that built and was judged counts, also from a run that was interrupted.
            for attempt in project.data.get("attempts", []):
                score = attempt.get("score")
                path = project.path(f"attempt{attempt.get('number')}_recipe.json")
                if not (attempt.get("run") or {}).get("ok") or score is None or not path.exists():
                    continue
                if best is None or float(score) > best[1]:
                    best = (Recipe.model_validate_json(path.read_text()), float(score), project.id)
        return best

    def _reference_frame(self, video: DownloadedVideo, part: TutorialPart, best: Attempt, project: Project,
                         info: dict[str, Any] | None) -> Path | None:
        if info is None:
            return None
        t = part.end - 15.0
        raw = best.comparison.get("reference_time")
        if raw:
            try:
                t = min(part.end, max(part.start, parse_timestamp(str(raw))))
            except ValueError:
                pass
        cache = self._lesson_dir(video) / "storyboard"
        return storyboard_frame(info, t, cache, project.path("tutorial_frame.png"))

    # -- 6. skill ---------------------------------------------------------------------------------------------
    def store_skill(self, recipe: Recipe, part: TutorialPart, video: DownloadedVideo, built: bool,
                    score: float | None, index: int, project: Project) -> str | None:
        from lucius.ids import new_id
        from lucius.skills.schema import ActionTemplate, SkillDefinition, SkillExample, SkillPhase
        from lucius.timeutil import now

        if not recipe.steps:
            return None
        library = self.app.library
        skill_id = f"lesson_{video.video_id}_{index + 1:02d}".replace("-", "_").lower()
        names = [o.get("name", "") for o in recipe.objects if o.get("name")]
        definition = SkillDefinition(
            skill_id=skill_id, name=recipe.title, purpose=(recipe.summary + " " + recipe.expected_result).strip(),
            categories=["tutorial_recipe"], object_class=(names[0].lower() if len(names) == 1 else None),
            triggers=sorted({n.lower() for n in names} | {w.lower() for w in part.task_text.split() if len(w) > 3}),
            applicable_contexts=["recipe"],
            prerequisites=[f"lesson_{video.video_id}_{index:02d}".replace("-", "_").lower()] if index > 0 else [],
            phases=[SkillPhase(name="primary_form", description=recipe.summary, actions=[
                ActionTemplate(action_type=s.action, description=s.note, args=s.args, object_ref=None,
                               evidence=[f"{video.url}&t={s.video_time}"] if s.video_time else [])
                for s in recipe.steps])],
            source_class="external_video",
            notes=[f"recipe from {video.url} chapter '{part.task_text}' ({_clock(part.start)}-{_clock(part.end)})",
                   f"project {project.id}"])
        if library.exists(skill_id):
            library.new_version(skill_id, definition, change_note="relearned from the tutorial", created_by="lesson")
        else:
            library.create(definition, created_by="lesson", change_note="learned from a tutorial chapter")
        skill = library.get(skill_id)
        library.add_example(SkillExample(
            id=new_id("example"), skill_id=skill_id, skill_version=skill.current_version, role="demonstration",
            source_class="external_video", t_start=part.start, t_end=part.end, evidence_weight=0.6,
            instance_signature=f"{video.video_id}:{int(part.start)}",
            summary={"video": video.url, "chapter": part.task_text, "project": project.id}, created_at=now()))
        success = built and (score or 0.0) >= self.pass_score
        run_id = record_run(self.app.db, task_text=f"lesson: {part.task_text}", mode="validation",
                            status="success" if success else "failure", environment="blender_headless",
                            metrics={"visual_score": score, "built": built, "project": project.id}, arm="lesson")
        library.record_use(skill_id, success=success, run_id=run_id,
                           instance_signature=f"{video.video_id}:{int(part.start)}", objective=built,
                           environment="blender_headless", role="validation", source_class="external_video",
                           detail={"visual_score": score, "judge": "model comparison with the tutorial clip",
                                   "project": project.id})
        try:
            self.app.retriever.refresh()
        except Exception as exc:  # indexing is best effort; the skill exists either way
            log.warning("skill index refresh failed: %s", exc)
        return skill_id

    # -- the whole lesson ------------------------------------------------------------------------------------
    def learn(self, video: DownloadedVideo, parts: list[TutorialPart], *, redo: bool = False,
              on_progress: Callable[[str], None] | None = None) -> list[ChapterResult]:
        info = json.loads(Path(video.info_path).read_text()) if video.info_path and Path(video.info_path).suffix == ".json" \
            else None
        state = self._state(video)
        results: list[ChapterResult] = []
        backend = self.app.headless_backend()
        runner = self.runner(backend)
        start_from: Path | None = None
        scene_before = ("Blender's default scene: Cube (mesh, 2 x 2 x 2 m at the origin), Camera (at 7.36, -6.93, "
                        "4.96 looking at the origin) and Light (point, at 4.08, 1.0, 5.9).")
        uses_camera = False
        for index, part in enumerate(parts):
            if part.skipped:
                continue
            key = f"{int(part.start)}-{int(part.end)}"
            done = state["chapters"].get(key)
            if done and not redo and done.get("status") in ("learned", "partial", "nothing_to_learn"):
                if done.get("blend") and Path(done["blend"]).exists():
                    start_from = Path(done["blend"])
                scene_before = done.get("scene_after") or scene_before
                uses_camera = uses_camera or bool(done.get("uses_camera"))
                continue
            chapter_index = part.chapter.index if part.chapter else index
            try:
                result = self.learn_part(video, part, runner, start_from=start_from, scene_before=scene_before,
                                         uses_camera=uses_camera, index=chapter_index, info=info,
                                         on_progress=on_progress)
            except QuotaExhausted as exc:
                unavailable = isinstance(exc, ModelUnavailable)
                results.append(ChapterResult(
                    chapter=part.task_text, project_id="", status="unavailable" if unavailable else "quota",
                    notes=[f"{'models overloaded' if unavailable else 'model quota exhausted'}: {exc}; "
                           "run the same command again later to continue from this chapter"]))
                break
            results.append(result)
            recipe_path = self.projects.get(result.project_id).path("recipe.json") if result.project_id else None
            if result.blend is not None:
                start_from = result.blend
                scene_before = RecipeRun(ok=True, scene=runner.scene()).scene_text()
            if recipe_path is not None and recipe_path.exists():
                uses_camera = uses_camera or "add_camera" in Recipe.model_validate_json(recipe_path.read_text()).actions_used()
            state["chapters"][key] = {**result.to_dict(), "blend": str(result.blend) if result.blend else None,
                                      "scene_after": scene_before, "uses_camera": uses_camera}
            self._save_state(video, state)
            if result.out_of_quota:
                results.append(ChapterResult(chapter="(next chapters)", project_id="", status="quota",
                                             notes=["the model's quota ran out; run the same command again later "
                                                    "to continue (and to practise this chapter more, with --redo)"]))
                break
        if info is not None:
            thumbnail(info, self._lesson_dir(video))
        self.overview(video)
        return results

    def overview(self, video: DownloadedVideo) -> Path | None:
        """One picture of the whole lesson: per chapter, the tutorial's frame next to Lucius' best result."""
        best: dict[tuple[int, int], dict[str, Any]] = {}
        for summary in self.projects.list(kind="lesson", limit=1000):
            data = self.projects.get(summary["id"]).data
            src = data.get("source") or {}
            if src.get("video_id") != video.video_id or data.get("status") not in ("learned", "partial"):
                continue
            key = (int(src.get("start", 0)), int(src.get("end", 0)))
            if key not in best or (data.get("score") or 0) > (best[key].get("score") or 0):
                best[key] = data
        if not best:
            return None
        tiles: list[tuple[Path | None, str]] = []
        for key in sorted(best):
            data = best[key]
            folder = self.projects.get(data["id"]).dir
            chapter = (data.get("title") or "").split(" — ")[-1]
            frame = folder / "tutorial_frame.png"
            tiles.append((frame if frame.exists() else None, f"Tutorial: {chapter}"))
            render = folder / (data.get("final_render") or "")
            tiles.append((render if data.get("final_render") and render.exists() else None,
                          f"Lucius: {data.get('score')}/10 ({data.get('status')})"))
        return contact_sheet(tiles, self._lesson_dir(video) / "overview.png", tile_w=400, tile_h=300,
                             title=f"{video.title[:70]} - what Lucius rebuilt", columns=2)

    def runner(self, backend: Any) -> RecipeRunner:
        from lucius.executor.safety import ActionValidator

        cfg = self.app.config
        return RecipeRunner(backend, lambda: ActionValidator(
            cfg.safety, bridge_actions=backend.bridge_actions, gui_only=backend.gui_only,
            background=backend.background, output_dirs=[str(cfg.exports_dir), str(cfg.projects_dir)]))


def _notes_text(notes: dict[str, Any], limit: int = 60000) -> str:
    lines = []
    for op in notes.get("operations", []):
        parts = [f"[{op.get('time')}]", str(op.get("operation"))]
        if op.get("object"):
            parts.append(f"on {op['object']}")
        if op.get("selection"):
            parts.append(f"| selected: {op['selection']}")
        if op.get("values"):
            parts.append(f"| values: {op['values']} ({op.get('value_source')})")
        if op.get("effect"):
            parts.append(f"| effect: {op['effect']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)[:limit]


def _objects_text(objects: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {o.get('name')}: {o.get('shape')}, size {o.get('size')}, at {o.get('location')}"
                     + (f", material {o.get('material')}" if o.get("material") else "") for o in objects) or "(none)"
