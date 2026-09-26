"""Teaching Lucius without a model API: a teacher writes the recipe, Lucius builds and keeps it.

The teacher can be a person or an assistant working in the same checkout (for example Claude Code
reading the tutorial's narration and looking at its preview frames). Lucius does what it does in a
lesson -- builds the recipe in headless Blender, continuing from the previous chapter's scene,
renders it, puts the tutorial's frame (or the reference images) next to the renders, saves a project
-- and when the teacher gives a score, keeps the recipe as the chapter's skill (or as a made thing),
with the teacher recorded as the judge. A failing step is reported with the scene and what the steps
before it selected, for the teacher to correct.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucius.errors import NotFoundError, ValidationError
from lucius.ingestion.download import parse_timestamp
from lucius.lessons.projects import ProjectStore, contact_sheet
from lucius.lessons.recipe import Recipe, RecipeStep
from lucius.lessons.references import storyboard_frame
from lucius.lessons.runner import RecipeRun

if TYPE_CHECKING:
    from lucius.app import Lucius
    from lucius.ingestion.download import DownloadedVideo
    from lucius.ingestion.tutorial import TutorialPart

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
COURSE_INFO_KEYS = ("title", "duration", "chapters", "channel", "language", "license")


def load_recipe(path: str | Path) -> Recipe:
    """A recipe file: {"title", "summary", "objects", "steps": [{"action", "args": {...}, "note", "video_time"}],
    "expected_result"}; ``args`` may also be a JSON string."""
    data = json.loads(Path(path).read_text())
    steps = []
    for i, item in enumerate(data.get("steps") or []):
        args = item.get("args") or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        if not isinstance(args, dict) or not item.get("action"):
            raise ValidationError(f"step {i} needs an action and an args object")
        steps.append(RecipeStep(action=str(item["action"]), args=args, note=str(item.get("note") or ""),
                                video_time=item.get("video_time")))
    return Recipe(title=str(data.get("title") or Path(path).stem), summary=str(data.get("summary") or ""),
                  objects=list(data.get("objects") or []), steps=steps,
                  expected_result=str(data.get("expected_result") or ""), source=dict(data.get("source") or {}))


@dataclass
class TeachResult:
    project_id: str
    ok: bool
    error: str | None = None
    context: str = ""
    scene: str = ""
    renders: list[Path] = field(default_factory=list)
    sheet: Path | None = None
    skill_id: str | None = None
    status: str = "trial"

    def to_dict(self) -> dict[str, Any]:
        return {"project_id": self.project_id, "ok": self.ok, "status": self.status, "error": self.error,
                "steps_before_failure": self.context or None, "scene": self.scene,
                "renders": [str(p) for p in self.renders], "sheet": str(self.sheet) if self.sheet else None,
                "skill_id": self.skill_id}


class Teacher:
    def __init__(self, app: Lucius, *, name: str = "teacher", render_samples: int = 32) -> None:
        from lucius.lessons.learner import LessonLearner

        self.app = app
        self.name = name
        self.render_samples = render_samples
        self.learner = LessonLearner(app, render_samples=render_samples)
        self.projects = ProjectStore(app.config.projects_dir)

    # -- tutorial chapters -----------------------------------------------------------------------------
    def video(self, video_id: str) -> DownloadedVideo:
        from lucius.ingestion.tutorial import TutorialImporter

        info = self.app.config.data_dir / "downloads" / f"{video_id}.info.json"
        if not info.exists():
            raise NotFoundError(f"no saved metadata for {video_id} (run `lucius tutorial <url> --plan-only` first)")
        meta = json.loads(info.read_text())
        return TutorialImporter.from_info(meta.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
                                          info)

    def part(self, video: DownloadedVideo, chapter: int) -> TutorialPart:
        from lucius.ingestion.tutorial import TutorialPart

        match = next((c for c in video.chapters if c.index + 1 == chapter), None)
        if match is None:
            raise NotFoundError(f"chapter {chapter} not found ({len(video.chapters)} chapters)")
        return TutorialPart(title=f"{video.title} — {chapter}. {match.title}", start=match.start, end=match.end,
                            task_text=match.title, chapter=match)

    def start_scene(self, video: DownloadedVideo, part: TutorialPart) -> tuple[Path | None, bool]:
        """The saved scene of the latest earlier chapter (a course builds one project), or the default scene."""
        state = self.learner._state(video)
        earlier = [(int(key.split("-")[0]), entry) for key, entry in state.get("chapters", {}).items()
                   if int(key.split("-")[0]) < int(part.start) and entry.get("blend")
                   and Path(entry["blend"]).exists()]
        if earlier:
            return Path(max(earlier, key=lambda item: item[0])[1]["blend"]), False
        return None, True

    def frames(self, video: DownloadedVideo, start: float, end: float, dest: Path, every: float = 10.0) -> Path:
        """The tutorial's preview frames between two times, in one picture (the teacher's eyes on the video)."""
        info = json.loads(Path(video.info_path).read_text())
        cache = self.learner._lesson_dir(video) / "storyboard"
        tiles: list[tuple[Path | None, str]] = []
        t = start
        work = dest.parent / f".{dest.stem}_frames"
        work.mkdir(parents=True, exist_ok=True)
        while t <= end and len(tiles) < 48:
            frame = storyboard_frame(info, t, cache, work / f"{int(t)}.png")
            tiles.append((frame, _clock(t)))
            t += every
        sheet = contact_sheet(tiles, dest, tile_w=320, tile_h=180, columns=4,
                              title=f"{video.title[:60]} {_clock(start)}-{_clock(end)}")
        shutil.rmtree(work, ignore_errors=True)
        return sheet

    # -- building ----------------------------------------------------------------------------------------
    def teach_chapter(self, recipe: Recipe, video_id: str, chapter: int, *, score: float | None = None,
                      frame_at: str | None = None, note: str = "", frame_image: Path | None = None) -> TeachResult:
        video = self.video(video_id)
        part = self.part(video, chapter)
        start_from, default_cube = self.start_scene(video, part)
        project = self.projects.create(
            "lesson", f"{video.title[:50]} — {part.task_text}", teacher=self.name,
            source={"kind": "tutorial", "video": video.url, "video_id": video.video_id, "video_title": video.title,
                    "chapter": part.task_text, "chapter_index": chapter - 1, "start": part.start, "end": part.end,
                    "teacher": self.name}, start_from=str(start_from) if start_from else None)
        result, run = self._build(project, recipe, start_from=start_from, default_cube=default_cube,
                                  scene_camera="add_camera" in recipe.actions_used() or self._uses_camera(video))
        info = json.loads(Path(video.info_path).read_text())
        t = part.end - 15.0
        if frame_at:
            t = min(part.end, max(part.start, parse_timestamp(frame_at)))
        if frame_image is not None and Path(frame_image).is_file():
            frame: Path | None = Path(shutil.copyfile(frame_image, project.path("tutorial_frame.png")))
        else:
            frame = storyboard_frame(info, t, self.learner._lesson_dir(video) / "storyboard",
                                     project.path("tutorial_frame.png"))
        tiles = [(frame, f"Tutorial at {_clock(t)}")] + [(r, f"Lucius ({r.stem.split('_')[-1]})")
                                                          for r in result.renders]
        result.sheet = contact_sheet(tiles, project.path("sheet.png"), title=part.task_text)
        project.data.update(sheet="sheet.png", recipe_steps=len(recipe.steps), actions=sorted(recipe.actions_used()),
                            final_render=result.renders[0].name if result.renders else None, note=note)
        if run.ok and score is not None:
            result.skill_id = self.learner.store_skill(recipe, part, video, True, score, chapter - 1, project,
                                                       judge=f"{self.name}: renders compared with the tutorial")
            result.status = "learned" if score >= self.learner.pass_score else "partial"
            project.data.update(status=result.status, score=score, skill_id=result.skill_id,
                                blend="scene.blend")
            state = self.learner._state(video)
            key = f"{int(part.start)}-{int(part.end)}"
            state["chapters"][key] = {"chapter": part.task_text, "project_id": project.id, "status": result.status,
                                      "score": score, "skill_id": result.skill_id, "teacher": self.name,
                                      "blend": str(project.path("scene.blend").resolve()),
                                      "scene_after": run.scene_text(), "recipe_digest": recipe_digest(recipe),
                                      "uses_camera": "add_camera" in recipe.actions_used() or self._uses_camera(video)}
            self.learner._save_state(video, state)
            project.save()
            self.learner.overview(video)
        else:
            project.data["status"] = "trial" if run.ok else "failed"
        project.save()
        return result

    # -- course packs ------------------------------------------------------------------------------------
    def install_course(self, folder: str | Path, *, redo: bool = False,
                       on_progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
        """Replay a course pack on this machine, without a model API.

        A pack (``course.json`` plus one recipe per chapter) holds what a teacher kept for a tutorial.
        Every chapter is built again in order, each continuing from the scene of the one before, and
        kept with the teacher's score: the same skills, projects and renders as where it was taught.
        A chapter already kept from the same recipe is skipped (``redo`` builds it again); the first
        chapter that fails stops the replay, since the later ones build on it.
        """
        folder = Path(folder)
        course = json.loads((folder / "course.json").read_text())
        video_id = str(course["video_id"])
        info = self.app.config.data_dir / "downloads" / f"{video_id}.info.json"
        if not info.exists():
            # Only what a lesson needs (title, duration, chapters): nothing is downloaded.
            info.parent.mkdir(parents=True, exist_ok=True)
            meta = {"id": video_id, "webpage_url": course.get("url") or f"https://www.youtube.com/watch?v={video_id}"}
            meta.update({key: course[key] for key in COURSE_INFO_KEYS if key in course})
            info.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        video = self.video(video_id)
        default_name, results = self.name, []
        try:
            for lesson in course.get("lessons") or []:
                chapter = int(lesson["chapter"])
                recipe = load_recipe(folder / lesson["recipe"])
                part = self.part(video, chapter)
                kept = self.learner._state(video).get("chapters", {}).get(f"{int(part.start)}-{int(part.end)}") or {}
                if (not redo and kept.get("recipe_digest") == recipe_digest(recipe) and kept.get("blend")
                        and Path(kept["blend"]).exists() and self.app.library.exists(str(kept.get("skill_id")))):
                    results.append({"chapter": chapter, "title": part.task_text, "status": "already learned",
                                    "score": kept.get("score"), "project_id": kept.get("project_id")})
                    continue
                if on_progress:
                    on_progress(f"chapter {chapter}: {part.task_text} ({len(recipe.steps)} steps)")
                self.name = str(lesson.get("teacher") or course.get("teacher") or default_name)
                frame = folder / lesson["frame"] if lesson.get("frame") else None
                result = self.teach_chapter(recipe, video_id, chapter, score=float(lesson["score"]),
                                            frame_at=lesson.get("frame_at"), note=str(lesson.get("note") or ""),
                                            frame_image=frame)
                results.append({"chapter": chapter, "title": part.task_text, "status": result.status,
                                 "score": lesson["score"] if result.ok else None, "project_id": result.project_id,
                                 "skill_id": result.skill_id, "error": result.error,
                                 "sheet": str(result.sheet) if result.sheet else None})
                if not result.ok:
                    break
        finally:
            self.name = default_name
        return results

    def teach_task(self, recipe: Recipe, task: str, *, references: list[str | Path] | None = None,
                   score: float | None = None, note: str = "") -> TeachResult:
        project = self.projects.create("task", task, task=task, teacher=self.name)
        copies = []
        for i, ref in enumerate(Path(r) for r in references or []):
            if not ref.is_file() or ref.suffix.lower() not in IMAGE_TYPES:
                raise ValidationError(f"reference image {ref} must be an existing PNG, JPEG or WebP file")
            copy = project.path(f"reference{i + 1}{ref.suffix.lower()}")
            shutil.copyfile(ref, copy)
            copies.append(copy)
        project.data["references"] = [c.name for c in copies]
        result, run = self._build(project, recipe, start_from=None, default_cube=False,
                                  scene_camera="add_camera" in recipe.actions_used(), clear=True)
        tiles = [(c, f"Reference {i + 1}") for i, c in enumerate(copies)]
        tiles += [(r, f"Lucius ({r.stem.split('_')[-1]})") for r in result.renders]
        result.sheet = contact_sheet(tiles, project.path("sheet.png"), title=f"Task: {task}")
        project.data.update(sheet="sheet.png", recipe_steps=len(recipe.steps), actions=sorted(recipe.actions_used()),
                            final_render=result.renders[0].name if result.renders else None, note=note)
        if run.ok and score is not None:
            from lucius.lessons.maker import Maker, MakeResult

            made = MakeResult(project_id=project.id, status="made", score=score)
            result.skill_id = Maker(self.app)._store_skill(task, recipe, made, project.id)
            result.status = "made" if score >= 6.0 else "rough"
            project.data.update(status=result.status, score=score, skill_id=result.skill_id, blend="scene.blend")
        else:
            project.data["status"] = "trial" if run.ok else "failed"
        project.save()
        return result

    def _build(self, project: Any, recipe: Recipe, *, start_from: Path | None, default_cube: bool,
               scene_camera: bool, clear: bool = False) -> tuple[TeachResult, RecipeRun]:
        backend = self.app.headless_backend()
        runner = self.learner.runner(backend)
        if clear:
            runner._bridge("reset_scene", {"keep_camera_light": False})
            run = runner.run(recipe, prepare=False)
        else:
            run = runner.run(recipe, start_from=start_from, default_cube=default_cube)
        (project.path("recipe.json")).write_text(recipe.model_dump_json(indent=1))
        result = TeachResult(project_id=project.id, ok=run.ok, error=run.error_text() or None,
                             context=run.context_text() if not run.ok else "", scene=run.scene_text())
        project.add_attempt({"number": 1, "score": None, "run": run.to_dict(), "fixes": 0, "renders": [],
                             "teacher": self.name})
        if run.ok:
            frame = subject_names(run.scene.get("objects", []))
            views = [("scene", "three_quarter", "camera")] if scene_camera and run.scene.get("camera") else []
            lit = any(o.get("type") == "LIGHT" for o in run.scene.get("objects", []))
            if not views and lit and "add_light" in recipe.actions_used():
                views.append(("auto_lit", "three_quarter", "lit"))   # the lighting itself, before there is a camera
            views += [("auto", "three_quarter", "three_quarter"), ("auto", "front", "front"), ("auto", "top", "top")]
            final = "set_render" in recipe.actions_used()
            for camera, view, label in views[:3]:
                path = project.path(f"render_{label}.png")
                if camera == "scene" and final:
                    # A finished shot: the scene's own camera, size, samples and colour settings.
                    runner._bridge("render_image", {"path": str(path.resolve()), "camera": "scene",
                                                    "scene_settings": True})
                    result.renders.append(path)
                    continue
                if camera == "auto_lit":
                    runner._bridge("render_image", {"path": str(path.resolve()), "camera": "auto", "view": view,
                                                    "frame": frame, "samples": self.render_samples,
                                                    "lights": "scene"})
                    result.renders.append(path)
                    continue
                result.renders.append(runner.render(path, camera=camera, view=view, frame=frame or None,
                                                    samples=self.render_samples))
            runner.save_blend(project.path("scene.blend"))
            project.data["attempts"][-1]["renders"] = [r.name for r in result.renders]
        return result, run

    def _uses_camera(self, video: DownloadedVideo) -> bool:
        return any(entry.get("uses_camera") for entry in self.learner._state(video).get("chapters", {}).values())


def recipe_digest(recipe: Recipe) -> str:
    """What a recipe does (its actions and values, not its notes): the same digest means the same build."""
    steps = [[step.action, step.args] for step in recipe.steps]
    return hashlib.sha256(json.dumps(steps, sort_keys=True, default=str).encode()).hexdigest()[:16]


def subject_names(objects: list[dict[str, Any]]) -> list[str]:
    """The meshes a preview should frame: all of them except a ground (a flat plane much larger than the rest --
    a table or tablecloth would otherwise shrink the subject to a speck)."""
    meshes = [o for o in objects if o.get("type") == "MESH"]

    def ground(o: dict[str, Any]) -> bool:
        x, y, z = (o.get("dimensions") or [0, 0, 0])[:3]
        others = [max((m.get("dimensions") or [0, 0, 0])[:2]) for m in meshes if m is not o]
        return z <= 0.01 * max(x, y, 1e-9) and bool(others) and max(x, y) > 2.5 * max(others)

    subjects = [o["name"] for o in meshes if not ground(o)]
    return subjects or [o["name"] for o in meshes]


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
