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
    source = dict(data.get("source") or {})
    if data.get("aliases"):
        source["aliases"] = [str(a) for a in data["aliases"]]   # other names for what it makes ("espada")
    return Recipe(title=str(data.get("title") or Path(path).stem), summary=str(data.get("summary") or ""),
                  objects=list(data.get("objects") or []), steps=steps,
                  expected_result=str(data.get("expected_result") or ""), source=source)


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
                                  scene_camera="add_camera" in recipe.actions_used() or self._uses_camera(video),
                                  animate=score is not None)
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
                            final_render=_final_render(result.renders), note=note,
                            frame_at=_clock(t))
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

    def store_object(self, task: str, recipe: Recipe, score: float, project_id: str) -> str:
        """Keep a taught object (the teacher built it and judged the renders): a skill Lucius can rebuild
        without a model (``lucius make`` recalls it by name or alias) and that planners adapt."""
        from lucius.ids import new_id
        from lucius.lessons.runner import record_run
        from lucius.skills.schema import ActionTemplate, SkillDefinition, SkillExample, SkillPhase
        from lucius.timeutil import now

        library = self.app.library
        skill_id = f"object_{_slug(recipe.title or task)}"
        aliases = [str(a) for a in recipe.source.get("aliases") or []]
        definition = SkillDefinition(
            skill_id=skill_id, name=recipe.title or task, purpose=f"{task}: {recipe.summary}".strip(),
            categories=["made_recipe", "taught_object"],
            triggers=sorted(set(words(" ".join([task, recipe.title, *aliases])))), applicable_contexts=["recipe"],
            phases=[SkillPhase(name="primary_form", description=recipe.summary, actions=[
                ActionTemplate(action_type=s.action, description=s.note, args=s.args, object_ref=None)
                for s in recipe.steps])],
            source_class="human_correction" if self.name not in ("", "teacher") else "user_demo",
            notes=[f"taught for the task '{task}' by {self.name}", f"project {project_id}",
                   f"digest {recipe_digest(recipe)}", *(f"alias {a}" for a in aliases)])
        if library.exists(skill_id):
            library.new_version(skill_id, definition, change_note=f"taught again for '{task}'", created_by=self.name)
            if library.get(skill_id).status.value == "disabled":
                library.set_disabled(skill_id, False)
        else:
            library.create(definition, created_by=self.name, change_note=f"taught for '{task}'")
        skill = library.get(skill_id)
        library.add_example(SkillExample(
            id=new_id("example"), skill_id=skill_id, skill_version=skill.current_version, role="demonstration",
            source_class="user_demo", evidence_weight=0.8, instance_signature=f"task:{project_id}",
            summary={"task": task, "project": project_id, "teacher": self.name}, created_at=now()))
        success = score >= 6.0
        run_id = record_run(self.app.db, task_text=task, mode="validation", status="success" if success else "failure",
                            environment="blender_headless", metrics={"visual_score": score, "project": project_id},
                            arm="teacher")
        library.record_use(skill_id, success=success, run_id=run_id, instance_signature=f"task:{project_id}",
                           objective=True, environment="blender_headless", role="validation",
                           source_class="agent_success",
                           detail={"visual_score": score, "judge": f"{self.name}: renders compared with the request",
                                   "project": project_id})
        try:
            self.app.retriever.refresh()
        except Exception:  # indexing is best effort
            pass
        return skill_id

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
        if course.get("kind") == "objects":
            return self._install_objects(folder, course, redo=redo, on_progress=on_progress)
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

    def _install_objects(self, folder: Path, course: dict[str, Any], *, redo: bool,
                         on_progress: Callable[[str], None] | None) -> list[dict[str, Any]]:
        """An objects pack: things a teacher made (not tutorial chapters), each built from scratch and kept."""
        default_name, results = self.name, []
        try:
            for lesson in course.get("lessons") or []:
                recipe = load_recipe(folder / lesson["recipe"])
                skill_id = f"object_{_slug(recipe.title or lesson['task'])}"
                if not redo and self.app.library.exists(skill_id) and any(
                        n == f"digest {recipe_digest(recipe)}" for n in self.app.library.get(skill_id).definition.notes):
                    results.append({"task": lesson["task"], "status": "already learned", "skill_id": skill_id})
                    continue
                if on_progress:
                    on_progress(f"{lesson['task']} ({len(recipe.steps)} steps)")
                self.name = str(lesson.get("teacher") or course.get("teacher") or default_name)
                refs = [folder / r for r in lesson.get("references") or []]
                result = self.teach_task(recipe, lesson["task"], references=refs, score=float(lesson["score"]),
                                         note=str(lesson.get("note") or ""))
                results.append({"task": lesson["task"], "status": result.status, "score": lesson["score"],
                                "project_id": result.project_id, "skill_id": result.skill_id, "error": result.error,
                                "sheet": str(result.sheet) if result.sheet else None})
        finally:
            self.name = default_name
        return results

    def export_objects(self, skill_ids: list[str], dest: str | Path) -> Path:
        """Write taught objects as an objects pack: each recipe, its score and its references."""
        dest = Path(dest)
        (dest / "renders").mkdir(parents=True, exist_ok=True)
        lessons = []
        for skill_id in skill_ids:
            skill = self.app.library.get(skill_id)
            notes = skill.definition.notes
            project_id = next(n.split(" ", 1)[1] for n in notes if n.startswith("project "))
            project = self.projects.get(project_id)
            recipe = json.loads(project.path("recipe.json").read_text())
            recipe = {k: recipe[k] for k in ("title", "summary", "objects", "expected_result", "steps") if k in recipe}
            aliases = [n.split(" ", 1)[1] for n in notes if n.startswith("alias ")]
            if aliases:
                recipe["aliases"] = aliases
            name = f"{_slug(recipe.get('title') or skill_id)}.json"
            (dest / name).write_text(json.dumps(recipe, indent=1, ensure_ascii=False) + "\n")
            lesson: dict[str, Any] = {"task": project.data.get("task") or skill.definition.name, "recipe": name,
                                      "score": project.data.get("score"), "teacher": project.data.get("teacher"),
                                      "note": project.data.get("note") or ""}
            refs = []
            for ref in project.data.get("references") or []:
                target = dest / "references" / f"{_slug(skill_id)}_{ref}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project.path(ref), target)
                refs.append(target.relative_to(dest).as_posix())
            if refs:
                lesson["references"] = refs
            if project.data.get("final_render") and project.path(project.data["final_render"]).exists():
                from PIL import Image

                Image.open(project.path(project.data["final_render"])).convert("RGB").save(
                    dest / "renders" / f"{_slug(skill_id)}.jpg", quality=88)
                lesson["render"] = f"renders/{_slug(skill_id)}.jpg"
            lessons.append(lesson)
        course = {"kind": "objects", "teacher": self.name,
                  "about": "Objects taught to Lucius beyond the tutorials, with the techniques it learned. Replay with "
                           f"`lucius teach course {dest.as_posix()}`; then `lucius make \"a sword\"` rebuilds one "
                           "without a model.", "lessons": lessons}
        (dest / "course.json").write_text(json.dumps(course, indent=1, ensure_ascii=False) + "\n")
        return dest

    def export_course(self, video_id: str, dest: str | Path) -> Path:
        """Write the chapters kept for a tutorial as a course pack (see ``install_course``): the recipes, the
        scores and teachers, each chapter's tutorial frame, the final render and the overview."""
        from PIL import Image

        video = self.video(video_id)
        dest = Path(dest)
        (dest / "frames").mkdir(parents=True, exist_ok=True)
        chapters = self.learner._state(video).get("chapters", {})
        lessons = []
        last_render = last_video = None
        for key, entry in sorted(chapters.items(), key=lambda item: int(item[0].split("-")[0])):
            project = self.projects.get(str(entry["project_id"]))
            source = project.data.get("source") or {}
            chapter = int(source.get("chapter_index", 0)) + 1
            recipe = json.loads(project.path("recipe.json").read_text())
            recipe = {k: recipe[k] for k in ("title", "summary", "objects", "expected_result", "steps") if k in recipe}
            name = f"{chapter:02d}_{_slug(recipe.get('title') or entry.get('chapter') or key)}.json"
            (dest / name).write_text(json.dumps(recipe, indent=1, ensure_ascii=False) + "\n")
            lesson: dict[str, Any] = {"chapter": chapter, "title": entry.get("chapter"), "recipe": name,
                                      "score": entry.get("score"), "teacher": entry.get("teacher") or self.name}
            frame = project.path("tutorial_frame.png")
            if frame.exists():
                Image.open(frame).convert("RGB").save(dest / "frames" / f"{chapter:02d}.jpg", quality=88)
                lesson["frame"] = f"frames/{chapter:02d}.jpg"
            if project.data.get("frame_at"):
                lesson["frame_at"] = project.data["frame_at"]
            lesson["note"] = project.data.get("note") or ""
            lessons.append(lesson)
            if project.data.get("final_render") and project.path(project.data["final_render"]).exists():
                last_render = project.path(project.data["final_render"])
            if project.data.get("video") and project.path(project.data["video"]).exists():
                last_video = project.path(project.data["video"])
        info = json.loads(Path(video.info_path).read_text())
        course = {"video_id": video.video_id, "url": video.url, "title": video.title,
                  "duration": info.get("duration") or video.duration, "language": video.language,
                  "channel": info.get("channel"),
                  "chapters": [{"title": c.get("title"), "start_time": c.get("start_time"), "end_time": c.get("end_time")}
                               for c in info.get("chapters") or []],
                  "teacher": self.name,
                  "about": f"What Lucius kept from this tutorial: one recipe per chapter, each continuing from the scene "
                           f"of the one before. Replay with `lucius teach course {dest.as_posix()}` (no API needed).",
                  "lessons": lessons}
        (dest / "course.json").write_text(json.dumps(course, indent=1, ensure_ascii=False) + "\n")
        if last_render is not None:
            Image.open(last_render).convert("RGB").save(dest / "final.jpg", quality=90)
        if last_video is not None:
            shutil.copyfile(last_video, dest / "final.mp4")   # the course's animation, as Lucius rendered it
        overview = self.learner.overview(video)
        if overview is not None:
            Image.open(overview).convert("RGB").save(dest / "overview.jpg", quality=85)
        return dest

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
                                  scene_camera="add_camera" in recipe.actions_used(), clear=True,
                                  animate=score is not None)
        tiles = [(c, f"Reference {i + 1}") for i, c in enumerate(copies)]
        tiles += [(r, f"Lucius ({r.stem.split('_')[-1]})") for r in result.renders]
        result.sheet = contact_sheet(tiles, project.path("sheet.png"), title=f"Task: {task}")
        project.data.update(sheet="sheet.png", recipe_steps=len(recipe.steps), actions=sorted(recipe.actions_used()),
                            final_render=_final_render(result.renders), note=note)
        if run.ok and score is not None:
            result.skill_id = self.store_object(task, recipe, score, project.id)
            result.status = "made" if score >= 6.0 else "rough"
            project.data.update(status=result.status, score=score, skill_id=result.skill_id, blend="scene.blend")
        else:
            project.data["status"] = "trial" if run.ok else "failed"
        project.save()
        return result

    def _build(self, project: Any, recipe: Recipe, *, start_from: Path | None, default_cube: bool,
               scene_camera: bool, clear: bool = False, animate: bool = False) -> tuple[TeachResult, RecipeRun]:
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
        if run.ok and scene_camera and run.scene.get("camera") and self._animated(run):
            # An animation: the camera's shot at its start, middle and end (and, when kept, the video).
            self._render_animation(runner, project, result, run, video=animate)
            runner.save_blend(project.path("scene.blend"))
            project.data["attempts"][-1]["renders"] = [r.name for r in result.renders]
        elif run.ok:
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

    @staticmethod
    def _animated(run: RecipeRun) -> bool:
        animation = run.scene.get("animation") or {}
        return bool(animation.get("animated")) and animation.get("end", 0) > animation.get("start", 0)

    def _render_animation(self, runner: Any, project: Any, result: TeachResult, run: RecipeRun, *,
                          video: bool) -> None:
        animation = run.scene["animation"]
        start, end = int(animation["start"]), int(animation["end"])
        rx, ry = (animation.get("resolution") or [1920, 1080])[:2]
        width = 800
        height = max(16, round(width * ry / rx))
        for label, number in (("start", start), ("middle", (start + end) // 2), ("end", end)):
            path = project.path(f"render_frame_{label}.png")
            runner._bridge("render_image", {"path": str(path.resolve()), "camera": "scene", "width": width,
                                            "height": height, "samples": self.render_samples,
                                            "frame_number": number, "lights": "scene"})
            result.renders.append(path)
        if video:
            percentage = max(10, min(100, round(640 / rx * 100)))
            path = project.path("animation.mp4")
            done = runner._bridge("render_animation", {"path": str(path.resolve()), "step": 2,
                                                       "samples": max(8, self.render_samples // 2),
                                                       "percentage": percentage})
            project.data["video"] = path.name
            project.data["video_frames"] = done.get("frames") if isinstance(done, dict) else None

    def _uses_camera(self, video: DownloadedVideo) -> bool:
        return any(entry.get("uses_camera") for entry in self.learner._state(video).get("chapters", {}).values())


def _final_render(renders: list[Path]) -> str | None:
    """The picture that stands for a build: an animation's middle frame (its start is usually a rest pose),
    otherwise the first view."""
    middle = [r for r in renders if r.stem == "render_frame_middle"]
    return (middle or renders or [None])[0].name if renders else None


STOPWORDS = {"make", "build", "model", "create", "draw", "render", "with", "and", "the", "for", "from", "like",
             "this", "that", "image", "reference", "please", "some", "one", "una", "uno", "unos", "unas", "con",
             "del", "las", "los", "que", "para", "haz", "hazme", "crea", "creame", "modela", "dibuja", "como",
             "esta", "este", "imagen", "referencia", "por", "favor", "pon", "hacer", "quiero", "want", "can", "you"}


def words(text: str) -> list[str]:
    """The content words of a request, lower case and without accents ("Una ESPADA" -> ["espada"])."""
    import re
    import unicodedata

    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [w for w in re.findall(r"[a-z]{3,}", plain) if w not in STOPWORDS]


def _slug(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "chapter"


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
