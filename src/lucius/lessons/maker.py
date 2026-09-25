"""Make something new -- "a sword", or what a reference image shows -- with what Lucius has learned.

The plan is a recipe written by a model, but only with the techniques Lucius has learned: the
actions used by tutorial recipes it rebuilt successfully (validated lesson skills), plus basic
object handling. The learned recipes are given as worked examples of how the tutorials use those
techniques and at what sizes. ``allow_unlearned`` lifts the restriction (and the project says so).

Then the same loop as a lesson: build in Blender, fix failing steps, render, have the model judge
the renders against the task (and the reference image), revise, keep the best attempt. The result
is a project folder for a person to judge; their rating confirms or rejects the skill it becomes.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucius.errors import ProviderError, ValidationError
from lucius.lessons.catalogue import BASIC_ACTIONS, RECIPE_ACTIONS, catalogue
from lucius.lessons.learner import RECIPE_RULES, QuotaExhausted, _quota
from lucius.lessons.projects import ProjectStore, contact_sheet
from lucius.lessons.recipe import Recipe, parse_recipe, recipe_schema
from lucius.lessons.runner import RecipeRun, RecipeRunner, record_run
from lucius.logging_setup import get_logger
from lucius.providers.base import ImageInput

if TYPE_CHECKING:
    from lucius.app import Lucius

log = get_logger("lessons.maker")

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
LEARNED_STATUSES = ("validated", "high_confidence")

MAKE_SYSTEM = (
    "You plan how Lucius builds what is asked in Blender, as a recipe of the actions listed -- the techniques it "
    "has learned from tutorials, plus basic object handling. The learned recipes show how the tutorials used "
    "those techniques and at what sizes: reuse their approach and values where they fit. Build at real-world "
    "scale, resting on the ground (z = 0), as separate named parts where the thing has parts, and give it "
    "materials only if set_material is available. If a reference image is given, match its shape, proportions "
    "and colours. If the task needs a technique that is not available, get as close as you can with what is, "
    "and name what is missing in missing_techniques."
)

CRITIQUE_SYSTEM = (
    "Judge renders of what Lucius built in Blender for a task (and against the reference image, if given). Be "
    "honest and specific: does it read as the requested object at first glance, are the proportions and parts "
    "right, does it match the reference? score 0-10: 10 excellent; 7 clearly the requested object with small "
    "problems; 4 recognisable but wrong in important ways; 1 not the requested object. List the problems with "
    "a fix stated in terms of the recipe (which step, which value). looks_like says plainly what a person would "
    "say the render shows."
)

REVISE_SYSTEM = (
    "Improve a Blender recipe from a critique of its result. Apply the fixes, keep what works, keep the recipe "
    "runnable (selections consistent with the geometry each step creates) and return the whole revised recipe."
)

FIX_SYSTEM = (
    "A Blender recipe failed while executing. Correct it so that it runs and still builds the same thing; change "
    "only what is needed and return the whole corrected recipe."
)


def critique_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "verdict": {"type": "string", "enum": ["good", "close", "poor"]},
            "looks_like": {"type": "string"},
            "matches": {"type": "array", "items": {"type": "string"}},
            "differences": {"type": "array", "items": {
                "type": "object",
                "properties": {"object": {"type": "string"}, "problem": {"type": "string"}, "fix": {"type": "string"}},
                "required": ["object", "problem", "fix"], "additionalProperties": False}},
        },
        "required": ["score", "verdict", "looks_like", "matches", "differences"], "additionalProperties": False,
    }


def make_schema(actions: list[str]) -> dict[str, Any]:
    schema = recipe_schema(actions)
    schema["properties"]["missing_techniques"] = {"type": "array", "items": {"type": "string"}}
    schema["required"] = [*schema["required"], "missing_techniques"]
    return schema


@dataclass
class Techniques:
    actions: set[str] = field(default_factory=set)
    modifiers: set[str] = field(default_factory=set)
    sources: dict[str, list[str]] = field(default_factory=dict)       # action -> lesson titles
    recipes: list[dict[str, Any]] = field(default_factory=list)        # learned recipes (examples)

    def allowed(self, allow_unlearned: bool) -> list[str]:
        if allow_unlearned:
            return list(RECIPE_ACTIONS)
        return [a for a in RECIPE_ACTIONS if a in BASIC_ACTIONS or a in self.actions]

    def to_dict(self) -> dict[str, Any]:
        return {"actions": sorted(self.actions), "modifiers": sorted(self.modifiers),
                "sources": {k: v[:3] for k, v in sorted(self.sources.items())},
                "recipes": [{k: r[k] for k in ("skill_id", "title", "status", "score")} for r in self.recipes]}


@dataclass
class MakeResult:
    project_id: str
    status: str
    score: float | None = None
    looks_like: str = ""
    sheet: Path | None = None
    final_render: Path | None = None
    blend: Path | None = None
    skill_id: str | None = None
    missing_techniques: list[str] = field(default_factory=list)
    attempts: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"project_id": self.project_id, "status": self.status, "score": self.score, "looks_like": self.looks_like,
                "sheet": str(self.sheet) if self.sheet else None,
                "final_render": str(self.final_render) if self.final_render else None,
                "blend": str(self.blend) if self.blend else None, "skill_id": self.skill_id,
                "missing_techniques": self.missing_techniques, "attempts": self.attempts, "notes": self.notes}


class Maker:
    def __init__(self, app: Lucius, *, iterations: int = 3, max_fixes: int = 3, target_score: float = 8.0,
                 pass_score: float = 6.0, render_samples: int = 24) -> None:
        self.app = app
        self.iterations = iterations
        self.max_fixes = max_fixes
        self.target_score = target_score
        self.pass_score = pass_score
        self.render_samples = render_samples
        self.projects = ProjectStore(app.config.projects_dir)

    # -- what has been learned --------------------------------------------------------------------------
    def techniques(self) -> Techniques:
        from lucius.lessons.recipe import RecipeStep

        out = Techniques()
        for skill in self.app.library.list(include_inactive=False):
            d = skill.definition
            if "tutorial_recipe" not in d.categories and "made_recipe" not in d.categories:
                continue
            steps = [RecipeStep(action=a.action_type, args=a.args, note=a.description)
                     for phase in d.phases for a in phase.actions if a.action_type in RECIPE_ACTIONS]
            score = next((e.summary.get("visual_score") for e in reversed(self.app.library.examples(skill.id))
                          if e.role == "validation"), None)
            out.recipes.append({"skill_id": skill.id, "title": d.name, "status": skill.status.value, "score": score,
                                "recipe": Recipe(title=d.name, summary=d.purpose, steps=steps)})
            if skill.status.value not in LEARNED_STATUSES or "tutorial_recipe" not in d.categories:
                continue  # only lessons Lucius passed teach it a technique
            for step in steps:
                out.actions.add(step.action)
                out.sources.setdefault(step.action, [])
                if d.name not in out.sources[step.action]:
                    out.sources[step.action].append(d.name)
                if step.action == "add_modifier" and step.args.get("type"):
                    out.modifiers.add(str(step.args["type"]).upper())
        return out

    def _examples(self, techniques: Techniques, task: str, limit_chars: int = 30000) -> str:
        words = set(re.findall(r"[a-z]{3,}", task.lower()))

        def relevance(item: dict[str, Any]) -> tuple[int, int]:
            text = f"{item['title']} {item['recipe'].summary}".lower()
            return (-len(words & set(re.findall(r"[a-z]{3,}", text))),
                    0 if item["status"] in LEARNED_STATUSES else 1)

        blocks, used = [], 0
        for item in sorted(techniques.recipes, key=relevance):
            block = f"## {item['title']} ({item['status']}, judged {item['score']}/10)\n{item['recipe'].compact(80)}"
            if used + len(block) > limit_chars:
                continue
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks) or "(no learned recipes yet)"

    # -- model calls ---------------------------------------------------------------------------------------
    def _call(self, purpose: str, *, system: str, prompt: str, schema: dict[str, Any],
              images: list[ImageInput] | None = None, max_tokens: int = 32000) -> dict[str, Any]:
        provider = self.app.providers.vlm if images else self.app.providers.llm
        try:
            result = provider.complete_json(purpose=purpose, system=system, prompt=prompt, schema=schema,
                                            images=images or [], max_tokens=max_tokens)
        except ProviderError as exc:
            if _quota(exc):
                raise QuotaExhausted(exc.message) from exc
            raise
        return {**result.data, "_model": result.model}

    # -- make ----------------------------------------------------------------------------------------------
    def make(self, task: str, *, references: list[str | Path] | None = None, allow_unlearned: bool = False,
             backend: Any = None, on_progress: Callable[[str], None] | None = None) -> MakeResult:
        say = on_progress or (lambda message: log.info(message))
        refs = [Path(r) for r in references or []]
        for ref in refs:
            if not ref.is_file() or ref.suffix.lower() not in IMAGE_TYPES:
                raise ValidationError(f"reference image {ref} must be an existing PNG, JPEG or WebP file")
        techniques = self.techniques()
        allowed = techniques.allowed(allow_unlearned)
        project = self.projects.create("task", task, task=task, allow_unlearned=allow_unlearned,
                                       techniques=techniques.to_dict(), allowed_actions=allowed)
        ref_copies = []
        for i, ref in enumerate(refs):
            copy = project.path(f"reference{i + 1}{ref.suffix.lower()}")
            shutil.copyfile(ref, copy)
            ref_copies.append(copy)
        project.data["references"] = [p.name for p in ref_copies]
        project.save()
        ref_images = [ImageInput(p.read_bytes(), IMAGE_TYPES[p.suffix.lower()], label=f"Reference image {i + 1}")
                      for i, p in enumerate(ref_copies)]
        result = MakeResult(project_id=project.id, status="failed")
        backend = backend or self.app.headless_backend()
        from lucius.lessons.learner import LessonLearner

        runner = LessonLearner(self.app).runner(backend)
        modifiers = sorted(techniques.modifiers) if not allow_unlearned else []
        rules = RECIPE_RULES + (f"\nModifier types learned: {', '.join(modifiers) or 'none'}; use only these."
                                if not allow_unlearned and "add_modifier" in allowed else "")
        say(f"planning '{task}' with {len(allowed)} actions ({len(techniques.actions)} learned from tutorials)")
        prompt = "\n\n".join([
            f"Task: {task}", f"Actions available:\n{catalogue(allowed)}",
            f"Learned recipes (how the tutorials built things):\n{self._examples(techniques, task)}",
            ("Reference image(s) attached: match them." if ref_images else ""),
            "Write the recipe."])
        try:
            data = self._call("make_plan", system=MAKE_SYSTEM + "\n\n" + rules, prompt=prompt,
                              schema=make_schema(allowed), images=ref_images)
        except QuotaExhausted as exc:
            result.status = "quota"
            result.notes.append(f"model quota exhausted: {exc}")
            project.data.update(status="quota", notes=result.notes)
            project.save()
            return result
        recipe, problems = parse_recipe(data, allowed=allowed, source={"kind": "task", "task": task,
                                                                        "model": data.get("_model")})
        result.missing_techniques = [str(m) for m in data.get("missing_techniques") or []]
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for number in range(1, self.iterations + 1):
            try:
                run, recipe, problems, fixes = self._build(runner, recipe, problems, allowed, rules, say, number)
            except QuotaExhausted as exc:
                result.notes.append(f"model quota exhausted: {exc}")
                break
            renders = self._render(runner, project, recipe, number) if run.ok else []
            critique: dict[str, Any] = {}
            if renders:
                try:
                    critique = self._critique(task, recipe, renders, ref_images)
                except QuotaExhausted as exc:
                    result.notes.append(f"model quota exhausted: {exc}")
                except ProviderError as exc:
                    critique = {"score": None, "error": exc.message[:200], "differences": []}
            score = critique.get("score") if run.ok else None
            attempt = {"number": number, "recipe": recipe, "run": run, "renders": renders, "critique": critique,
                       "score": score, "fixes": fixes}
            attempts.append(attempt)
            (project.path(f"attempt{number}_recipe.json")).write_text(recipe.model_dump_json(indent=1))
            project.add_attempt({"number": number, "score": score, "fixes": fixes, "run": run.to_dict(),
                                 "renders": [p.name for p in renders], "critique": critique, "problems": problems})
            say(f"attempt {number}: {'built' if run.ok else 'failed: ' + run.error_text()}; score {score}; "
                f"{critique.get('looks_like', '')}")
            if best is None or (score or -1) > (best["score"] or -1):
                best = attempt
            if not run.ok or (score or 0) >= self.target_score or number == self.iterations \
                    or not critique.get("differences"):
                break
            try:
                recipe, problems = self._revise(recipe, run, critique, allowed, rules, task)
            except (QuotaExhausted, ProviderError) as exc:
                result.notes.append(f"revision stopped: {exc}")
                break
        result.attempts = len(attempts)
        if best is None:
            project.data.update(status="failed", notes=result.notes)
            project.save()
            return result
        final_run: RecipeRun = best["run"]
        if best is not attempts[-1] and final_run.ok:
            final_run = runner.run(best["recipe"])
        (project.path("recipe.json")).write_text(best["recipe"].model_dump_json(indent=1))
        if final_run.ok:
            try:
                result.blend = runner.save_blend(project.path("scene.blend"))
            except Exception as exc:
                result.notes.append(f"scene not saved: {exc}")
        result.score = best["score"]
        result.looks_like = best["critique"].get("looks_like", "")
        result.final_render = best["renders"][0] if best["renders"] else None
        tiles = [(p, f"Reference {i + 1}") for i, p in enumerate(ref_copies)]
        tiles += [(a["renders"][0] if a["renders"] else None,
                   f"Attempt {a['number']}: " + (f"{a['score']}/10" if a["score"] is not None else "failed"))
                  for a in attempts]
        if best["renders"][1:]:
            tiles.append((best["renders"][1], "best attempt, front"))
        result.sheet = contact_sheet(tiles, project.path("sheet.png"), title=f"Task: {task}")
        result.status = ("made" if final_run.ok and (result.score or 0) >= self.pass_score
                         else "rough" if final_run.ok else "failed")
        if final_run.ok:
            result.skill_id = self._store_skill(task, best["recipe"], result, project.id)
        project.data.update(status=result.status, score=result.score, looks_like=result.looks_like,
                            sheet="sheet.png", final_render=result.final_render.name if result.final_render else None,
                            blend="scene.blend" if result.blend else None, skill_id=result.skill_id,
                            missing_techniques=result.missing_techniques, notes=result.notes,
                            actions=sorted(best["recipe"].actions_used()),
                            learned_from={a: techniques.sources.get(a, ["basic"])
                                          for a in sorted(best["recipe"].actions_used())})
        project.save()
        return result

    def _build(self, runner: RecipeRunner, recipe: Recipe, problems: list[str], allowed: list[str], rules: str,
               say: Callable[[str], None], number: int) -> tuple[RecipeRun, Recipe, list[str], int]:
        run = runner.run(recipe)
        fixes = 0
        while not run.ok and fixes < self.max_fixes:
            fixes += 1
            say(f"attempt {number}: {run.error_text()} -> correcting ({fixes})")
            prompt = "\n\n".join([
                f"Actions available:\n{catalogue(allowed)}", f"Recipe ({recipe.title}):\n{recipe.compact()}",
                f"Failure: {run.error_text()}" + (f"\nOther problems: {'; '.join(problems)}" if problems else ""),
                f"Scene when it failed:\n{run.scene_text()}", "Return the corrected recipe."])
            try:
                data = self._call("make_fix", system=FIX_SYSTEM + "\n\n" + rules, prompt=prompt,
                                  schema=recipe_schema(allowed))
            except ProviderError as exc:
                log.warning("recipe correction failed: %s", exc.message)
                break
            recipe, problems = parse_recipe(data, allowed=allowed, source=recipe.source)
            run = runner.run(recipe)
        return run, recipe, problems, fixes

    def _render(self, runner: RecipeRunner, project: Any, recipe: Recipe, number: int) -> list[Path]:
        meshes = [o["name"] for o in runner.scene().get("objects", []) if o.get("type") == "MESH"]
        renders = []
        for view in ("three_quarter", "front"):
            path = project.path(f"attempt{number}_{view}.png")
            try:
                renders.append(runner.render(path, view=view, frame=meshes or None, samples=self.render_samples))
            except Exception as exc:
                log.warning("render failed: %s", exc)
        return renders

    def _critique(self, task: str, recipe: Recipe, renders: list[Path], refs: list[ImageInput]) -> dict[str, Any]:
        images = list(refs) + [ImageInput(p.read_bytes(), "image/png", label=f"Lucius render ({p.stem})")
                               for p in renders]
        prompt = f"Task: {task}\nWhat the recipe meant to build: {recipe.expected_result or recipe.summary}\n" \
                 + ("Compare with the reference image(s) first.\n" if refs else "") + "Judge the renders."
        data = self._call("make_critique", system=CRITIQUE_SYSTEM, prompt=prompt, schema=critique_schema(),
                          images=images, max_tokens=6000)
        data["score"] = max(0.0, min(10.0, float(data.get("score") or 0.0)))
        return data

    def _revise(self, recipe: Recipe, run: RecipeRun, critique: dict[str, Any], allowed: list[str], rules: str,
                task: str) -> tuple[Recipe, list[str]]:
        diffs = "\n".join(f"- {d.get('object')}: {d.get('problem')} -> fix: {d.get('fix')}"
                          for d in critique.get("differences", []))
        prompt = "\n\n".join([
            f"Task: {task}", f"Actions available:\n{catalogue(allowed)}", f"Recipe ({recipe.title}):\n{recipe.compact()}",
            f"Scene it built:\n{run.scene_text()}",
            f"Critique ({critique.get('score')}/10, looks like: {critique.get('looks_like')}):\n{diffs}",
            "Return the revised recipe."])
        data = self._call("make_revise", system=REVISE_SYSTEM + "\n\n" + rules, prompt=prompt,
                          schema=recipe_schema(allowed))
        return parse_recipe(data, allowed=allowed, source=recipe.source)

    def _store_skill(self, task: str, recipe: Recipe, result: MakeResult, project_id: str) -> str:
        from lucius.skills.library import slugify
        from lucius.skills.schema import ActionTemplate, SkillDefinition, SkillPhase

        library = self.app.library
        skill_id = f"made_{slugify(task)[:40]}_{project_id[-6:]}".replace("-", "_")
        definition = SkillDefinition(
            skill_id=skill_id, name=recipe.title or task, purpose=f"{task}: {recipe.summary}".strip(),
            categories=["made_recipe"], triggers=sorted({w for w in re.findall(r"[a-z]{3,}", task.lower())}),
            applicable_contexts=["recipe"],
            phases=[SkillPhase(name="primary_form", description=recipe.summary, actions=[
                ActionTemplate(action_type=s.action, description=s.note, args=s.args, object_ref=None)
                for s in recipe.steps])],
            source_class="agent_generated", notes=[f"made for the task '{task}'", f"project {project_id}"])
        library.create(definition, created_by="maker", change_note=f"made for '{task}'")
        # The model judged its own work: not objective. A person's rating (``rate``) is what can validate it.
        run_id = record_run(self.app.db, task_text=task, mode="execute",
                            status="success" if (result.score or 0) >= self.pass_score else "failure",
                            environment="blender_headless", metrics={"visual_score": result.score,
                                                                      "project": project_id}, arm="maker")
        library.record_use(skill_id, success=(result.score or 0) >= self.pass_score, run_id=run_id,
                           instance_signature=f"task:{project_id}", objective=False, environment="blender_headless",
                           role="validation", source_class="agent_success",
                           detail={"visual_score": result.score, "judge": "model critique", "project": project_id})
        try:
            self.app.retriever.refresh()
        except Exception as exc:
            log.warning("skill index refresh failed: %s", exc)
        return skill_id
