"""Learning a tutorial as recipes, rebuilding and judging it, and making new things with what was learned.

Imports are lazy: the planner needs :mod:`lucius.lessons.catalogue` while the executor (which the learner
uses) needs the planner.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ChapterResult": "learner", "LessonLearner": "learner", "QuotaExhausted": "learner",
    "Maker": "maker", "MakeResult": "maker",
    "Project": "projects", "ProjectStore": "projects", "contact_sheet": "projects",
    "Recipe": "recipe", "RecipeStep": "recipe", "parse_recipe": "recipe", "recipe_schema": "recipe",
}

__all__ = [*_EXPORTS, "rate_project"]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(f"lucius.lessons.{_EXPORTS[name]}"), name)
    raise AttributeError(name)


def rate_project(app: Any, project_id: str, *, good: bool, note: str = "") -> dict[str, Any]:
    """A person's verdict on a project: stored with it, and counted for (or against) the skill it produced."""
    import time

    from lucius.lessons.projects import ProjectStore

    project = ProjectStore(app.config.projects_dir).get(project_id)
    project.data["rating"] = {"good": good, "note": note, "at": time.time()}
    project.save()
    skill_id = project.data.get("skill_id")
    if skill_id and app.library.exists(skill_id):
        from lucius.lessons.runner import record_run

        app.library.review(skill_id, accept=good)
        run_id = record_run(app.db, task_text=f"review of {project.data.get('title')}", mode="review",
                            status="success" if good else "failure", environment="human_review",
                            metrics={"note": note, "project": project_id}, arm="person", backend="human")
        app.library.record_use(skill_id, success=good, run_id=run_id,
                               instance_signature=f"rating:{project_id}", objective=True, environment="human_review",
                               role="validation", source_class="human_correction" if good else "agent_failure",
                               detail={"rated_by": "person", "note": note, "project": project_id})
    return project.summary()
