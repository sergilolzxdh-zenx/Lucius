"""Recipes: an ordered list of concrete Blender actions that builds something.

A lesson turns each tutorial chapter into a recipe (what the tutor did, with the values they used or
that were estimated from the picture); a task ("make a sword") is planned as a recipe from the
techniques learned so far. Recipes are executed in Blender, rendered and judged; the one that
reproduces the result best is kept, as a skill.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from lucius.lessons.catalogue import RECIPE_ACTIONS


class RecipeStep(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    video_time: str | None = None      # where the tutorial shows it ("36:25")


class Recipe(BaseModel):
    title: str
    summary: str = ""
    objects: list[dict[str, str]] = Field(default_factory=list)    # [{name, description}]
    steps: list[RecipeStep] = Field(default_factory=list)
    expected_result: str = ""
    source: dict[str, Any] = Field(default_factory=dict)           # video/chapter, or the task

    def actions_used(self) -> set[str]:
        return {s.action for s in self.steps}

    def modifier_types(self) -> set[str]:
        return {str(s.args.get("type", "")).upper() for s in self.steps if s.action == "add_modifier"} - {""}

    def without_step(self, index: int) -> Recipe:
        """The recipe minus one step (a step that could not be made to work), noted in the summary."""
        steps = [s for i, s in enumerate(self.steps) if i != index]
        dropped = self.steps[index]
        note = f"[skipped: {dropped.action} {dropped.note or ''}]".strip()
        return self.model_copy(update={"steps": steps, "summary": f"{self.summary} {note}".strip()})

    def object_names(self) -> list[str]:
        """Objects the recipe creates or changes, in order of first mention."""
        names: list[str] = []
        for step in self.steps:
            for key in ("name", "object", "new_name"):
                value = step.args.get(key)
                if isinstance(value, str) and value not in names and not (key == "name" and step.action in (
                        "set_material", "add_modifier", "add_scatter")):
                    names.append(value)
        return names

    def compact(self, limit: int | None = None) -> str:
        """One line per step, for prompts."""
        lines = []
        for i, step in enumerate(self.steps[:limit]):
            args = json.dumps(step.args, separators=(",", ":"))
            lines.append(f"{i}. {step.action} {args}" + (f"  # {step.note}" if step.note else ""))
        if limit is not None and len(self.steps) > limit:
            lines.append(f"... ({len(self.steps) - limit} more steps)")
        return "\n".join(lines)


def recipe_schema(actions: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    """JSON schema for a model's recipe. ``args`` is a JSON object written as a string: argument sets differ
    per action, and a string keeps the schema small for constrained decoding; it is parsed and checked here."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "objects": {"type": "array", "items": {
                "type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"], "additionalProperties": False}},
            "steps": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(actions or RECIPE_ACTIONS)},
                    "args": {"type": "string", "description": "the action's arguments as a JSON object"},
                    "note": {"type": "string"},
                    "video_time": {"type": ["string", "null"]},
                },
                "required": ["action", "args", "note", "video_time"], "additionalProperties": False}},
            "expected_result": {"type": "string"},
        },
        "required": ["title", "summary", "objects", "steps", "expected_result"], "additionalProperties": False,
    }


def parse_recipe(data: dict[str, Any], *, allowed: tuple[str, ...] | list[str] | set[str] | None = None,
                 source: dict[str, Any] | None = None) -> tuple[Recipe, list[str]]:
    """A model's recipe -> Recipe. Steps with unreadable arguments or actions outside ``allowed`` are dropped
    and reported (the problems go back to the model when it revises the recipe)."""
    problems: list[str] = []
    steps: list[RecipeStep] = []
    for i, item in enumerate(data.get("steps") or []):
        action = str(item.get("action") or "")
        if action not in RECIPE_ACTIONS or (allowed is not None and action not in allowed):
            problems.append(f"step {i}: action {action!r} is not available")
            continue
        raw = item.get("args")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except json.JSONDecodeError as exc:
            problems.append(f"step {i} ({action}): args are not valid JSON ({exc.msg})")
            continue
        if not isinstance(args, dict):
            problems.append(f"step {i} ({action}): args must be a JSON object")
            continue
        steps.append(RecipeStep(action=action, args=args, note=str(item.get("note") or ""),
                                video_time=item.get("video_time")))
    objects = [{"name": str(o.get("name", "")), "description": str(o.get("description", ""))}
               for o in data.get("objects") or [] if isinstance(o, dict)]
    return Recipe(title=str(data.get("title") or "untitled"), summary=str(data.get("summary") or ""),
                  objects=objects, steps=steps, expected_result=str(data.get("expected_result") or ""),
                  source=source or {}), problems
