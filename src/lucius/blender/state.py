"""Typed, availability-aware view over a Blender state capture."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BlenderState(BaseModel):
    """A state capture. Absent information is listed in ``unavailable`` -- never guessed."""

    ts: float | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    geometry_changed: list[str] = Field(default_factory=list)

    @classmethod
    def from_capture(cls, data: dict[str, Any] | None) -> BlenderState | None:
        if not data:
            return None
        return cls(ts=data.get("ts"), values=data.get("values", {}), unavailable=data.get("unavailable", {}),
                   geometry_changed=data.get("geometry_changed", []))

    def get(self, key: str) -> Any:
        return self.values.get(key)

    @property
    def mode(self) -> str | None:
        return self.values.get("mode")

    @property
    def active_tool(self) -> str | None:
        return self.values.get("active_tool")

    @property
    def workspace(self) -> str | None:
        return self.values.get("workspace")

    @property
    def active_object(self) -> str | None:
        return self.values.get("active_object")

    @property
    def selected_objects(self) -> list[str]:
        return list(self.values.get("selected_objects") or [])

    @property
    def named_view(self) -> str | None:
        view = self.values.get("viewport") or {}
        return view.get("named_view")

    @property
    def perspective(self) -> str | None:
        view = self.values.get("viewport") or {}
        return view.get("perspective")

    @property
    def modifiers(self) -> list[str]:
        summary = self.values.get("active_object_summary") or {}
        return [m.get("type") for m in summary.get("modifiers", [])]

    def compact(self) -> dict[str, Any]:
        """Small, JSON-friendly summary stored alongside trajectory steps."""
        out = {k: self.values.get(k) for k in ("mode", "active_tool", "workspace", "active_object") if k in self.values}
        if self.selected_objects:
            out["selected"] = self.selected_objects[:20]
        if self.named_view:
            out["view"] = self.named_view
        if self.perspective:
            out["perspective"] = self.perspective
        summary = self.values.get("active_object_summary") or {}
        if summary.get("dimensions"):
            out["dimensions"] = summary["dimensions"]
        if summary.get("mesh"):
            out["mesh"] = {k: summary["mesh"][k] for k in ("verts", "faces") if k in summary["mesh"]}
        if summary.get("modifiers"):
            out["modifiers"] = [m.get("type") for m in summary["modifiers"]]
        if isinstance(self.values.get("edit_selection"), dict):
            out["edit_selection"] = self.values["edit_selection"]
        if self.geometry_changed:
            out["geometry_changed"] = self.geometry_changed
        return out
