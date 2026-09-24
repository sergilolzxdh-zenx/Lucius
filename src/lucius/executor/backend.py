"""Execution backends.

A backend executes validated :class:`PlanAction` s and exposes observations. ``BridgeBackend``
drives Blender through the Lucius Bridge add-on -- the same code path for the live GUI
(``environment='blender_live'``) and headless Blender (``'blender_headless'``). Layers stay
separate: Blender API actions go to the bridge, internal macros read live state first, and
observations never mutate the scene.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from lucius.blender.bridge import BlenderBridge
from lucius.blender.state import BlenderState
from lucius.errors import BlenderBridgeError
from lucius.planner.model import PlanAction


class ActionResult(BaseModel):
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    duration_s: float = 0.0


class ExecutionBackend(Protocol):
    name: str
    environment: str
    gui_available: bool
    bridge_actions: set[str]
    gui_only: set[str]
    background: bool

    def observe(self) -> BlenderState: ...

    def structure(self, names: list[str] | None = None) -> dict[str, Any]: ...

    def execute(self, action: PlanAction) -> ActionResult: ...

    def snapshot(self, tag: str) -> None: ...

    def restore(self, tag: str) -> None: ...

    def reset(self) -> None: ...


class BridgeBackend:
    name = "bridge"

    def __init__(self, bridge: BlenderBridge) -> None:
        if not bridge.connected:
            bridge.connect()
        self.bridge = bridge
        self.background = bridge.background
        self.environment = "blender_headless" if self.background else "blender_live"
        self.gui_available = not self.background
        self.bridge_actions = set((bridge.info.get("actions") or {}).keys())
        self.gui_only = set(bridge.info.get("gui_only_actions") or [])

    def observe(self) -> BlenderState:
        return self.bridge.get_state()

    def structure(self, names: list[str] | None = None) -> dict[str, Any]:
        return self.bridge.inspect_structure(names=names)

    def execute(self, action: PlanAction) -> ActionResult:
        started = time.monotonic()
        try:
            if action.layer == "blender_api":
                result = self.bridge.execute(action.name, action.args)
            elif action.layer == "internal":
                result = self._internal(action)
            elif action.layer == "observation":
                result = {"observed": action.args}
            else:
                return ActionResult(ok=False, error={"code": "unsupported_layer", "message": action.layer})
        except BlenderBridgeError as exc:
            return ActionResult(ok=False, error=exc.to_dict(), duration_s=time.monotonic() - started)
        return ActionResult(ok=True, result=result or {}, duration_s=time.monotonic() - started)

    def _internal(self, action: PlanAction) -> dict[str, Any]:
        args = action.args
        if action.name == "scale_to_size":
            obj, axis, size = args["object"], args["axis"], float(args["size"])
            info = next((o for o in self.structure([obj]).get("objects", []) if o.get("name") == obj), None)
            if info is None:
                raise BlenderBridgeError(f"object {obj} not found", code="object_not_found")
            current = float((info.get("mesh_size") or info.get("dimensions"))["xyz".index(axis)])
            if current <= 1e-9:
                raise BlenderBridgeError(f"{obj} has no extent along {axis}", code="degenerate")
            factor = size / current
            vec = [1.0, 1.0, 1.0]
            vec["xyz".index(axis)] = factor
            if args.get("space") == "edit":
                result = self._perform("scale_selection", {"object": obj, "factor": vec, "pivot": "median"})
            else:
                result = self._perform("transform_object", {"object": obj, "scale": vec, "relative": True})
            return {"factor": round(factor, 6), "from": current, "to": size, **result}
        if action.name == "restore_snapshot":
            return self.bridge.execute("restore", {"tag": args["tag"]})
        if action.name == "snapshot":
            return self.bridge.execute("snapshot", {"tag": args["tag"]})
        raise BlenderBridgeError(f"unknown internal action {action.name}", code="unknown_internal")

    def _perform(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A bridge action a macro decided on (the GUI backend types it instead when it can)."""
        return self.bridge.execute(name, args)

    def snapshot(self, tag: str) -> None:
        self.bridge.execute("snapshot", {"tag": tag})

    def restore(self, tag: str) -> None:
        self.bridge.execute("restore", {"tag": tag})

    def reset(self) -> None:
        self.bridge.execute("reset_scene", {"keep_camera_light": True})
