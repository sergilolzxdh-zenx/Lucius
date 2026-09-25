"""Action safety boundary (sections 72-73).

All planned actions -- including anything a model proposed -- are untrusted until validated here.
Blender actions are restricted to the bridge allowlist; GUI actions to key sequences that stay
inside Blender (no quit/close/OS shortcuts, no typing into consoles); OS-level actions are
rejected unless explicitly enabled.
"""

from __future__ import annotations

import json
import os
from typing import Any

from lucius.config import SafetyConfig
from lucius.errors import ActionRejected
from lucius.planner.model import PlanAction

INTERNAL_ACTIONS = {"scale_to_size", "restore_snapshot", "snapshot"}
OBSERVATION_ACTIONS = {"observe_views"}
BLOCKED_KEY_COMBOS = [
    # the OS and window management
    {"ALT", "F4"}, {"CTRL", "Q"}, {"CTRL", "W"}, {"CTRL", "ALT", "DEL"}, {"CTRL", "ALT", "T"}, {"CTRL", "SHIFT", "ESC"},
    {"OSKEY"}, {"CTRL", "F4"}, {"ALT", "TAB"},
    # files (saving goes through the add-on's path allowlist), new/open discard the user's scene
    {"CTRL", "S"}, {"CTRL", "O"}, {"CTRL", "N"}, {"F4"},
    # editors that run or edit Python, preferences, rendering
    {"SHIFT", "F4"}, {"SHIFT", "F11"}, {"ALT", "P"}, {"CTRL", "COMMA"}, {"F12"},
]
ALLOWED_TEXT = set("0123456789.-")
GUI_MODIFIERS = {"CTRL", "SHIFT", "ALT"}
GUI_BUTTONS = {"left", "right", "middle"}


class ActionValidator:
    def __init__(self, config: SafetyConfig, *, bridge_actions: set[str], gui_only: set[str], background: bool,
                 output_dirs: list[str] | None = None) -> None:
        self.config = config
        self.output_dirs = list(output_dirs or [])   # Lucius' own output folders (renders, project files)
        self.bridge_actions = bridge_actions
        self.gui_only = gui_only
        self.background = background
        self.count = 0

    def validate(self, action: PlanAction) -> None:
        self.count += 1
        if self.count > self.config.max_actions_per_run:
            raise ActionRejected("action budget for this run exhausted", limit=self.config.max_actions_per_run)
        try:
            encoded = json.dumps(action.args)
        except (TypeError, ValueError) as exc:
            raise ActionRejected("action arguments are not plain data", action=action.name) from exc
        if len(encoded) > 20000:
            raise ActionRejected("action arguments too large", action=action.name)
        if action.layer == "internal":
            if action.name not in INTERNAL_ACTIONS:
                raise ActionRejected(f"unknown internal action {action.name}", action=action.name)
            return
        if action.layer == "observation":
            if action.name not in OBSERVATION_ACTIONS:
                raise ActionRejected(f"unknown observation {action.name}", action=action.name)
            return
        if action.layer == "blender_api":
            if action.name not in self.bridge_actions:
                raise ActionRejected(f"{action.name} is not an allowlisted Blender action", action=action.name)
            if action.name in self.gui_only and self.background:
                raise ActionRejected(f"{action.name} needs an interactive viewport", action=action.name)
            if action.name == "save_file":
                self._check_save_path(str(action.args.get("path", "")), (".blend",))
            elif action.name == "render_image":
                self._check_save_path(str(action.args.get("path", "")), (".png", ".jpg", ".jpeg"))
            return
        if action.layer == "gui":
            if not self.config.allow_gui_actions:
                raise ActionRejected("GUI actions are disabled by configuration")
            self._check_keys(action.args.get("sequence", []))
            return
        raise ActionRejected(f"layer {action.layer} is not permitted (OS actions are disabled)", layer=action.layer)

    def _check_save_path(self, path: str, extensions: tuple[str, ...]) -> None:
        allowed = [os.path.realpath(d) for d in [*self.config.allowed_save_dirs, *self.output_dirs]]
        real = os.path.realpath(path)
        if not path.lower().endswith(extensions) or not any(real == d or real.startswith(d + os.sep) for d in allowed):
            raise ActionRejected("save path outside the allowed directories", path=path)

    @staticmethod
    def _check_keys(sequence: list[Any]) -> None:
        from lucius.executor.gui import ALLOWED_KEYS

        if len(sequence) > 200:
            raise ActionRejected("GUI sequence too long")
        for item in sequence:
            if not isinstance(item, dict):
                raise ActionRejected("GUI sequence items must be structured key events")
            kind = item.get("kind")
            modifiers = {str(k).upper() for k in item.get("modifiers", [])}
            if not modifiers <= GUI_MODIFIERS:
                raise ActionRejected(f"modifier keys {sorted(modifiers - GUI_MODIFIERS)} not allowed")
            if kind == "key":
                key = str(item.get("key", "")).upper()
                if key not in ALLOWED_KEYS:
                    raise ActionRejected(f"key {key!r} is not allowlisted")
                combo = modifiers | {key}
                if any(block <= combo for block in BLOCKED_KEY_COMBOS):
                    raise ActionRejected(f"blocked key combination {sorted(combo)}")
            elif kind == "text":
                if not set(str(item.get("text", ""))) <= ALLOWED_TEXT:
                    raise ActionRejected("GUI text input is limited to numeric values")
            elif kind == "pointer":
                # Viewport-relative (0..1) by default; window coordinates come from the add-on's projection.
                # The actuator also refuses any point outside the 3D viewport.
                if item.get("space", "view3d") == "view3d" and not all(
                        0.0 <= float(item.get(axis, 0.5)) <= 1.0 for axis in ("x", "y")):
                    raise ActionRejected("pointer targets are fractions of the 3D viewport")
            elif kind in ("click", "drag"):
                if item.get("button", "left") not in GUI_BUTTONS:
                    raise ActionRejected(f"mouse button {item.get('button')!r} not allowed")
                if kind == "drag" and not all(abs(float(item.get(k, 0.0))) <= 1.0 for k in ("dx", "dy")):
                    raise ActionRejected("drags are limited to one viewport size")
            elif kind == "scroll":
                if abs(int(item.get("clicks", 0))) > 20:
                    raise ActionRejected("scrolling is limited to 20 clicks per event")
            elif kind == "wait":
                if not 0.0 <= float(item.get("s", 0.0)) <= 5.0:
                    raise ActionRejected("waits are limited to 5 s")
            else:
                raise ActionRejected(f"GUI event kind {kind!r} not allowed")
