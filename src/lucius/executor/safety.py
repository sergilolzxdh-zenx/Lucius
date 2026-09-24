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
    {"ALT", "F4"}, {"CTRL", "Q"}, {"CTRL", "W"}, {"CTRL", "ALT", "DEL"}, {"CTRL", "ALT", "T"}, {"CTRL", "SHIFT", "ESC"},
    {"OSKEY"}, {"CTRL", "F4"}, {"ALT", "TAB"},
]
ALLOWED_TEXT = set("0123456789.-")


class ActionValidator:
    def __init__(self, config: SafetyConfig, *, bridge_actions: set[str], gui_only: set[str], background: bool) -> None:
        self.config = config
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
                self._check_save_path(str(action.args.get("path", "")))
            return
        if action.layer == "gui":
            if not self.config.allow_gui_actions:
                raise ActionRejected("GUI actions are disabled by configuration")
            self._check_keys(action.args.get("sequence", []))
            return
        raise ActionRejected(f"layer {action.layer} is not permitted (OS actions are disabled)", layer=action.layer)

    def _check_save_path(self, path: str) -> None:
        allowed = [os.path.realpath(d) for d in self.config.allowed_save_dirs]
        real = os.path.realpath(path)
        if not path.endswith(".blend") or not any(real == d or real.startswith(d + os.sep) for d in allowed):
            raise ActionRejected("save path outside the allowed directories", path=path)

    @staticmethod
    def _check_keys(sequence: list[Any]) -> None:
        for item in sequence:
            if not isinstance(item, dict):
                raise ActionRejected("GUI sequence items must be structured key events")
            kind = item.get("kind")
            if kind == "key":
                combo = {str(k).upper() for k in item.get("modifiers", [])} | {str(item.get("key", "")).upper()}
                if any(block <= combo for block in BLOCKED_KEY_COMBOS):
                    raise ActionRejected(f"blocked key combination {sorted(combo)}")
            elif kind == "text":
                if not set(str(item.get("text", ""))) <= ALLOWED_TEXT:
                    raise ActionRejected("GUI text input is limited to numeric values")
            elif kind not in ("wait",):
                raise ActionRejected(f"GUI event kind {kind!r} not allowed")
