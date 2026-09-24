"""Capture privacy policy.

Recording is scoped to Blender: frames are only grabbed and input only kept while an allowed
window is focused. Blocked title patterns (password managers...) always win.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lucius.config import PrivacyConfig
from lucius.recorder.window import WindowInfo


@dataclass(frozen=True)
class PrivacyDecision:
    allowed: bool
    is_blender: bool
    reason: str


class PrivacyFilter:
    def __init__(self, config: PrivacyConfig, *, capture_only_blender: bool = True) -> None:
        self.config = config
        self.capture_only_blender = capture_only_blender
        self._allowed_processes = {p.lower() for p in config.allowed_processes}
        self._allowed_titles = [re.compile(p) for p in config.allowed_title_patterns]
        self._blocked_titles = [re.compile(p) for p in config.blocked_title_patterns]

    def is_blender(self, window: WindowInfo | None) -> bool:
        if window is None:
            return False
        process = (window.process or "").lower()
        if process in self._allowed_processes or process.startswith("blender"):
            return True
        title = window.title or ""
        return any(p.search(title) for p in self._allowed_titles)

    def decide(self, window: WindowInfo | None) -> PrivacyDecision:
        if window is None:
            return PrivacyDecision(False, False, "no_active_window")
        title = window.title or ""
        if any(p.search(title) for p in self._blocked_titles):
            return PrivacyDecision(False, False, "blocked_title")
        blender = self.is_blender(window)
        if blender:
            return PrivacyDecision(True, True, "blender_window")
        if self.capture_only_blender:
            return PrivacyDecision(False, False, "not_blender")
        return PrivacyDecision(True, False, "capture_all_enabled")
