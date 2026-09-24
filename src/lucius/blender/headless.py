"""Headless Blender process management.

Starts the Lucius Bridge in a separate process -- either the ``bpy`` Python module or a
``blender -b`` binary -- so practice, skill validation and benchmark runs execute against a
real Blender without a GUI (10U: reproduction in a Blender test scene).
"""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lucius.blender import ADDON_DIR
from lucius.blender.bridge import BlenderBridge
from lucius.errors import BlenderUnavailable
from lucius.logging_setup import get_logger

log = get_logger("blender.headless")

_BOOT = (
    "import sys; sys.path.insert(0, {addon_dir!r}); import bpy; "
    "bpy.ops.wm.read_factory_settings(use_empty=False); "
    "import lucius_bridge; lucius_bridge.run_headless(port=0, ready_file={ready!r})"
)


def headless_available() -> str | None:
    """Return the backend that would be used ('bpy' or a blender path), or None."""
    if importlib.util.find_spec("bpy") is not None:
        return "bpy"
    return shutil.which("blender")


class HeadlessBlender:
    def __init__(self, *, allowed_save_dirs: list[str] | None = None,
                 allowed_read_dirs: list[str] | None = None, startup_timeout: float = 90.0) -> None:
        self.allowed_save_dirs = allowed_save_dirs or []
        self.allowed_read_dirs = allowed_read_dirs or []
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.bridge: BlenderBridge | None = None
        self._workdir: Path | None = None

    def start(self) -> BlenderBridge:
        backend = headless_available()
        if backend is None:
            raise BlenderUnavailable("headless Blender unavailable: install `bpy` or put `blender` on PATH")
        self._workdir = Path(tempfile.mkdtemp(prefix="lucius_blender_"))
        ready = self._workdir / "ready.json"
        token = secrets.token_urlsafe(24)
        code = _BOOT.format(addon_dir=str(ADDON_DIR), ready=str(ready))
        if backend == "bpy":
            cmd = [sys.executable, "-c", code]
        else:
            cmd = [backend, "-b", "--factory-startup", "--python-expr", code]
        env = dict(os.environ)
        env["LUCIUS_BRIDGE_TOKEN"] = token  # via environment, never argv
        env["LUCIUS_ALLOWED_SAVE_DIRS"] = os.pathsep.join(self.allowed_save_dirs)
        env["LUCIUS_ALLOWED_READ_DIRS"] = os.pathsep.join(self.allowed_read_dirs)
        log_path = self._workdir / "blender.log"
        log_handle = log_path.open("wb")
        self.process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env,
                                        cwd=self._workdir)
        deadline = time.monotonic() + self.startup_timeout
        while not ready.exists():
            if self.process.poll() is not None:
                raise BlenderUnavailable("headless Blender exited during startup",
                                         log=log_path.read_text(errors="replace")[-4000:])
            if time.monotonic() > deadline:
                self.stop()
                raise BlenderUnavailable("headless Blender did not become ready in time")
            time.sleep(0.05)
        port = json.loads(ready.read_text())["port"]
        self.bridge = BlenderBridge("127.0.0.1", port, token)
        self.bridge.connect()
        return self.bridge

    def stop(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def __enter__(self) -> BlenderBridge:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
