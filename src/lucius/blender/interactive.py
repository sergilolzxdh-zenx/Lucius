"""Start an interactive (windowed) Blender with the Lucius Bridge, for keyboard/mouse control.

Normally you start Blender yourself with the add-on installed; this launcher is for tests and for
driving a private Blender on a virtual display (``Xvfb``). The bridge token goes through the
environment, never the command line.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from lucius.blender import ADDON_DIR
from lucius.blender.bridge import BlenderBridge
from lucius.errors import BlenderUnavailable

_BOOT = (
    "import sys, json, os; sys.path.insert(0, {addon_dir!r}); import bpy, lucius_bridge; "
    "lucius_bridge.register()\n"
    "def _start():\n"
    "    server = lucius_bridge.start_server(port=0, write_discovery=False)\n"
    "    tmp = {ready!r} + '.tmp'\n"
    "    open(tmp, 'w').write(json.dumps({{'port': server.port, 'pid': os.getpid()}}))\n"
    "    os.replace(tmp, {ready!r})\n"
    "bpy.app.timers.register(_start, first_interval=1.0)"
)


def blender_binary() -> str | None:
    """A Blender executable (``LUCIUS_BLENDER`` or ``blender`` on PATH); the ``bpy`` module has no UI."""
    return os.environ.get("LUCIUS_BLENDER") or shutil.which("blender")


class InteractiveBlender:
    def __init__(self, binary: str | None = None, *, display: str | None = None, startup_timeout: float = 120.0,
                 extra_args: list[str] | None = None) -> None:
        self.binary = binary or blender_binary()
        self.extra_args = list(extra_args or [])
        self.display = display
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.bridge: BlenderBridge | None = None
        self.pid: int | None = None
        self._workdir: Path | None = None

    def start(self) -> BlenderBridge:
        if not self.binary:
            raise BlenderUnavailable("no Blender executable: set LUCIUS_BLENDER or put blender on PATH")
        self._workdir = Path(tempfile.mkdtemp(prefix="lucius_gui_blender_"))
        ready = self._workdir / "ready.json"
        token = secrets.token_urlsafe(24)
        env = dict(os.environ, LUCIUS_BRIDGE_TOKEN=token)
        if self.display:
            env["DISPLAY"] = self.display
        env.pop("WAYLAND_DISPLAY", None)  # keyboard/mouse synthesis targets X11 windows
        log_handle = (self._workdir / "blender.log").open("wb")
        self.process = subprocess.Popen(
            [self.binary, "--factory-startup", *self.extra_args, "--python-expr",
             _BOOT.format(addon_dir=str(ADDON_DIR), ready=str(ready))],
            stdout=log_handle, stderr=subprocess.STDOUT, env=env, cwd=self._workdir)
        deadline = time.monotonic() + self.startup_timeout
        while not ready.exists():
            if self.process.poll() is not None:
                raise BlenderUnavailable("Blender exited during startup",
                                         log=(self._workdir / "blender.log").read_text(errors="replace")[-4000:])
            if time.monotonic() > deadline:
                self.stop()
                raise BlenderUnavailable("Blender did not become ready in time")
            time.sleep(0.1)
        info = json.loads(ready.read_text())
        self.pid = info["pid"]
        self.bridge = BlenderBridge("127.0.0.1", info["port"], token, request_timeout=60.0)
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

    def __enter__(self) -> InteractiveBlender:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
