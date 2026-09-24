from __future__ import annotations

import importlib.util
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from lucius.config import load_config
from lucius.events import EventBus
from lucius.sessions import SessionStore
from lucius.storage import Database, FrameStore

HAS_BPY = importlib.util.find_spec("bpy") is not None or shutil.which("blender") is not None


@pytest.fixture
def config(tmp_path: Path):
    return load_config(tmp_path / "data")


@pytest.fixture
def db(config):
    database = Database(config.db_path)
    yield database
    database.close()


@pytest.fixture
def frames(config):
    return FrameStore(config.frames_dir)


@pytest.fixture
def sessions(db, frames):
    return SessionStore(db, frames)


@pytest.fixture
def bus(db):
    return EventBus(db)


def _free_display() -> int:
    for number in range(140, 200):
        if not Path(f"/tmp/.X11-unix/X{number}").exists() and not Path(f"/tmp/.X{number}-lock").exists():
            return number
    raise RuntimeError("no free X display number")


@pytest.fixture
def xvfb(monkeypatch):
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb not installed")
    number = _free_display()
    proc = subprocess.Popen(["Xvfb", f":{number}", "-screen", "0", "1280x800x24", "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while not Path(f"/tmp/.X11-unix/X{number}").exists():
        if time.time() > deadline or proc.poll() is not None:
            proc.kill()
            pytest.skip("Xvfb failed to start")
        time.sleep(0.05)
    monkeypatch.setenv("DISPLAY", f":{number}")
    yield f":{number}"
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def headless_blender():
    if not HAS_BPY:
        pytest.skip("Blender (bpy module or blender binary) not available")
    from lucius.blender.headless import HeadlessBlender

    blender = HeadlessBlender()
    bridge = blender.start()
    yield bridge
    blender.stop()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "headless_blender" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.bpy)
        if "xvfb" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.xvfb)
