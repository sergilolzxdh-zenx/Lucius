"""The control center renders every view without script errors (real Chromium, real server)."""

from __future__ import annotations

import os
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

from lucius.app import Lucius
from lucius.api import create_app
from tests.fixtures.demos import sword_blockout_demo
from tests.test_e2e_learning import record

playwright = pytest.importorskip("playwright.sync_api")
CHROMIUM = next((p for p in (os.environ.get("LUCIUS_TEST_CHROMIUM"), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                             shutil.which("chromium"), shutil.which("chromium-browser")) if p and Path(p).exists()), None)
pytestmark = [pytest.mark.ui, pytest.mark.skipif(CHROMIUM is None, reason="needs a Chromium binary")]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    app = Lucius(data_dir=tmp_path_factory.mktemp("ui") / "data", background_processing=False)
    session_id = record(app, sword_blockout_demo(t0=1_700_000_000.0))
    record(app, sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False,
                                    t0=1_700_100_000.0))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(app, token="ui-token"), host="127.0.0.1", port=port,
                                        log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", app, session_id
    srv.should_exit = True
    thread.join(timeout=5)
    app.close()


def test_every_view_renders_without_errors(server):
    base, app, session_id = server
    failure_id = app.failures.list()[0].id
    routes = ["agent", "watch", f"sessions/{session_id}", "import", "skills", "skills/hard_surface_blade_blockout",
              "memory", "memory/Semantic", "memory/Preferences", "memory/Learning graph", "memory/Retrieval debug",
              "failures", f"failures/{failure_id}", "practice", "practice/hard_surface", "datasets", "benchmarks",
              "settings"]
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        for route in routes:
            page.goto(f"{base}/#/{route}")
            page.wait_for_function("document.querySelector('#view h1') !== null", timeout=10000)
            assert "Could not load" not in page.inner_text("#view"), route
        assert not errors, errors

        # A human edit through the timeline editor reaches the store.
        page.goto(f"{base}/#/sessions/{session_id}")
        page.wait_for_selector("text=Relabel")
        page.fill("#view .card input:not([type])", "Setup by hand")
        page.click("text=Relabel")
        page.wait_for_timeout(600)
        segment = app.segments.for_session(session_id)[0]
        assert segment.title == "Setup by hand" and segment.locked
        browser.close()
