from __future__ import annotations

import json
import os
import time

import pytest

from lucius.config import PrivacyConfig
from lucius.recorder import AgentActionLedger, DemonstrationRecorder, ExpectedInput, RecorderSources
from lucius.recorder.compression import MouseMoveCompressor
from lucius.recorder.keys import hotkey_label, normalize_key
from lucius.recorder.privacy import PrivacyFilter
from lucius.recorder.recorder import recover_interrupted_sessions
from lucius.recorder.window import WindowInfo
from lucius.provenance import DataPolicy
from lucius.sessions import CapturedEvent, EventKind, SessionKind, SessionStatus


def test_key_normalization():
    assert normalize_key(char="e") == "E"
    assert normalize_key(name="ctrl_l") == "CTRL"
    assert normalize_key(char="\x1a") == "Z"  # Ctrl+Z control character
    assert normalize_key(keypad_keysym=0xFFB1) == "NUMPAD_1"
    assert normalize_key(vk=0x63, system="win32") == "NUMPAD_3"
    # X11 keysym for 'e' equals VK_NUMPAD5 on Windows: must not be confused off Windows.
    assert normalize_key(char="e", vk=0x65, system="linux") == "E"
    assert normalize_key(char="1") == "ONE"
    assert normalize_key(name="f3") == "F3"
    assert hotkey_label("Z", {"SHIFT", "CTRL"}) == "CTRL+SHIFT+Z"


def test_privacy_filter_scopes_capture_to_blender():
    f = PrivacyFilter(PrivacyConfig())
    blender = WindowInfo(title="sword.blend - Blender 4.5", process="blender")
    assert f.decide(blender).allowed and f.decide(blender).is_blender
    assert not f.decide(WindowInfo(title="Inbox - Mail", process="thunderbird")).allowed
    assert not f.decide(WindowInfo(title="Blender password vault", process="keepass")).allowed
    assert not f.decide(None).allowed
    permissive = PrivacyFilter(PrivacyConfig(), capture_only_blender=False)
    assert permissive.decide(WindowInfo(title="Terminal", process="bash")).allowed


def test_mouse_compressor_keeps_turns_and_rate_limits():
    c = MouseMoveCompressor(max_hz=10, min_distance=5, turn_degrees=45)
    kept = [c.offer(0.0 + i * 0.001, float(i), 0.0) for i in range(100)]  # 100 moves in 0.1s
    assert kept[0] and sum(kept) < 10
    assert c.offer(0.101, 99.0, 30.0)  # sharp turn, far enough
    pending = c.flush_pending()
    assert pending is None or pending[1:] == (99.0, 30.0)


def test_ledger_attributes_injected_events_to_agent():
    ledger = AgentActionLedger()
    ledger.begin("a1", [ExpectedInput("key_down", key="E"), ExpectedInput("mouse_down", button="LEFT", x=10, y=10)])
    t = time.time()
    assert ledger.attribute(t, "key_down", {"key": "E"}) == "agent"
    assert ledger.attribute(t, "key_down", {"key": "E"}) == "human"  # expectation consumed
    assert ledger.attribute(t, "mouse_down", {"button": "LEFT", "x": 11, "y": 9}) == "agent"
    assert ledger.agent_active(t)
    ledger.end("a1")
    assert not ledger.agent_active(time.time() + 5)


def test_crash_recovery_replays_journal(config, sessions, bus):
    session = sessions.create(user_id="u", kind=SessionKind.LIVE_DEMO, policy=DataPolicy.for_live_demo())
    journal = config.journal_dir / f"{session.id}.jsonl"
    events = [CapturedEvent(seq=i, ts=session.start_time + i, kind=EventKind.KEY_DOWN, payload={"key": "G"})
              for i in range(4)]
    sessions.append_events(session.id, events[:2])  # only the first two reached SQLite
    with journal.open("w") as handle:
        for e in events:
            handle.write(json.dumps({"session_id": session.id, **e.model_dump(mode="json")}) + "\n")
        handle.write('{"truncated": ')  # crash mid-write
    (config.journal_dir / f"{session.id}.lock").write_text(json.dumps({"pid": 999999, "host": os.uname().nodename}))
    recovered = recover_interrupted_sessions(sessions, config.journal_dir, bus)
    assert recovered == [session.id]
    after = sessions.get(session.id)
    assert after.status == SessionStatus.INTERRUPTED
    assert [e.seq for e in sessions.events(session.id)] == [0, 1, 2, 3]
    assert after.end_time == pytest.approx(session.start_time + 3)
    assert after.meta["recovered"]["replayed_events"] == 2


# -- real capture under Xvfb ------------------------------------------------------------------------

class _XWindows:
    """Creates titled top-level windows on the test X server and switches focus between them."""

    def __init__(self):
        from Xlib import X, Xatom, display

        self.X = X
        self.d = display.Display()
        self.screen = self.d.screen()
        self.Xatom = Xatom

    def create(self, title: str, x: int, y: int, w: int, h: int, color: int):
        win = self.screen.root.create_window(x, y, w, h, 0, self.screen.root_depth, background_pixel=color)
        win.set_wm_name(title)
        win.change_property(self.d.intern_atom("_NET_WM_NAME"), self.d.intern_atom("UTF8_STRING"), 8,
                            title.encode())
        win.change_property(self.d.intern_atom("_NET_WM_PID"), self.Xatom.CARDINAL, 32, [os.getpid()])
        win.map()
        self.d.sync()
        return win

    def focus(self, win):
        win.raise_window()
        self.d.set_input_focus(win, self.X.RevertToParent, self.X.CurrentTime)
        self.d.sync()

    def key(self, keysym_name: str, modifiers: tuple[str, ...] = ()):
        from Xlib import XK
        from Xlib.ext import xtest

        codes = [self.d.keysym_to_keycode(XK.string_to_keysym(m)) for m in modifiers]
        code = self.d.keysym_to_keycode(XK.string_to_keysym(keysym_name))
        for c in codes:
            xtest.fake_input(self.d, self.X.KeyPress, c)
        xtest.fake_input(self.d, self.X.KeyPress, code)
        xtest.fake_input(self.d, self.X.KeyRelease, code)
        for c in reversed(codes):
            xtest.fake_input(self.d, self.X.KeyRelease, c)
        self.d.sync()

    def click(self, x: int, y: int, button: int = 1):
        from Xlib.ext import xtest

        xtest.fake_input(self.d, self.X.MotionNotify, x=x, y=y)
        xtest.fake_input(self.d, self.X.ButtonPress, button)
        xtest.fake_input(self.d, self.X.ButtonRelease, button)
        self.d.sync()

    def drag(self, x0: int, y0: int, x1: int, y1: int, button: int = 2, steps: int = 20):
        from Xlib.ext import xtest

        xtest.fake_input(self.d, self.X.MotionNotify, x=x0, y=y0)
        xtest.fake_input(self.d, self.X.ButtonPress, button)
        self.d.sync()
        for i in range(1, steps + 1):
            xtest.fake_input(self.d, self.X.MotionNotify, x=x0 + (x1 - x0) * i // steps, y=y0 + (y1 - y0) * i // steps)
            self.d.sync()
            time.sleep(0.01)
        xtest.fake_input(self.d, self.X.ButtonRelease, button)
        self.d.sync()


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_real_capture_under_xvfb(xvfb, config, sessions, bus):
    from lucius.recorder.sources import MssGrabber, PynputInputSource
    from lucius.recorder.window import X11WindowProvider

    xw = _XWindows()
    blender = xw.create("sword.blend - Blender 4.5", 0, 0, 900, 700, 0x334455)
    secret = xw.create("Password Manager", 910, 0, 360, 400, 0xAA2222)
    xw.focus(blender)

    config.recording.fps = 8
    provider = X11WindowProvider()
    sources = RecorderSources(window=provider, grabber_factory=MssGrabber, input=PynputInputSource())
    recorder = DemonstrationRecorder(config, sessions, bus, sources)
    session = recorder.start(task_text="blockout test")
    assert _wait_for(lambda: recorder.status()["blender_focused"])

    xw.key("e")
    xw.key("z", ("Control_L",))
    xw.drag(300, 300, 420, 360, button=2)
    xw.click(450, 320)
    xw.key("KP_1")
    time.sleep(0.6)

    xw.focus(secret)
    time.sleep(0.4)
    xw.key("s")
    xw.key("e")
    xw.key("c")
    time.sleep(0.4)
    xw.focus(blender)
    time.sleep(0.4)
    result = recorder.stop()

    assert result.status == SessionStatus.FINALIZED
    events = sessions.events(session.id)
    keys = [(e.payload.get("key"), e.payload.get("modifiers"), e.blender_active) for e in events
            if e.kind == EventKind.KEY_DOWN]
    assert ("E", [], True) in keys
    assert ("Z", ["CTRL"], True) in keys
    assert ("NUMPAD_1", [], True) in keys
    # Nothing typed into the password manager was kept.
    assert not any(k == "S" for k, _m, _b in keys)
    assert any(e.kind == EventKind.FOCUS_LOST for e in events)
    windows = [e.payload for e in events if e.kind == EventKind.WINDOW_CONTEXT]
    assert any(w.get("title", "").endswith("Blender 4.5") for w in windows)
    assert all("title" not in w for w in windows if not w["allowed"])
    downs = [e for e in events if e.kind == EventKind.MOUSE_DOWN]
    assert {d.payload["button"] for d in downs} >= {"MIDDLE", "LEFT"}
    moves = [e for e in events if e.kind == EventKind.MOUSE_MOVE]
    assert 2 <= len(moves) < 25  # drag path compressed
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    stored_frames = sessions.frames_for(session.id)
    assert stored_frames and all(f.width == 900 and f.height == 700 for f in stored_frames)
    assert result.environment.resolution == "1280x800"
    assert not (config.journal_dir / f"{session.id}.lock").exists()
