"""Keyboard and mouse control of a live Blender: the ``gui`` execution layer.

Plans stay semantic (typed bridge actions). ``GuiBackend`` performs each action the way a person
would -- pointer over the 3D viewport, Blender's default shortcuts, numbers typed into the modal
operator -- whenever the default keymap expresses it *exactly* (``G Z 0.5 ⏎`` is exactly a 0.5
translation along Z). Everything else (region selections, adding modifiers) goes through the add-on,
which also observes: after every keystroke sequence the operator log must show the operator the
sequence meant, with the values that were typed. A mismatch is undone and, by default, the action
is performed through the add-on instead; the result records which path ran.

Safety:
* input is only sent while a Blender window is focused (or, without a window manager, under the pointer);
* the pointer never leaves the 3D viewport, and only allowlisted keys are pressed (no OS/quit/file shortcuts);
* every injected event is registered in the ``AgentActionLedger`` so recordings attribute it to the agent;
* if the pointer moves by itself between two injected events, a person is using the mouse: the
  actuator presses Esc (cancelling any modal operator) and stops with ``human_interference``.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from lucius.blender.bridge import BlenderBridge
from lucius.errors import BlenderBridgeError, LuciusError
from lucius.executor.backend import ActionResult, BridgeBackend
from lucius.logging_setup import get_logger
from lucius.planner.model import PlanAction
from lucius.recorder.ledger import AgentActionLedger, ExpectedInput
from lucius.recorder.window import Bounds

log = get_logger("executor.gui")

MODIFIER_KEYS = ("CTRL", "SHIFT", "ALT")
_LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)}
DIGIT_KEYS = {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE", "6": "SIX", "7": "SEVEN",
              "8": "EIGHT", "9": "NINE"}
TEXT_KEYS = {**DIGIT_KEYS, ".": "PERIOD", "-": "MINUS"}
NAMED_KEYS = {"TAB", "RET", "ESC", "DEL", "BACK_SPACE", "SPACE", "UP_ARROW", "DOWN_ARROW", "LEFT_ARROW", "RIGHT_ARROW",
              "HOME", "END", "PAGE_UP", "PAGE_DOWN", "PERIOD", "MINUS", "COMMA",
              *(f"F{i}" for i in range(1, 13)), *(f"NUMPAD_{i}" for i in range(10)), "NUMPAD_PERIOD",
              "NUMPAD_PLUS", "NUMPAD_MINUS", "NUMPAD_ASTERIX", "NUMPAD_SLASH", "NUMPAD_ENTER"}
ALLOWED_KEYS = _LETTERS | set(DIGIT_KEYS.values()) | NAMED_KEYS
BUTTONS = ("left", "right", "middle")


class GuiError(LuciusError):
    code = "gui_error"


class GuiInterference(GuiError):
    code = "human_interference"


# -- input injection ---------------------------------------------------------------------------------------

class InputInjector(Protocol):
    def press(self, key: str) -> None: ...

    def release(self, key: str) -> None: ...

    def move(self, x: int, y: int) -> None: ...

    def position(self) -> tuple[int, int]: ...

    def button(self, button: str, down: bool) -> None: ...

    def scroll(self, clicks: int) -> None: ...


# Numeric keypad codes pynput cannot name: X11 keysyms, Windows virtual keys, macOS virtual key codes.
_KEYPAD = {
    "linux": {**{f"NUMPAD_{i}": 0xFFB0 + i for i in range(10)}, "NUMPAD_PERIOD": 0xFFAE, "NUMPAD_PLUS": 0xFFAB,
              "NUMPAD_MINUS": 0xFFAD, "NUMPAD_ASTERIX": 0xFFAA, "NUMPAD_SLASH": 0xFFAF, "NUMPAD_ENTER": 0xFF8D},
    "win32": {**{f"NUMPAD_{i}": 0x60 + i for i in range(10)}, "NUMPAD_PERIOD": 0x6E, "NUMPAD_PLUS": 0x6B,
              "NUMPAD_MINUS": 0x6D, "NUMPAD_ASTERIX": 0x6A, "NUMPAD_SLASH": 0x6F},
    "darwin": {"NUMPAD_0": 0x52, "NUMPAD_1": 0x53, "NUMPAD_2": 0x54, "NUMPAD_3": 0x55, "NUMPAD_4": 0x56,
               "NUMPAD_5": 0x57, "NUMPAD_6": 0x58, "NUMPAD_7": 0x59, "NUMPAD_8": 0x5B, "NUMPAD_9": 0x5C,
               "NUMPAD_PERIOD": 0x41, "NUMPAD_PLUS": 0x45, "NUMPAD_MINUS": 0x4E, "NUMPAD_ASTERIX": 0x43,
               "NUMPAD_SLASH": 0x4B, "NUMPAD_ENTER": 0x4C},
}
_PYNPUT_NAMED = {"TAB": "tab", "RET": "enter", "ESC": "esc", "DEL": "delete", "BACK_SPACE": "backspace",
                 "SPACE": "space", "UP_ARROW": "up", "DOWN_ARROW": "down", "LEFT_ARROW": "left",
                 "RIGHT_ARROW": "right", "HOME": "home", "END": "end", "PAGE_UP": "page_up",
                 "PAGE_DOWN": "page_down", "CTRL": "ctrl", "SHIFT": "shift", "ALT": "alt",
                 **{f"F{i}": f"f{i}" for i in range(1, 13)}}
_PYNPUT_CHARS = {**{v: k for k, v in DIGIT_KEYS.items()}, "PERIOD": ".", "MINUS": "-", "COMMA": ","}


class PynputInjector:
    """Synthesised input through pynput: SendInput on Windows, Quartz events on macOS (which needs the
    Accessibility permission). On X11 use ``XTestInjector``."""

    def __init__(self) -> None:
        try:
            from pynput import keyboard, mouse
        except Exception as exc:  # ImportError, or no display to connect to
            raise GuiError(f"keyboard/mouse control unavailable: {exc}") from exc
        self._keyboard_mod = keyboard
        self._keyboard = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right, "middle": mouse.Button.middle}
        platform = "linux" if sys.platform.startswith("linux") else sys.platform
        self._keypad = _KEYPAD.get(platform, {})

    def _key(self, name: str) -> Any:
        keyboard = self._keyboard_mod
        if name in _PYNPUT_NAMED:
            return getattr(keyboard.Key, _PYNPUT_NAMED[name])
        if name in self._keypad:
            return keyboard.KeyCode.from_vk(self._keypad[name])
        if name in _PYNPUT_CHARS:
            return keyboard.KeyCode.from_char(_PYNPUT_CHARS[name])
        if len(name) == 1 and name in _LETTERS:
            return keyboard.KeyCode.from_char(name.lower())
        raise GuiError(f"no key mapping for {name}")

    def press(self, key: str) -> None:
        self._keyboard.press(self._key(key))

    def release(self, key: str) -> None:
        self._keyboard.release(self._key(key))

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (int(x), int(y))

    def position(self) -> tuple[int, int]:
        x, y = self._mouse.position
        return int(x), int(y)

    def button(self, button: str, down: bool) -> None:
        (self._mouse.press if down else self._mouse.release)(self._buttons[button])

    def scroll(self, clicks: int) -> None:
        self._mouse.scroll(0, clicks)


# X11 keysym names for Blender-style key names.
X11_KEYSYMS = {**{k: k.lower() for k in _LETTERS}, **{v: k for k, v in DIGIT_KEYS.items()},
               "PERIOD": "period", "MINUS": "minus", "COMMA": "comma", "TAB": "Tab", "RET": "Return",
               "ESC": "Escape", "DEL": "Delete", "BACK_SPACE": "BackSpace", "SPACE": "space", "UP_ARROW": "Up",
               "DOWN_ARROW": "Down", "LEFT_ARROW": "Left", "RIGHT_ARROW": "Right", "HOME": "Home", "END": "End",
               "PAGE_UP": "Prior", "PAGE_DOWN": "Next", "CTRL": "Control_L", "SHIFT": "Shift_L", "ALT": "Alt_L",
               **{f"F{i}": f"F{i}" for i in range(1, 13)}, **{f"NUMPAD_{i}": f"KP_{i}" for i in range(10)},
               "NUMPAD_PERIOD": "KP_Decimal", "NUMPAD_PLUS": "KP_Add", "NUMPAD_MINUS": "KP_Subtract",
               "NUMPAD_ASTERIX": "KP_Multiply", "NUMPAD_SLASH": "KP_Divide", "NUMPAD_ENTER": "KP_Enter"}
X11_BUTTONS = {"left": 1, "middle": 2, "right": 3}


class XTestInjector:
    """X11 input through the XTEST extension, using the keycodes of the current keyboard layout.

    pynput's X11 backend types characters by temporarily remapping a spare keycode, which Blender does
    not decode (and which briefly changes the user's keyboard mapping); this injector never remaps.
    """

    def __init__(self) -> None:
        try:
            from Xlib import X, XK, display
            from Xlib.ext import xtest
        except ImportError as exc:
            raise GuiError("python-xlib is not installed") from exc
        try:
            self._display = display.Display()
        except Exception as exc:
            raise GuiError(f"no X display: {exc}") from exc
        if not self._display.query_extension("XTEST"):
            raise GuiError("the X server has no XTEST extension")
        self._X, self._XK, self._xtest = X, XK, xtest
        self._codes: dict[str, int] = {}

    def _code(self, key: str) -> int:
        if key not in self._codes:
            name = X11_KEYSYMS.get(key)
            keysym = self._XK.string_to_keysym(name) if name else 0
            code = self._display.keysym_to_keycode(keysym) if keysym else 0
            if not code:
                raise GuiError(f"the keyboard layout has no key for {key}")
            self._codes[key] = code
        return self._codes[key]

    def _send(self, event_type: int, detail: int = 0, **kw: int) -> None:
        self._xtest.fake_input(self._display, event_type, detail, **kw)
        self._display.sync()

    def press(self, key: str) -> None:
        self._send(self._X.KeyPress, self._code(key))

    def release(self, key: str) -> None:
        self._send(self._X.KeyRelease, self._code(key))

    def move(self, x: int, y: int) -> None:
        self._send(self._X.MotionNotify, x=int(x), y=int(y))

    def position(self) -> tuple[int, int]:
        pointer = self._display.screen().root.query_pointer()
        return int(pointer.root_x), int(pointer.root_y)

    def button(self, button: str, down: bool) -> None:
        self._send(self._X.ButtonPress if down else self._X.ButtonRelease, X11_BUTTONS[button])

    def scroll(self, clicks: int) -> None:
        wheel = 4 if clicks > 0 else 5
        for _ in range(abs(int(clicks))):
            self._send(self._X.ButtonPress, wheel)
            self._send(self._X.ButtonRelease, wheel)


def create_injector() -> InputInjector:
    if sys.platform.startswith("linux"):
        return XTestInjector()
    return PynputInjector()


# -- where Blender is on screen -------------------------------------------------------------------------------

class WindowLocator(Protocol):
    def bounds(self) -> Bounds: ...

    def focused(self) -> bool: ...

    def activate(self) -> None: ...


class X11BlenderWindow:
    """Blender's top-level window on X11, found by the process id the add-on reports."""

    def __init__(self, pid: int | None) -> None:
        try:
            from Xlib import X, display
        except ImportError as exc:
            raise GuiError("python-xlib is not installed") from exc
        self._X = X
        self._display = display.Display()
        self._root = self._display.screen().root
        self._pid_atom = self._display.intern_atom("_NET_WM_PID")
        self._active_atom = self._display.intern_atom("_NET_ACTIVE_WINDOW")
        self.pid = pid
        self._window = None

    def _candidates(self):
        stack = [(self._root, 0)]
        while stack:
            window, depth = stack.pop()
            try:
                children = window.query_tree().children
            except Exception:
                continue
            for child in children:
                yield child
                if depth < 2:
                    stack.append((child, depth + 1))

    def _find(self):
        best, best_area = None, 0
        for window in self._candidates():
            try:
                attrs = window.get_attributes()
                if attrs.map_state != self._X.IsViewable:
                    continue
                pid = window.get_full_property(self._pid_atom, self._X.AnyPropertyType)
                pid = int(pid.value[0]) if pid is not None and len(pid.value) else None
                name = window.get_wm_name() or ""
                wm_class = window.get_wm_class() or ()
            except Exception:
                continue
            if self.pid is not None and pid is not None:
                matches = pid == self.pid
            else:
                matches = "blender" in (str(name) + " ".join(wm_class)).lower()
            if not matches:
                continue
            geometry = window.get_geometry()
            if geometry.width * geometry.height > best_area:
                best, best_area = window, geometry.width * geometry.height
        if best is None:
            raise GuiError("no Blender window found on this display")
        return best

    @property
    def window(self):
        if self._window is None:
            self._window = self._find()
        return self._window

    def bounds(self) -> Bounds:
        geometry = self.window.get_geometry()
        origin = self.window.translate_coords(self._root, 0, 0)
        return Bounds(x=-origin.x, y=-origin.y, width=geometry.width, height=geometry.height)

    def _is_ours(self, window) -> bool:
        target = self.window.id
        for _ in range(8):
            if window is None or isinstance(window, int) or window.id == self._root.id:
                return False
            if window.id == target:
                return True
            try:
                window = window.query_tree().parent
            except Exception:
                return False
        return False

    def focused(self) -> bool:
        active = self._root.get_full_property(self._active_atom, self._X.AnyPropertyType)
        if active is not None and len(active.value) and active.value[0]:
            return self._is_ours(self._display.create_resource_object("window", active.value[0]))
        focus = self._display.get_input_focus().focus
        if isinstance(focus, int) or focus == self._X.PointerRoot:
            # No window manager: keys go to the window under the pointer.
            child = self._root.query_pointer().child
            return self._is_ours(child) if child else False
        return self._is_ours(focus)

    def activate(self) -> None:
        from Xlib import protocol

        event = protocol.event.ClientMessage(window=self.window, client_type=self._active_atom,
                                             data=(32, [2, self._X.CurrentTime, 0, 0, 0]))
        self._root.send_event(event, event_mask=self._X.SubstructureRedirectMask | self._X.SubstructureNotifyMask)
        self._display.flush()


class ActiveWindowLocator:
    """Windows and macOS: the focused window, which must be Blender (activation is left to the user)."""

    def __init__(self) -> None:
        from lucius.recorder.window import create_window_provider

        self.provider = create_window_provider()

    def _blender(self):
        window = self.provider.active_window()
        name = f"{window.process or ''} {window.title or ''}".lower() if window else ""
        return window if window is not None and "blender" in name and window.bounds is not None else None

    def bounds(self) -> Bounds:
        window = self._blender()
        if window is None:
            raise GuiError("Blender is not the focused window")
        return window.bounds

    def focused(self) -> bool:
        return self._blender() is not None

    def activate(self) -> None:
        raise GuiError("focus Blender yourself before the agent acts (automatic activation is X11-only)")


def create_locator(pid: int | None) -> WindowLocator:
    if sys.platform.startswith("linux"):
        return X11BlenderWindow(pid)
    return ActiveWindowLocator()


@dataclass
class Screen:
    """Maps Blender window coordinates (origin bottom-left, window pixels) to screen coordinates."""

    layout: dict[str, Any]
    bounds: Bounds

    @property
    def window(self) -> dict[str, Any]:
        return self.layout["windows"][0]

    def to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = self.bounds.width / max(1, self.window["width"])
        sy = self.bounds.height / max(1, self.window["height"])
        return round(self.bounds.x + wx * sx), round(self.bounds.y + (self.window["height"] - wy) * sy)

    def view3d(self) -> dict[str, Any]:
        for area in self.window["areas"]:
            if area["type"] == "VIEW_3D":
                region = next((r for r in area["regions"] if r["type"] == "WINDOW"), area)
                return region
        raise GuiError("no 3D viewport is open in Blender's window")

    def view3d_point(self, fx: float = 0.5, fy: float = 0.5) -> tuple[int, int]:
        """A point of the viewport; fx, fy in 0..1 from its top-left corner (clamped inside)."""
        r = self.view3d()
        fx, fy = min(0.97, max(0.03, fx)), min(0.97, max(0.03, fy))
        return self.to_screen(r["x"] + fx * r["width"], r["y"] + (1.0 - fy) * r["height"])

    def inside_view3d(self, sx: int, sy: int) -> bool:
        r = self.view3d()
        x0, y1 = self.to_screen(r["x"], r["y"])
        x1, y0 = self.to_screen(r["x"] + r["width"], r["y"] + r["height"])
        return x0 <= sx <= x1 and y0 <= sy <= y1


# -- the actuator --------------------------------------------------------------------------------------------

def key_sequence(keys: list[str], text: str | None = None) -> list[dict[str, Any]]:
    """['CTRL+B'] + typed '0.1' -> structured key events (the format the safety validator checks)."""
    events: list[dict[str, Any]] = []
    for combo in keys:
        parts = combo.upper().split("+")
        events.append({"kind": "key", "key": parts[-1], "modifiers": parts[:-1]})
    if text:
        events.append({"kind": "text", "text": text})
    return events


def number(value: float) -> str:
    """A value as typed into a modal operator: plain decimal, no exponent."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


class GuiActuator:
    def __init__(self, injector: InputInjector, locator: WindowLocator, ledger: AgentActionLedger | None = None, *,
                 event_delay_s: float = 0.05, interference_px: float = 4.0, activate_window: bool = False) -> None:
        self.injector = injector
        self.locator = locator
        self.ledger = ledger
        self.event_delay_s = event_delay_s
        self.interference_px = interference_px
        self.activate_window = activate_window
        self.stop_requested = threading.Event()
        self._last_pointer: tuple[int, int] | None = None

    def stop(self) -> None:
        """Kill switch (UI, CLI, signal handler): the next event is not sent."""
        self.stop_requested.set()

    def ensure_focus(self) -> None:
        if self.locator.focused():
            return
        if self.activate_window:
            self.locator.activate()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if self.locator.focused():
                    return
                time.sleep(0.05)
        raise GuiError("Blender is not the focused window; the agent does not type into other applications")

    def perform(self, sequence: list[dict[str, Any]], screen: Screen, action_id: str) -> dict[str, Any]:
        expected = self._expected(sequence)
        # Interference is judged within one sequence: between actions Blender itself may re-place the
        # cursor (modal operators restore it), and every sequence starts by placing the pointer anyway.
        self._last_pointer = None
        if self.ledger is not None:
            self.ledger.begin(action_id, expected, duration_s=2.0 + len(sequence) * (self.event_delay_s + 0.05))
        sent = 0
        try:
            for event in sequence:
                if self.stop_requested.is_set():
                    raise GuiInterference("stopped by the user")
                self._check_pointer()
                self._dispatch(event, screen)
                sent += 1
                time.sleep(self.event_delay_s)
            self._check_pointer()
        except GuiInterference:
            self._cancel()
            raise
        finally:
            if self.ledger is not None:
                self.ledger.end(action_id)
        return {"events": sent}

    def _check_pointer(self) -> None:
        if self._last_pointer is None:
            return
        x, y = self.injector.position()
        if math.dist((x, y), self._last_pointer) > self.interference_px:
            raise GuiInterference("the pointer moved while the agent was acting: a person is using the mouse",
                                  expected=list(self._last_pointer), actual=[x, y])

    def _cancel(self) -> None:
        # Esc cancels any modal operator the sequence started; nothing is left half-applied.
        try:
            self.injector.press("ESC")
            self.injector.release("ESC")
        except Exception:
            log.warning("could not send Esc after interference")
        self._last_pointer = None

    def _move(self, x: int, y: int, screen: Screen) -> None:
        if not screen.inside_view3d(x, y):
            raise GuiError("pointer target outside the 3D viewport", x=x, y=y)
        self.injector.move(x, y)
        self._last_pointer = (x, y)

    def _dispatch(self, event: dict[str, Any], screen: Screen) -> None:
        kind = event["kind"]
        if kind == "pointer":
            if event.get("space") == "window":
                x, y = screen.to_screen(float(event["x"]), float(event["y"]))
            else:
                x, y = screen.view3d_point(float(event.get("x", 0.5)), float(event.get("y", 0.5)))
            self._move(x, y, screen)
        elif kind == "key":
            modifiers = [m.upper() for m in event.get("modifiers", [])]
            for modifier in modifiers:
                self.injector.press(modifier)
            self.injector.press(event["key"])
            self.injector.release(event["key"])
            for modifier in reversed(modifiers):
                self.injector.release(modifier)
        elif kind == "text":
            for char in str(event["text"]):
                key = TEXT_KEYS[char]
                self.injector.press(key)
                self.injector.release(key)
                time.sleep(self.event_delay_s / 2)
        elif kind == "click":
            modifiers = [m.upper() for m in event.get("modifiers", [])]
            for modifier in modifiers:
                self.injector.press(modifier)
            self.injector.button(event.get("button", "left"), True)
            self.injector.button(event.get("button", "left"), False)
            for modifier in reversed(modifiers):
                self.injector.release(modifier)
        elif kind == "drag":
            region = screen.view3d()
            start = self._last_pointer or screen.view3d_point()
            dx, dy = float(event.get("dx", 0.0)) * region["width"], float(event.get("dy", 0.0)) * region["height"]
            modifiers = [m.upper() for m in event.get("modifiers", [])]
            for modifier in modifiers:
                self.injector.press(modifier)
            self.injector.button(event.get("button", "middle"), True)
            steps = max(2, int(max(abs(dx), abs(dy)) // 12))
            for i in range(1, steps + 1):
                self._move(round(start[0] + dx * i / steps), round(start[1] + dy * i / steps), screen)
                time.sleep(0.01)
            self.injector.button(event.get("button", "middle"), False)
            for modifier in reversed(modifiers):
                self.injector.release(modifier)
        elif kind == "scroll":
            self.injector.scroll(int(event.get("clicks", 1)))
        elif kind == "wait":
            time.sleep(min(5.0, max(0.0, float(event.get("s", 0.1)))))
        else:
            raise GuiError(f"unknown GUI event {kind!r}")

    @staticmethod
    def _expected(sequence: list[dict[str, Any]]) -> list[ExpectedInput]:
        out: list[ExpectedInput] = []
        for event in sequence:
            if event["kind"] == "key":
                keys = [*[m.upper() for m in event.get("modifiers", [])], event["key"]]
                out += [ExpectedInput("key_down", key=k) for k in keys] + [ExpectedInput("key_up", key=k) for k in keys]
            elif event["kind"] == "text":
                for char in str(event["text"]):
                    out += [ExpectedInput("key_down", key=TEXT_KEYS[char]), ExpectedInput("key_up", key=TEXT_KEYS[char])]
            elif event["kind"] in ("click", "drag"):
                button = event.get("button", "left" if event["kind"] == "click" else "middle")
                out += [ExpectedInput("mouse_down", button=button), ExpectedInput("mouse_up", button=button)]
            elif event["kind"] == "scroll":
                out.append(ExpectedInput("scroll"))
        return out


# -- translating bridge actions into input -----------------------------------------------------------------

@dataclass
class Expectation:
    """What must appear in Blender's operator log (or state) after a sequence."""

    idname: str | None = None
    checks: list[tuple[str | None, str, Any]] = field(default_factory=list)   # (macro idname, property, value)
    state: dict[str, Any] = field(default_factory=dict)                        # e.g. {"mode": "EDIT_MESH"}
    tolerance: float = 1e-4


@dataclass
class GuiPlan:
    sequence: list[dict[str, Any]]
    expect: Expectation
    keys: list[str]
    select_first: str | None = None       # click this object before the keys
    # Events before a modal operator must be running. Axis letters and numbers are only sent once Blender
    # reports the modal: otherwise they would be shortcuts of their own (X deletes, Z opens a pie menu).
    modal_after: int = 0
    mode: str | None = None               # the mode the keys were chosen for


AXES = "xyz"
VIEW_KEYS = {"FRONT": "NUMPAD_1", "BACK": "CTRL+NUMPAD_1", "RIGHT": "NUMPAD_3", "LEFT": "CTRL+NUMPAD_3",
             "TOP": "NUMPAD_7", "BOTTOM": "CTRL+NUMPAD_7"}
ORBIT_KEYS = {"ORBITLEFT": "NUMPAD_4", "ORBITRIGHT": "NUMPAD_6", "ORBITUP": "NUMPAD_8", "ORBITDOWN": "NUMPAD_2"}
PIVOTS = {"median": "MEDIAN_POINT", "bbox_center": "BOUNDING_BOX_CENTER"}
# Scene-changing actions: a keyboard attempt that does not verify is rolled back from a snapshot.
MUTATING = {"extrude", "translate_selection", "scale_selection", "transform_object", "inset", "bevel",
            "delete_objects"}


def _single_axis(vec: list[float], neutral: float) -> tuple[str, float] | None:
    moved = [(AXES[i], float(v)) for i, v in enumerate(vec) if abs(float(v) - neutral) > 1e-9]
    return moved[0] if len(moved) == 1 else None


def translate(action: PlanAction, layout: dict[str, Any]) -> GuiPlan | None:
    """The exact keyboard form of a bridge action in Blender's default keymap, or None."""
    name, a = action.name, action.args
    mode = layout.get("mode") or ""
    obj = a.get("object")
    select_first = None
    if obj and obj != layout.get("active_object"):
        if mode != "OBJECT" or name == "select_objects":
            return None
        select_first = obj  # click it in the viewport first (object mode only)

    def plan(keys: list[str], text: str | None, expect: Expectation) -> GuiPlan:
        return GuiPlan(key_sequence(keys, text) + ([{"kind": "key", "key": "RET", "modifiers": []}] if text else []),
                       expect, keys + ([f"{text} RET"] if text else []), select_first,
                       modal_after=1 if text else 0, mode=mode)

    if name == "set_mode":
        target = {"OBJECT": "OBJECT", "EDIT": "EDIT_MESH"}.get(a.get("mode"))
        if target is None or mode not in ("OBJECT", "EDIT_MESH"):
            return None
        if mode == target:
            return GuiPlan([], Expectation(state={"mode": target}), [], select_first)
        return plan(["TAB"], None, Expectation("OBJECT_OT_editmode_toggle", state={"mode": target}))
    if name == "select_all":
        prefix = "MESH" if mode == "EDIT_MESH" else "OBJECT"
        select = a.get("action", "SELECT") == "SELECT"
        return plan(["A" if select else "ALT+A"], None,
                    Expectation(f"{prefix}_OT_select_all", [(None, "action", "SELECT" if select else "DESELECT")]))
    if name == "select_objects":
        names = [n for n in a.get("names", []) if isinstance(n, str)]
        if mode != "OBJECT" or len(names) != 1 or not a.get("deselect_others", True) or \
                (a.get("active") not in (None, names[0])):
            return None
        return GuiPlan([], Expectation("VIEW3D_OT_select", state={"active_object": names[0]}), ["click"],
                       select_first=names[0])
    if name in ("extrude", "translate_selection") and mode == "EDIT_MESH":
        axis = _single_axis(a.get("offset") or [0, 0, 0], 0.0)
        if axis is None:
            return None
        vec = [0.0, 0.0, 0.0]
        vec[AXES.index(axis[0])] = axis[1]
        if name == "extrude":
            # Extrusion starts constrained to the normal's Z: a first Z press only removes that constraint,
            # the second selects global Z. X and Y select their global axis at once (measured in Blender 5.0).
            keys = ["E", "Z", "Z"] if axis[0] == "z" else ["E", axis[0].upper()]
            return plan(keys, number(axis[1]), Expectation("MESH_OT_extrude_region_move", [
                ("TRANSFORM_OT_translate", "value", vec), ("TRANSFORM_OT_translate", "orient_type", "GLOBAL")]))
        return plan(["G", axis[0].upper()], number(axis[1]),
                    Expectation("TRANSFORM_OT_translate", [(None, "value", vec), (None, "orient_type", "GLOBAL")]))
    if name == "scale_selection" and mode == "EDIT_MESH":
        if PIVOTS.get(a.get("pivot", "median")) != layout.get("pivot_point"):
            return None
        factor = [float(v) for v in a.get("factor") or [1, 1, 1]]
        if max(factor) - min(factor) < 1e-9:
            return plan(["S"], number(factor[0]), Expectation("TRANSFORM_OT_resize", [(None, "value", factor)]))
        axis = _single_axis(factor, 1.0)
        if axis is None:
            return None
        return plan(["S", axis[0].upper()], number(axis[1]), Expectation("TRANSFORM_OT_resize", [(None, "value", factor)]))
    if name == "transform_object" and mode == "OBJECT" and a.get("relative"):
        given = {k: a.get(k) for k in ("location", "rotation", "scale") if a.get(k) is not None}
        if len(given) != 1:
            return None
        kind, vec = next(iter(given.items()))
        vec = [float(v) for v in vec]
        if kind == "scale" and max(vec) - min(vec) < 1e-9:
            return plan(["S"], number(vec[0]), Expectation("TRANSFORM_OT_resize", [(None, "value", vec)]))
        axis = _single_axis(vec, 1.0 if kind == "scale" else 0.0)
        if axis is None:
            return None
        if kind == "rotation":
            # Typed rotation is in degrees; Blender stores the operator value in radians (sign follows the view).
            return plan(["R", axis[0].upper()], number(math.degrees(axis[1])),
                        Expectation("TRANSFORM_OT_rotate", [(None, "orient_axis", axis[0].upper()),
                                                            (None, "value", axis[1])]))
        key = {"location": "G", "scale": "S"}[kind]
        idname = {"location": "TRANSFORM_OT_translate", "scale": "TRANSFORM_OT_resize"}[kind]
        return plan([key, axis[0].upper()], number(axis[1]), Expectation(idname, [(None, "value", vec)]))
    if name == "inset" and mode == "EDIT_MESH" and not a.get("depth"):
        # Operators remember their last options within a session; anything not typed is checked instead.
        return plan(["I"], number(a["thickness"]), Expectation("MESH_OT_inset", [
            (None, "thickness", float(a["thickness"])), (None, "depth", 0.0), (None, "use_individual", False),
            (None, "use_outset", False), (None, "use_relative_offset", False)]))
    if name == "bevel" and mode == "EDIT_MESH" and a.get("affect", "EDGES") == "EDGES":
        segments = int(a.get("segments", 1))
        # Typed segments even for 1: the bevel remembers the previous count (seen live: 3 after a 3-segment bevel).
        # Clamp overlap is not checked: Blender's Ctrl+B defaults to off, the add-on's bevel clamps; they differ
        # only where bevels would overlap.
        keys = key_sequence(["CTRL+B"], number(a["offset"])) + key_sequence(["S"], str(segments))
        keys.append({"kind": "key", "key": "RET", "modifiers": []})
        return GuiPlan(keys, Expectation("MESH_OT_bevel", [
            (None, "offset", float(a["offset"])), (None, "segments", segments), (None, "offset_type", "OFFSET"),
            (None, "affect", "EDGES"), (None, "profile", 0.5)]),
            ["CTRL+B", number(a["offset"]), "S", str(segments), "RET"], select_first, modal_after=1, mode=mode)
    if name == "set_view":
        view = str(a.get("view", "FRONT")).upper()
        if view not in VIEW_KEYS or (a.get("ortho", True) != bool(layout.get("auto_perspective", True))):
            return None
        return plan([VIEW_KEYS[view]], None, Expectation(state={"viewport.named_view": view}))
    if name == "orbit_view":
        step = math.radians(float(layout.get("orbit_step_deg", 15.0)))
        presses = float(a.get("angle", step)) / step
        if abs(presses - round(presses)) > 1e-3 or round(presses) < 1:
            return None
        return plan([ORBIT_KEYS[a["direction"]]] * round(presses), None, Expectation(state={"viewport.changed": True}))
    if name == "frame_selected":
        return plan(["NUMPAD_PERIOD"], None, Expectation(state={"viewport.changed": True}))
    if name in ("undo", "redo"):
        steps = int(a.get("steps", 1))
        return plan(["CTRL+Z" if name == "undo" else "CTRL+SHIFT+Z"] * steps, None, Expectation())
    if name == "delete_objects" and mode == "OBJECT":
        names = a.get("names") or []
        if names != [layout.get("active_object")]:
            return None
        return plan(["DEL"], None, Expectation("OBJECT_OT_delete"))
    return None


def _close(a: Any, b: Any, tolerance: float) -> bool:
    if isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and len(a) >= len(b) and all(_close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(b, float) or isinstance(a, float):
        try:
            return abs(float(a) - float(b)) <= tolerance * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


# -- the backend ---------------------------------------------------------------------------------------------

class GuiBackend(BridgeBackend):
    """A live Blender driven by keyboard and mouse, observed (and, as a fallback, driven) by the add-on."""

    name = "gui"

    def __init__(self, bridge: BlenderBridge, actuator: GuiActuator, *, fallback_to_bridge: bool = True,
                 verify_timeout_s: float = 5.0) -> None:
        super().__init__(bridge)
        if self.background:
            raise GuiError("keyboard/mouse control needs an interactive Blender, not a background one")
        self.environment = "blender_live_gui"
        self.gui_available = True
        self.actuator = actuator
        self.fallback_to_bridge = fallback_to_bridge
        self.verify_timeout_s = verify_timeout_s
        self.stats = {"gui": 0, "bridge": 0, "fallbacks": 0}
        self._prepared = False

    def layout(self) -> dict[str, Any]:
        return self.bridge.request("gui_layout")

    def screen(self, layout: dict[str, Any] | None = None) -> Screen:
        return Screen(layout or self.layout(), self.actuator.locator.bounds())

    def execute(self, action: PlanAction) -> ActionResult:
        started = time.monotonic()
        if action.layer == "gui":
            try:
                result = self._run(action.id, list(action.args.get("sequence", [])),
                                   Expectation(**action.args["expect"]) if action.args.get("expect") else Expectation(),
                                   select_first=None)
            except (GuiError, BlenderBridgeError) as exc:
                return ActionResult(ok=False, error=exc.to_dict(), duration_s=time.monotonic() - started)
            return ActionResult(ok=bool(result["verified"] is not False), result=result,
                                error=None if result["verified"] is not False else
                                {"code": "gui_unverified", "message": "the expected operator did not run"},
                                duration_s=time.monotonic() - started)
        if action.layer != "blender_api":
            return super().execute(action)
        try:
            layout = self.layout()
            plan = translate(action, layout)
        except BlenderBridgeError as exc:
            return ActionResult(ok=False, error=exc.to_dict(), duration_s=time.monotonic() - started)
        if plan is None:
            self.stats["bridge"] += 1
            outcome = super().execute(action)
            outcome.result = {**outcome.result, "via": "bridge", "gui": "no exact keyboard form"}
            return outcome
        snapshot = f"gui_{action.id}" if action.name in MUTATING else None
        try:
            if snapshot:
                self.bridge.execute("snapshot", {"tag": snapshot})
            result = self._run(action.id, plan.sequence, plan.expect, select_first=plan.select_first,
                               modal_after=plan.modal_after, mode=plan.mode)
            result["keys"] = plan.keys
        except GuiInterference as exc:
            # A person took the mouse: never fall back behind their back.
            return ActionResult(ok=False, error=exc.to_dict(), duration_s=time.monotonic() - started)
        except (GuiError, BlenderBridgeError) as exc:
            result = {"verified": False, "error": exc.to_dict(), "keys": plan.keys}
        if result.get("verified") is not False:
            self.stats["gui"] += 1
            return ActionResult(ok=True, result={**result, "via": "gui"}, duration_s=time.monotonic() - started)
        self._cancel_modal()
        if result.get("changed") and snapshot:
            # Put the scene back exactly (undo could also revert add-on edits that pushed no undo step).
            self.bridge.execute("restore", {"tag": snapshot})
            if layout.get("mode") == "EDIT_MESH" and action.args.get("object"):
                self.bridge.execute("set_mode", {"object": action.args["object"], "mode": "EDIT"})
        if not self.fallback_to_bridge:
            return ActionResult(ok=False, result=result, duration_s=time.monotonic() - started,
                                error={"code": "gui_unverified", "message": "the keyboard action did not verify",
                                       "details": result})
        self.stats["fallbacks"] += 1
        outcome = super().execute(action)
        outcome.result = {**outcome.result, "via": "bridge_fallback", "gui_attempt": result}
        return outcome

    def _perform(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Macros (``scale_to_size``) compute their action from live state; it is then typed like any other."""
        outcome = self.execute(PlanAction(id=f"macro_{time.monotonic_ns()}", layer="blender_api", name=name,
                                          args=args, action_type=name))
        if not outcome.ok:
            raise BlenderBridgeError((outcome.error or {}).get("message", f"{name} failed"),
                                     code=(outcome.error or {}).get("code", "gui_error"))
        return outcome.result

    def _select_by_click(self, name: str, screen: Screen, action_id: str) -> None:
        point = self.bridge.request("project", {"object": name})
        if not point.get("inside_region"):
            raise GuiError(f"{name} is not visible in the viewport")
        self.actuator.perform([{"kind": "pointer", "space": "window", "x": point["x"], "y": point["y"]},
                               {"kind": "click", "button": "left"}], screen, f"{action_id}:select")
        deadline = time.monotonic() + self.verify_timeout_s
        while time.monotonic() < deadline:
            if self.layout().get("active_object") == name:
                return
            time.sleep(0.05)
        raise GuiError(f"clicking did not make {name} active (another object is in front of it?)")

    def _run(self, action_id: str, sequence: list[dict[str, Any]], expect: Expectation, *, select_first: str | None,
             modal_after: int = 0, mode: str | None = None) -> dict[str, Any]:
        layout = self.layout()
        self.actuator.ensure_focus()
        screen = self.screen(layout)
        if not self._prepared:
            # An open popup (the splash screen, a menu left open) takes every key press; Esc closes it and
            # does nothing in the viewport otherwise.
            self.actuator.perform([{"kind": "pointer", "x": 0.5, "y": 0.5},
                                   {"kind": "key", "key": "ESC", "modifiers": []}], screen, f"{action_id}:prepare")
            time.sleep(0.3)
            self._prepared = True
        running = self._modal()
        if running:
            self._cancel_modal()
            if self._modal():
                raise GuiError(f"a modal operator is running in Blender ({', '.join(running)}); finish or cancel it")
        before = self.bridge.request("operator_log", {"limit": 16})["operators"]
        if select_first:
            self._select_by_click(select_first, screen, action_id)
        if mode is not None and self.layout().get("mode") != mode:
            raise GuiError(f"Blender is in {self.layout().get('mode')} mode; the keys were chosen for {mode}")
        before_view = self.bridge.get_state(include_objects=False).get("viewport") \
            if any(k.startswith("viewport.") for k in expect.state) else None
        pointer = [] if any(e["kind"] == "pointer" for e in sequence[:1]) else [{"kind": "pointer", "x": 0.5, "y": 0.5}]
        if not sequence:
            sent = {"events": 0}
        elif modal_after:
            sent = self.actuator.perform(pointer + sequence[:modal_after], screen, action_id)
            if not self._wait_modal(min(3.0, self.verify_timeout_s)):
                self._cancel_modal()
                raise GuiError("the operator did not start (no modal operator running); the remaining keys were "
                               "not sent", keys_sent=modal_after)
            # The pointer stays where it is: moving it now would move the geometry before the typed value.
            rest = self.actuator.perform(sequence[modal_after:], screen, f"{action_id}:modal")
            sent = {"events": sent["events"] + rest["events"]}
        else:
            sent = self.actuator.perform(pointer + sequence, screen, action_id)
        return {**sent, **self._verify(expect, before, before_view)}

    def _wait_modal(self, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._modal():
                return True
            time.sleep(0.03)
        return False

    def _verify(self, expect: Expectation, before: list[dict[str, Any]],
                before_view: dict[str, Any] | None) -> dict[str, Any]:
        if expect.idname is None and not expect.state:
            return {"verified": None, "changed": True}  # nothing observable to check (e.g. undo)
        seen = {e["id"]: e for e in before}
        deadline = time.monotonic() + self.verify_timeout_s
        found = None
        problems: list[str] = []
        while time.monotonic() < deadline:
            log_now = self.bridge.request("operator_log", {"limit": 16})["operators"]
            new = [e for e in log_now if e["id"] not in seen or e != seen[e["id"]]]
            found = next((e for e in reversed(new) if e["idname"] == expect.idname), None) if expect.idname else None
            state_ok, problems = self._state_ok(expect, before_view)
            if (expect.idname is None or found is not None) and state_ok:
                break
            time.sleep(0.05)
        else:
            missing = [f"{expect.idname} not seen (a menu or popup open in Blender takes key presses)"]
            return {"verified": False, "changed": bool(found), "problems": problems or missing}
        mismatches = []
        for macro, prop, value in expect.checks:
            props = found["properties"] if macro is None else next(
                (m["properties"] for m in found.get("macros", []) if m["idname"] == macro), {})
            if not _close(props.get(prop), value, expect.tolerance):
                mismatches.append(f"{macro or found['idname']}.{prop}={props.get(prop)!r}, wanted {value!r}")
        if mismatches:
            return {"verified": False, "changed": True, "operator": found["idname"], "problems": mismatches}
        return {"verified": True, "changed": True, **({"operator": found["idname"]} if found else {})}

    def _state_ok(self, expect: Expectation, before_view: dict[str, Any] | None) -> tuple[bool, list[str]]:
        if not expect.state:
            return True, []
        layout = self.layout()
        view = (self.bridge.get_state(include_objects=False).get("viewport") or {}) \
            if any(k.startswith("viewport.") for k in expect.state) else {}
        problems = []
        for key, wanted in expect.state.items():
            if key == "viewport.changed":
                actual = bool(before_view is not None and view and (view.get("rotation") != before_view.get("rotation")
                                                                   or view.get("location") != before_view.get("location")))
            elif key.startswith("viewport."):
                actual = view.get(key.split(".", 1)[1])
            else:
                actual = layout.get(key)
            same = actual.lower() == wanted.lower() if isinstance(actual, str) and isinstance(wanted, str) else \
                actual == wanted
            if not same:
                problems.append(f"{key}={actual!r}, wanted {wanted!r}")
        return not problems, problems

    def _modal(self) -> list[str]:
        return [m for w in self.layout().get("windows", []) for m in w.get("modal_operators", [])]

    def _cancel_modal(self) -> None:
        """Esc a modal operator a failed sequence may have left running (it would take every later key)."""
        try:
            if self._modal():
                self.actuator.injector.press("ESC")
                self.actuator.injector.release("ESC")
                time.sleep(0.2)
        except BlenderBridgeError as exc:
            log.warning("could not check for a running modal operator: %s", exc.message)
