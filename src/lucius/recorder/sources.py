"""Capture sources: screen frames and raw input.

Both are protocols so the recorder can be driven by the real OS backends (``mss``, ``pynput``)
or by any other source (e.g. a VNC stream) without changes to the recorder itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from lucius.errors import CaptureUnavailable
from lucius.recorder.keys import normalize_key
from lucius.recorder.window import Bounds
from lucius.timeutil import now


@dataclass
class RawInput:
    kind: str                      # mouse_move, mouse_down, mouse_up, scroll, key_down, key_up
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)
    injected: bool | None = None   # set when the OS tells us the event was synthetic


class FrameGrabber(Protocol):
    def grab(self, region: Bounds | None) -> Image.Image: ...

    def screen_size(self) -> tuple[int, int]: ...

    def close(self) -> None: ...


class InputSource(Protocol):
    def start(self, callback: Callable[[RawInput], None]) -> None: ...

    def stop(self) -> None: ...


class MssGrabber:
    """Screen grabber backed by ``mss``. Create it on the thread that will use it."""

    def __init__(self) -> None:
        try:
            import mss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CaptureUnavailable("mss is not installed (pip install lucius[capture])") from exc
        factory = getattr(mss, "MSS", None) or mss.mss
        try:
            self._sct = factory()
        except Exception as exc:
            raise CaptureUnavailable(f"screen capture unavailable: {exc}") from exc
        monitor = self._sct.monitors[0]
        self._screen = Bounds(x=monitor["left"], y=monitor["top"], width=monitor["width"], height=monitor["height"])

    def screen_size(self) -> tuple[int, int]:
        return self._screen.width, self._screen.height

    def grab(self, region: Bounds | None) -> Image.Image:
        area = self._clip(region) if region is not None else self._screen
        shot = self._sct.grab({"left": area.x, "top": area.y, "width": area.width, "height": area.height})
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def _clip(self, region: Bounds) -> Bounds:
        s = self._screen
        x0, y0 = max(region.x, s.x), max(region.y, s.y)
        x1 = min(region.x + region.width, s.x + s.width)
        y1 = min(region.y + region.height, s.y + s.height)
        if x1 <= x0 or y1 <= y0:
            return s
        return Bounds(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

    def close(self) -> None:
        self._sct.close()


_BUTTONS = {"left": "LEFT", "right": "RIGHT", "middle": "MIDDLE", "x1": "BUTTON4", "x2": "BUTTON5"}


def _prompt_xorg_stop(base: Any) -> Any:
    """An X11 pynput listener class whose ``stop()`` takes effect immediately.

    pynput 1.8 sends the RECORD disable request on the recording connection, which is blocked
    reading and only flushes it when the next X event arrives: on a quiet display the listener
    keeps running after ``stop()``. Sending the request on the control connection and flushing it
    ends the recording at once, so no input is received after capture stops.
    """

    class PromptStopListener(base):  # type: ignore[misc, valid-type]
        def _stop_platform(self) -> None:
            if not hasattr(self, "_context"):
                self.wait()
            try:
                self._display_stop.record_disable_context(self._context)
                self._display_stop.flush()
            except Exception:  # connection already gone: the listener thread is ending anyway
                pass

    PromptStopListener.__name__ = base.__name__
    return PromptStopListener


class PynputInputSource:
    """Global mouse/keyboard listener. Only the recorder decides what is kept."""

    def __init__(self) -> None:
        try:
            from pynput import keyboard, mouse  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CaptureUnavailable(f"pynput unavailable: {exc}") from exc
        self._keyboard_mod = keyboard
        self._mouse_mod = mouse
        self._listeners: list[Any] = []

    @staticmethod
    def _injected(args: tuple[Any, ...]) -> bool | None:
        # pynput >= 1.8 passes an ``injected`` flag on platforms that report it.
        return bool(args[-1]) if args and isinstance(args[-1], bool) else None

    def _key_payload(self, key: Any, raw: dict[str, Any]) -> dict[str, Any]:
        name = getattr(key, "name", None)
        char = getattr(key, "char", None)
        vk = getattr(key, "vk", None)
        if name is None and vk is None and hasattr(key, "value"):
            vk = getattr(key.value, "vk", None)
        payload = {"key": normalize_key(name=name, char=char, vk=vk, keypad_keysym=raw.get("keypad_keysym")),
                   "char": char if char and char.isprintable() else None}
        if raw.get("keycode") is not None:
            payload["keycode"] = raw["keycode"]
        return payload

    def _keyboard_listener_class(self) -> Any:
        base = self._keyboard_mod.Listener
        if not base.__module__.endswith("_xorg"):
            return base

        class RawXorgListener(_prompt_xorg_stop(base)):  # type: ignore[misc]
            """Keeps the raw keycode and the NumLock-level keysym of each event (pynput drops them)."""

            raw: dict[str, Any] = {}

            def _handle_message(self, display: Any, event: Any, injected: Any) -> None:
                self.raw = {"keycode": event.detail, "keypad_keysym": display.keycode_to_keysym(event.detail, 1)}
                super()._handle_message(display, event, injected)

        return RawXorgListener

    def _mouse_listener_class(self) -> Any:
        base = self._mouse_mod.Listener
        return _prompt_xorg_stop(base) if base.__module__.endswith("_xorg") else base

    def start(self, callback: Callable[[RawInput], None]) -> None:
        def on_move(x: float, y: float, *args: Any) -> None:
            callback(RawInput("mouse_move", now(), {"x": x, "y": y}, self._injected(args)))

        def on_click(x: float, y: float, button: Any, pressed: bool, *args: Any) -> None:
            kind = "mouse_down" if pressed else "mouse_up"
            callback(RawInput(kind, now(), {"x": x, "y": y, "button": _BUTTONS.get(button.name, button.name.upper())},
                              self._injected(args)))

        def on_scroll(x: float, y: float, dx: float, dy: float, *args: Any) -> None:
            callback(RawInput("scroll", now(), {"x": x, "y": y, "dx": dx, "dy": dy}, self._injected(args)))

        key_listener: Any = None

        def raw() -> dict[str, Any]:
            return getattr(key_listener, "raw", {}) or {}

        def on_press(key: Any, *args: Any) -> None:
            callback(RawInput("key_down", now(), self._key_payload(key, raw()), self._injected(args)))

        def on_release(key: Any, *args: Any) -> None:
            callback(RawInput("key_up", now(), self._key_payload(key, raw()), self._injected(args)))

        mouse_listener = self._mouse_listener_class()(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        key_listener = self._keyboard_listener_class()(on_press=on_press, on_release=on_release)
        for listener in (mouse_listener, key_listener):
            listener.daemon = True
            listener.start()
            if hasattr(listener, "wait"):
                listener.wait()
            self._listeners.append(listener)

    def stop(self) -> None:
        for listener in self._listeners:
            listener.stop()
        for listener in self._listeners:
            listener.join(timeout=2.0)
        self._listeners.clear()
