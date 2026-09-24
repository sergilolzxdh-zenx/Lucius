"""Active-window context (application, process, title, bounds, monitor, DPI).

Platform backends are imported lazily: importing Lucius never requires a display. Each backend
reports only what the platform exposes; anything it cannot determine is ``None``.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Protocol

from pydantic import BaseModel, Field

from lucius.errors import CaptureUnavailable


class Bounds(BaseModel):
    x: int
    y: int
    width: int
    height: int

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


class MonitorInfo(BaseModel):
    index: int
    x: int
    y: int
    width: int
    height: int
    primary: bool | None = None
    dpi_scale: float | None = None


class WindowInfo(BaseModel):
    window_id: str | None = None
    title: str | None = None
    process: str | None = None
    pid: int | None = None
    bounds: Bounds | None = None
    active: bool = True
    monitor: int | None = None
    extras: dict[str, str] = Field(default_factory=dict)


class WindowContextProvider(Protocol):
    name: str

    def active_window(self) -> WindowInfo | None: ...

    def monitors(self) -> list[MonitorInfo]: ...

    def dpi_scale(self) -> float | None: ...


def monitor_for(bounds: Bounds | None, monitors: list[MonitorInfo]) -> int | None:
    if bounds is None:
        return None
    cx, cy = bounds.x + bounds.width / 2, bounds.y + bounds.height / 2
    for mon in monitors:
        if mon.x <= cx < mon.x + mon.width and mon.y <= cy < mon.y + mon.height:
            return mon.index
    return None


def _process_name(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        with open(f"/proc/{pid}/comm") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


class X11WindowProvider:
    name = "x11"

    def __init__(self) -> None:
        try:
            from Xlib import X, display  # type: ignore[import-not-found]
            from Xlib.error import XError  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CaptureUnavailable("python-xlib is not installed") from exc
        try:
            self._display = display.Display()
        except Exception as exc:  # DisplayNameError, ConnectionClosedError...
            raise CaptureUnavailable(f"no X display: {exc}") from exc
        self._X = X
        self._XError = XError
        self._root = self._display.screen().root
        self._atoms = {name: self._display.intern_atom(name) for name in (
            "_NET_ACTIVE_WINDOW", "_NET_WM_NAME", "_NET_WM_PID", "UTF8_STRING", "WM_NAME")}

    def _prop(self, window, atom_name, prop_type):
        prop = window.get_full_property(self._atoms[atom_name], prop_type)
        return None if prop is None else prop.value

    def _active_id(self):
        try:
            value = self._prop(self._root, "_NET_ACTIVE_WINDOW", self._X.AnyPropertyType)
            if value is not None and len(value) and value[0]:
                return self._display.create_resource_object("window", value[0])
        except self._XError:
            pass
        # No EWMH window manager: fall back to the input focus, walking up to a named window.
        focus = self._display.get_input_focus().focus
        if isinstance(focus, int):
            return None
        return focus

    def _named_ancestor(self, window):
        current = window
        for _ in range(8):
            try:
                if current.get_wm_name() or self._prop(current, "_NET_WM_NAME", self._atoms["UTF8_STRING"]):
                    return current
                tree = current.query_tree()
            except self._XError:
                return window
            if tree.parent == 0 or tree.parent == self._root:
                return current
            current = tree.parent
        return current

    def active_window(self) -> WindowInfo | None:
        try:
            window = self._active_id()
            if window is None:
                return None
            window = self._named_ancestor(window)
            name = self._prop(window, "_NET_WM_NAME", self._atoms["UTF8_STRING"])
            title = name.decode("utf-8", "replace") if isinstance(name, bytes) else name
            if not title:
                title = window.get_wm_name()
                if isinstance(title, bytes):
                    title = title.decode("utf-8", "replace")
            pid_value = self._prop(window, "_NET_WM_PID", self._X.AnyPropertyType)
            pid = int(pid_value[0]) if pid_value is not None and len(pid_value) else None
            geom = window.get_geometry()
            coords = window.translate_coords(self._root, 0, 0)
            bounds = Bounds(x=-coords.x, y=-coords.y, width=geom.width, height=geom.height)
            wm_class = window.get_wm_class()
            process = _process_name(pid) or (wm_class[1] if wm_class else None)
            return WindowInfo(window_id=hex(window.id), title=title, process=process, pid=pid, bounds=bounds,
                              monitor=monitor_for(bounds, self.monitors()),
                              extras={"wm_class": ".".join(wm_class)} if wm_class else {})
        except self._XError:
            return None

    def monitors(self) -> list[MonitorInfo]:
        geom = self._root.get_geometry()
        return [MonitorInfo(index=0, x=0, y=0, width=geom.width, height=geom.height, primary=True,
                            dpi_scale=self.dpi_scale())]

    def dpi_scale(self) -> float | None:
        try:
            resources = self._root.get_full_property(self._display.intern_atom("RESOURCE_MANAGER"),
                                                     self._X.AnyPropertyType)
            if resources is not None:
                text = resources.value.decode() if isinstance(resources.value, bytes) else str(resources.value)
                for line in text.splitlines():
                    if line.startswith("Xft.dpi:"):
                        return round(float(line.split(":", 1)[1]) / 96.0, 3)
        except Exception:
            return None
        return None


class Win32WindowProvider:
    name = "win32"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise CaptureUnavailable("win32 provider requires Windows")
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        try:
            self._user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
        except (AttributeError, OSError):
            pass

    def active_window(self) -> WindowInfo | None:
        ctypes, wintypes, user32 = self._ctypes, self._wintypes, self._user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process = None
        handle = self._kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(1024)
            if self._kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                process = os.path.basename(path.value)
            self._kernel32.CloseHandle(handle)
        bounds = Bounds(x=rect.left, y=rect.top, width=rect.right - rect.left, height=rect.bottom - rect.top)
        return WindowInfo(window_id=hex(hwnd), title=buffer.value, process=process, pid=pid.value, bounds=bounds,
                          monitor=monitor_for(bounds, self.monitors()))

    def monitors(self) -> list[MonitorInfo]:
        user32 = self._user32
        return [MonitorInfo(index=0, x=user32.GetSystemMetrics(76), y=user32.GetSystemMetrics(77),
                            width=user32.GetSystemMetrics(78), height=user32.GetSystemMetrics(79),
                            dpi_scale=self.dpi_scale())]

    def dpi_scale(self) -> float | None:
        try:
            return round(self._user32.GetDpiForSystem() / 96.0, 3)
        except AttributeError:
            return None


class QuartzWindowProvider:
    name = "quartz"

    def __init__(self) -> None:
        try:
            import Quartz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CaptureUnavailable("pyobjc-framework-Quartz is required on macOS") from exc
        self._Q = Quartz

    def active_window(self) -> WindowInfo | None:
        Q = self._Q
        windows = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID)
        for info in windows or []:
            if info.get("kCGWindowLayer", 1) != 0:
                continue
            b = info.get("kCGWindowBounds", {})
            bounds = Bounds(x=int(b.get("X", 0)), y=int(b.get("Y", 0)), width=int(b.get("Width", 0)),
                            height=int(b.get("Height", 0)))
            return WindowInfo(window_id=str(info.get("kCGWindowNumber")), title=info.get("kCGWindowName"),
                              process=info.get("kCGWindowOwnerName"), pid=info.get("kCGWindowOwnerPID"),
                              bounds=bounds)
        return None

    def monitors(self) -> list[MonitorInfo]:
        Q = self._Q
        bounds = Q.CGDisplayBounds(Q.CGMainDisplayID())
        return [MonitorInfo(index=0, x=int(bounds.origin.x), y=int(bounds.origin.y),
                            width=int(bounds.size.width), height=int(bounds.size.height), primary=True)]

    def dpi_scale(self) -> float | None:
        return None


def create_window_provider() -> WindowContextProvider:
    if sys.platform == "win32":
        return Win32WindowProvider()
    if sys.platform == "darwin":
        return QuartzWindowProvider()
    if os.environ.get("DISPLAY"):
        return X11WindowProvider()
    raise CaptureUnavailable("no supported window system (X11 DISPLAY, Windows or macOS required)")


def os_version() -> str:
    return platform.platform()
