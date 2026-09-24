"""Client for the Lucius Bridge add-on (see ``addon/lucius_bridge/protocol.py``)."""

from __future__ import annotations

import itertools
import json
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lucius.blender.state import BlenderState
from lucius.config import BlenderConfig
from lucius.errors import BlenderBridgeError, BlenderUnavailable
from lucius.logging_setup import get_logger

log = get_logger("blender.bridge")

PushHandler = Callable[[str, float, dict[str, Any]], None]


def discovery_file() -> Path:
    base = os.environ.get("LUCIUS_BRIDGE_DIR") or os.path.join(os.path.expanduser("~"), ".config", "lucius")
    return Path(base) / "bridge.json"


class BlenderBridge:
    """Thread-safe request/response client with a push-event subscription.

    One socket carries both responses (matched by request id) and pushed observations; a
    reader thread demultiplexes them.
    """

    def __init__(self, host: str, port: int, token: str, *, connect_timeout: float = 2.0,
                 request_timeout: float = 30.0) -> None:
        self.host = host
        self.port = port
        self._token = token
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._sock: socket.socket | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._push_handlers: list[PushHandler] = []
        self._reader: threading.Thread | None = None
        self._closed = threading.Event()
        self.info: dict[str, Any] = {}

    # -- construction ----------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: BlenderConfig) -> BlenderBridge:
        token, port = config.bridge_token, config.bridge_port
        if token is None:
            path = discovery_file()
            if not path.exists():
                raise BlenderUnavailable(
                    "no Blender bridge found: install and enable the Lucius Bridge add-on in Blender",
                    discovery_file=str(path),
                )
            data = json.loads(path.read_text())
            token, port = data["token"], data.get("port", port)
        return cls(config.bridge_host, port, token, connect_timeout=config.connect_timeout_s,
                   request_timeout=config.request_timeout_s)

    # -- connection ------------------------------------------------------------------------
    def connect(self) -> dict[str, Any]:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except OSError as exc:
            raise BlenderUnavailable(f"cannot reach Blender bridge at {self.host}:{self.port}: {exc}") from exc
        sock.settimeout(None)
        self._sock = sock
        self._closed.clear()
        self._reader = threading.Thread(target=self._read_loop, name="lucius-bridge-reader", daemon=True)
        self._reader.start()
        self.info = self._request("hello", {}, token=self._token)
        return self.info

    @property
    def connected(self) -> bool:
        return self._sock is not None and not self._closed.is_set()

    @property
    def capabilities(self) -> dict[str, bool]:
        return dict(self.info.get("capabilities", {}))

    @property
    def background(self) -> bool:
        return bool(self.info.get("background"))

    def close(self) -> None:
        self._closed.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        with self._pending_lock:
            for event, box in self._pending.values():
                box["error"] = {"code": "disconnected", "message": "bridge connection closed"}
                event.set()
            self._pending.clear()

    def __enter__(self) -> BlenderBridge:
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport -------------------------------------------------------------------------
    def _read_loop(self) -> None:
        buffer = b""
        sock = self._sock
        try:
            while sock is not None and not self._closed.is_set():
                chunk = sock.recv(1 << 16)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        self._dispatch(json.loads(line))
        except (OSError, ValueError) as exc:
            if not self._closed.is_set():
                log.warning("bridge reader stopped: %s", exc)
        finally:
            self.close()

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "push" in message:
            for handler in list(self._push_handlers):
                try:
                    handler(message["push"], float(message.get("ts", 0.0)), message.get("data") or {})
                except Exception:
                    log.exception("bridge push handler failed")
            return
        req_id = message.get("id")
        with self._pending_lock:
            entry = self._pending.pop(req_id, None)
        if entry is not None:
            event, box = entry
            box.update(message)
            event.set()

    def _request(self, cmd: str, args: dict[str, Any], *, token: str | None = None,
                 timeout: float | None = None) -> Any:
        if self._sock is None:
            raise BlenderUnavailable("bridge not connected")
        req_id = next(self._ids)
        event, box = threading.Event(), {}
        with self._pending_lock:
            self._pending[req_id] = (event, box)
        message: dict[str, Any] = {"id": req_id, "cmd": cmd, "args": args}
        if token is not None:
            message["token"] = token
        data = (json.dumps(message) + "\n").encode()
        try:
            with self._send_lock:
                self._sock.sendall(data)
        except OSError as exc:
            self.close()
            raise BlenderUnavailable(f"bridge send failed: {exc}") from exc
        if not event.wait(timeout or self.request_timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise BlenderBridgeError(f"bridge request {cmd} timed out", code="timeout", cmd=cmd)
        if not box.get("ok"):
            error = box.get("error") or {}
            raise BlenderBridgeError(error.get("message", "bridge error"), code=error.get("code", "bridge_error"),
                                     cmd=cmd, **(error.get("details") or {}))
        return box.get("result")

    # -- API -------------------------------------------------------------------------------
    def request(self, cmd: str, args: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        return self._request(cmd, args or {}, timeout=timeout)

    def get_state(self, include_objects: bool = True) -> BlenderState:
        state = BlenderState.from_capture(self._request("get_state", {"include_objects": include_objects}))
        assert state is not None
        return state

    def inspect_structure(self, names: list[str] | None = None, views: tuple[str, ...] = ("front", "side", "top"),
                          max_triangles: int = 20000) -> dict[str, Any]:
        return self._request("inspect_structure", {"names": names, "views": list(views),
                                                   "max_triangles": max_triangles})

    def execute(self, action: str, args: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
        return self._request("execute", {"action": action, "args": args or {}}, timeout=timeout)

    def subscribe(self, handler: PushHandler) -> None:
        self._push_handlers.append(handler)
        self._request("subscribe", {"enabled": True})

    def unsubscribe(self, handler: PushHandler) -> None:
        if handler in self._push_handlers:
            self._push_handlers.remove(handler)
        if not self._push_handlers and self.connected:
            self._request("subscribe", {"enabled": False})

    def set_recording_indicator(self, active: bool) -> None:
        self._request("set_recording", {"active": active})
