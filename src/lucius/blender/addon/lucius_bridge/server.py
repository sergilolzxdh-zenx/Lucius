"""Loopback JSON-lines server exposing the bridge to Lucius.

Security model: binds to 127.0.0.1 only, every connection must authenticate with the shared
token before any other command, and only the allowlisted commands below exist -- there is no
code evaluation path.

``bpy`` is not thread-safe, so socket threads never touch it directly: calls go through a
dispatcher. Inside the Blender GUI the :class:`TimerDispatcher` runs them on the main thread
via ``bpy.app.timers``; in headless module/background mode the :class:`InlineDispatcher`
serialises them with a lock (there is no competing main loop).
"""

import hmac
import json
import queue
import socket
import threading
import time
import traceback

from .protocol import MAX_LINE_BYTES, PROTOCOL_VERSION, BridgeCommandError, encode


class InlineDispatcher:
    def __init__(self):
        self._lock = threading.Lock()

    def call(self, fn, *args, timeout=60.0):
        with self._lock:
            return fn(*args)


class TimerDispatcher:
    def __init__(self, interval=0.02):
        self._queue = queue.Queue()
        self.interval = interval

    def pump(self):
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            try:
                fn, args, box, done = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                box["result"] = fn(*args)
            except BaseException as exc:  # returned to the waiting socket thread
                box["error"] = exc
            finally:
                done.set()
        return self.interval

    def call(self, fn, *args, timeout=60.0):
        box, done = {}, threading.Event()
        self._queue.put((fn, args, box, done))
        if not done.wait(timeout):
            raise BridgeCommandError("timeout", "Blender main thread did not run the command in time")
        if "error" in box:
            raise box["error"]
        return box["result"]


class _Client:
    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.authenticated = False
        self.subscribed = False
        self.lock = threading.Lock()

    def send(self, message):
        data = encode(message)
        with self.lock:
            self.sock.sendall(data)


class BridgeServer:
    def __init__(self, token, dispatcher, commands, host="127.0.0.1", port=47821):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the bridge only listens on loopback interfaces")
        if not token or len(token) < 16:
            raise ValueError("a bridge token of at least 16 characters is required")
        self.token = token
        self.dispatcher = dispatcher
        self.commands = commands
        self.host = host
        self.port = port
        self._sock = None
        self._clients = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self.recording_indicator = False

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        threading.Thread(target=self._accept_loop, name="lucius-bridge-accept", daemon=True).start()
        return self.port

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.sock.close()
            except OSError:
                pass

    @property
    def client_count(self):
        with self._clients_lock:
            return len(self._clients)

    def push(self, kind, data):
        message = {"push": kind, "ts": time.time(), "data": data}
        with self._clients_lock:
            targets = [c for c in self._clients if c.subscribed]
        for client in targets:
            try:
                client.send(message)
            except OSError:
                self._drop(client)

    # -- connection handling -----------------------------------------------------------------
    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                sock, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client = _Client(sock, addr)
            with self._clients_lock:
                self._clients.append(client)
            threading.Thread(target=self._client_loop, args=(client,), name="lucius-bridge-client",
                             daemon=True).start()

    def _drop(self, client):
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)
        try:
            client.sock.close()
        except OSError:
            pass

    def _client_loop(self, client):
        buffer = b""
        try:
            while not self._stop.is_set():
                chunk = client.sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > MAX_LINE_BYTES:
                    break
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        if not self._handle_line(client, line):
                            return
        except OSError:
            pass
        finally:
            self._drop(client)

    def _handle_line(self, client, line):
        try:
            request = json.loads(line)
            req_id = request.get("id")
            cmd = request.get("cmd")
            args = request.get("args") or {}
        except (ValueError, AttributeError):
            client.send({"id": None, "ok": False, "error": {"code": "bad_request", "message": "invalid JSON"}})
            return False
        if not client.authenticated:
            token = str(request.get("token") or "")
            if cmd != "hello" or not hmac.compare_digest(token.encode(), self.token.encode()):
                client.send({"id": req_id, "ok": False,
                             "error": {"code": "unauthorized", "message": "authenticate with hello + token"}})
                return False
            client.authenticated = True
        try:
            if cmd == "subscribe":
                client.subscribed = bool(args.get("enabled", True))
                result = {"subscribed": client.subscribed}
            elif cmd == "set_recording":
                self.recording_indicator = bool(args.get("active"))
                result = {"recording": self.recording_indicator}
            elif cmd in self.commands:
                handler, main_thread = self.commands[cmd]
                if main_thread:
                    # A render or a heavy modifier may keep Blender's main thread busy for minutes.
                    action = args.get("action") if cmd == "execute" else None
                    slow = action in ("render_image", "apply_modifier", "add_scatter", "bind_to_armature", "bake_texture",
                                      "bind_modifier")
                    timeout = 7200.0 if action in ("render_animation", "bake_fluid") else 600.0 if slow else 60.0
                    result = self.dispatcher.call(handler, args, timeout=timeout)
                else:
                    result = handler(args)
            else:
                raise BridgeCommandError("unknown_command", f"unknown command {cmd!r}")
            client.send({"id": req_id, "ok": True, "result": result})
        except BridgeCommandError as exc:
            client.send({"id": req_id, "ok": False, "error": exc.to_dict()})
        except Exception as exc:
            client.send({"id": req_id, "ok": False, "error": {
                "code": "blender_error", "message": f"{type(exc).__name__}: {exc}",
                "details": {"traceback": traceback.format_exc(limit=5)}}})
        return True


def build_commands(capabilities):
    """Command table: name -> (handler(args), must_run_on_main_thread)."""
    from . import actions, gui, state, structure

    def hello(_args):
        import bpy

        return {
            "protocol": PROTOCOL_VERSION,
            "blender_version": bpy.app.version_string,
            "background": bool(bpy.app.background),
            "capabilities": capabilities(),
            "actions": actions.describe_actions(),
            "gui_only_actions": sorted(actions.GUI_ONLY),
        }

    return {
        "hello": (hello, True),
        "ping": (lambda _a: {"pong": time.time()}, False),
        "get_state": (lambda a: state.capture_state(include_objects=bool(a.get("include_objects", True))), True),
        "inspect_structure": (lambda a: structure.inspect_structure(
            names=a.get("names"), views=tuple(a.get("views") or ("front", "side", "top")),
            max_triangles=int(a.get("max_triangles", 20000))), True),
        "execute": (lambda a: actions.execute_action(a.get("action"), a.get("args") or {}), True),
        "scene_summary": (lambda _a: state.scene_summary(), True),
        # Read-only screen information for keyboard/mouse actuation (interactive sessions only).
        "gui_layout": (gui.gui_layout, True),
        "operator_log": (gui.operator_log, True),
        "project": (gui.project, True),
    }
