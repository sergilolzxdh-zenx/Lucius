"""Lucius Bridge: the Blender side of the Lucius agent.

Installed as a regular Blender add-on it (1) serves Blender state and allowlisted actions to
Lucius over a loopback socket and (2) pushes observed operators, undo/redo and state changes
while Lucius is recording a demonstration. It also shows a visible recording indicator in
Blender's status bar whenever Lucius is recording.

For headless use (practice, validation, tests) :func:`run_headless` starts the same server
without any UI, from ``blender -b`` or from the ``bpy`` Python module.
"""

bl_info = {
    "name": "Lucius Bridge",
    "author": "Lucius",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Lucius",
    "description": "Connects Blender to the Lucius learning agent (state capture, observed operators, safe actions)",
    "category": "System",
}

import json
import os
import secrets
import threading

import bpy
from bpy.app.handlers import persistent

from . import state
from .server import BridgeServer, InlineDispatcher, TimerDispatcher, build_commands
from .watch import ActivityWatcher

_runtime = {"server": None, "watcher": None, "dispatcher": None, "poll_hz": 4.0}


def discovery_path():
    base = os.environ.get("LUCIUS_BRIDGE_DIR") or os.path.join(os.path.expanduser("~"), ".config", "lucius")
    return os.path.join(base, "bridge.json")


def _write_discovery(port, token):
    path = discovery_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"host": "127.0.0.1", "port": port, "token": token, "pid": os.getpid(),
                   "blender_version": bpy.app.version_string}, handle)


def _capabilities():
    gui = not bpy.app.background
    return {"gui": gui, "undo": gui, "view_control": gui, "operator_log": gui, "snapshots": True,
            "structure": True}


def start_server(port=47821, token=None, dispatcher=None, write_discovery=True):
    if _runtime["server"] is not None:
        return _runtime["server"]
    token = token or os.environ.get("LUCIUS_BRIDGE_TOKEN") or secrets.token_urlsafe(24)
    dispatcher = dispatcher or (InlineDispatcher() if bpy.app.background else TimerDispatcher())
    server = BridgeServer(token, dispatcher, build_commands(_capabilities), port=port)
    server.start()
    watcher = ActivityWatcher(server.push, lambda: state.capture_state(include_objects=True),
                              state.light_state_key)
    _runtime.update(server=server, watcher=watcher, dispatcher=dispatcher)
    if isinstance(dispatcher, TimerDispatcher):
        bpy.app.timers.register(dispatcher.pump, persistent=True)
        bpy.app.timers.register(_poll_activity, first_interval=0.5, persistent=True)
    if write_discovery:
        _write_discovery(server.port, token)
    return server


def stop_server():
    server = _runtime["server"]
    if server is None:
        return
    server.stop()
    dispatcher = _runtime["dispatcher"]
    if isinstance(dispatcher, TimerDispatcher) and bpy.app.timers.is_registered(dispatcher.pump):
        bpy.app.timers.unregister(dispatcher.pump)
    if bpy.app.timers.is_registered(_poll_activity):
        bpy.app.timers.unregister(_poll_activity)
    _runtime.update(server=None, watcher=None, dispatcher=None)


def _poll_activity():
    server, watcher = _runtime["server"], _runtime["watcher"]
    if server is None or watcher is None:
        return None
    if server.client_count:
        try:
            watcher.poll(bpy.context.window_manager)
        except Exception as exc:  # never let observation break the user's Blender session
            print(f"[lucius] activity poll failed: {exc}")
    return 1.0 / _runtime["poll_hz"]


@persistent
def _on_undo(*_args):
    if _runtime["watcher"] is not None:
        _runtime["watcher"].on_undo(redo=False)


@persistent
def _on_redo(*_args):
    if _runtime["watcher"] is not None:
        _runtime["watcher"].on_undo(redo=True)


@persistent
def _on_depsgraph(_scene, depsgraph):
    watcher = _runtime["watcher"]
    if watcher is None:
        return
    changed = [u.id.name for u in depsgraph.updates if isinstance(u.id, bpy.types.Object) and u.is_updated_geometry]
    if changed:
        watcher.note_geometry(changed)


def run_headless(port=0, token=None, ready_file=None):
    """Serve forever without UI (``blender -b --python-expr`` or the ``bpy`` module)."""
    server = start_server(port=port, token=token, dispatcher=InlineDispatcher(), write_discovery=False)
    if ready_file:
        tmp = ready_file + ".tmp"
        with open(tmp, "w") as handle:
            json.dump({"port": server.port, "pid": os.getpid(), "blender_version": bpy.app.version_string}, handle)
        os.replace(tmp, ready_file)
    threading.Event().wait()


# -- UI --------------------------------------------------------------------------------------

class LUCIUS_OT_bridge_start(bpy.types.Operator):
    bl_idname = "lucius.bridge_start"
    bl_label = "Start Lucius Bridge"

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        start_server(port=prefs.port)
        return {"FINISHED"}


class LUCIUS_OT_bridge_stop(bpy.types.Operator):
    bl_idname = "lucius.bridge_stop"
    bl_label = "Stop Lucius Bridge"

    def execute(self, _context):
        stop_server()
        return {"FINISHED"}


class LUCIUS_PT_panel(bpy.types.Panel):
    bl_label = "Lucius"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lucius"

    def draw(self, _context):
        layout = self.layout
        server = _runtime["server"]
        if server is None:
            layout.label(text="Bridge stopped", icon="UNLINKED")
            layout.operator("lucius.bridge_start")
            return
        layout.label(text=f"Bridge on 127.0.0.1:{server.port}", icon="LINKED")
        layout.label(text=f"Clients: {server.client_count}")
        if server.recording_indicator:
            layout.label(text="Lucius is RECORDING", icon="REC")
        layout.operator("lucius.bridge_stop")


class LuciusBridgePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    port: bpy.props.IntProperty(name="Port", default=47821, min=1024, max=65535)
    autostart: bpy.props.BoolProperty(name="Start bridge with Blender", default=True)

    def draw(self, _context):
        self.layout.prop(self, "port")
        self.layout.prop(self, "autostart")
        self.layout.label(text=f"Discovery file: {discovery_path()}")


def _draw_status(self, _context):
    server = _runtime["server"]
    if server is not None and server.recording_indicator:
        self.layout.label(text="Lucius recording", icon="REC")


_CLASSES = (LUCIUS_OT_bridge_start, LUCIUS_OT_bridge_stop, LUCIUS_PT_panel, LuciusBridgePreferences)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.app.handlers.undo_post.append(_on_undo)
    bpy.app.handlers.redo_post.append(_on_redo)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.types.STATUSBAR_HT_header.append(_draw_status)
    prefs = bpy.context.preferences.addons.get(__package__)
    if prefs is not None and prefs.preferences.autostart and not bpy.app.background:
        bpy.app.timers.register(lambda: (start_server(port=prefs.preferences.port), None)[1], first_interval=1.0)


def unregister():
    stop_server()
    bpy.types.STATUSBAR_HT_header.remove(_draw_status)
    for handlers, fn in ((bpy.app.handlers.undo_post, _on_undo), (bpy.app.handlers.redo_post, _on_redo),
                         (bpy.app.handlers.depsgraph_update_post, _on_depsgraph)):
        if fn in handlers:
            handlers.remove(fn)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
