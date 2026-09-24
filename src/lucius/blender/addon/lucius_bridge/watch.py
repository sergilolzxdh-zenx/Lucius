"""Observation of human activity inside Blender.

* Operators the user runs from the UI land in ``window_manager.operators`` (the same list the
  Info editor shows). Diffing successive snapshots yields *observed* operations with their
  exact parameters -- the strongest evidence a demonstration can carry.
* ``undo_post`` / ``redo_post`` handlers report undo and redo, which downstream become the
  primary failure/correction signal.
* ``depsgraph_update_post`` accumulates which objects had geometry changes.
"""

import time

MAX_PROPERTY_DEPTH = 2
MAX_LIST_ITEMS = 64


def sanitize_value(value, depth=0):
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if hasattr(value, "bl_rna"):
        if depth >= MAX_PROPERTY_DEPTH:
            return str(value)
        return sanitize_properties(value, depth + 1)
    try:
        items = list(value)
    except TypeError:
        return str(value)
    return [sanitize_value(v, depth + 1) for v in items[:MAX_LIST_ITEMS]]


def sanitize_properties(props, depth=0):
    out = {}
    for prop in props.bl_rna.properties:
        name = prop.identifier
        if name == "rna_type":
            continue
        try:
            out[name] = sanitize_value(getattr(props, name), depth)
        except Exception as exc:
            out[name] = f"<unreadable: {type(exc).__name__}>"
    return out


def operator_snapshot(op):
    info = {"idname": op.bl_idname, "name": op.name, "properties": sanitize_properties(op.properties)}
    macros = getattr(op, "macros", None)
    if macros:
        info["macros"] = [
            {"idname": m.bl_idname, "properties": sanitize_properties(m.properties)} for m in macros
        ]
    return info


class OperatorLogWatcher:
    """Turns successive ``(identity, snapshot)`` lists into new-operator and adjustment events.

    Identity is ``(pointer, idname)``. The newest operator can be re-run with new parameters
    from the redo panel (F9), which mutates it in place: that is reported as an adjustment.
    """

    def __init__(self):
        self._seen = set()
        self._last_identity = None
        self._last_props = None

    def diff(self, entries):
        events = []
        identities = [identity for identity, _snap in entries]
        for identity, snap in entries:
            if identity not in self._seen:
                events.append({"kind": "blender_operator", "operator": snap})
        if entries:
            last_identity, last_snap = entries[-1]
            if (last_identity == self._last_identity and last_snap.get("properties") != self._last_props
                    and not any(e["operator"] is last_snap for e in events)):
                events.append({"kind": "blender_operator_adjusted", "operator": last_snap})
            self._last_identity = last_identity
            self._last_props = last_snap.get("properties")
        self._seen = set(identities)
        return events


def collect_operator_entries(window_manager):
    entries = []
    for op in window_manager.operators:
        try:
            entries.append(((op.as_pointer(), op.bl_idname), operator_snapshot(op)))
        except Exception:
            continue
    return entries


class ActivityWatcher:
    """Polls Blender for state changes and forwards them to ``push(kind, data)``."""

    def __init__(self, push, capture_state, light_state_key):
        self.push = push
        self.capture_state = capture_state
        self.light_state_key = light_state_key
        self.operators = OperatorLogWatcher()
        self._last_key = None
        self._geometry_changed = set()
        self._primed = False

    def note_geometry(self, names):
        self._geometry_changed.update(names)

    def poll(self, window_manager):
        if not self._primed:
            # Operators already in the log before recording started are history, not events.
            self.operators.diff(collect_operator_entries(window_manager))
            self._primed = True
        for event in self.operators.diff(collect_operator_entries(window_manager)):
            kind = event.pop("kind")
            event["geometry_changed"] = sorted(self._geometry_changed)
            self.push(kind, event)
            self._geometry_changed.clear()
        self.push_state_if_changed()

    def push_state_if_changed(self, force=False, reason=None):
        state = self.capture_state()
        key = self.light_state_key(state)
        if force or key != self._last_key:
            self._last_key = key
            if self._geometry_changed:
                state["geometry_changed"] = sorted(self._geometry_changed)
            if reason:
                state["reason"] = reason
            self.push("blender_state", state)

    def on_undo(self, redo=False):
        self.push("redo" if redo else "undo", {"ts": time.time()})
        self.push_state_if_changed(force=True, reason="redo" if redo else "undo")
