"""Wire protocol shared by the add-on and the Lucius client (stdlib only).

Newline-delimited JSON over a loopback TCP socket:

    request  {"id": 7, "token": "...", "cmd": "get_state", "args": {}}
    response {"id": 7, "ok": true, "result": {...}}
             {"id": 7, "ok": false, "error": {"code": "...", "message": "..."}}
    push     {"push": "blender_operator", "ts": 1712345678.12, "data": {...}}
"""

import json

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 64 * 1024 * 1024


def encode(message):
    return (json.dumps(message, separators=(",", ":"), default=_default) + "\n").encode("utf-8")


def _default(value):
    # mathutils vectors/quaternions/colors and bpy arrays are sequences of floats.
    try:
        return [float(v) for v in value]
    except TypeError:
        return str(value)


class BridgeCommandError(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
