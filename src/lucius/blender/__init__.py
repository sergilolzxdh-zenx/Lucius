"""Blender integration: the bridge client, headless launcher and typed state views.

The add-on that runs *inside* Blender lives in ``addon/lucius_bridge`` and depends only on the
standard library and ``bpy``; it is installed into Blender, never imported by Lucius itself.
"""

from pathlib import Path

ADDON_DIR = Path(__file__).parent / "addon"
ADDON_PACKAGE = ADDON_DIR / "lucius_bridge"
