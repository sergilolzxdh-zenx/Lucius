"""Demonstration capture (WATCH ME)."""

from __future__ import annotations

from lucius.blender.bridge import BlenderBridge
from lucius.config import LuciusConfig
from lucius.errors import BlenderUnavailable, CaptureUnavailable
from lucius.recorder.ledger import AgentActionLedger, ExpectedInput
from lucius.recorder.recorder import DemonstrationRecorder, RecorderSources, recover_interrupted_sessions
from lucius.recorder.sources import MssGrabber, PynputInputSource
from lucius.recorder.window import create_window_provider


def detect_sources(config: LuciusConfig, *, connect_blender: bool = True) -> RecorderSources:
    """Probe every capture backend; unavailable ones are reported, never faked."""
    sources = RecorderSources()
    try:
        sources.window = create_window_provider()
    except CaptureUnavailable as exc:
        sources.notes["window"] = exc.message
    try:
        MssGrabber().close()
        sources.grabber_factory = MssGrabber
    except CaptureUnavailable as exc:
        sources.notes["screen"] = exc.message
    try:
        sources.input = PynputInputSource()
    except CaptureUnavailable as exc:
        sources.notes["input"] = exc.message
    if connect_blender:
        try:
            bridge = BlenderBridge.from_config(config.blender)
            bridge.connect()
            sources.bridge = bridge
        except BlenderUnavailable as exc:
            sources.notes["blender_bridge"] = exc.message
    return sources


__all__ = [
    "AgentActionLedger", "DemonstrationRecorder", "ExpectedInput", "RecorderSources", "detect_sources",
    "recover_interrupted_sessions",
]
