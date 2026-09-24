"""Structured error hierarchy.

Every subsystem raises a subclass of :class:`LuciusError` carrying a stable ``code`` and a
JSON-serialisable ``details`` mapping, so failures can be persisted (processing jobs, run
transitions) and surfaced in the UI without string parsing.
"""

from __future__ import annotations

from typing import Any


class LuciusError(Exception):
    code: str = "lucius_error"

    def __init__(self, message: str, *, code: str | None = None, **details: Any) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(LuciusError):
    code = "config_error"


class StorageError(LuciusError):
    code = "storage_error"


class NotFoundError(StorageError):
    code = "not_found"


class ConflictError(StorageError):
    code = "conflict"


class ValidationError(LuciusError):
    code = "validation_error"


class RecorderError(LuciusError):
    code = "recorder_error"


class CaptureUnavailable(RecorderError):
    """A capture backend (screen, input, window) is not available on this machine."""

    code = "capture_unavailable"


class BlenderBridgeError(LuciusError):
    code = "blender_bridge_error"


class BlenderUnavailable(BlenderBridgeError):
    code = "blender_unavailable"


class ProviderError(LuciusError):
    code = "provider_error"


class ProviderUnavailable(ProviderError):
    """No provider is configured for a capability (LLM, VLM, embeddings, evaluation)."""

    code = "provider_unavailable"


class ProviderRefusal(ProviderError):
    code = "provider_refusal"


class ActionRejected(LuciusError):
    """The safety layer refused an action before it reached any actuator."""

    code = "action_rejected"


class ExecutionError(LuciusError):
    code = "execution_error"


class InvalidTransition(ExecutionError):
    code = "invalid_transition"


class MediaError(LuciusError):
    code = "media_error"


class PolicyViolation(LuciusError):
    """Data-use policy (consent, license, training eligibility) forbids the operation."""

    code = "policy_violation"
