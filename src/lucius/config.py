"""Runtime configuration.

Configuration is a single validated pydantic model. It is loaded from (in increasing priority)
defaults, ``<data_dir>/config.json`` and ``LUCIUS_*`` environment variables. Only
non-secret settings are persisted: provider credentials are always resolved from the
environment by the provider SDKs themselves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from lucius.errors import ConfigError


class RecordingConfig(BaseModel):
    fps: float = Field(default=6.0, ge=0.5, le=30.0, description="Learning-oriented frame rate.")
    image_format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = Field(default=82, ge=40, le=100)
    max_frame_width: int = Field(default=1600, ge=320)
    skip_identical_frames: bool = True
    # Mouse-move compression: keep at most this many samples per second, plus direction changes.
    mouse_move_hz: float = Field(default=12.0, ge=1.0, le=120.0)
    mouse_min_distance_px: float = Field(default=6.0, ge=0.0)
    blender_state_poll_hz: float = Field(default=4.0, ge=0.5, le=30.0)
    flush_interval_s: float = Field(default=0.25, gt=0.0, le=5.0)
    capture_only_blender: bool = True


class PrivacyConfig(BaseModel):
    # Window titles/process names that may be captured. Blender is always allowed.
    allowed_processes: list[str] = Field(default_factory=lambda: ["blender", "blender.exe", "Blender"])
    allowed_title_patterns: list[str] = Field(default_factory=lambda: [r"\bBlender\b"])
    # Keystrokes typed while a non-allowed window is focused are dropped entirely.
    drop_input_outside_allowed: bool = True
    # Titles matching these are never captured even if otherwise allowed.
    blocked_title_patterns: list[str] = Field(
        default_factory=lambda: [r"(?i)password", r"(?i)1password", r"(?i)keychain", r"(?i)bitwarden"]
    )


class ProviderConfig(BaseModel):
    llm: Literal["anthropic", "gemini", "none"] = "none"
    vlm: Literal["anthropic", "gemini", "none"] = "none"
    evaluation: Literal["anthropic", "gemini", "none"] = "none"
    embeddings: Literal["hashing", "sentence-transformers"] = "hashing"
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    anthropic_server_fallbacks: bool = True
    # Gemini: the model must be chosen explicitly (`lucius models --provider gemini` lists them).
    gemini_model: str | None = None
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    sentence_transformers_model: str = "all-MiniLM-L6-v2"
    hashing_dim: int = Field(default=512, ge=64, le=8192)
    max_retries: int = Field(default=2, ge=0, le=8)
    # Client-side cap on model requests per minute (None: unlimited). Free-tier keys have per-minute quotas.
    requests_per_minute: int | None = Field(default=None, ge=1, le=10000)
    # Upper bound on images sent to a VLM in a single request.
    max_images_per_call: int = Field(default=6, ge=1, le=20)


class ProcessingConfig(BaseModel):
    workers: int = Field(default=2, ge=1, le=16)
    pause_threshold_s: float = Field(default=4.0, gt=0.0)
    navigation_min_steps: int = Field(default=2, ge=1)
    visual_change_threshold: float = Field(default=0.18, gt=0.0, le=1.0)
    model_refinement: bool = True
    representative_frames_per_segment: int = Field(default=3, ge=1, le=10)


class SafetyConfig(BaseModel):
    allow_gui_actions: bool = True
    allow_os_actions: bool = False
    allowed_save_dirs: list[str] = Field(default_factory=list)
    max_actions_per_run: int = Field(default=400, ge=1)


class BlenderConfig(BaseModel):
    bridge_host: str = "127.0.0.1"
    bridge_port: int = Field(default=47821, ge=1024, le=65535)
    bridge_token: str | None = None
    connect_timeout_s: float = 2.0
    request_timeout_s: float = 30.0


class LuciusConfig(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path(os.environ.get("LUCIUS_DATA_DIR", ".lucius")))
    user_id: str = "local"
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    blender: BlenderConfig = Field(default_factory=BlenderConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lucius.sqlite3"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def journal_dir(self) -> Path:
        return self.data_dir / "journal"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.frames_dir, self.media_dir, self.exports_dir, self.journal_dir):
            path.mkdir(parents=True, exist_ok=True)

    def save(self) -> Path:
        self.ensure_dirs()
        path = self.data_dir / "config.json"
        payload = self.model_dump(mode="json", exclude={"data_dir"})
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


def _apply_env(raw: dict) -> dict:
    """Map ``LUCIUS_SECTION__FIELD=value`` environment variables onto the raw config dict."""
    for key, value in os.environ.items():
        if not key.startswith("LUCIUS_") or "__" not in key:
            continue
        section, _, field = key[len("LUCIUS_"):].lower().partition("__")
        raw.setdefault(section, {})
        if isinstance(raw[section], dict):
            raw[section][field] = value
    return raw


def load_config(data_dir: str | Path | None = None, **overrides: object) -> LuciusConfig:
    base = Path(data_dir) if data_dir is not None else Path(os.environ.get("LUCIUS_DATA_DIR", ".lucius"))
    raw: dict = {}
    config_file = base / "config.json"
    if config_file.exists():
        try:
            raw = json.loads(config_file.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid config file {config_file}: {exc}") from exc
    raw = _apply_env(raw)
    raw.update(overrides)
    raw["data_dir"] = base
    try:
        config = LuciusConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ConfigError("invalid configuration", errors=json.loads(exc.json())) from exc
    config.ensure_dirs()
    return config
