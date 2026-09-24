"""Trainer interfaces (behaviour cloning, SFT, LoRA, preference learning, DPO).

These are contracts, not implementations: registering a backend is how a real trainer is
plugged in. ``TrainerRegistry.launch`` refuses to run without a registered backend rather than
pretending to train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from lucius.errors import LuciusError

TrainingStrategy = Literal["behaviour_cloning", "sft", "lora", "preference", "dpo"]


class TrainingJobSpec(BaseModel):
    strategy: TrainingStrategy
    dataset_dir: str
    formatted_files: dict[str, str] = Field(default_factory=dict)
    base_model: str | None = None
    target_skills: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    evaluation_benchmarks: list[str] = Field(default_factory=list)


class TrainerBackend(Protocol):
    name: str
    strategies: set[str]

    def launch(self, spec: TrainingJobSpec) -> str: ...        # returns an external job id

    def status(self, job_id: str) -> dict[str, Any]: ...


class TrainerRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, TrainerBackend] = {}

    def register(self, backend: TrainerBackend) -> None:
        self._backends[backend.name] = backend

    def available(self) -> dict[str, list[str]]:
        return {name: sorted(b.strategies) for name, b in self._backends.items()}

    def launch(self, spec: TrainingJobSpec, backend: str | None = None) -> str:
        candidates = [b for n, b in self._backends.items() if (backend in (None, n)) and spec.strategy in b.strategies]
        if not candidates:
            raise LuciusError(f"no trainer backend registered for {spec.strategy}", code="no_trainer_backend",
                              registered=self.available())
        if not Path(spec.dataset_dir).exists():
            raise LuciusError("dataset directory does not exist", code="dataset_missing", path=spec.dataset_dir)
        return candidates[0].launch(spec)
