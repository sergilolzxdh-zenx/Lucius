"""Dataset -> imitation-learning formats.

Only actions with strong provenance become supervised targets: observed or human-confirmed
steps performed by a human (demonstrations and takeover corrections). Inferred and
model-inferred actions are kept as context but never as labels.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from lucius.storage.jsonl import read_jsonl, write_jsonl

LABEL_SOURCES = {"observed", "human_confirmed"}
HISTORY = 6
SYSTEM = ("You operate Blender for a modelling task. Given the task, the current Blender state and recent actions, "
          "output the next action as JSON with action_type and params.")


def _label_ok(step: dict[str, Any]) -> bool:
    return (step.get("actor") == "human" and step.get("action_source") in LABEL_SOURCES
            and not (step.get("meta") or {}).get("cancelled") and "undone_by" not in (step.get("meta") or {})
            and step.get("action_type") not in ("unknown_action",))


def behaviour_cloning_pairs(samples: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for sample in samples:
        steps = sample.get("trajectory_steps", [])
        segments = {s["id"]: s for s in sample.get("segments", [])}
        for i, step in enumerate(steps):
            if not _label_ok(step):
                continue
            history = [{"action_type": s["action_type"], "params": (s.get("action_payload") or {}).get("params", {})}
                       for s in steps[max(0, i - HISTORY):i] if not (s.get("meta") or {}).get("cancelled")]
            segment = segments.get(step.get("segment_id") or "")
            yield {
                "observation": {"task": sample.get("task_text"), "task_class": sample.get("task_class"),
                                "state": step.get("state_before"), "phase": segment.get("label") if segment else None,
                                "frame_id": step.get("frame_before_id"), "history": history},
                "action": {"action_type": step["action_type"],
                           "params": (step.get("action_payload") or {}).get("params", {})},
                "provenance": {"session_id": sample.get("session_id"), "step_id": step["id"],
                               "source": step.get("action_source"), "evidence": step.get("evidence_kind"),
                               "correction": bool((step.get("meta") or {}).get("during_takeover"))},
            }


def sft_samples(samples: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for pair in behaviour_cloning_pairs(samples):
        obs = pair["observation"]
        user = json.dumps({"task": obs["task"], "phase": obs["phase"], "state": obs["state"], "recent": obs["history"]},
                          sort_keys=True)
        yield {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                            {"role": "assistant", "content": json.dumps(pair["action"], sort_keys=True)}],
               "provenance": pair["provenance"]}


def write_training_files(dataset_dir: Path, out_dir: Path) -> dict[str, int]:
    samples = list(read_jsonl(Path(dataset_dir) / "samples.jsonl"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"behaviour_cloning": write_jsonl(out_dir / "bc_pairs.jsonl", behaviour_cloning_pairs(samples)),
            "sft": write_jsonl(out_dir / "sft.jsonl", sft_samples(samples))}
