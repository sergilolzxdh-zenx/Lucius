"""Explainable, evidence-based confidence (section 61).

Confidence is a Beta-posterior mean over weighted evidence. Every score comes with the counts
that produced it, so "why is this 0.72?" always has an answer. Weights encode evidence quality:
a directly observed user demonstration counts more than a VLM reading of a compressed video.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    demonstrations: float = 0.0          # weighted count (source x evidence quality)
    demonstration_count: int = 0
    distinct_sources: int = 0
    distinct_instances: int = 0
    successes: int = 0
    failures: int = 0
    validations: int = 0                 # objective checkpoint-verified reproductions
    human_confirmations: int = 0
    human_rejections: int = 0
    contradictions: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    notes: list[str] = Field(default_factory=list)


def score(ev: Evidence) -> tuple[float, dict[str, Any]]:
    # Generalisation bonus only once the same thing worked on more than one instance.
    instance_bonus = 0.5 * max(0, ev.distinct_instances - 1)
    alpha = (ev.prior_alpha + 0.6 * ev.demonstrations + 1.0 * ev.successes + 1.5 * ev.validations
             + 1.5 * ev.human_confirmations + instance_bonus)
    beta = ev.prior_beta + 1.0 * ev.failures + 2.0 * ev.human_rejections + 1.0 * ev.contradictions
    value = alpha / (alpha + beta)
    breakdown = ev.model_dump()
    breakdown.update(alpha=round(alpha, 3), beta=round(beta, 3), score=round(value, 4),
                     formula="alpha/(alpha+beta); alpha=1+0.6*demos+succ+1.5*valid+1.5*confirm+0.5*(instances-1);"
                             " beta=1+fail+2*reject+contradictions")
    return round(value, 4), breakdown


def explain(breakdown: dict[str, Any]) -> str:
    parts = []
    if breakdown.get("demonstration_count"):
        parts.append(f"{breakdown['demonstration_count']} demonstration(s) (weight {breakdown['demonstrations']:.2f})")
    for key, label in (("successes", "successful use(s)"), ("failures", "failed use(s)"),
                       ("validations", "validation(s)"), ("human_confirmations", "human confirmation(s)"),
                       ("human_rejections", "human rejection(s)"), ("distinct_instances", "distinct instance(s)")):
        if breakdown.get(key):
            parts.append(f"{breakdown[key]} {label}")
    return ", ".join(parts) or "no evidence yet"
