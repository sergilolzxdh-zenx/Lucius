"""Cross-demonstration generalisation (section 22, 10N).

Given the per-instance definitions extracted from several demonstrations, produce one skill
definition that separates:

* the common workflow (steps present in every instance -> required, frequency 1.0),
* instance-specific behaviour (steps present in some -> optional, with their frequency),
* alternative implementations (aligned regions where instances disagree -> variants),
* parameters (identical values -> invariant; varying values -> ranges with a median default).

Alignment is a progressive pairwise alignment of action tokens against the heaviest instance.
"""

from __future__ import annotations

import difflib
import statistics
from typing import Any

from lucius.skills.schema import (
    ActionTemplate,
    Checkpoint,
    FailureCondition,
    ParamSpec,
    RecoveryAction,
    SkillDefinition,
    SkillPhase,
    SkillVariant,
)
from lucius.trajectory import vocabulary as vocab


def _token(action: ActionTemplate) -> str:
    sel = action.selection
    region = f"{sel.kind}:{sel.axis or ''}" if sel is not None else "-"
    return f"{action.action_type}|{action.args.get('axis', '')}|{action.args.get('type', '')}|{region}"


def _same_family(a: list[ActionTemplate], b: list[ActionTemplate]) -> bool:
    fa = {vocab.spec(x.action_type).family for x in a}
    fb = {vocab.spec(x.action_type).family for x in b}
    return bool(fa & fb) or {"edit", "modifier"} <= (fa | fb)


def _merge_phase(name: str, instances: list[tuple[str, str, SkillPhase]]) -> tuple[SkillPhase, list[SkillVariant]]:
    """``instances``: (example_id, source_class, phase) with the heaviest instance first."""
    base_id, _base_source, base = instances[0]
    consensus: list[tuple[ActionTemplate, set[str]]] = [(a.model_copy(deep=True), {base_id}) for a in base.actions]
    variants: list[SkillVariant] = []
    for ex_id, source, phase in instances[1:]:
        tokens_a = [_token(a) for a, _s in consensus]
        tokens_b = [_token(a) for a in phase.actions]
        matcher = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
        inserts: list[tuple[int, list[ActionTemplate]]] = []
        for op, a0, a1, b0, b1 in matcher.get_opcodes():
            if op == "equal":
                for k in range(a1 - a0):
                    consensus[a0 + k][1].add(ex_id)
                    consensus[a0 + k][0].evidence = sorted(set(consensus[a0 + k][0].evidence) | set(phase.actions[b0 + k].evidence))
            elif op == "replace":
                ours = [a for a, _s in consensus[a0:a1]]
                theirs = phase.actions[b0:b1]
                if _same_family(ours, theirs):
                    variants.append(SkillVariant(
                        variant_id=f"{name}_v{len(variants) + 1}_{ex_id[-6:]}",
                        name=" + ".join(a.action_type for a in theirs), phase=name,
                        description=f"alternative to {' + '.join(a.action_type for a in ours)}",
                        actions=[a.model_copy(deep=True) for a in theirs], source_classes=[source], evidence=[ex_id],
                    ))
                else:
                    inserts.append((a1, [a.model_copy(deep=True) for a in theirs]))
            elif op == "insert":
                inserts.append((a0, [a.model_copy(deep=True) for a in phase.actions[b0:b1]]))
        for position, actions in sorted(inserts, key=lambda x: x[0], reverse=True):
            for offset, action in enumerate(actions):
                consensus.insert(position + offset, (action, {ex_id}))
    total = len(instances)
    merged: list[ActionTemplate] = []
    for action, support in consensus:
        action.frequency = round(len(support) / total, 3)
        action.optional = len(support) < total
        merged.append(action)
    checkpoints = sorted({c for _i, _s, p in instances for c in p.checkpoints})
    preconditions = {c.model_dump_json(): c for _i, _s, p in instances for c in p.preconditions}
    return SkillPhase(name=name, description=base.description, actions=merged, checkpoints=checkpoints,
                      preconditions=list(preconditions.values())), _dedupe_variants(variants)


def _dedupe_variants(variants: list[SkillVariant]) -> list[SkillVariant]:
    by_key: dict[str, SkillVariant] = {}
    for v in variants:
        key = v.phase + "|" + ",".join(_token(a) for a in v.actions)
        if key in by_key:
            existing = by_key[key]
            existing.evidence = sorted(set(existing.evidence) | set(v.evidence))
            existing.source_classes = sorted(set(existing.source_classes) | set(v.source_classes))
        else:
            by_key[key] = v
    return list(by_key.values())


def _merge_params(instances: list[SkillDefinition]) -> list[ParamSpec]:
    by_name: dict[str, list[ParamSpec]] = {}
    order: list[str] = []
    for inst in instances:
        for p in inst.parameters:
            if p.name not in by_name:
                order.append(p.name)
            by_name.setdefault(p.name, []).append(p)
    out = []
    for name in order:
        specs = by_name[name]
        values = [v for s in specs for v in (s.observed_values or ([s.default] if s.default is not None else []))]
        first = specs[0].model_copy(deep=True)
        first.observed_values = values[-50:]
        numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric and len(numeric) == len(values):
            lo, hi = min(numeric), max(numeric)
            first.invariant = len(values) >= 2 and (hi - lo) <= 1e-3 * max(1.0, abs(hi))
            first.range = (lo, hi)
            median = statistics.median(numeric)
            first.default = int(round(median)) if first.kind == "int" else round(median, 4)
        elif values:
            first.invariant = len(values) >= 2 and all(v == values[0] for v in values)
            first.choices = sorted({str(v) for v in values}) if first.kind in ("enum", "str") else first.choices
        out.append(first)
    return out


def _merge_checkpoints(instances: list[SkillDefinition]) -> list[Checkpoint]:
    by_id: dict[str, Checkpoint] = {}
    for inst in instances:
        for cp in inst.checkpoints:
            if cp.id not in by_id:
                by_id[cp.id] = cp.model_copy(deep=True)
                continue
            existing = by_id[cp.id]
            existing.derived_from = sorted(set(existing.derived_from) | set(cp.derived_from))
            for bound, pick in (("min", min), ("max", max)):
                a, b = existing.check.get(bound), cp.check.get(bound)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    existing.check[bound] = pick(a, b)
            if isinstance(existing.check.get("views"), list) and isinstance(cp.check.get("views"), list):
                existing.check["views"] = sorted(set(existing.check["views"]) | set(cp.check["views"]))
    return list(by_id.values())


def _merge_failures(instances: list[SkillDefinition]) -> tuple[list[FailureCondition], list[RecoveryAction]]:
    conditions: dict[str, FailureCondition] = {}
    recoveries: dict[str, RecoveryAction] = {}
    for inst in instances:
        for fc in inst.failure_conditions:
            if fc.id in conditions:
                existing = conditions[fc.id]
                existing.failure_ids = sorted(set(existing.failure_ids) | set(fc.failure_ids))
                existing.confidence = max(existing.confidence, fc.confidence)
            else:
                conditions[fc.id] = fc.model_copy(deep=True)
        for rc in inst.recovery_actions:
            if rc.id in recoveries:
                recoveries[rc.id].evidence = sorted(set(recoveries[rc.id].evidence) | set(rc.evidence))
            else:
                recoveries[rc.id] = rc.model_copy(deep=True)
    return list(conditions.values()), list(recoveries.values())


def generalize(instances: list[tuple[str, str, float, SkillDefinition]]) -> SkillDefinition:
    """``instances``: (example_id, source_class, weight, per-instance definition)."""
    if not instances:
        raise ValueError("nothing to generalise")
    ordered = sorted(instances, key=lambda x: x[2], reverse=True)
    base = ordered[0][3].model_copy(deep=True)
    if len(ordered) == 1:
        return base
    phase_names: list[str] = []
    for _id, _src, _w, inst in ordered:
        for phase in inst.phases:
            if phase.name not in phase_names:
                phase_names.append(phase.name)
    phases, variants = [], []
    for name in phase_names:
        per_instance = [(ex_id, src, next(p for p in inst.phases if p.name == name))
                        for ex_id, src, _w, inst in ordered if any(p.name == name for p in inst.phases)]
        phase, phase_variants = _merge_phase(name, per_instance)
        if len(per_instance) < len(ordered):
            phase.description = (phase.description + f" (present in {len(per_instance)}/{len(ordered)} examples)").strip()
        phases.append(phase)
        variants += phase_variants
    defs = [inst for _i, _s, _w, inst in ordered]
    base.phases = phases
    base.parameters = _merge_params(defs)
    base.checkpoints = _merge_checkpoints(defs)
    base.failure_conditions, base.recovery_actions = _merge_failures(defs)
    base.variants = _dedupe_variants(base.variants + variants + [v for d in defs[1:] for v in d.variants])
    base.triggers = sorted({t for d in defs for t in d.triggers})
    base.categories = list(dict.fromkeys(c for d in defs for c in d.categories))
    base.applicable_contexts = sorted({c for d in defs for c in d.applicable_contexts})
    base.notes = [f"generalised from {len(ordered)} examples"]
    return base


def human_edited_fields(edits: list[dict[str, Any]]) -> set[str]:
    """Top-level definition fields a human has edited (generalisation must preserve them)."""
    fields: set[str] = set()
    for edit in edits:
        after = edit.get("after") or {}
        if isinstance(after, dict):
            fields |= set(after)
    return fields
