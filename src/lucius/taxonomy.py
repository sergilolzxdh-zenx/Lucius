"""Extensible taxonomies (segment labels, intents, object classes).

Seed terms are marked ``system_seeded``; users and models can add terms at runtime, which are
persisted with their source. Nothing downstream assumes the taxonomy is closed.
"""

from __future__ import annotations

import re

from lucius.storage.db import Database
from lucius.timeutil import now

SEGMENT_LABELS = {
    "scene_setup": "Preparing the scene: clearing defaults, adding base objects, symmetry setup",
    "reference_alignment": "Loading and aligning reference images",
    "primary_blockout": "Establishing primary forms and proportions",
    "secondary_forms": "Adding secondary shapes on top of the primary form",
    "detail_pass": "Bevels, subdivision, shading and other fine detail",
    "inspection": "Viewing the model from several angles without editing it",
    "corrective_pass": "Undoing or redoing work and trying an alternative",
    "verification": "Final checks after the last edit",
    "navigation": "Moving the viewport without a clear inspection pattern",
    "recovery": "Returning to a previous state after a mistake",
    "unlabeled": "No confident label yet",
}

INTENT_CATEGORIES = {
    "inspect": "Look at the model to evaluate it",
    "create": "Add new geometry or objects",
    "refine": "Adjust existing geometry",
    "correct": "Fix a previous mistake",
    "verify": "Confirm the result meets the goal",
    "navigate": "Move the view without an evaluative purpose",
    "recover": "Return to a known-good state",
    "setup": "Prepare the scene or tools",
}

INTENT_TARGETS = {
    "silhouette": "Overall outline of the object",
    "proportions": "Relative dimensions of parts",
    "blade_width": "Width of a blade",
    "blade_length": "Length of a blade",
    "tip_profile": "Shape of a tip",
    "edge_detail": "Bevels and edge treatment",
    "edge_flow": "Topology loops",
    "topology": "Mesh structure",
    "symmetry": "Mirror symmetry",
    "primary_form": "Main volume of an object",
    "secondary_form": "Secondary shapes",
    "reference_alignment": "Match with reference images",
    "face_proportions": "Facial proportions",
    "brush_radius": "Sculpt brush size",
    "scene": "Scene organisation",
    "object_selection": "Which object is being worked on",
}

# Object classes recognised in task text: class -> (keywords, categories)
OBJECT_CLASSES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "blade_weapon": (("sword", "blade", "dagger", "knife", "katana", "sabre", "saber", "longsword"),
                     ("hard-surface", "weapon")),
    "axe": (("axe", "hatchet"), ("hard-surface", "weapon")),
    "shield": (("shield",), ("hard-surface", "weapon")),
    "furniture": (("table", "chair", "desk", "stool", "shelf", "bench"), ("hard-surface", "prop")),
    "container": (("cup", "mug", "bottle", "vase", "barrel", "crate", "box"), ("hard-surface", "prop")),
    "architecture": (("house", "wall", "door", "window", "pillar", "column", "stairs", "building"),
                     ("hard-surface", "architecture")),
    "vehicle": (("car", "truck", "ship", "boat", "plane", "spaceship"), ("hard-surface", "vehicle")),
    "character_head": (("head", "face", "bust", "skull"), ("organic", "character")),
    "character_body": (("body", "character", "torso", "hand", "arm", "leg"), ("organic", "character")),
    "creature": (("creature", "monster", "dragon", "animal"), ("organic", "creature")),
    "primitive_study": (("cube", "sphere", "cylinder", "primitive"), ("fundamentals",)),
}


def classify_task(text: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Map task text to an object class and categories by keyword (deterministic)."""
    if not text:
        return None, ()
    words = set(re.findall(r"[a-z]+", text.lower()))
    for name, (keywords, categories) in OBJECT_CLASSES.items():
        if any(k in words or k + "s" in words for k in keywords):
            return name, categories
    return None, ()


class Taxonomy:
    def __init__(self, db: Database) -> None:
        self.db = db

    def seed(self) -> None:
        rows = []
        t = now()
        for kind, terms in (("segment_label", SEGMENT_LABELS), ("intent_category", INTENT_CATEGORIES),
                            ("intent_target", INTENT_TARGETS)):
            rows += [(kind, term, desc, None, "system_seeded", t) for term, desc in terms.items()]
        rows += [("object_class", name, ", ".join(kw), None, "system_seeded", t)
                 for name, (kw, _c) in OBJECT_CLASSES.items()]
        self.db.executemany(
            "INSERT OR IGNORE INTO taxonomy_terms(kind, term, description, parent, source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)", rows)

    def ensure(self, kind: str, term: str, *, source: str, description: str | None = None) -> bool:
        """Add a term if missing. Returns True when the taxonomy grew."""
        term = term.strip().lower().replace(" ", "_")
        cur = self.db.execute(
            "INSERT OR IGNORE INTO taxonomy_terms(kind, term, description, parent, source, created_at)"
            " VALUES (?, ?, ?, NULL, ?, ?)", (kind, term, description, source, now()))
        return cur.rowcount > 0

    def terms(self, kind: str) -> dict[str, dict[str, str | None]]:
        rows = self.db.query("SELECT term, description, source FROM taxonomy_terms WHERE kind = ? ORDER BY term", (kind,))
        return {r["term"]: {"description": r["description"], "source": r["source"]} for r in rows}
