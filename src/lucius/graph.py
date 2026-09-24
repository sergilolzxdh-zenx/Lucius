"""Learning and provenance graph (sections 31, 80, 10Y).

Edges connect heterogeneous nodes -- skills, failures, episodes, sessions, media, segments,
runs, references -- so the system can answer "where did this skill come from, which
demonstrations support it, which failures modified it". Re-asserting an edge increments its
evidence count rather than duplicating it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lucius.ids import new_id
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

RELATIONS = {
    "requires", "follows", "parent_of", "variant_of", "causes_failure", "corrected_by", "co_used", "merged_into",
    "extracted_from", "supported_by", "validated_by", "conditioned_by", "modified_by", "promoted_to", "demonstrated_in",
    "derived_from", "recovers_with",
}


class Edge(BaseModel):
    id: str
    src_kind: str
    src_id: str
    rel: str
    dst_kind: str
    dst_id: str
    weight: float
    evidence_count: int
    meta: dict[str, Any]
    created_at: float
    updated_at: float


def _edge(row: Any) -> Edge:
    return Edge(id=row["id"], src_kind=row["src_kind"], src_id=row["src_id"], rel=row["rel"],
                dst_kind=row["dst_kind"], dst_id=row["dst_id"], weight=row["weight"],
                evidence_count=row["evidence_count"], meta=loads(row["meta"], {}), created_at=row["created_at"],
                updated_at=row["updated_at"])


class Graph:
    def __init__(self, db: Database) -> None:
        self.db = db

    def link(self, src: tuple[str, str], rel: str, dst: tuple[str, str], *, weight: float = 1.0,
             meta: dict[str, Any] | None = None) -> None:
        if rel not in RELATIONS:
            raise ValueError(f"unknown relation {rel!r}")
        t = now()
        self.db.execute(
            "INSERT INTO graph_edges (id, src_kind, src_id, rel, dst_kind, dst_id, weight, evidence_count, meta,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,1,?,?,?)"
            " ON CONFLICT(src_kind, src_id, rel, dst_kind, dst_id) DO UPDATE SET"
            " evidence_count = evidence_count + 1, weight = weight + excluded.weight, updated_at = excluded.updated_at",
            (new_id("edge"), src[0], src[1], rel, dst[0], dst[1], weight, dumps(meta or {}), t, t))

    def unlink(self, src: tuple[str, str], rel: str, dst: tuple[str, str]) -> None:
        self.db.execute("DELETE FROM graph_edges WHERE src_kind=? AND src_id=? AND rel=? AND dst_kind=? AND dst_id=?",
                        (src[0], src[1], rel, dst[0], dst[1]))

    def outgoing(self, kind: str, node_id: str, rels: set[str] | None = None) -> list[Edge]:
        rows = self.db.query("SELECT * FROM graph_edges WHERE src_kind = ? AND src_id = ?", (kind, node_id))
        return [_edge(r) for r in rows if rels is None or r["rel"] in rels]

    def incoming(self, kind: str, node_id: str, rels: set[str] | None = None) -> list[Edge]:
        rows = self.db.query("SELECT * FROM graph_edges WHERE dst_kind = ? AND dst_id = ?", (kind, node_id))
        return [_edge(r) for r in rows if rels is None or r["rel"] in rels]

    def neighbourhood(self, kind: str, node_id: str, depth: int = 2, limit: int = 200) -> dict[str, Any]:
        """Nodes/edges around a node, for provenance views and the learning-graph UI."""
        seen = {(kind, node_id)}
        frontier = [(kind, node_id)]
        edges: dict[str, Edge] = {}
        for _ in range(depth):
            nxt = []
            for k, n in frontier:
                for edge in self.outgoing(k, n) + self.incoming(k, n):
                    edges[edge.id] = edge
                    for node in ((edge.src_kind, edge.src_id), (edge.dst_kind, edge.dst_id)):
                        if node not in seen and len(seen) < limit:
                            seen.add(node)
                            nxt.append(node)
            frontier = nxt
        return {"nodes": [{"kind": k, "id": n} for k, n in seen], "edges": [e.model_dump() for e in edges.values()]}
