from __future__ import annotations

import math
from typing import Iterable

import networkx as nx

from core.events import (
    PLACEMENT_CHANGED,
    RISK_BATCH_UPDATED,
    RISK_UPDATED,
    ROAD_BLOCKED,
    EventBus,
)
from core.models import CityEdge, CityNode


RISK_MULTIPLIER = {0: 1.0, 1: 1.2, 2: 1.5}


class CityGraph:
    """Single source-of-truth graph shared across all modules."""

    def __init__(self) -> None:
        self._g = nx.Graph()
        self.events = EventBus()
        self._ambulance_positions: dict[str, str] = {}
        self.events.subscribe(PLACEMENT_CHANGED, self._on_placement_changed)

    def add_node(self, node: CityNode) -> None:
        self._g.add_node(
            node.node_id,
            location_type=node.location_type.value,
            population_density=float(node.population_density),
            risk_index=int(node.risk_index),
            accessible=bool(node.accessible),
            grid_row=node.grid_row,
            grid_col=node.grid_col,
            officer_allocation=max(0, int(node.officer_allocation)),
        )

    def grid_position(self, node_id: str) -> tuple[int, int] | None:
        if node_id not in self._g:
            return None
        data = self._g.nodes[node_id]
        row = data.get("grid_row")
        col = data.get("grid_col")
        if row is None or col is None:
            return None
        return int(row), int(col)

    def grid_dimensions(self) -> tuple[int, int] | None:
        positions = [self.grid_position(n) for n in self._g.nodes]
        positions = [p for p in positions if p is not None]
        if not positions:
            return None
        rows = max(p[0] for p in positions) + 1
        cols = max(p[1] for p in positions) + 1
        return rows, cols

    def set_edge_cost(self, node_a: str, node_b: str, base_cost: float) -> None:
        if not self._g.has_edge(node_a, node_b):
            raise KeyError(f"Road ({node_a}, {node_b}) not found.")
        self._g[node_a][node_b]["base_cost"] = float(base_cost)

    def add_edge(self, edge: CityEdge) -> None:
        self._g.add_edge(
            edge.node_a,
            edge.node_b,
            base_cost=float(edge.base_cost),
            blocked=bool(edge.blocked),
        )

    def nodes(self) -> list[str]:
        return list(self._g.nodes)

    def edges(self) -> list[tuple[str, str]]:
        return list(self._g.edges)

    def neighbors(self, node_id: str) -> list[str]:
        return list(self._g.neighbors(node_id))

    def adjacency_map(self) -> dict[str, list[str]]:
        return {node: list(self._g.neighbors(node)) for node in self._g.nodes}

    def set_risk(self, node_id: str, risk_index: int) -> None:
        if risk_index not in RISK_MULTIPLIER:
            raise ValueError("risk_index must be 0, 1, or 2")
        self._g.nodes[node_id]["risk_index"] = risk_index
        self.events.publish(RISK_UPDATED, {"node_id": node_id, "risk_index": risk_index})

    def set_risks_bulk(self, predictions: dict[str, int]) -> None:
        """Apply many risk updates atomically and emit a single batch event.

        Subscribers that need to react to *any* risk change should listen on
        ``RISK_BATCH_UPDATED`` instead of ``RISK_UPDATED`` so they fire once
        per full prediction pass. Per-node updates would otherwise trigger
        e.g. C3's GA to re-run dozens of times per simulation step.
        """
        applied: dict[str, int] = {}
        for node_id, risk_index in predictions.items():
            if risk_index not in RISK_MULTIPLIER:
                raise ValueError("risk_index must be 0, 1, or 2")
            if node_id not in self._g:
                continue
            self._g.nodes[node_id]["risk_index"] = int(risk_index)
            applied[str(node_id)] = int(risk_index)
        self.events.publish(RISK_BATCH_UPDATED, {"predictions": applied})

    def set_officer_allocation_bulk(self, allocation: dict[str, int]) -> None:
        """Write police officer counts per node (Challenge 5).

        Every graph node gets a count; missing keys default to 0 so the
        shared graph remains the single source of truth for policing coverage.
        """
        for node_id in self._g.nodes:
            count = int(allocation.get(str(node_id), 0))
            self._g.nodes[node_id]["officer_allocation"] = max(0, count)

    def officer_allocation_map(self) -> dict[str, int]:
        """Current officer counts keyed by node id (same keys as C5's allocation dict)."""
        return {
            str(n): max(0, int(self._g.nodes[n].get("officer_allocation", 0)))
            for n in self._g.nodes
        }

    def set_location_type(self, node_id: str, location_type: str) -> None:
        if node_id not in self._g:
            raise KeyError(f"Node {node_id} not found.")
        self._g.nodes[node_id]["location_type"] = location_type

    def set_accessible(self, node_id: str, accessible: bool) -> None:
        if node_id not in self._g:
            raise KeyError(f"Node {node_id} not found.")
        self._g.nodes[node_id]["accessible"] = bool(accessible)

    def set_ambulance_placement(self, ambulance_id: str, node_id: str) -> None:
        self.events.publish(
            PLACEMENT_CHANGED,
            {"ambulance_id": ambulance_id, "node_id": node_id},
        )

    def ambulance_positions(self) -> dict[str, str]:
        return dict(self._ambulance_positions)

    def block_road(self, node_a: str, node_b: str, blocked: bool = True) -> None:
        if not self._g.has_edge(node_a, node_b):
            raise KeyError(f"Road ({node_a}, {node_b}) not found.")
        self._g[node_a][node_b]["blocked"] = blocked
        if blocked:
            self.events.publish(ROAD_BLOCKED, {"edge": (node_a, node_b)})

    def remove_edge(self, node_a: str, node_b: str) -> None:
        self._g.remove_edge(node_a, node_b)

    def is_blocked(self, node_a: str, node_b: str) -> bool:
        return bool(self._g[node_a][node_b]["blocked"])

    def edge_cost(self, node_a: str, node_b: str) -> float:
        edge = self._g[node_a][node_b]
        if edge["blocked"]:
            return math.inf
        base_cost = float(edge["base_cost"])
        risk_a = int(self._g.nodes[node_a]["risk_index"])
        risk_b = int(self._g.nodes[node_b]["risk_index"])
        risk = max(risk_a, risk_b)
        return base_cost * RISK_MULTIPLIER[risk]

    def shortest_hops(self, source: str, cutoff: int | None = None) -> dict[str, int]:
        active_graph = self.to_networkx(include_blocked=False)
        return nx.single_source_shortest_path_length(active_graph, source=source, cutoff=cutoff)

    def shortest_path(self, source: str, target: str) -> list[str]:
        temp = nx.Graph()
        temp.add_nodes_from(self._g.nodes(data=True))
        for u, v in self._g.edges:
            w = self.edge_cost(u, v)
            if math.isfinite(w):
                temp.add_edge(u, v, weight=w)
        return nx.shortest_path(temp, source=source, target=target, weight="weight")

    def to_networkx(self, include_blocked: bool = False) -> nx.Graph:
        out = nx.Graph()
        if include_blocked:
            out.add_nodes_from(self._g.nodes(data=True))
        else:
            for node_id, attrs in self._g.nodes(data=True):
                if bool(attrs.get("accessible", True)):
                    out.add_node(node_id, **attrs)
        for u, v in self._g.edges:
            if u not in out or v not in out:
                continue
            if not include_blocked and self.is_blocked(u, v):
                continue
            out.add_edge(u, v, weight=self.edge_cost(u, v), base_cost=self._g[u][v]["base_cost"])
        return out

    def connected_components(self) -> Iterable[set[str]]:
        g = self.to_networkx(include_blocked=False)
        return nx.connected_components(g)

    def _on_placement_changed(self, payload: dict[str, str]) -> None:
        ambulance_id = payload.get("ambulance_id")
        node_id = payload.get("node_id")
        if not ambulance_id or not node_id:
            return
        if node_id not in self._g:
            return
        self._ambulance_positions[str(ambulance_id)] = str(node_id)

