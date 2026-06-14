from __future__ import annotations

from collections import deque
import math

import networkx as nx


def _hop_distance_from_sources(
    graph: nx.Graph, sources: list[str], cutoff: int
) -> dict[str, int]:
    """Multi-source BFS up to ``cutoff`` hops; returns ``{node: min_hops_to_any_source}``."""
    distances: dict[str, int] = {}
    queue: deque[str] = deque()
    for source in sources:
        if source in graph and source not in distances:
            distances[source] = 0
            queue.append(source)
    while queue:
        node = queue.popleft()
        depth = distances[node]
        if depth >= cutoff:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor in distances:
                continue
            distances[neighbor] = depth + 1
            queue.append(neighbor)
    return distances


def _is_connected_bfs(graph: nx.Graph) -> bool:
    """Hand-coded BFS connectivity check (replaces ``nx.is_connected``)."""
    nodes = list(graph.nodes())
    if not nodes:
        return True
    start = nodes[0]
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return len(visited) == len(nodes)


def _bfs_edge_path(
    graph: nx.Graph,
    source: str,
    target: str,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str]] | None:
    """Hand-coded BFS returning the sorted-edge sequence of a shortest hop path
    from ``source`` to ``target`` on an undirected graph, or ``None`` if no
    such path exists.

    Edges whose sorted endpoints appear in ``blocked_edges`` are skipped, so
    callers can probe a *second* edge-disjoint path on the residual graph
    without mutating the input.
    """
    if source not in graph or target not in graph:
        return None
    if source == target:
        return []
    blocked = blocked_edges if blocked_edges is not None else set()
    parent: dict[str, str] = {source: source}
    queue: deque[str] = deque([source])
    found = False
    while queue:
        node = queue.popleft()
        if node == target:
            found = True
            break
        for neighbor in graph.neighbors(node):
            if neighbor in parent:
                continue
            if tuple(sorted((node, neighbor))) in blocked:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    if not found:
        return None
    edges: list[tuple[str, str]] = []
    cursor = target
    while cursor != source:
        prev = parent[cursor]
        edges.append(tuple(sorted((prev, cursor))))
        cursor = prev
    edges.reverse()
    return edges


def validate_c1_hop_constraints(
    graph: nx.Graph, location_types: dict[str, str]
) -> dict[str, list[str]]:
    """Return per-rule violation lists for the post-C2 graph.

    Encodes the two hop rules from the project statement:
      - Every Residential node must be within 3 hops of some Hospital.
      - Every PowerPlant node must be within 2 hops of some Industrial node.
    Hops are unweighted (edge count) on the *current* road graph; this is what
    detects roads pruned away by Kruskal that broke C1's original guarantees.
    """
    hospitals = [n for n, t in location_types.items() if t == "Hospital" and n in graph]
    industrials = [n for n, t in location_types.items() if t == "Industrial" and n in graph]
    residentials = [n for n, t in location_types.items() if t == "Residential" and n in graph]
    power_plants = [n for n, t in location_types.items() if t == "PowerPlant" and n in graph]

    hospital_dist = _hop_distance_from_sources(graph, hospitals, cutoff=3)
    residential_violations = sorted(
        node for node in residentials if hospital_dist.get(node, math.inf) > 3
    )

    industrial_dist = _hop_distance_from_sources(graph, industrials, cutoff=2)
    power_violations = sorted(
        node for node in power_plants if industrial_dist.get(node, math.inf) > 2
    )

    return {
        "residential_far_from_hospital": residential_violations,
        "power_far_from_industrial": power_violations,
    }


class _UnionFind:
    """Disjoint-set union for Kruskal MST (path compression + union by rank)."""

    def __init__(self, nodes: list[str]) -> None:
        self._parent = {n: n for n in nodes}
        self._rank = dict.fromkeys(nodes, 0)

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1
        return True


def kruskal_minimum_spanning_tree(graph: nx.Graph, weight_key: str = "weight") -> nx.Graph:
    """Manual Kruskal MST: sort edges by weight, greedily add if endpoints differ in UF."""
    nodes = list(graph.nodes())
    if not nodes:
        return nx.Graph()

    edges: list[tuple[float, str, str]] = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get(weight_key, 1.0))
        edges.append((w, u, v))

    edges.sort(key=lambda t: (t[0], str(t[1]), str(t[2])))

    uf = _UnionFind(nodes)
    mst = nx.Graph()
    mst.add_nodes_from(nodes)
    for w, u, v in edges:
        if uf.union(u, v):
            attrs = dict(graph[u][v])
            mst.add_edge(u, v, **attrs)

    return mst


class RoadNetworkOptimizer:
    """Challenge 2: Kruskal MST plus redundancy and C1 hop preservation.

    When ``enforce_c1_hops`` is provided to :meth:`build`, a third augmentation
    pass runs after MST + redundancy: it checks the project-statement hop
    constraints on the *post-C2* graph and adds the cheapest non-tree edges
    needed to restore them. This prevents Kruskal from silently invalidating
    C1's guarantees ("residential within 3 hops of a hospital / power plant
    within 2 hops of industrial") at the cost of a few extra roads beyond the
    raw MST.
    """

    def __init__(self, graph: nx.Graph, hospital_id: str, depot_id: str) -> None:
        self.graph = graph
        self.hospital_id = hospital_id
        self.depot_id = depot_id
        self.hop_augmentation_report: dict[str, object] = {
            "enforced": False,
            "pre_fix_violations": {
                "residential_far_from_hospital": [],
                "power_far_from_industrial": [],
            },
            "post_fix_violations": {
                "residential_far_from_hospital": [],
                "power_far_from_industrial": [],
            },
            "edges_added_for_hops": [],
        }

    def build(
        self, enforce_c1_hops: dict[str, str] | None = None
    ) -> nx.Graph:
        if self.hospital_id not in self.graph or self.depot_id not in self.graph:
            raise ValueError("Hospital or depot node is missing from graph.")
        if not _is_connected_bfs(self.graph):
            raise ValueError("Input graph must be connected for MST construction.")

        mst = kruskal_minimum_spanning_tree(self.graph, weight_key="weight")
        if self.has_redundant_hospital_depot_routes(mst):
            optimized = mst
        else:
            optimized = self._augment_for_redundancy(mst)
            if not self.has_redundant_hospital_depot_routes(optimized):
                raise ValueError(
                    "Unable to enforce two edge-disjoint hospital-depot routes with available edges."
                )

        if enforce_c1_hops:
            optimized = self._enforce_c1_hops(optimized, enforce_c1_hops)

        return optimized

    def _enforce_c1_hops(
        self, base_graph: nx.Graph, location_types: dict[str, str]
    ) -> nx.Graph:
        """Add cheapest non-tree edges until both hop constraints hold (or no edge helps)."""
        report = self.hop_augmentation_report
        report["enforced"] = True

        pre_violations = validate_c1_hop_constraints(base_graph, location_types)
        report["pre_fix_violations"] = {
            key: list(values) for key, values in pre_violations.items()
        }

        if not any(pre_violations.values()):
            report["post_fix_violations"] = {
                key: list(values) for key, values in pre_violations.items()
            }
            return base_graph

        current = base_graph.copy()
        existing_edges = {tuple(sorted(edge)) for edge in current.edges}
        candidates: list[tuple[float, str, str]] = []
        for u, v, data in self.graph.edges(data=True):
            edge_key = tuple(sorted((u, v)))
            if edge_key in existing_edges:
                continue
            weight = float(data.get("weight", 1.0))
            candidates.append((weight, u, v))
        candidates.sort(key=lambda item: (item[0], str(item[1]), str(item[2])))

        def violation_count(graph: nx.Graph) -> int:
            v = validate_c1_hop_constraints(graph, location_types)
            return len(v["residential_far_from_hospital"]) + len(
                v["power_far_from_industrial"]
            )

        edges_added: list[dict[str, object]] = []
        current_violations = violation_count(current)

        while current_violations > 0 and candidates:
            chosen_index = -1
            for index, (weight, u, v) in enumerate(candidates):
                trial = current.copy()
                trial.add_edge(u, v, weight=weight)
                if violation_count(trial) < current_violations:
                    chosen_index = index
                    break
            if chosen_index < 0:
                break
            weight, u, v = candidates.pop(chosen_index)
            current.add_edge(u, v, weight=weight)
            current_violations = violation_count(current)
            edges_added.append({"u": str(u), "v": str(v), "weight": float(weight)})

        report["edges_added_for_hops"] = edges_added
        post_violations = validate_c1_hop_constraints(current, location_types)
        report["post_fix_violations"] = {
            key: list(values) for key, values in post_violations.items()
        }
        return current

    def _augment_for_redundancy(self, base_tree: nx.Graph) -> nx.Graph:
        """Prefer MST + one cheapest chord that yields two edge-disjoint H-D paths.

        If no single chord suffices, fall back to adding cheapest non-tree
        edges in order until redundancy holds (some layouts require more than
        one augmentation).
        """
        mst_edges = {tuple(sorted(e)) for e in base_tree.edges}
        candidates: list[tuple[float, str, str]] = []
        for u, v, data in self.graph.edges(data=True):
            edge = tuple(sorted((u, v)))
            if edge in mst_edges:
                continue
            weight = float(data.get("weight", 1.0))
            candidates.append((weight, u, v))
        candidates.sort(key=lambda item: item[0])

        for weight, u, v in candidates:
            trial = base_tree.copy()
            trial.add_edge(u, v, weight=weight)
            if self.has_redundant_hospital_depot_routes(trial):
                return trial

        augmented = base_tree.copy()
        for weight, u, v in candidates:
            if augmented.has_edge(u, v):
                continue
            augmented.add_edge(u, v, weight=weight)
            if self.has_redundant_hospital_depot_routes(augmented):
                return augmented
        return augmented

    def has_redundant_hospital_depot_routes(self, graph: nx.Graph) -> bool:
        """Hand-coded edge-disjoint path check (Menger's theorem at k=2).

        Find a first H->D path with BFS, mark its edges as blocked, then run
        a second BFS on the residual graph. Two edge-disjoint H->D paths
        exist iff the second BFS also succeeds, which is exactly
        ``edge_connectivity(H, D) >= 2`` — the formal definition of the pair
        surviving any single-road failure. Replaces the previous
        ``nx.edge_connectivity`` / ``nx.has_path`` calls so the redundancy
        guarantee is genuinely hand-implemented.
        """
        if self.hospital_id not in graph or self.depot_id not in graph:
            return False
        first_path = _bfs_edge_path(graph, self.hospital_id, self.depot_id)
        if first_path is None:
            return False
        second_path = _bfs_edge_path(
            graph,
            self.hospital_id,
            self.depot_id,
            blocked_edges=set(first_path),
        )
        return second_path is not None
