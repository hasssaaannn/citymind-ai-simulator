from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math

import networkx as nx


@dataclass
class EmergencyRouter:
    """Challenge 4: A* routing with event-driven replanning."""

    city_graph: object
    current_position: str | None = None
    pending_civilians: list[str] = field(default_factory=list)
    reached_civilians: list[str] = field(default_factory=list)
    unreachable_civilians: list[str] = field(default_factory=list)
    current_target: str | None = None
    current_path: list[str] = field(default_factory=list)
    event_log: list[str] = field(default_factory=list)

    def plan_route(self, start_node: str, civilians: list[str]) -> list[str]:
        self.current_position = start_node
        # Preserve order while removing duplicates and excluding the start node.
        self.pending_civilians = [node for node in dict.fromkeys(civilians) if node != start_node]
        self.reached_civilians = []
        self.unreachable_civilians = []
        self.current_target = None
        self.current_path = [start_node]
        self.event_log.append(
            f"C4 plan initialized at {start_node} with civilians: {self.pending_civilians}"
        )
        self.replan_from_current_position()
        return list(self.current_path)

    def advance_one_step(self) -> bool:
        if self.current_position is None:
            return False

        if not self.current_path or len(self.current_path) == 1:
            if self.current_target is None and self.pending_civilians:
                self.replan_from_current_position()
            if not self.current_path or len(self.current_path) == 1:
                return False

        next_node = self.current_path[1]
        if self.city_graph.is_blocked(self.current_position, next_node):
            self.event_log.append(
                f"C4 step blocked on ({self.current_position}, {next_node}); replanning."
            )
            self.replan_from_current_position()
            return False

        previous = self.current_position
        self.current_position = next_node
        self.current_path = self.current_path[1:]
        self.event_log.append(f"C4 moved {previous} -> {self.current_position}")

        if self.current_target is not None and self.current_position == self.current_target:
            self.reached_civilians.append(self.current_target)
            if self.current_target in self.pending_civilians:
                self.pending_civilians.remove(self.current_target)
            self.event_log.append(f"C4 reached civilian {self.current_target}")
            self.current_target = None
            self.current_path = [self.current_position]
            self.replan_from_current_position()
        return True

    def on_road_blocked(self, payload: dict[str, tuple[str, str]]) -> None:
        blocked_edge = payload.get("edge")
        if blocked_edge is None:
            return
        normalized = tuple(sorted(blocked_edge))
        if normalized in self.current_path_edges():
            self.event_log.append(f"C4 road blocked on active route: {blocked_edge}; replanning.")
            self.replan_from_current_position()

    def replan_from_current_position(self) -> list[str]:
        if self.current_position is None:
            return []
        if not self.pending_civilians:
            self.current_target = None
            self.current_path = [self.current_position]
            self.event_log.append("C4 mission complete.")
            return list(self.current_path)

        active_graph = self.city_graph.to_networkx(include_blocked=False)
        target, path = self._select_next_target(active_graph)
        if target is None or path is None:
            self.current_target = None
            self.current_path = [self.current_position]
            self.event_log.append("C4 no reachable civilians remain.")
            return list(self.current_path)

        self.current_target = target
        self.current_path = path
        self.event_log.append(f"C4 planned route to {target}: {path}")
        return list(self.current_path)

    def current_path_edges(self) -> set[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for idx in range(len(self.current_path) - 1):
            edges.add(tuple(sorted((self.current_path[idx], self.current_path[idx + 1]))))
        return edges

    def _select_next_target(self, active_graph: nx.Graph) -> tuple[str | None, list[str] | None]:
        if self.current_position is None:
            return None, None

        best_target: str | None = None
        best_path: list[str] | None = None
        best_cost: float | None = None

        for target in list(self.pending_civilians):
            result = self._astar_search(active_graph, self.current_position, target)
            if result is None:
                if target in self.pending_civilians:
                    self.pending_civilians.remove(target)
                    self.unreachable_civilians.append(target)
                    self.event_log.append(
                        f"C4 civilian unreachable from {self.current_position}: {target}"
                    )
                continue
            path, cost = result
            if (
                best_cost is None
                or cost < best_cost
                or (cost == best_cost and target < (best_target or target))
            ):
                best_target = target
                best_path = path
                best_cost = cost

        return best_target, best_path

    def _astar_search(
        self,
        graph: nx.Graph,
        source: str,
        goal: str,
    ) -> tuple[list[str], float] | None:
        """Hand-rolled A* on the active (unblocked, risk-weighted) graph.

        Returns ``(path, cost)`` for the optimum route from ``source`` to
        ``goal`` or ``None`` if no route exists. The grid-Manhattan
        heuristic is admissible *and* consistent on this graph because the
        smallest possible step cost is 0.8 (residential edge under the
        lowest risk multiplier of 1.0) while Manhattan distance changes by
        at most 1 per neighbour move. Consistency lets us safely keep a
        closed set so each node is expanded at most once.
        """
        if source == goal:
            return [source], 0.0
        if source not in graph or goal not in graph:
            return None

        # Tie-breaker counter keeps heap ordering stable when two entries
        # share the same f-score (heapq otherwise compares the node ids,
        # which would error on non-comparable types).
        counter = itertools.count()
        start_h = self._heuristic(source, goal)
        open_heap: list[tuple[float, int, str]] = [(start_h, next(counter), source)]
        g_score: dict[str, float] = {source: 0.0}
        came_from: dict[str, str] = {}
        closed: set[str] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current), g_score[current]
            if current in closed:
                continue
            closed.add(current)

            for neighbour, edge_data in graph[current].items():
                if neighbour in closed:
                    continue
                step_cost = float(edge_data.get("weight", 1.0))
                if not math.isfinite(step_cost):
                    # Blocked / impassable edges live as +inf weights in the
                    # snapshot graph; treat them as missing entirely.
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbour, math.inf):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f_score, next(counter), neighbour))

        return None

    @staticmethod
    def _reconstruct_path(came_from: dict[str, str], goal: str) -> list[str]:
        path = [goal]
        node = goal
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path

    def _heuristic(self, node: str, goal: str) -> float:
        node_pos = self.city_graph.grid_position(node)
        goal_pos = self.city_graph.grid_position(goal)
        if node_pos is None or goal_pos is None:
            # Fallback remains admissible when coordinates are unavailable.
            return 0.0
        # Manhattan hop count times minimum possible edge weight (0.8 residential
        # discount) so h never exceeds true shortest-path cost on the weighted grid.
        manhattan = float(abs(node_pos[0] - goal_pos[0]) + abs(node_pos[1] - goal_pos[1]))
        return 0.8 * manhattan

