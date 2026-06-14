from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx
import pygame

from core.models import LocationType

from ui import theme
from ui.controller import SimulationController


OVERLAY_LAYOUT = 1
OVERLAY_ROADS = 2
OVERLAY_COVERAGE = 3
OVERLAY_RISK = 4

OVERLAY_NAMES = {
    OVERLAY_LAYOUT: "Layout",
    OVERLAY_ROADS: "Road Network",
    OVERLAY_COVERAGE: "Ambulance Coverage",
    OVERLAY_RISK: "Crime Risk",
}


@dataclass(slots=True)
class CanvasRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)


class GridRenderer:
    """Renders CityGraph state onto a pygame surface."""

    NODE_RADIUS = 13

    def __init__(self, canvas: CanvasRect) -> None:
        self.canvas = canvas
        self._coverage_cache: dict[str, int] = {}
        self._coverage_signature: tuple = ()

    def resize(self, canvas: CanvasRect) -> None:
        self.canvas = canvas

    def screen_positions(
        self, controller: SimulationController
    ) -> dict[str, tuple[int, int]]:
        positions = controller.node_positions
        if not positions:
            return {}
        padding = 78
        width = max(self.canvas.width - padding * 2, 1)
        height = max(self.canvas.height - padding * 2, 1)

        dims = controller.grid_dimensions()
        screen_positions: dict[str, tuple[int, int]] = {}
        if dims is not None:
            rows, cols = dims
            if cols == 1 and rows == 1:
                return {
                    node: (self.canvas.x + padding, self.canvas.y + padding)
                    for node in positions
                }
            # Pick a uniform cell size so grid stays square.
            step_x = width / max(cols - 1, 1)
            step_y = height / max(rows - 1, 1)
            step = min(step_x, step_y)
            total_w = step * max(cols - 1, 1)
            total_h = step * max(rows - 1, 1)
            origin_x = self.canvas.x + (self.canvas.width - total_w) / 2
            origin_y = self.canvas.y + (self.canvas.height - total_h) / 2
            for node_id, (col, row) in positions.items():
                screen_x = int(origin_x + col * step)
                screen_y = int(origin_y + row * step)
                screen_positions[node_id] = (screen_x, screen_y)
            return screen_positions

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)

        for node_id, (x, y) in positions.items():
            nx_norm = (x - min_x) / span_x
            ny_norm = (y - min_y) / span_y
            screen_x = int(self.canvas.x + padding + nx_norm * width)
            screen_y = int(self.canvas.y + padding + ny_norm * height)
            screen_positions[node_id] = (screen_x, screen_y)
        return screen_positions

    def draw(
        self,
        surface: pygame.Surface,
        controller: SimulationController,
        overlay_mode: int,
        font: pygame.font.Font,
        pulse: float,
    ) -> None:
        pygame.draw.rect(surface, theme.PANEL_BG, self.canvas.rect, border_radius=10)
        pygame.draw.rect(surface, theme.PANEL_BORDER, self.canvas.rect, width=1, border_radius=10)

        positions = self.screen_positions(controller)
        if not positions:
            return

        self._draw_grid_guidelines(surface, controller, positions)
        active_path_edges = self._active_path_edges(controller)
        self._draw_edges(
            surface,
            controller,
            positions,
            overlay_mode=overlay_mode,
            active_path_edges=active_path_edges,
        )
        self._draw_active_route_arrow(surface, controller, positions, pulse)

        coverage_map = None
        if overlay_mode == OVERLAY_COVERAGE:
            coverage_map = self._compute_coverage(controller)

        ambulance_set = set(controller.ambulance_nodes())
        hospital_id = self._hospital_id(controller)

        for node_id, (sx, sy) in positions.items():
            color = self._node_color(controller, node_id, overlay_mode, coverage_map)
            # Soft shadow + dual-ring style for better depth and readability.
            pygame.draw.circle(surface, (10, 12, 18), (sx + 2, sy + 2), self.NODE_RADIUS + 1)
            pygame.draw.circle(surface, color, (sx, sy), self.NODE_RADIUS)
            pygame.draw.circle(surface, (20, 22, 28), (sx, sy), self.NODE_RADIUS, 2)
            pygame.draw.circle(surface, (255, 255, 255), (sx - 4, sy - 4), 2)

            label = font.render(node_id, True, theme.HUD_TEXT)
            surface.blit(label, (sx - label.get_width() // 2, sy + self.NODE_RADIUS + 2))

        # Ambulances (pulsing outer ring)
        for amb_node in ambulance_set:
            if amb_node not in positions:
                continue
            sx, sy = positions[amb_node]
            pulse_radius = int(self.NODE_RADIUS + 6 + 3 * math.sin(pulse))
            pygame.draw.circle(
                surface,
                theme.AMBULANCE_MARKER,
                (sx, sy),
                pulse_radius,
                2,
            )

        # Current C4 position marker
        if controller.router is not None and controller.router.current_position in positions:
            sx, sy = positions[controller.router.current_position]
            pygame.draw.circle(surface, theme.HIGHLIGHT, (sx, sy), self.NODE_RADIUS + 4, 2)

        # Hospital marker
        if hospital_id and hospital_id in positions:
            sx, sy = positions[hospital_id]
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy), self.NODE_RADIUS + 8, 1)

        # Civilian state markers (dot overlay)
        self._draw_civilian_markers(surface, controller, positions)
        # Police allocation markers (small numeric badges).
        self._draw_police_markers(surface, controller, positions, font)

    def _draw_grid_guidelines(
        self,
        surface: pygame.Surface,
        controller: SimulationController,
        positions: dict[str, tuple[int, int]],
    ) -> None:
        dims = controller.grid_dimensions()
        if dims is None or not positions:
            return
        xs = sorted({p[0] for p in positions.values()})
        ys = sorted({p[1] for p in positions.values()})
        if len(xs) < 2 or len(ys) < 2:
            return
        guide_color = (40, 45, 58)
        pad = 12
        for x in xs:
            pygame.draw.line(
                surface,
                guide_color,
                (x, ys[0] - pad),
                (x, ys[-1] + pad),
                1,
            )
        for y in ys:
            pygame.draw.line(
                surface,
                guide_color,
                (xs[0] - pad, y),
                (xs[-1] + pad, y),
                1,
            )

    def _draw_edges(
        self,
        surface: pygame.Surface,
        controller: SimulationController,
        positions: dict[str, tuple[int, int]],
        overlay_mode: int,
        active_path_edges: set[tuple[str, str]],
    ) -> None:
        for u, v in controller.city_graph.edges():
            if u not in positions or v not in positions:
                continue
            start = positions[u]
            end = positions[v]
            blocked = controller.is_blocked(u, v)
            edge_key = tuple(sorted((u, v)))
            active = edge_key in active_path_edges

            if blocked:
                self._draw_dashed_line(surface, theme.EDGE_BLOCKED, start, end, width=4, dash_length=10)
            elif active:
                pygame.draw.line(surface, theme.EDGE_ACTIVE, start, end, 5)
            else:
                edge_cost = controller.city_graph.edge_cost(u, v)
                width = self._road_width(edge_cost=edge_cost, overlay_mode=overlay_mode)
                color = self._road_color(edge_cost=edge_cost, overlay_mode=overlay_mode)
                pygame.draw.line(surface, color, start, end, width)

    def _draw_dashed_line(
        self,
        surface: pygame.Surface,
        color: tuple[int, int, int],
        start: tuple[int, int],
        end: tuple[int, int],
        width: int = 2,
        dash_length: int = 8,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        if distance == 0:
            return
        steps = max(int(distance // (dash_length * 2)), 1)
        for i in range(steps):
            t0 = (i * 2) / (steps * 2)
            t1 = (i * 2 + 1) / (steps * 2)
            sx = int(x1 + dx * t0)
            sy = int(y1 + dy * t0)
            ex = int(x1 + dx * t1)
            ey = int(y1 + dy * t1)
            pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)

    def _node_color(
        self,
        controller: SimulationController,
        node_id: str,
        overlay_mode: int,
        coverage_map: dict[str, int] | None,
    ) -> tuple[int, int, int]:
        if overlay_mode == OVERLAY_RISK:
            return theme.risk_color(controller.risk_index_for(node_id))
        if overlay_mode == OVERLAY_COVERAGE and coverage_map is not None:
            idx = coverage_map.get(node_id)
            if idx is None:
                return (120, 120, 120)
            return theme.coverage_color(idx)
        return theme.location_color(controller.location_type_for(node_id))

    def _road_width(self, edge_cost: float, overlay_mode: int) -> int:
        if edge_cost <= 1.0:
            return 2 if overlay_mode != OVERLAY_ROADS else 3
        if edge_cost <= 1.3:
            return 3 if overlay_mode != OVERLAY_ROADS else 4
        return 4 if overlay_mode != OVERLAY_ROADS else 5

    def _road_color(self, edge_cost: float, overlay_mode: int) -> tuple[int, int, int]:
        if overlay_mode != OVERLAY_ROADS:
            return theme.EDGE_DEFAULT
        if edge_cost <= 1.0:
            return (120, 136, 170)
        if edge_cost <= 1.3:
            return (175, 150, 105)
        return (210, 120, 95)

    def _draw_civilian_markers(
        self,
        surface: pygame.Surface,
        controller: SimulationController,
        positions: dict[str, tuple[int, int]],
    ) -> None:
        router = controller.router
        if router is None:
            return
        pending = set(router.pending_civilians)
        reached = set(router.reached_civilians)
        unreachable = set(router.unreachable_civilians)
        for node_id, (sx, sy) in positions.items():
            marker_color = None
            if node_id in reached:
                marker_color = theme.CIVILIAN_REACHED
            elif node_id in unreachable:
                marker_color = theme.CIVILIAN_UNREACHABLE
            elif node_id in pending:
                marker_color = theme.CIVILIAN_PENDING
            if marker_color is None:
                continue
            pygame.draw.circle(surface, marker_color, (sx - 14, sy - 14), 4)

    def _draw_police_markers(
        self,
        surface: pygame.Surface,
        controller: SimulationController,
        positions: dict[str, tuple[int, int]],
        font: pygame.font.Font,
    ) -> None:
        allocation = controller.police_officer_allocation()
        if not allocation:
            return
        for node_id, officer_count in allocation.items():
            if officer_count <= 0 or node_id not in positions:
                continue
            sx, sy = positions[node_id]
            bx, by = sx + 14, sy - 14
            pygame.draw.circle(surface, (10, 14, 22), (bx + 1, by + 1), 9)
            pygame.draw.circle(surface, theme.POLICE_MARKER, (bx, by), 9)
            pygame.draw.circle(surface, (18, 24, 36), (bx, by), 9, 1)

            label = str(officer_count) if officer_count < 10 else "9+"
            text = font.render(label, True, theme.POLICE_TEXT)
            surface.blit(text, (bx - text.get_width() // 2, by - text.get_height() // 2))

    def _draw_active_route_arrow(
        self,
        surface: pygame.Surface,
        controller: SimulationController,
        positions: dict[str, tuple[int, int]],
        pulse: float,
    ) -> None:
        router = controller.router
        if router is None or len(router.current_path) < 2:
            return

        route_points = [positions[node] for node in router.current_path if node in positions]
        if len(route_points) < 2:
            return

        segments: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        total_length = 0.0
        for start, end in zip(route_points, route_points[1:]):
            seg_len = math.hypot(end[0] - start[0], end[1] - start[1])
            if seg_len <= 1e-6:
                continue
            segments.append((start, end, seg_len))
            total_length += seg_len
        if total_length <= 1e-6:
            return

        # Continuous loop over the visible active route.
        # Keep arrow movement intentionally slow for readability.
        travel = (pulse * 30.0) % total_length
        head_x = float(route_points[0][0])
        head_y = float(route_points[0][1])
        dir_x, dir_y = 1.0, 0.0
        traversed = 0.0
        for start, end, seg_len in segments:
            if traversed + seg_len >= travel:
                ratio = (travel - traversed) / seg_len
                head_x = start[0] + (end[0] - start[0]) * ratio
                head_y = start[1] + (end[1] - start[1]) * ratio
                dir_x = (end[0] - start[0]) / seg_len
                dir_y = (end[1] - start[1]) / seg_len
                break
            traversed += seg_len

        head = (int(head_x), int(head_y))
        tail_len = 12.0
        wing_len = 8.0

        tail = (
            int(head_x - dir_x * tail_len),
            int(head_y - dir_y * tail_len),
        )
        perp_x, perp_y = -dir_y, dir_x
        wing_a = (
            int(head_x - dir_x * wing_len + perp_x * wing_len * 0.7),
            int(head_y - dir_y * wing_len + perp_y * wing_len * 0.7),
        )
        wing_b = (
            int(head_x - dir_x * wing_len - perp_x * wing_len * 0.7),
            int(head_y - dir_y * wing_len - perp_y * wing_len * 0.7),
        )

        pygame.draw.line(surface, (18, 20, 24), tail, head, 4)
        pygame.draw.line(surface, theme.EDGE_ACTIVE, tail, head, 2)
        pygame.draw.polygon(surface, theme.EDGE_ACTIVE, [head, wing_a, wing_b])

    def _active_path_edges(self, controller: SimulationController) -> set[tuple[str, str]]:
        router = controller.router
        if router is None or not router.current_path:
            return set()
        edges: set[tuple[str, str]] = set()
        for a, b in zip(router.current_path, router.current_path[1:]):
            edges.add(tuple(sorted((a, b))))
        return edges

    def _compute_coverage(self, controller: SimulationController) -> dict[str, int]:
        ambulances = controller.ambulance_nodes()
        signature = (tuple(ambulances), self._graph_signature(controller))
        if signature == self._coverage_signature and self._coverage_cache:
            return self._coverage_cache
        self._coverage_signature = signature
        if not ambulances:
            self._coverage_cache = {}
            return self._coverage_cache

        graph = controller.city_graph.to_networkx(include_blocked=False)
        assignment: dict[str, int] = {}
        for idx, amb_node in enumerate(ambulances):
            if amb_node not in graph:
                continue
            try:
                distances = nx.single_source_dijkstra_path_length(graph, amb_node, weight="weight")
            except nx.NetworkXError:
                continue
            for node, dist in distances.items():
                current = assignment.get(node)
                if current is None:
                    assignment[node] = idx
                else:
                    existing_amb = ambulances[current]
                    if existing_amb in graph:
                        existing_dist = nx.single_source_dijkstra_path_length(
                            graph, existing_amb, weight="weight"
                        ).get(node, math.inf)
                    else:
                        existing_dist = math.inf
                    if dist < existing_dist:
                        assignment[node] = idx
        self._coverage_cache = assignment
        return assignment

    def _graph_signature(self, controller: SimulationController) -> tuple:
        blocked = tuple(
            sorted(
                (u, v)
                for u, v in controller.city_graph.edges()
                if controller.is_blocked(u, v)
            )
        )
        return blocked

    def _hospital_id(self, controller: SimulationController) -> str | None:
        c2 = controller.stage_results.get("c2")
        if isinstance(c2, dict):
            hospital = c2.get("hospital_id")
            if hospital is not None:
                return str(hospital)
        for node_id in controller.city_graph.nodes():
            if controller.location_type_for(node_id) == LocationType.HOSPITAL.value:
                return node_id
        return None


class LegendRenderer:
    def __init__(self) -> None:
        self.layout_entries: list[tuple[str, tuple[int, int, int]]] = [
            (location.value, theme.location_color(location.value))
            for location in LocationType
        ]

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        overlay_mode: int,
    ) -> None:
        title = font.render(f"Legend ({OVERLAY_NAMES.get(overlay_mode, '-')})", True, theme.HUD_TEXT)
        surface.blit(title, (x, y))
        line_height = font.get_linesize() + 2
        entries = self._entries_for_mode(overlay_mode)
        for idx, (label, color) in enumerate(entries):
            offset_y = y + (idx + 1) * line_height
            self._draw_symbol(
                surface=surface,
                overlay_mode=overlay_mode,
                entry_index=idx,
                color=color,
                x=x + 7,
                y=offset_y + 8,
            )
            text = font.render(label.replace("_", " ").title(), True, theme.HUD_TEXT)
            surface.blit(text, (x + 20, offset_y))

    def _draw_symbol(
        self,
        surface: pygame.Surface,
        overlay_mode: int,
        entry_index: int,
        color: tuple[int, int, int],
        x: int,
        y: int,
    ) -> None:
        if overlay_mode == OVERLAY_ROADS:
            if entry_index == 0:  # blocked road dashed
                pygame.draw.line(surface, color, (x - 6, y), (x - 1, y), 3)
                pygame.draw.line(surface, color, (x + 1, y), (x + 6, y), 3)
            else:
                width = 4 if entry_index == 1 else 2 + (entry_index - 2)
                pygame.draw.line(surface, color, (x - 6, y), (x + 6, y), width)
            return

        if overlay_mode == OVERLAY_COVERAGE:
            if entry_index == 0:  # ambulance ring
                pygame.draw.circle(surface, color, (x, y), 6, 2)
            elif entry_index == 1:  # police badge
                pygame.draw.circle(surface, color, (x, y), 6)
                pygame.draw.circle(surface, (20, 22, 28), (x, y), 6, 1)
            elif entry_index in (2, 3, 4):  # coverage zones
                pygame.draw.circle(surface, color, (x, y), 5)
            else:  # civilians
                pygame.draw.circle(surface, color, (x, y), 4)
            return

        if overlay_mode == OVERLAY_RISK:
            # Heatmap style rounded squares
            pygame.draw.rect(surface, color, pygame.Rect(x - 6, y - 5, 12, 10), border_radius=3)
            return

        # Layout default: node circles
        pygame.draw.circle(surface, color, (x, y), 5)
        pygame.draw.circle(surface, (20, 22, 28), (x, y), 5, 1)

    def _entries_for_mode(self, overlay_mode: int) -> list[tuple[str, tuple[int, int, int]]]:
        if overlay_mode == OVERLAY_ROADS:
            return [
                ("blocked road", theme.EDGE_BLOCKED),
                ("active route", theme.EDGE_ACTIVE),
                ("low cost road", (120, 136, 170)),
                ("medium cost road", (175, 150, 105)),
                ("high cost road", (210, 120, 95)),
            ]
        if overlay_mode == OVERLAY_COVERAGE:
            return [
                ("ambulance", theme.AMBULANCE_MARKER),
                ("police officers", theme.POLICE_MARKER),
                ("coverage zone A", theme.coverage_color(0)),
                ("coverage zone B", theme.coverage_color(1)),
                ("coverage zone C", theme.coverage_color(2)),
                ("civilian pending", theme.CIVILIAN_PENDING),
                ("civilian reached", theme.CIVILIAN_REACHED),
            ]
        if overlay_mode == OVERLAY_RISK:
            return [
                ("low risk", theme.risk_color(0)),
                ("medium risk", theme.risk_color(1)),
                ("high risk", theme.risk_color(2)),
            ]
        return self.layout_entries + [("police officers", theme.POLICE_MARKER)]
