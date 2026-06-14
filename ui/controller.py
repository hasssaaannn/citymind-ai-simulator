from __future__ import annotations

from dataclasses import dataclass, field
import random

import networkx as nx

from challenges.c5_crime import CrimeRiskPredictor
from core.city_graph import CityGraph
from main import (
    SIMULATION_MODE_COMPLETE,
    SIMULATION_MODE_STRICT,
    RunConfig,
    _block_random_road,
    apply_residential_edge_costs,
    build_initial_city,
    run_c1_stage,
    run_c2_stage,
    run_c3_stage,
    run_c4_stage,
)


@dataclass(slots=True)
class UIEvent:
    step: int
    kind: str
    message: str


def _fmt_nodes(nodes: list[str], limit: int = 3) -> str:
    """Compact representation for logs; keeps a stable width regardless of N."""
    if len(nodes) <= limit:
        return "[" + ",".join(nodes) + "]"
    return "[" + ",".join(nodes[:limit]) + f",+{len(nodes) - limit}]"


@dataclass
class SimulationController:
    """Drives C1-C5 pipeline step-by-step using main.py's stage functions.

    The controller mirrors the headless simulation loop in :func:`main.run_integrated_simulation`
    so the GUI shows exactly the same integrated behaviour: per-step floods,
    per-step C5 risk refresh (with batched C3 recompute), C4 movement, and
    optional dual-mode termination (``strict20`` vs ``complete``).
    """

    config: RunConfig
    city_graph: CityGraph = field(init=False)
    stage_results: dict[str, object] = field(init=False, default_factory=dict)
    router: object = field(init=False, default=None)
    c3_optimizer: object = field(init=False, default=None)
    c3_result: object = field(init=False, default=None)
    c5_predictor: CrimeRiskPredictor = field(init=False, default=None)
    node_positions: dict[str, tuple[float, float]] = field(init=False, default_factory=dict)
    events: list[UIEvent] = field(init=False, default_factory=list)
    current_step: int = field(init=False, default=0)
    _rng: random.Random = field(init=False, default=None)
    _completed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reset(self.config)

    def reset(self, config: RunConfig) -> None:
        self.config = config
        self.events = []
        self.current_step = 0
        self._completed = False
        self._rng = random.Random(config.seed + 997)

        self.city_graph = build_initial_city(config)
        self._log(
            "INIT",
            f"nodes={config.node_count} seed={config.seed} "
            f"steps={config.simulation_steps} mode={config.simulation_mode}",
        )

        c1 = run_c1_stage(self.city_graph, seed=config.seed)
        reweighted_edges = apply_residential_edge_costs(self.city_graph)
        c1["residential_edges_reweighted"] = reweighted_edges
        self.stage_results["c1"] = c1
        self._log(
            "C1",
            f"valid={c1['valid']} conflict={c1['conflict_rule']} "
            f"residential_edges={reweighted_edges}",
        )

        c2 = run_c2_stage(self.city_graph)
        self.stage_results["c2"] = c2
        hop_check = c2.get("c1_hop_check", {})
        post_violations = hop_check.get("post_fix_violations", {}) if isinstance(hop_check, dict) else {}
        residual = sum(len(v) for v in post_violations.values()) if isinstance(post_violations, dict) else 0
        self._log(
            "C2",
            f"hospital={c2['hospital_id']} depot={c2['depot_id']} "
            f"edges={c2['optimized_edge_count']} redundant={c2['redundant_hospital_depot_routes']} "
            f"hop_added={c2.get('added_edges_for_hops', 0)} hop_residual={residual}",
        )

        self.c5_predictor = CrimeRiskPredictor(random_state=config.seed)
        c5 = self.c5_predictor.run(self.city_graph)
        self.stage_results["c5"] = {
            "accuracy": float(c5.accuracy),
            "risk_counts": dict(c5.risk_counts),
            "officer_allocation": dict(c5.officer_allocation),
            "class_balance": dict(c5.class_balance),
            "warnings": list(c5.warnings),
            "fallback_used": bool(c5.fallback_used),
        }
        self._log("C5", f"accuracy={c5.accuracy:.2f} risk_counts={dict(c5.risk_counts)}")

        self.c3_optimizer, self.c3_result = run_c3_stage(self.city_graph, config=config)
        self.stage_results["c3"] = {
            "ambulance_nodes": list(self.c3_result.ambulance_nodes),
            "fitness": float(self.c3_result.fitness),
        }
        self._log(
            "C3",
            f"ambulances={_fmt_nodes(self.c3_result.ambulance_nodes)} "
            f"worst={self.c3_result.fitness:.2f}",
        )

        self.router = run_c4_stage(self.city_graph, c2_details=self.stage_results["c2"])
        self._log(
            "C4",
            f"start={self.router.current_position} "
            f"path_len={len(self.router.current_path)}",
        )

        self.node_positions = self._compute_layout()

    def step(self, ignore_step_limit: bool = False) -> bool:
        """Advance one simulation tick.

        Set ``ignore_step_limit=True`` to manually run beyond the configured
        step cap. The controller also honours :attr:`RunConfig.simulation_mode`:
        in ``complete`` mode it keeps stepping past the strict cap until every
        civilian is reached or proven unreachable (bounded by
        ``completion_step_cap``).
        """
        mode_completion = (
            self.config.simulation_mode == SIMULATION_MODE_COMPLETE
            and self.router is not None
            and bool(self.router.pending_civilians)
        )
        allow_overrun = ignore_step_limit or mode_completion

        if self._completed:
            if not (allow_overrun and self.router and self.router.pending_civilians):
                return False
            self._completed = False

        strict_cap = self.config.simulation_steps
        hard_cap = max(strict_cap, self.config.completion_step_cap)

        if not allow_overrun and self.current_step >= strict_cap:
            self._completed = True
            return False
        if self.current_step >= hard_cap:
            self._completed = True
            self._log(
                "DONE",
                f"hard_cap_reached step={self.current_step} "
                f"pending={len(self.router.pending_civilians) if self.router else 0}",
            )
            return False

        self.current_step += 1
        step = self.current_step
        self._log("TICK", f"step {step} mode={self.config.simulation_mode}")

        if self.config.flood_every_n_steps > 0 and step % self.config.flood_every_n_steps == 0:
            blocked = _block_random_road(self.city_graph, self.router, self._rng, self.config)
            if blocked is not None:
                self._log("ROAD_BLOCKED", f"{blocked[0]}-{blocked[1]}")

        risk_refresh_recomputed = False
        if self.config.risk_refresh_every_step and self.c5_predictor is not None:
            try:
                c5_step = self.c5_predictor.run(self.city_graph)
                self.stage_results["c5"] = {
                    "accuracy": float(c5_step.accuracy),
                    "risk_counts": dict(c5_step.risk_counts),
                    "officer_allocation": dict(c5_step.officer_allocation),
                    "class_balance": dict(c5_step.class_balance),
                    "warnings": list(c5_step.warnings),
                    "fallback_used": bool(c5_step.fallback_used),
                }
                self._log(
                    "C5_REFRESH",
                    f"acc={c5_step.accuracy:.2f} risk_counts={dict(c5_step.risk_counts)} "
                    f"fallback={c5_step.fallback_used}",
                )
                last = self.c3_optimizer.last_result
                if last is not None:
                    self.c3_result = last
                    self.stage_results["c3"] = {
                        "ambulance_nodes": list(last.ambulance_nodes),
                        "fitness": float(last.fitness),
                    }
                    self._log(
                        "C3_RECOMPUTE",
                        f"trigger=risk_refresh "
                        f"ambulances={_fmt_nodes(last.ambulance_nodes)} "
                        f"worst={last.fitness:.2f}",
                    )
                    risk_refresh_recomputed = True
            except Exception as exc:  # defensive: keep the GUI alive
                self._log("C5_REFRESH_ERROR", str(exc))

        if (
            not risk_refresh_recomputed
            and self.config.c3_recompute_interval > 0
            and step % self.config.c3_recompute_interval == 0
        ):
            recomputed = self.c3_optimizer.recompute_placement()
            self.c3_result = recomputed
            self.stage_results["c3"] = {
                "ambulance_nodes": list(recomputed.ambulance_nodes),
                "fitness": float(recomputed.fitness),
            }
            self._log(
                "C3_RECOMPUTE",
                f"trigger=interval "
                f"ambulances={_fmt_nodes(recomputed.ambulance_nodes)} "
                f"worst={recomputed.fitness:.2f}",
            )

        reached_before = list(self.router.reached_civilians)
        unreachable_before = list(self.router.unreachable_civilians)
        log_before = len(self.router.event_log)

        moved = self.router.advance_one_step()
        kind = "C4_MOVE" if moved else "C4_IDLE"
        target = self.router.current_target if self.router.current_target is not None else "-"
        self._log(
            kind,
            f"pos={self.router.current_position} target={target} pending={len(self.router.pending_civilians)}",
        )

        for entry in self.router.event_log[log_before:]:
            if "replanning" in entry.lower():
                compact = entry.replace("Replanning ", "replan ")
                if len(compact) > 90:
                    compact = compact[:87] + "..."
                self._log("C4_REPLAN", compact)

        for reached in self.router.reached_civilians[len(reached_before):]:
            self._log("C4_REACHED", f"civilian={reached}")
        for unreachable in self.router.unreachable_civilians[len(unreachable_before):]:
            self._log("C4_UNREACHABLE", f"civilian={unreachable}")

        # Termination logic per mode.
        mission_resolved = self.router is not None and not self.router.pending_civilians
        if self.config.simulation_mode == SIMULATION_MODE_COMPLETE:
            if step >= strict_cap and mission_resolved:
                self._completed = True
                self._log(
                    "DONE",
                    f"mode=complete step={step} reached={len(self.router.reached_civilians)} "
                    f"unreachable={len(self.router.unreachable_civilians)} pending=0",
                )
            elif step >= hard_cap:
                self._completed = True
                self._log(
                    "DONE",
                    f"mode=complete hard_cap step={step} "
                    f"pending={len(self.router.pending_civilians)}",
                )
        else:  # strict mode
            if step >= strict_cap and not allow_overrun:
                self._completed = True
                self._log(
                    "DONE",
                    f"mode=strict20 step={step} reached={len(self.router.reached_civilians)} "
                    f"pending={len(self.router.pending_civilians)}",
                )
            elif allow_overrun and mission_resolved:
                self._completed = True
                self._log(
                    "DONE",
                    f"mode=strict20 run_all step={step} pending=0",
                )
        return True

    def ambulance_nodes(self) -> list[str]:
        result = self.stage_results.get("c3")
        if isinstance(result, dict):
            return list(result.get("ambulance_nodes", []))
        return []

    def police_officer_allocation(self) -> dict[str, int]:
        return self.city_graph.officer_allocation_map()

    def risk_index_for(self, node_id: str) -> int:
        graph = self.city_graph.to_networkx(include_blocked=True)
        return int(graph.nodes[node_id].get("risk_index", 0))

    def location_type_for(self, node_id: str) -> str:
        graph = self.city_graph.to_networkx(include_blocked=True)
        return str(graph.nodes[node_id].get("location_type", ""))

    def is_blocked(self, u: str, v: str) -> bool:
        return self.city_graph.is_blocked(u, v)

    def _compute_layout(self) -> dict[str, tuple[float, float]]:
        """Use grid coordinates when available; fall back to spring layout."""
        graph = self.city_graph.to_networkx(include_blocked=True)
        if graph.number_of_nodes() == 0:
            return {}
        grid_positions: dict[str, tuple[float, float]] = {}
        for node in graph.nodes():
            pos = self.city_graph.grid_position(str(node))
            if pos is None:
                grid_positions = {}
                break
            grid_positions[str(node)] = (float(pos[1]), float(pos[0]))
        if grid_positions:
            return grid_positions
        positions = nx.spring_layout(graph, seed=self.config.seed, iterations=120)
        return {str(node): (float(pos[0]), float(pos[1])) for node, pos in positions.items()}

    def grid_dimensions(self) -> tuple[int, int] | None:
        return self.city_graph.grid_dimensions()

    _EVENT_BUFFER_LIMIT = 2000

    def _log(self, kind: str, message: str) -> None:
        self.events.append(UIEvent(step=self.current_step, kind=kind, message=message))
        overflow = len(self.events) - self._EVENT_BUFFER_LIMIT
        if overflow > 0:
            del self.events[:overflow]
