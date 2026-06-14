from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import time
import networkx as nx

from challenges.c1_layout import CityLayoutCSP
from challenges.c2_roads import RoadNetworkOptimizer
from challenges.c3_ambulance import AmbulancePlacementGA, PlacementResult
from challenges.c4_routing import EmergencyRouter
from challenges.c5_crime import CrimeRiskPredictor, CrimeRiskRunResult
from core.city_graph import CityGraph
from core.events import RISK_BATCH_UPDATED, ROAD_BLOCKED
from core.models import CityEdge, CityNode, LocationType


SIMULATION_MODE_STRICT = "strict20"
SIMULATION_MODE_COMPLETE = "complete"
VALID_SIMULATION_MODES = (SIMULATION_MODE_STRICT, SIMULATION_MODE_COMPLETE)


def _near_square_factors(n: int) -> tuple[int, int]:
    """Return (rows, cols) with rows*cols == n and rows as close to cols as possible."""
    best = (1, n)
    for rows in range(2, int(n ** 0.5) + 1):
        if n % rows == 0:
            best = (rows, n // rows)
    return best


@dataclass(slots=True)
class RunConfig:
    seed: int = 42
    grid_rows: int = 0  # 0 => derive from node_count or default 5
    grid_cols: int = 0  # 0 => derive from node_count or default 5
    node_count: int = 0  # 0 => use grid_rows * grid_cols (or defaults)
    simulation_steps: int = 20
    ambulance_count: int = 3
    flood_every_n_steps: int = 4
    c3_recompute_interval: int = 5
    preferred_block_active_route_probability: float = 0.7
    output_dir: str = "run_outputs"
    simulation_mode: str = SIMULATION_MODE_STRICT
    risk_refresh_every_step: bool = True
    completion_step_cap: int = 500

    def __post_init__(self) -> None:
        if self.grid_rows and self.grid_cols:
            if self.grid_rows < 2 or self.grid_cols < 2:
                raise ValueError("grid_rows and grid_cols must each be at least 2.")
        elif self.node_count:
            if self.node_count < 6:
                raise ValueError("node_count must be at least 6 so all location types fit.")
            self.grid_rows, self.grid_cols = _near_square_factors(self.node_count)
            if self.grid_rows < 2 or self.grid_cols < 2:
                # Fall back to 2xK for prime totals.
                self.grid_rows = 2
                self.grid_cols = max(3, (self.node_count + 1) // 2)
        else:
            self.grid_rows, self.grid_cols = 5, 5
        expected_cells = self.grid_rows * self.grid_cols
        if expected_cells < 6:
            raise ValueError("Grid must hold at least 6 cells so every location type can be represented.")
        self.node_count = expected_cells
        if self.simulation_mode not in VALID_SIMULATION_MODES:
            raise ValueError(
                f"simulation_mode must be one of {VALID_SIMULATION_MODES}, got {self.simulation_mode!r}."
            )
        if self.completion_step_cap < self.simulation_steps:
            self.completion_step_cap = self.simulation_steps


@dataclass(slots=True)
class PipelineResult:
    summary_path: str
    event_log_path: str
    snapshot_path: str
    summary: dict[str, object]


def grid_node_id(row: int, col: int, cols: int) -> str:
    """Node IDs are sequential Node0..NodeN-1 in row-major order.

    Grid geometry is kept separately on the CityNode (grid_row/grid_col) so the
    renderer can still place nodes on a grid without parsing the ID.
    """
    return f"Node{row * cols + col}"


def build_initial_city(config: RunConfig) -> CityGraph:
    """Build an empty grid-based city graph.

    Per CityMind project spec, the city is modelled as a grid where nodes are
    locations and edges connect 4-directional neighbours (N/S/E/W). Location
    types are assigned by Challenge 1. Roads start at a neutral base cost of
    1.0; residential-edge discounts are applied after C1 (see apply_residential_edge_costs).
    """
    rng = random.Random(config.seed)
    graph = CityGraph()
    rows, cols = config.grid_rows, config.grid_cols
    for r in range(rows):
        for c in range(cols):
            graph.add_node(
                CityNode(
                    node_id=grid_node_id(r, c, cols),
                    location_type=LocationType.RESIDENTIAL,
                    population_density=float(rng.randint(15, 95)),
                    grid_row=r,
                    grid_col=c,
                )
            )
    for r in range(rows):
        for c in range(cols):
            here = grid_node_id(r, c, cols)
            if c + 1 < cols:
                graph.add_edge(CityEdge(here, grid_node_id(r, c + 1, cols), base_cost=1.0))
            if r + 1 < rows:
                graph.add_edge(CityEdge(here, grid_node_id(r + 1, c, cols), base_cost=1.0))
    return graph


def apply_residential_edge_costs(city_graph: CityGraph) -> int:
    """Set cost=0.8 on any edge where at least one endpoint is residential.

    Implements the spec rule: "Roads through residential zones cost 0.8".
    Returns the number of edges re-priced so the caller can log the change.
    """
    adjusted = 0
    graph = city_graph.to_networkx(include_blocked=True)
    residential_value = LocationType.RESIDENTIAL.value
    for u, v in city_graph.edges():
        if (
            graph.nodes[u].get("location_type") == residential_value
            or graph.nodes[v].get("location_type") == residential_value
        ):
            city_graph.set_edge_cost(u, v, 0.8)
            adjusted += 1
    return adjusted


def run_c1_stage(city_graph: CityGraph, seed: int | None = None) -> dict[str, object]:
    adjacency = city_graph.adjacency_map()
    solver = CityLayoutCSP(
        adjacency=adjacency,
        node_ids=city_graph.nodes(),
        rng_seed=seed,
    )
    result = solver.solve()
    for node_id, location_type in result.assignment.items():
        city_graph.set_location_type(node_id, location_type.value)
    return {
        "valid": result.valid,
        "conflict_rule": result.conflict_rule,
        "conflict_details": result.conflict_details or [],
        "type_counts": dict(result.type_counts),
    }


def run_c2_stage(city_graph: CityGraph) -> dict[str, object]:
    hospitals = _node_ids_by_location_type(city_graph, LocationType.HOSPITAL.value)
    depots = _node_ids_by_location_type(city_graph, LocationType.AMBULANCE_DEPOT.value)
    if not hospitals or not depots:
        raise ValueError("C2 requires at least one hospital and one ambulance depot from C1 layout.")
    hospital_id = hospitals[0]
    depot_id = depots[0]

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(city_graph.nodes())
    for u, v in city_graph.edges():
        nx_graph.add_edge(
            u,
            v,
            weight=city_graph.edge_cost(u, v),
            base_cost=city_graph.edge_cost(u, v),
        )

    # Snapshot C1's location types so the optimizer can re-validate the hop
    # constraints after Kruskal prunes edges and add cheapest non-tree edges
    # back if any hop guarantees were broken.
    full_graph = city_graph.to_networkx(include_blocked=True)
    location_types = {
        str(node_id): str(attrs.get("location_type", ""))
        for node_id, attrs in full_graph.nodes(data=True)
    }

    optimizer = RoadNetworkOptimizer(nx_graph, hospital_id=hospital_id, depot_id=depot_id)
    optimized = optimizer.build(enforce_c1_hops=location_types)

    optimized_edges = {tuple(sorted(edge)) for edge in optimized.edges()}
    removed_edges = 0
    for u, v in list(city_graph.edges()):
        if tuple(sorted((u, v))) not in optimized_edges:
            city_graph.remove_edge(u, v)
            removed_edges += 1

    # Persist any new edges the hop-pass added so the live CityGraph matches
    # the optimizer's idea of the road network. At C2 time risk_index is 0
    # everywhere, so weight == base_cost.
    existing_edges = {tuple(sorted(e)) for e in city_graph.edges()}
    persisted_added_edges = 0
    for u, v in optimized.edges():
        if tuple(sorted((u, v))) in existing_edges:
            continue
        edge_data = optimized[u][v]
        base_cost = float(edge_data.get("base_cost", edge_data.get("weight", 1.0)))
        city_graph.add_edge(CityEdge(u, v, base_cost=base_cost))
        persisted_added_edges += 1

    hop_report = optimizer.hop_augmentation_report
    edges_added_for_hops = (
        len(hop_report.get("edges_added_for_hops", []))
        if isinstance(hop_report, dict)
        else 0
    )

    return {
        "hospital_id": hospital_id,
        "depot_id": depot_id,
        "optimized_edge_count": len(optimized_edges),
        "removed_edge_count": removed_edges,
        "added_edges_for_hops": edges_added_for_hops,
        "persisted_added_edges": persisted_added_edges,
        "redundant_hospital_depot_routes": optimizer.has_redundant_hospital_depot_routes(optimized),
        "c1_hop_check": hop_report,
    }


def run_c5_stage(city_graph: CityGraph, seed: int) -> CrimeRiskRunResult:
    predictor = CrimeRiskPredictor(random_state=seed)
    return predictor.run(city_graph)


def run_c3_stage(city_graph: CityGraph, config: RunConfig) -> tuple[AmbulancePlacementGA, PlacementResult]:
    optimizer = AmbulancePlacementGA(random_state=config.seed)
    result = optimizer.optimize(city_graph, ambulance_count=config.ambulance_count)
    # Subscribe to the *batched* risk event so a full C5 pass triggers a
    # single GA recompute instead of one per node update. Per-node
    # RISK_UPDATED would re-run the GA dozens of times per simulation step.

    def _safe_c3_recompute(_payload: dict[str, object]) -> None:
        try:
            optimizer.recompute_placement()
        except Exception:
            pass

    city_graph.events.subscribe(RISK_BATCH_UPDATED, _safe_c3_recompute)
    return optimizer, result


def run_c4_stage(city_graph: CityGraph, c2_details: dict[str, object]) -> EmergencyRouter:
    hospital_id = str(c2_details["hospital_id"])
    router = EmergencyRouter(city_graph=city_graph)
    city_graph.events.subscribe(ROAD_BLOCKED, router.on_road_blocked)
    full_graph = city_graph.to_networkx(include_blocked=True)
    civilians = [
        n
        for n in city_graph.nodes()
        if str(full_graph.nodes[n].get("location_type")) == LocationType.RESIDENTIAL.value
    ]
    router.plan_route(start_node=hospital_id, civilians=civilians)
    return router


SIMULATION_EVENT_BUFFER_LIMIT = 10000


def run_integrated_simulation(
    city_graph: CityGraph,
    router: EmergencyRouter,
    c3_optimizer: AmbulancePlacementGA,
    config: RunConfig,
    c5_predictor: CrimeRiskPredictor | None = None,
) -> dict[str, object]:
    """Run the integrated CityMind simulation loop.

    Two modes are supported (controlled by ``config.simulation_mode``):

    - ``strict20``: stops after exactly ``simulation_steps`` ticks.
    - ``complete``: keeps stepping past ``simulation_steps`` until every
      civilian has been reached or proven unreachable, bounded by
      ``completion_step_cap`` to avoid runaway loops.

    When ``risk_refresh_every_step`` is enabled, the C5 predictor runs after
    each flood event so risk weights actually shift during simulation. The
    bulk-update emits a single ``RISK_BATCH_UPDATED`` event, which is what
    triggers a (single) C3 recompute per refresh.
    """
    rng = random.Random(config.seed + 997)
    events: list[dict[str, object]] = []
    c3_recomputes = 0
    c5_refreshes = 0

    def record(event: dict[str, object]) -> None:
        events.append(event)
        overflow = len(events) - SIMULATION_EVENT_BUFFER_LIMIT
        if overflow > 0:
            del events[:overflow]

    mode = config.simulation_mode
    strict_cap = config.simulation_steps
    hard_cap = max(strict_cap, config.completion_step_cap)

    # Track per-step recomputes so the post-step periodic recompute does not
    # double-run the GA when a risk refresh already triggered it that tick.
    step = 0
    executed_steps = 0
    while True:
        next_step = step + 1
        if mode == SIMULATION_MODE_STRICT and next_step > strict_cap:
            break
        if mode == SIMULATION_MODE_COMPLETE:
            if next_step > hard_cap:
                break
            if next_step > strict_cap and not router.pending_civilians:
                # Mission terminal: every civilian reached or marked unreachable.
                break
        step = next_step
        executed_steps = step

        print(f"[STEP {step}] tick")
        record({"step": step, "event": "SIM_TICK", "mode": mode})

        if config.flood_every_n_steps > 0 and step % config.flood_every_n_steps == 0:
            blocked = _block_random_road(city_graph, router, rng, config)
            if blocked is not None:
                print(f"[STEP {step}] flood blocked edge={blocked[0]}-{blocked[1]}")
                record({"step": step, "event": "ROAD_BLOCKED", "edge": list(blocked)})

        risk_refresh_recomputed = False
        if config.risk_refresh_every_step and c5_predictor is not None:
            try:
                c5_step_result = c5_predictor.run(city_graph)
                c5_refreshes += 1
                risk_refresh_recomputed = True  # C3 recomputed via RISK_BATCH_UPDATED.
                print(
                    f"[STEP {step}] c5 risk refresh "
                    f"counts={dict(c5_step_result.risk_counts)} "
                    f"acc={c5_step_result.accuracy:.2f}"
                )
                record(
                    {
                        "step": step,
                        "event": "C5_RISK_REFRESH",
                        "risk_counts": {
                            int(k): int(v) for k, v in c5_step_result.risk_counts.items()
                        },
                        "accuracy": float(c5_step_result.accuracy),
                        "class_balance": dict(c5_step_result.class_balance),
                        "fallback_used": bool(c5_step_result.fallback_used),
                        "warnings": list(c5_step_result.warnings),
                    }
                )
                # The bus-driven recompute already updated the GA; reflect it
                # in stage events for traceability.
                last = c3_optimizer.last_result
                if last is not None:
                    c3_recomputes += 1
                    record(
                        {
                            "step": step,
                            "event": "C3_RECOMPUTE",
                            "trigger": "risk_refresh",
                            "ambulance_nodes": list(last.ambulance_nodes),
                            "fitness": float(last.fitness),
                        }
                    )
            except Exception as exc:  # defensive: keep the sim alive on rare ML hiccups
                record(
                    {
                        "step": step,
                        "event": "C5_RISK_REFRESH_ERROR",
                        "error": str(exc),
                    }
                )

        if (
            not risk_refresh_recomputed
            and config.c3_recompute_interval > 0
            and step % config.c3_recompute_interval == 0
        ):
            recomputed = c3_optimizer.recompute_placement()
            c3_recomputes += 1
            print(
                f"[STEP {step}] c3 recompute nodes={recomputed.ambulance_nodes} "
                f"worst={recomputed.fitness:.2f}"
            )
            record(
                {
                    "step": step,
                    "event": "C3_RECOMPUTE",
                    "trigger": "interval",
                    "ambulance_nodes": list(recomputed.ambulance_nodes),
                    "fitness": float(recomputed.fitness),
                }
            )

        moved = router.advance_one_step()
        current_target = router.current_target if router.current_target is not None else "-"
        print(
            f"[STEP {step}] c4 moved={moved} pos={router.current_position} "
            f"target={current_target} pending={len(router.pending_civilians)}"
        )
        record(
            {
                "step": step,
                "event": "C4_ADVANCE",
                "moved": moved,
                "current_position": router.current_position,
                "pending_civilians": len(router.pending_civilians),
            }
        )

        # Completion mode: terminate as soon as the mission is fully resolved
        # (no more pending civilians) once we are past the strict cap.
        if (
            mode == SIMULATION_MODE_COMPLETE
            and step >= strict_cap
            and not router.pending_civilians
        ):
            break

    return {
        "events": events,
        "c3_recompute_count": c3_recomputes,
        "c5_refresh_count": c5_refreshes,
        "executed_steps": executed_steps,
        "c4_event_log_size": len(router.event_log),
        "c4_replan_count": sum(1 for item in router.event_log if "replanning" in item.lower()),
    }


def run_citymind(config: RunConfig) -> PipelineResult:
    stage_timings: dict[str, float] = {}
    stage_results: dict[str, object] = {}

    city_graph = build_initial_city(config)
    print(
        f"[INIT] seed={config.seed} nodes={config.node_count} "
        f"steps={config.simulation_steps} ambulances={config.ambulance_count} "
        f"mode={config.simulation_mode}"
    )

    started = time.perf_counter()
    stage_results["c1"] = run_c1_stage(city_graph, seed=config.seed)
    reweighted = apply_residential_edge_costs(city_graph)
    stage_results["c1"]["residential_edges_reweighted"] = reweighted
    stage_timings["c1_seconds"] = time.perf_counter() - started
    print(
        f"[C1] valid={stage_results['c1']['valid']} "
        f"counts={stage_results['c1']['type_counts']} "
        f"conflict_rule={stage_results['c1']['conflict_rule']} "
        f"residential_edges={reweighted}"
    )

    started = time.perf_counter()
    stage_results["c2"] = run_c2_stage(city_graph)
    stage_timings["c2_seconds"] = time.perf_counter() - started
    hop_check = stage_results["c2"].get("c1_hop_check", {})
    post_violations = hop_check.get("post_fix_violations", {}) if isinstance(hop_check, dict) else {}
    print(
        f"[C2] hospital={stage_results['c2']['hospital_id']} "
        f"depot={stage_results['c2']['depot_id']} "
        f"edges={stage_results['c2']['optimized_edge_count']} "
        f"redundant={stage_results['c2']['redundant_hospital_depot_routes']} "
        f"hop_added={stage_results['c2'].get('added_edges_for_hops', 0)} "
        f"hop_residual={sum(len(v) for v in post_violations.values())}"
    )

    started = time.perf_counter()
    c5_predictor = CrimeRiskPredictor(random_state=config.seed)
    c5_result = c5_predictor.run(city_graph)
    stage_timings["c5_seconds"] = time.perf_counter() - started
    stage_results["c5"] = {
        "accuracy": float(c5_result.accuracy),
        "risk_counts": dict(c5_result.risk_counts),
        "officer_allocation": dict(c5_result.officer_allocation),
        "class_balance": dict(c5_result.class_balance),
        "warnings": list(c5_result.warnings),
        "fallback_used": bool(c5_result.fallback_used),
    }
    print(
        f"[C5] accuracy={c5_result.accuracy:.2f} "
        f"risk_counts={dict(c5_result.risk_counts)}"
    )

    started = time.perf_counter()
    c3_optimizer, c3_result = run_c3_stage(city_graph, config=config)
    stage_timings["c3_seconds"] = time.perf_counter() - started
    stage_results["c3"] = asdict(c3_result)
    print(f"[C3] nodes={c3_result.ambulance_nodes} worst={c3_result.fitness:.2f}")

    started = time.perf_counter()
    router = run_c4_stage(city_graph, c2_details=stage_results["c2"])
    stage_timings["c4_init_seconds"] = time.perf_counter() - started
    stage_results["c4_init"] = {
        "start_position": router.current_position,
        "initial_target": router.current_target,
        "initial_path": list(router.current_path),
    }
    print(f"[C4] start={router.current_position} initial_path={router.current_path}")

    started = time.perf_counter()
    sim_result = run_integrated_simulation(
        city_graph=city_graph,
        router=router,
        c3_optimizer=c3_optimizer,
        config=config,
        c5_predictor=c5_predictor if config.risk_refresh_every_step else None,
    )
    stage_timings["simulation_seconds"] = time.perf_counter() - started

    snapshot = _build_final_snapshot(city_graph, router, c3_result)
    summary = {
        "config": asdict(config),
        "stage_results": stage_results,
        "stage_timings": stage_timings,
        "simulation": {
            "mode": config.simulation_mode,
            "configured_steps": config.simulation_steps,
            "executed_steps": int(sim_result.get("executed_steps", config.simulation_steps)),
            "completion_step_cap": config.completion_step_cap,
            "event_count": len(sim_result["events"]),
            "c3_recompute_count": int(sim_result["c3_recompute_count"]),
            "c5_refresh_count": int(sim_result.get("c5_refresh_count", 0)),
            "c4_event_log_size": int(sim_result["c4_event_log_size"]),
            "c4_replan_count": int(sim_result["c4_replan_count"]),
        },
        "final_snapshot": snapshot,
    }
    paths = _write_run_artifacts(config, summary, sim_result["events"])
    print(
        f"[DONE] mode={config.simulation_mode} "
        f"executed_steps={summary['simulation']['executed_steps']} "
        f"events={summary['simulation']['event_count']} "
        f"replans={summary['simulation']['c4_replan_count']} "
        f"reached={len(snapshot['c4_reached_civilians'])} "
        f"unreachable={len(snapshot['c4_unreachable_civilians'])}"
    )
    return PipelineResult(
        summary_path=paths["summary_path"],
        event_log_path=paths["event_log_path"],
        snapshot_path=paths["snapshot_path"],
        summary=summary,
    )


def _write_run_artifacts(
    config: RunConfig,
    summary: dict[str, object],
    event_stream: list[dict[str, object]],
) -> dict[str, str]:
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Directory key includes seed, grid shape, step cap, mode, and a wall-clock
    # timestamp so distinct configs cannot overwrite each other and repeat
    # invocations of the same config remain independently traceable.
    run_dir = out_root / (
        f"seed_{config.seed}"
        f"_grid_{config.grid_rows}x{config.grid_cols}"
        f"_nodes_{config.node_count}"
        f"_steps_{config.simulation_steps}"
        f"_mode_{config.simulation_mode}"
        f"_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    event_log_path = run_dir / "event_log.jsonl"
    snapshot_path = run_dir / "final_snapshot.json"

    # Embed a config fingerprint in the summary for unambiguous provenance.
    summary_with_provenance = dict(summary)
    summary_with_provenance["provenance"] = {
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "config_fingerprint": asdict(config),
    }

    summary_path.write_text(json.dumps(summary_with_provenance, indent=2), encoding="utf-8")
    with event_log_path.open("w", encoding="utf-8") as handle:
        for event in event_stream:
            handle.write(json.dumps(event) + "\n")
    snapshot_path.write_text(json.dumps(summary["final_snapshot"], indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "event_log_path": str(event_log_path),
        "snapshot_path": str(snapshot_path),
    }


def _build_final_snapshot(
    city_graph: CityGraph,
    router: EmergencyRouter,
    c3_result: PlacementResult,
) -> dict[str, object]:
    graph = city_graph.to_networkx(include_blocked=True)
    risk_counts = {0: 0, 1: 0, 2: 0}
    for _node, attrs in graph.nodes(data=True):
        risk = int(attrs.get("risk_index", 0))
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    blocked_edges = [
        [u, v] for u, v in city_graph.edges() if city_graph.is_blocked(u, v)
    ]
    return {
        "risk_counts": risk_counts,
        "officer_allocation": city_graph.officer_allocation_map(),
        "ambulance_nodes": list(c3_result.ambulance_nodes),
        "ambulance_positions": city_graph.ambulance_positions(),
        "ambulance_worst_case_response_time": float(c3_result.fitness),
        "c4_current_position": router.current_position,
        "c4_reached_civilians": list(router.reached_civilians),
        "c4_unreachable_civilians": list(router.unreachable_civilians),
        "c4_pending_civilians": list(router.pending_civilians),
        "blocked_edges": blocked_edges,
    }


def _block_random_road(
    city_graph: CityGraph,
    router: EmergencyRouter,
    rng: random.Random,
    config: RunConfig,
) -> tuple[str, str] | None:
    unblocked_edges = [
        edge for edge in city_graph.edges() if not city_graph.is_blocked(edge[0], edge[1])
    ]
    if not unblocked_edges:
        return None

    active_nx = city_graph.to_networkx(include_blocked=False)
    if active_nx.number_of_nodes() == 0 or active_nx.number_of_edges() == 0:
        return None
    bridge_set = {tuple(sorted(e)) for e in nx.bridges(active_nx)}
    safe_edges = [e for e in unblocked_edges if tuple(sorted(e)) not in bridge_set]
    if not safe_edges:
        return None

    active_path_edges = list(router.current_path_edges())
    safe_active = [e for e in active_path_edges if tuple(sorted(e)) not in bridge_set]
    pick_active = (
        safe_active
        and rng.random() < config.preferred_block_active_route_probability
    )
    if pick_active:
        edge = rng.choice(safe_active)
        u, v = edge
    else:
        u, v = rng.choice(safe_edges)
    city_graph.block_road(u, v, blocked=True)
    return tuple(sorted((u, v)))


def _node_ids_by_location_type(city_graph: CityGraph, location_type: str) -> list[str]:
    graph = city_graph.to_networkx(include_blocked=True)
    return sorted(
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("location_type")) == location_type
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CityMind non-UI orchestration pipeline.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic execution.")
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps.")
    parser.add_argument("--rows", type=int, default=None, help="Grid rows (>=2).")
    parser.add_argument("--cols", type=int, default=None, help="Grid columns (>=2).")
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Total grid cells; auto-selects a near-square rows/cols pair (overridden by --rows/--cols).",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_SIMULATION_MODES,
        default=None,
        help=(
            "Simulation mode: 'strict20' stops at the configured step count; "
            "'complete' continues until every civilian has been reached or "
            "marked unreachable (bounded by --completion-cap)."
        ),
    )
    parser.add_argument(
        "--completion-cap",
        type=int,
        default=None,
        help="Hard upper bound on steps in 'complete' mode (default: 500).",
    )
    parser.add_argument(
        "--no-risk-refresh",
        action="store_true",
        help="Disable per-step C5 risk refresh during the simulation loop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides: dict[str, object] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.steps is not None:
        overrides["simulation_steps"] = args.steps
    if args.rows is not None:
        overrides["grid_rows"] = args.rows
    if args.cols is not None:
        overrides["grid_cols"] = args.cols
    if args.nodes is not None and "grid_rows" not in overrides and "grid_cols" not in overrides:
        overrides["node_count"] = args.nodes
    if args.mode is not None:
        overrides["simulation_mode"] = args.mode
    if args.completion_cap is not None:
        overrides["completion_step_cap"] = args.completion_cap
    if args.no_risk_refresh:
        overrides["risk_refresh_every_step"] = False

    config = RunConfig(**overrides) if overrides else RunConfig()

    result = run_citymind(config)
    print("Run summary written:", result.summary_path)
    print("Event log written:", result.event_log_path)
    print("Snapshot written:", result.snapshot_path)

