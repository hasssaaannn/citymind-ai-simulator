from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random

import networkx as nx


@dataclass(slots=True)
class PlacementResult:
    ambulance_nodes: list[str]
    fitness: float
    generations_run: int
    population_size: int


class AmbulancePlacementGA:
    """Challenge 3: place ambulances using a genetic algorithm."""

    def __init__(
        self,
        population_size: int = 80,
        max_generations: int = 500,
        tournament_size: int = 5,
        mutation_rate: float = 0.2,
        random_state: int = 42,
        no_improvement_limit: int = 50,
    ) -> None:
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.no_improvement_limit = no_improvement_limit
        self._rng = random.Random(random_state)

        self._last_graph = None
        self._last_ambulance_count = 3
        self._last_result: PlacementResult | None = None
        self._fitness_cache: dict[frozenset[str], float] = {}

    @property
    def last_result(self) -> PlacementResult | None:
        return self._last_result

    def optimize(self, city_graph, ambulance_count: int = 3) -> PlacementResult:
        self._last_graph = city_graph
        self._last_ambulance_count = ambulance_count

        nodes = city_graph.nodes()
        if ambulance_count <= 0:
            raise ValueError("ambulance_count must be positive.")
        if len(nodes) < ambulance_count:
            raise ValueError("ambulance_count cannot exceed number of graph nodes.")

        graph = city_graph.to_networkx(include_blocked=False)

        pop_size = self.population_size
        gen_cap = self.max_generations
        patience = self.no_improvement_limit
        self._fitness_cache.clear()

        population = [self._random_candidate(nodes, ambulance_count) for _ in range(pop_size)]

        best_candidate = population[0]
        best_fitness = self._fitness(graph, best_candidate)
        generations_run = 0
        no_improvement = 0

        for generation in range(1, gen_cap + 1):
            scored = [(candidate, self._fitness(graph, candidate)) for candidate in population]
            scored.sort(key=lambda item: item[1])
            current_best_candidate, current_best_fitness = scored[0]

            if current_best_fitness < best_fitness:
                best_candidate = list(current_best_candidate)
                best_fitness = current_best_fitness
                no_improvement = 0
            else:
                no_improvement += 1

            generations_run = generation
            if no_improvement >= patience:
                break

            next_population: list[list[str]] = [list(best_candidate)]
            while len(next_population) < pop_size:
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)
                child = self._crossover(parent_a, parent_b, nodes, ambulance_count)
                child = self._mutate(child, nodes)
                child = self._repair_unique(child, nodes, ambulance_count)
                next_population.append(child)
            population = next_population

        result = PlacementResult(
            ambulance_nodes=list(best_candidate),
            fitness=float(best_fitness),
            generations_run=generations_run,
            population_size=pop_size,
        )
        self._last_result = result
        for index, node_id in enumerate(result.ambulance_nodes):
            city_graph.set_ambulance_placement(f"AMB{index + 1}", node_id)
        return result

    def recompute_placement(self) -> PlacementResult:
        if self._last_graph is None:
            raise RuntimeError("Cannot recompute placement before optimize() is called once.")
        return self.optimize(self._last_graph, self._last_ambulance_count)

    def _fitness(self, graph: nx.Graph, candidate_nodes: list[str]) -> float:
        # Cache by unordered set: multi-source Dijkstra only depends on the
        # set of sources, so candidates with the same members reuse the same
        # score.
        key = frozenset(candidate_nodes)
        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached
        distances = self._multi_source_dijkstra(graph, candidate_nodes)
        if distances is None or len(distances) != graph.number_of_nodes():
            self._fitness_cache[key] = math.inf
            return math.inf
        score = float(max(distances.values()))
        self._fitness_cache[key] = score
        return score

    @staticmethod
    def _multi_source_dijkstra(
        graph: nx.Graph, sources: list[str]
    ) -> dict[str, float] | None:
        """Hand-coded multi-source Dijkstra on edge attribute ``weight``.

        Standard binary-heap Dijkstra seeded with every source at distance
        zero; returns ``{node: shortest distance from any source}`` for every
        reachable node, or ``None`` if any requested source is missing from
        the graph (matches the previous ``nx.multi_source_dijkstra_path_length``
        failure behavior). Replaces that black-box NetworkX call so the
        fitness function — i.e. the GA's evaluation signal — is genuinely
        hand-implemented.
        """
        if not sources:
            return None
        for src in sources:
            if src not in graph:
                return None
        dist: dict[str, float] = {src: 0.0 for src in sources}
        heap: list[tuple[float, str]] = [(0.0, src) for src in sources]
        heapq.heapify(heap)
        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist > dist[node]:
                continue
            for neighbor, edge_data in graph[node].items():
                weight = float(edge_data.get("weight", 1.0))
                # Dijkstra requires non-negative weights; skip negatives
                # rather than silently producing wrong distances.
                if weight < 0:
                    continue
                alt = current_dist + weight
                if alt < dist.get(neighbor, math.inf):
                    dist[neighbor] = alt
                    heapq.heappush(heap, (alt, neighbor))
        return dist

    def _random_candidate(self, nodes: list[str], ambulance_count: int) -> list[str]:
        return self._rng.sample(nodes, k=ambulance_count)

    def _tournament_select(self, scored_population: list[tuple[list[str], float]]) -> list[str]:
        sampled = self._rng.sample(scored_population, k=min(self.tournament_size, len(scored_population)))
        sampled.sort(key=lambda item: item[1])
        return list(sampled[0][0])

    def _crossover(
        self,
        parent_a: list[str],
        parent_b: list[str],
        all_nodes: list[str],
        ambulance_count: int,
    ) -> list[str]:
        if ambulance_count == 1:
            return [parent_a[0]]
        point = self._rng.randint(1, ambulance_count - 1)
        child = parent_a[:point] + parent_b[point:]
        return self._repair_unique(child, all_nodes, ambulance_count)

    def _mutate(self, candidate: list[str], all_nodes: list[str]) -> list[str]:
        out = list(candidate)
        if self._rng.random() >= self.mutation_rate:
            return out
        idx = self._rng.randrange(len(out))
        available = [node for node in all_nodes if node not in out or node == out[idx]]
        out[idx] = self._rng.choice(available)
        return out

    def _repair_unique(self, candidate: list[str], all_nodes: list[str], ambulance_count: int) -> list[str]:
        seen: set[str] = set()
        repaired: list[str] = []
        for node in candidate:
            if node not in seen:
                repaired.append(node)
                seen.add(node)
        available = [node for node in all_nodes if node not in seen]
        while len(repaired) < ambulance_count and available:
            node = self._rng.choice(available)
            repaired.append(node)
            available.remove(node)
        if len(repaired) < ambulance_count:
            raise ValueError("Unable to repair candidate with unique nodes.")
        return repaired[:ambulance_count]

