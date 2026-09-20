"""Topology-only initialization phases from the SCMCF algorithm."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class LocalOverlapResult:
    communities: list[list[int]]
    node_communities: list[list[int]]
    seed_order: list[int]


def impact_set_first_fit_coloring(
    n: int, edges: Iterable[tuple[int, int]]
) -> tuple[list[int], list[list[int]], list[int]]:
    in_neighbors = [set() for _ in range(n)]
    out_degree = [0] * n
    for source, target in edges:
        if source == target or source in in_neighbors[target]:
            continue
        in_neighbors[target].add(source)
        out_degree[source] += 1
    ordering = sorted(
        range(n),
        key=lambda node: (-(1 + len(in_neighbors[node])), -out_degree[node], node),
    )
    colors = [-1] * n
    occupied_by_color: list[set[int]] = []
    for node in ordering:
        impact = {node, *in_neighbors[node]}
        for color, occupied in enumerate(occupied_by_color):
            if impact.isdisjoint(occupied):
                colors[node] = color
                occupied.update(impact)
                break
        else:
            colors[node] = len(occupied_by_color)
            occupied_by_color.append(set(impact))
    classes: list[list[int]] = [[] for _ in occupied_by_color]
    for node, color in enumerate(colors):
        classes[color].append(node)
    return colors, classes, ordering


def local_modularity_overlapping_communities(
    n: int,
    edges: Iterable[tuple[int, int]],
    gamma_dir: float = 1.0,
) -> LocalOverlapResult:
    """Deterministic batched local expansion used for SCMCF Phase 1."""
    if n <= 0 or not math.isfinite(gamma_dir) or gamma_dir < 0.0:
        raise ValueError("n must be positive and gamma_dir finite and nonnegative")
    in_sets = [set() for _ in range(n)]
    out_sets = [set() for _ in range(n)]
    for source, target in edges:
        if not (0 <= source < n and 0 <= target < n):
            raise ValueError("edge endpoint outside the node set")
        if source != target:
            out_sets[source].add(target)
            in_sets[target].add(source)
    out_neighbors = [np.fromiter(sorted(x), dtype=np.int64) for x in out_sets]
    in_neighbors = [np.fromiter(sorted(x), dtype=np.int64) for x in in_sets]
    incident_edges = [
        np.concatenate((out_neighbors[i], in_neighbors[i])) for i in range(n)
    ]
    incident_nodes = [np.unique(x) for x in incident_edges]
    out_degree = np.asarray([x.size for x in out_neighbors], dtype=np.int64)
    in_degree = np.asarray([x.size for x in in_neighbors], dtype=np.int64)
    edge_count = int(out_degree.sum())
    if edge_count == 0:
        raise ValueError("at least one directed edge is required")

    closed_incident: list[np.ndarray] = []
    containing: list[list[int]] = [[] for _ in range(n)]
    for node in range(n):
        nodes = np.unique(
            np.concatenate(([node], in_neighbors[node], out_neighbors[node]))
        )
        closed_incident.append(nodes)
        for affected in nodes:
            containing[int(affected)].append(node)

    uncovered = np.ones(n, dtype=bool)
    counts = np.asarray([x.size for x in closed_incident], dtype=np.int64)
    sizes = counts.copy()
    seed_heap = [(-int(counts[i]), -int(sizes[i]), i) for i in range(n)]
    heapq.heapify(seed_heap)
    in_community = np.zeros(n, dtype=bool)
    boundary = np.zeros(n, dtype=bool)
    cross_edges = np.zeros(n, dtype=np.int64)
    versions = np.zeros(n, dtype=np.int64)
    communities: list[list[int]] = []
    seed_order: list[int] = []
    covered_count = 0

    while covered_count < n:
        while True:
            neg_count, _, seed = heapq.heappop(seed_heap)
            if uncovered[seed] and -neg_count == int(counts[seed]):
                break
        seed_order.append(seed)
        current: list[int] = []
        touched: list[int] = [seed]
        candidates: list[tuple[float, int, int]] = []

        def gain(node: int, community_out: int, community_in: int) -> float:
            product_change = (
                (community_out + int(out_degree[node]))
                * (community_in + int(in_degree[node]))
                - community_out * community_in
            )
            return cross_edges[node] / edge_count - (
                gamma_dir * product_change / (edge_count * edge_count)
            )

        def push(node: int, community_out: int, community_in: int) -> None:
            versions[node] += 1
            heapq.heappush(
                candidates, (-gain(node, community_out, community_in), node, int(versions[node]))
            )

        def add_batch(
            nodes: Sequence[int], community_out: int, community_in: int
        ) -> tuple[int, int]:
            batch = np.asarray(nodes, dtype=np.int64)
            in_community[batch] = True
            boundary[batch] = False
            current.extend(int(x) for x in batch)
            nonempty = [incident_edges[int(x)] for x in batch if incident_edges[int(x)].size]
            affected = np.empty(0, dtype=np.int64)
            if nonempty:
                all_edges = np.concatenate(nonempty)
                np.add.at(cross_edges, all_edges, 1)
                affected = np.unique(all_edges)
            next_out = community_out + int(out_degree[batch].sum())
            next_in = community_in + int(in_degree[batch].sum())
            for raw in affected:
                node = int(raw)
                if in_community[node]:
                    continue
                if not boundary[node]:
                    boundary[node] = True
                    touched.append(node)
                push(node, next_out, next_in)
            return next_out, next_in

        community_out, community_in = add_batch([seed], 0, 0)
        while candidates:
            accepted: list[int] = []
            while candidates:
                neg_gain, node, version = heapq.heappop(candidates)
                if not boundary[node] or in_community[node] or version != int(versions[node]):
                    continue
                if -neg_gain <= 0.0:
                    break
                accepted.append(node)
            if not accepted:
                break
            community_out, community_in = add_batch(
                accepted, community_out, community_in
            )
        community = sorted(current)
        communities.append(community)
        for node in community:
            if not uncovered[node]:
                continue
            uncovered[node] = False
            covered_count += 1
            for center in containing[node]:
                counts[center] -= 1
                heapq.heappush(
                    seed_heap, (-int(counts[center]), -int(sizes[center]), center)
                )
        reset = np.unique(np.asarray(touched + current, dtype=np.int64))
        in_community[reset] = False
        boundary[reset] = False
        cross_edges[reset] = 0

    node_communities: list[list[int]] = [[] for _ in range(n)]
    for community, members in enumerate(communities):
        for node in members:
            node_communities[node].append(community)
    return LocalOverlapResult(communities, node_communities, seed_order)


def overlapping_memberships_from_communities(
    communities: Sequence[Sequence[int]], n: int
) -> np.ndarray:
    memberships = np.zeros((n, len(communities)), dtype=float)
    for community, members in enumerate(communities):
        memberships[np.asarray(members, dtype=int), community] = 1.0
    totals = memberships.sum(axis=1)
    if np.any(totals == 0.0):
        raise ValueError("communities must cover every DM")
    return memberships / totals[:, None]
