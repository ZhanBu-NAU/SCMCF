"""Deterministic community procedures used by literature-inspired baselines.

The routines in this module keep the source papers' community semantics while
accepting the numerical preference profiles used by the common experiment
runner. They return binary ``(N, K_c)`` incidence matrices; a row may contain
more than one active community label.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import sparse


_CR_OCA_EXACT_MAX_N = 512
_DENSE_LABEL_ENTRY_LIMIT = 10_000_000


def _labels_from_groups(
    n: int, groups: list[set[int]]
) -> np.ndarray | sparse.csr_matrix:
    """Materialize an incidence matrix without forcing large cases dense."""
    if not groups:
        groups = [{i} for i in range(n)]
    row_ids: list[int] = []
    column_ids: list[int] = []
    for column, group in enumerate(groups):
        row_ids.extend(int(node) for node in group)
        column_ids.extend([column] * len(group))
    labels = sparse.csr_matrix(
        (np.ones(len(row_ids), dtype=np.uint8), (row_ids, column_ids)),
        shape=(n, len(groups)),
        dtype=np.uint8,
    )
    labels.sort_indices()
    return labels


def _binary_symmetric_support(adjacency: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(adjacency, dtype=float)
    support = ((matrix + matrix.T) > 0.0).astype(np.float64).tocsr()
    support.setdiag(0.0)
    support.eliminate_zeros()
    support.sort_indices()
    return support


def _fitness(internal_degree: float, total_degree: float, alpha: float) -> float:
    if internal_degree <= 0.0 or total_degree <= 0.0:
        return 0.0
    return float(internal_degree / (total_degree**alpha))


def lfm_overlapping_labels(
    adjacency: sparse.spmatrix,
    *,
    alpha: float = 1.0,
    seed_order: np.ndarray | None = None,
    edge_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Run a deterministic LFM-style local fitness expansion.

    The source LFM procedure starts from a seed, adds the boundary node with
    the largest positive fitness gain, and continues until no positive gain is
    available. Nodes already covered by an earlier community remain eligible as
    boundary nodes, which preserves crisp overlapping membership semantics.
    """
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    support = _binary_symmetric_support(adjacency)
    n = support.shape[0]
    if edge_weights is None:
        weights = np.ones(support.nnz, dtype=float)
    else:
        weights = np.asarray(edge_weights, dtype=float)
        if weights.shape != (support.nnz,) or np.any(weights < 0.0):
            raise ValueError("edge_weights must match the symmetric support nnz")
    # Keep the support in CSR form and attach the edge weights to its data.
    # Batch expansion can then aggregate all incident contributions in the
    # compiled sparse row-slicing kernel instead of iterating over edges in
    # Python.  The weighted support has the same sparsity pattern as ``support``.
    weighted_support = support.copy()
    weighted_support.data = weights
    # ``reduceat`` is undefined for an empty CSR row.  A bincount also makes
    # the degree calculation correct when preference filtering isolates a DM.
    support_rows = np.repeat(
        np.arange(n, dtype=np.int64), np.diff(support.indptr)
    )
    degree = np.bincount(
        support_rows, weights=weights, minlength=n
    ).astype(float, copy=False)
    neighbors = []
    neighbor_weights = []
    for i in range(n):
        start, end = support.indptr[i : i + 2]
        neighbors.append(support.indices[start:end])
        neighbor_weights.append(weights[start:end])
    if seed_order is None:
        seed_order = np.lexsort((np.arange(n), -degree))
    else:
        seed_order = np.asarray(seed_order, dtype=np.int64)
        if seed_order.shape != (n,) or set(seed_order.tolist()) != set(range(n)):
            raise ValueError("seed_order must be a permutation of all node ids")

    uncovered = np.ones(n, dtype=bool)
    communities: list[set[int]] = []
    in_community = np.zeros(n, dtype=bool)
    boundary = np.zeros(n, dtype=bool)
    link_to_community = np.zeros(n, dtype=np.float64)
    touched = np.empty(0, dtype=np.int64)
    for preferred_seed in seed_order:
        if not np.any(uncovered):
            break
        if not uncovered[int(preferred_seed)]:
            continue
        seed = int(preferred_seed)
        # Reset only entries touched by the previous local expansion.
        if touched.size:
            in_community[touched] = False
            boundary[touched] = False
            link_to_community[touched] = 0.0
        members: list[int] = [seed]
        boundary_set: set[int] = set()
        in_community[seed] = True
        touched_list = [seed]
        seed_neighbors = neighbors[seed]
        link_to_community[seed_neighbors] += neighbor_weights[seed]
        for candidate in seed_neighbors:
            candidate = int(candidate)
            if candidate != seed:
                boundary[candidate] = True
                boundary_set.add(candidate)
                touched_list.append(candidate)
        community_total_degree = float(degree[seed])
        community_internal_degree = 0.0

        while boundary_set:
            boundary_nodes = np.fromiter(boundary_set, dtype=np.int64)
            links = link_to_community[boundary_nodes]
            current_fitness = _fitness(
                community_internal_degree, community_total_degree, alpha
            )
            new_internal = community_internal_degree + 2.0 * links
            new_total = community_total_degree + degree[boundary_nodes]
            gains = new_internal / np.maximum(new_total, 1e-12) ** alpha - current_fitness
            order = np.lexsort(
                (boundary_nodes, -degree[boundary_nodes], -gains)
            )
            positive_positions = order[gains[order] > 0.0]
            if positive_positions.size == 0:
                break
            # Batch all currently positive boundary gains.  Their gains are
            # evaluated against the same frozen local state, matching the
            # local-stale heuristic while avoiding one full boundary rebuild
            # per accepted node.
            batch_nodes = boundary_nodes[positive_positions]
            # All selected entries came from the current boundary array;
            # clear them before updating the incident support so that members
            # of the same batch are never reintroduced as boundary candidates.
            boundary_set.difference_update(batch_nodes.tolist())
            boundary[batch_nodes] = False
            # The original local-stale batch update commits nodes in the
            # deterministic ``positive_positions`` order.  A node therefore
            # also receives contributions from earlier members of the same
            # batch.  Recover those lower-triangular within-batch links from
            # the flat CSR rows without constructing a dense or sparse b-by-b
            # submatrix.
            row_starts = weighted_support.indptr[batch_nodes]
            row_ends = weighted_support.indptr[batch_nodes + 1]
            row_lengths = row_ends - row_starts
            batch_indices = (
                np.concatenate(
                    [
                        weighted_support.indices[start:end]
                        for start, end in zip(row_starts, row_ends)
                    ]
                )
                if np.any(row_lengths)
                else np.empty(0, dtype=np.int64)
            )
            batch_data = (
                np.concatenate(
                    [
                        weighted_support.data[start:end]
                        for start, end in zip(row_starts, row_ends)
                    ]
                )
                if np.any(row_lengths)
                else np.empty(0, dtype=float)
            )
            batch_rows = np.repeat(
                np.arange(batch_nodes.size, dtype=np.int64),
                row_lengths,
            )
            sorted_order = np.argsort(batch_nodes, kind="stable")
            sorted_nodes = batch_nodes[sorted_order]
            positions = np.searchsorted(sorted_nodes, batch_indices)
            valid_positions = positions < batch_nodes.size
            safe_positions = np.minimum(positions, batch_nodes.size - 1)
            valid_positions &= (
                sorted_nodes[safe_positions] == batch_indices
            )
            batch_cols = np.full(batch_indices.size, -1, dtype=np.int64)
            batch_cols[valid_positions] = sorted_order[positions[valid_positions]]
            earlier = (batch_cols >= 0) & (batch_rows < batch_cols)
            within_batch_links = np.bincount(
                batch_cols[earlier],
                weights=batch_data[earlier],
                minlength=batch_nodes.size,
            )
            links = link_to_community[batch_nodes] + within_batch_links
            community_internal_degree += 2.0 * float(np.sum(links))
            community_total_degree += float(np.sum(degree[batch_nodes]))
            in_community[batch_nodes] = True
            members.extend(int(node) for node in batch_nodes)
            touched_list.extend(int(node) for node in batch_nodes)

            # ``weighted_support[batch_nodes]`` contains every incident
            # support edge once.  Updating the flat CSR data in one compiled
            # operation preserves the same additive semantics as the previous
            # per-node/per-edge loop.
            if batch_indices.size:
                np.add.at(
                    link_to_community,
                    batch_indices,
                    batch_data,
                )
                affected = np.unique(batch_indices)
                new_boundary = affected[~in_community[affected]]
                if new_boundary.size:
                    fresh_boundary = new_boundary[~boundary[new_boundary]]
                    if fresh_boundary.size:
                        boundary[fresh_boundary] = True
                        boundary_set.update(fresh_boundary.tolist())
                        touched_list.extend(fresh_boundary.tolist())

            # Remove a non-seed member when its deletion increases local
            # fitness.  Only the current community member list is inspected.
            while np.count_nonzero(in_community) > 1:
                member_array = np.flatnonzero(in_community)
                current_fitness = _fitness(
                    community_internal_degree, community_total_degree, alpha
                )
                candidate_members = member_array[member_array != seed]
                if candidate_members.size == 0:
                    break
                member_links = link_to_community[candidate_members]
                without_internal = community_internal_degree - 2.0 * member_links
                without_total = community_total_degree - degree[candidate_members]
                without_fitness = np.divide(
                    without_internal,
                    np.maximum(without_total, 1e-12) ** alpha,
                )
                contributions = current_fitness - without_fitness
                removable = np.flatnonzero(contributions < 0.0)
                if removable.size == 0:
                    break
                order = np.lexsort(
                    (
                        candidate_members[removable],
                        -degree[candidate_members[removable]],
                        contributions[removable],
                    )
                )
                remove_node = int(candidate_members[removable[order[0]]])
                in_community[remove_node] = False
                community_internal_degree -= 2.0 * link_to_community[remove_node]
                community_total_degree -= degree[remove_node]
                start = int(weighted_support.indptr[remove_node])
                end = int(weighted_support.indptr[remove_node + 1])
                remove_indices = weighted_support.indices[start:end]
                remove_data = weighted_support.data[start:end]
                if remove_indices.size:
                    np.add.at(link_to_community, remove_indices, -remove_data)
                    affected = np.unique(remove_indices)
                    reactivated = affected[
                        (~in_community[affected])
                        & (link_to_community[affected] > 0.0)
                    ]
                    if reactivated.size:
                        boundary[reactivated] = True
                        boundary_set.update(reactivated.tolist())

        members = np.flatnonzero(in_community).tolist()
        touched = np.fromiter(set(touched_list), dtype=np.int64)
        community = set(members)
        if not community:
            community = {seed}
        if community not in communities:
            communities.append(community)
        uncovered[list(community)] = False

    if not communities:
        communities = [{i} for i in range(n)]
    labels = _labels_from_groups(n, communities)
    covered = np.asarray(labels.getnnz(axis=1)).ravel() > 0
    empty_rows = np.flatnonzero(~covered)
    if empty_rows.size:
        extra = sparse.csr_matrix(
            (np.ones(empty_rows.size, dtype=np.uint8),
             (empty_rows, np.arange(empty_rows.size))),
            shape=(n, empty_rows.size),
        )
        labels = sparse.hstack((labels, extra), format="csr")
    return labels


def _pairwise_similarity(
    preferences: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(preferences, dtype=float)
    n = values.shape[0]
    if mask is None:
        differences = np.mean(np.abs(values[:, None, :] - values[None, :, :]), axis=2)
    else:
        active = np.asarray(mask, dtype=float)
        if active.shape != values.shape:
            raise ValueError("mask must have the same shape as preferences")
        shared = active[:, None, :] * active[None, :, :]
        counts = shared.sum(axis=2)
        differences = np.divide(
            np.sum(shared * np.abs(values[:, None, :] - values[None, :, :]), axis=2),
            counts,
            out=np.ones((n, n), dtype=float),
            where=counts > 0.0,
        )
        differences[counts <= 0.0] = 1.0
    return np.clip(1.0 - differences, 0.0, 1.0)


def symmetric_preference_support(
    adjacency: sparse.spmatrix,
    preferences: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return an undirected candidate support and edge similarities.

    The support is the symmetrized supplied trust topology. Similarities are
    evaluated on the current preference profile and can optionally filter the
    support, as in the similarity-thresholded trust graph used by Gai et al.
    """
    support = _binary_symmetric_support(adjacency)
    rows = np.repeat(np.arange(support.shape[0]), np.diff(support.indptr))
    cols = support.indices
    values = np.asarray(preferences, dtype=float)
    pair_mask = None
    if mask is not None:
        pair_mask = np.asarray(mask, dtype=float)
    if values.ndim != 2 or values.shape[0] != support.shape[0]:
        raise ValueError("preferences and adjacency must have the same N")
    similarity = np.zeros(rows.size, dtype=float)
    edge_chunk = 65_536
    for start in range(0, rows.size, edge_chunk):
        end = min(start + edge_chunk, rows.size)
        shared = (
            np.ones((end - start, values.shape[1]), dtype=float)
            if pair_mask is None
            else pair_mask[rows[start:end]] * pair_mask[cols[start:end]]
        )
        counts = shared.sum(axis=1)
        differences = np.divide(
            np.sum(
                shared * np.abs(values[rows[start:end]] - values[cols[start:end]]),
                axis=1,
            ),
            counts,
            out=np.ones(end - start, dtype=float),
            where=counts > 0.0,
        )
        similarity[start:end] = np.where(
            counts > 0.0, 1.0 - differences, 0.0
        )
    similarity = np.clip(similarity, 0.0, 1.0)
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        keep = similarity >= threshold
        filtered = sparse.csr_matrix(
            (np.ones(int(np.count_nonzero(keep))), (rows[keep], cols[keep])),
            shape=support.shape,
        )
        filtered.sort_indices()
        keep_keys = rows[keep] * support.shape[0] + cols[keep]
        filtered_rows = np.repeat(
            np.arange(filtered.shape[0]), np.diff(filtered.indptr)
        )
        filtered_keys = filtered_rows * support.shape[0] + filtered.indices
        order = np.searchsorted(keep_keys, filtered_keys)
        similarity = similarity[keep][order]
        support = filtered
    return support, similarity


def cr_oca_overlapping_labels(
    preferences: np.ndarray,
    *,
    adjacency: sparse.spmatrix | None = None,
    regularization: float = 1e-3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Construct CR-OCA-style crisp overlapping communities.

    The collaborative-representation coefficient matrix is obtained from one
    ridge solve on the numerical profile matrix for small instances. Self
    coefficients are removed; each DM then forms the source CR-OCA three-DM
    trust subgroup with its two highest positive/above-average coefficients.
    Subgroups sharing at least two DMs are recursively merged. For a large
    sparse trust support, the same local subgroup construction is evaluated on
    each DM's supported peers and merged through a pair index, avoiding a dense
    ``N x N`` coefficient matrix.
    """
    if regularization <= 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    values = np.asarray(preferences, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("preferences must have shape (N, L) with N >= 2")
    complete_profile = mask is None or np.all(np.asarray(mask, dtype=float) > 0.5)
    support = None
    if adjacency is not None:
        support = _binary_symmetric_support(adjacency)
    if support is not None:
        return _cr_oca_support_local_labels(
            values,
            support,
            regularization=regularization,
            mask=mask,
        )
    if not complete_profile:
        # For sparse profiles, pairwise similarity is used for
        # missing-cell-safe trust candidates rather than silently treating
        # missing entries as zero in the collaborative representation solve.
        if support is None:
            coefficients = _pairwise_similarity(values, mask)
        else:
            rows = np.repeat(np.arange(values.shape[0]), np.diff(support.indptr))
            cols = support.indices
            active = np.asarray(mask, dtype=float)[rows] * np.asarray(
                mask, dtype=float
            )[cols]
            counts = active.sum(axis=1)
            differences = np.divide(
                np.sum(active * np.abs(values[rows] - values[cols]), axis=1),
                counts,
                out=np.ones(rows.size, dtype=float),
                where=counts > 0.0,
            )
            coefficients = np.zeros((values.shape[0], values.shape[0]), dtype=float)
            coefficients[rows, cols] = np.where(
                counts > 0.0, np.clip(1.0 - differences, 0.0, 1.0), 0.0
            )
    else:
        design = values.T
        gram = design.T @ design
        regularized = gram + regularization * np.eye(values.shape[0])
        coefficients = np.linalg.solve(regularized, gram).T
    np.fill_diagonal(coefficients, 0.0)
    if support is not None:
        support = support.toarray().astype(bool)
        # CR-OCA scores identify similar peers, while the trust support defines
        # which peers can form a source-style local subgroup.
        coefficients = np.where(support, coefficients, 0.0)
    n = values.shape[0]
    groups: list[set[int]] = []
    for i in range(n):
        row = coefficients[i].copy()
        mean = float(np.mean(row))
        candidates = np.flatnonzero(row > mean)
        if support is not None:
            candidates = candidates[support[i, candidates]]
        if candidates.size < min(2, n - 1):
            candidates = np.argsort(-row, kind="stable")
            candidates = candidates[candidates != i]
            if support is not None:
                supported = candidates[support[i, candidates]]
                if supported.size:
                    candidates = supported
        selected = candidates[: min(2, n - 1)]
        groups.append({i, *map(int, selected)})

    changed = True
    while changed:
        changed = False
        for left, right in combinations(range(len(groups)), 2):
            if len(groups[left] & groups[right]) >= 2:
                groups[left] |= groups[right]
                del groups[right]
                changed = True
                break
        if changed:
            continue

    return _labels_from_groups(n, groups)


def _merge_overlapping_groups(groups: list[set[int]]) -> list[set[int]]:
    """Merge groups sharing at least two members through an indexed union-find."""
    if len(groups) <= 1:
        return groups
    parent = list(range(len(groups)))
    rank = [0] * len(groups)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    pair_owner: dict[tuple[int, int], int] = {}
    for index, group in enumerate(groups):
        members = sorted(group)
        for left, right in combinations(members, 2):
            previous = pair_owner.setdefault((left, right), index)
            if previous != index:
                union(index, previous)

    merged: dict[int, set[int]] = {}
    for index, group in enumerate(groups):
        root = find(index)
        merged.setdefault(root, set()).update(group)
    return list(merged.values())


def _cr_oca_support_local_labels(
    values: np.ndarray,
    support: sparse.csr_matrix,
    *,
    regularization: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scalable support-local CR-OCA approximation for large sparse graphs."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    active_mask = None if mask is None else np.asarray(mask, dtype=float)
    complete_profile = active_mask is None or np.all(active_mask > 0.5)
    support_rows = np.repeat(
        np.arange(n, dtype=np.int64), np.diff(support.indptr)
    )
    support_cols = support.indices.astype(np.int64, copy=False)
    coefficients_by_edge = np.empty(support_cols.size, dtype=float)
    edge_chunk = 65536
    for edge_start in range(0, support_cols.size, edge_chunk):
        edge_end = min(edge_start + edge_chunk, support_cols.size)
        edge_sources = support_rows[edge_start:edge_end]
        edge_targets = support_cols[edge_start:edge_end]
        shared = (
            np.ones(
                (edge_end - edge_start, values.shape[1]), dtype=float
            )
            if complete_profile
            else active_mask[edge_sources] * active_mask[edge_targets]
        )
        counts = shared.sum(axis=1)
        distances = np.divide(
            np.sum(
                shared
                * np.abs(
                    values[edge_sources] - values[edge_targets]
                ),
                axis=1,
            ),
            counts,
            out=np.ones(edge_end - edge_start, dtype=float),
            where=counts > 0.0,
        )
        coefficients_by_edge[edge_start:edge_end] = np.where(
            counts > 0.0,
            np.clip(1.0 - distances, 0.0, 1.0),
            0.0,
        )
    groups: list[set[int]] = []

    for i in range(n):
        start, end = support.indptr[i : i + 2]
        candidates = support.indices[start:end]
        if candidates.size == 0:
            groups.append({i})
            continue

        # A support-local similarity coefficient is the scalable large-graph
        # adapter for the source collaborative-representation score.  It uses
        # exactly the same supported peers and top-two subgroup rule as the
        # small-instance ridge path, while avoiding one dense solve per DM.
        coefficients = coefficients_by_edge[start:end]

        mean = float(np.mean(coefficients))
        selected = candidates[coefficients > mean]
        if selected.size < min(2, candidates.size):
            order = np.argsort(-coefficients, kind="stable")
            selected = candidates[order[: min(2, candidates.size)]]
        groups.append({i, *map(int, selected[:2])})

    groups = _merge_overlapping_groups(groups)
    return _labels_from_groups(n, groups)


def wdslpa_overlapping_labels(
    adjacency: sparse.spmatrix,
    preferences: np.ndarray,
    *,
    edge_trust: np.ndarray | None = None,
    edge_similarity: np.ndarray | None = None,
    iterations: int = 10,
    frequency_threshold: float = 0.10,
    max_labels: int = 32,
) -> np.ndarray:
    """Run a sparse weighted-directional SLPA approximation.

    Each iteration combines forward (incoming-edge) and reverse
    (outgoing-edge) propagation. Label distributions are kept as dictionaries
    and pruned by the source-style frequency threshold, avoiding an ``N x N``
    dense label matrix on larger networks.
    """
    if iterations <= 0 or max_labels <= 0:
        raise ValueError("iterations and max_labels must be positive")
    if not 0.0 < frequency_threshold <= 1.0:
        raise ValueError("frequency_threshold must lie in (0, 1]")
    matrix = sparse.csr_matrix(adjacency, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("adjacency must be square")
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(matrix.indptr))
    cols = matrix.indices.astype(np.int64, copy=False)
    edge_count = rows.size
    values = np.asarray(preferences, dtype=float)
    if values.shape[0] != n:
        raise ValueError("preferences and adjacency must have the same N")
    if edge_similarity is None:
        edge_similarity = np.clip(
            1.0 - np.mean(np.abs(values[rows] - values[cols]), axis=1),
            0.0,
            1.0,
        )
    else:
        edge_similarity = np.asarray(edge_similarity, dtype=float)
    if edge_similarity.shape != (edge_count,):
        raise ValueError("edge_similarity must match adjacency.nnz")
    trust = (
        np.ones(edge_count, dtype=float)
        if edge_trust is None
        else np.asarray(edge_trust, dtype=float)
    )
    if trust.shape != (edge_count,):
        raise ValueError("edge_trust must match adjacency.nnz")
    weights = np.clip(trust * edge_similarity, 0.0, None)

    # On a large graph, propagate a sparse label-distribution matrix instead
    # of materializing one Python dictionary for every edge/label pair.  The
    # recurrence is the same as the scalar path below: incoming and outgoing
    # evidence are averaged, row-normalized, thresholded, and capped.
    if n >= 0:
        out_totals = np.bincount(rows, weights=weights, minlength=n)
        in_totals = np.bincount(cols, weights=weights, minlength=n)
        out_scale = np.divide(
            weights,
            np.maximum(out_totals[rows], 1e-12),
            out=np.zeros_like(weights),
            where=out_totals[rows] > 1e-12,
        )
        in_scale = np.divide(
            weights,
            np.maximum(in_totals[cols], 1e-12),
            out=np.zeros_like(weights),
            where=in_totals[cols] > 1e-12,
        )
        outgoing_transition = sparse.csr_matrix(
            (out_scale, (rows, cols)), shape=(n, n)
        )
        incoming_transition = sparse.csr_matrix(
            (in_scale, (cols, rows)), shape=(n, n)
        )
        outgoing_transition.sum_duplicates()
        incoming_transition.sum_duplicates()
        labels = sparse.identity(n, format="csr", dtype=float)

        def prune(distribution: sparse.csr_matrix) -> sparse.csr_matrix:
            distribution = distribution.tocsr()
            distribution.sort_indices()
            row_ids_all = np.repeat(
                np.arange(n, dtype=np.int64), np.diff(distribution.indptr)
            )
            row_totals = np.asarray(distribution.sum(axis=1)).ravel()
            normalized = np.divide(
                distribution.data,
                row_totals[row_ids_all],
                out=np.zeros_like(distribution.data),
                where=row_totals[row_ids_all] > 1e-12,
            )
            keep_mask = normalized >= frequency_threshold
            counts = np.bincount(
                row_ids_all[keep_mask], minlength=n
            ).astype(np.int64, copy=False)
            # A row with no retained label keeps its largest propagated label;
            # an empty row falls back to its own identity label.
            fallback_rows = np.flatnonzero(counts == 0)
            if fallback_rows.size:
                argmax = np.asarray(distribution.argmax(axis=1)).ravel()
                nonempty = row_totals > 1e-12
                fallback_labels = np.where(
                    nonempty[fallback_rows], argmax[fallback_rows], fallback_rows
                )
                fallback_values = np.ones(fallback_rows.size, dtype=float)
            else:
                fallback_labels = np.empty(0, dtype=np.int64)
                fallback_values = np.empty(0, dtype=float)

            overfull_rows = np.flatnonzero(counts > max_labels)
            if overfull_rows.size:
                # The frequency threshold normally bounds this set by
                # ``floor(1 / threshold)``; retain a deterministic fallback
                # for unusually small thresholds.
                for row in overfull_rows:
                    start, end = distribution.indptr[row : row + 2]
                    positions = np.flatnonzero(
                        keep_mask[start:end]
                    ) + start
                    order = np.lexsort(
                        (
                            distribution.indices[positions],
                            -normalized[positions],
                        )
                    )[:max_labels]
                    keep_mask[positions] = False
                    keep_mask[positions[order]] = True

            kept_rows = row_ids_all[keep_mask]
            kept_labels = distribution.indices[keep_mask]
            kept_values = normalized[keep_mask]
            retained_sums = np.bincount(
                kept_rows, weights=kept_values, minlength=n
            )
            kept_values = kept_values / np.maximum(retained_sums[kept_rows], 1e-12)
            if fallback_rows.size:
                kept_rows = np.concatenate((kept_rows, fallback_rows))
                kept_labels = np.concatenate((kept_labels, fallback_labels))
                kept_values = np.concatenate((kept_values, fallback_values))
            result = sparse.csr_matrix(
                (kept_values, (kept_rows, kept_labels)), shape=(n, n)
            )
            result.sort_indices()
            return result

        for _ in range(iterations):
            evidence = 0.5 * (
                incoming_transition @ labels
                + outgoing_transition @ labels
            )
            labels = prune(evidence)
        active_label_ids = np.unique(labels.indices)
        if active_label_ids.size == 0:
            active_label_ids = np.arange(n, dtype=np.int64)
        column_map = np.full(n, -1, dtype=np.int64)
        column_map[active_label_ids] = np.arange(active_label_ids.size)
        remapped = labels.copy()
        remapped.indices = column_map[remapped.indices]
        remapped.data[:] = 1.0
        remapped.sort_indices()
        return remapped.astype(np.uint8)

    raise RuntimeError("unreachable sparse WDSLPA branch")
