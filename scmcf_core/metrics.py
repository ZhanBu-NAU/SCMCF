"""Consensus and trajectory metrics."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _beta_weights(
    preferences: np.ndarray, observation_mask: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(preferences, dtype=float)
    if values.ndim != 2:
        raise ValueError("preferences must have shape (N, L)")
    if observation_mask is None:
        return np.full(values.shape, 1.0 / values.shape[1], dtype=float)
    mask = np.asarray(observation_mask, dtype=float)
    if mask.shape != values.shape:
        raise ValueError("observation_mask must have shape (N, L)")
    if not np.all(np.isclose(mask, 0.0, atol=1e-12) | np.isclose(mask, 1.0, atol=1e-12)):
        raise ValueError("observation_mask must be binary")
    mask = mask > 0.5
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("each DM must have at least one observed preference")
    return mask.astype(float) / counts[:, None]


def collective_preference(
    preferences: np.ndarray, observation_mask: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(preferences, dtype=float)
    beta = _beta_weights(values, observation_mask)
    masses = beta.sum(axis=0)
    weighted_sum = np.sum(beta * values, axis=0)
    return np.divide(weighted_sum, masses, out=np.zeros_like(weighted_sum), where=masses > 0.0)


def disagreement(
    preferences: np.ndarray, observation_mask: np.ndarray | None = None
) -> float:
    preferences = np.asarray(preferences, dtype=float)
    beta = _beta_weights(preferences, observation_mask)
    mean = collective_preference(preferences, observation_mask)
    return float(np.sum(beta * (preferences - mean) ** 2) / preferences.shape[0])


def individual_consensus_levels(
    preferences: np.ndarray, observation_mask: np.ndarray | None = None
) -> np.ndarray:
    """Distance-based SNGDM consensus indices for all DMs.

    Preference scores lie in [0, 1], so one minus the mean absolute deviation
    from the equally aggregated collective preference also lies in [0, 1].
    """
    preferences = np.asarray(preferences, dtype=float)
    beta = _beta_weights(preferences, observation_mask)
    mean = collective_preference(preferences, observation_mask)
    return 1.0 - np.sum(beta * np.abs(preferences - mean), axis=1)


def consensus_levels(
    preferences: np.ndarray, observation_mask: np.ndarray | None = None
) -> tuple[float, float]:
    """Return the mean and minimum individual consensus levels."""
    individual = individual_consensus_levels(preferences, observation_mask)
    return float(individual.mean()), float(individual.min())


def total_preference_adjustment(
    initial_preferences: np.ndarray,
    final_preferences: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> float:
    """Total absolute initial-to-final preference adjustment."""
    initial = np.asarray(initial_preferences, dtype=float)
    final = np.asarray(final_preferences, dtype=float)
    if initial.shape != final.shape:
        raise ValueError("initial and final preference profiles must have the same shape")
    if observation_mask is None:
        return float(np.sum(np.abs(final - initial)))
    mask = np.asarray(observation_mask, dtype=float)
    if mask.shape != initial.shape:
        raise ValueError("observation_mask must have the same shape as preferences")
    return float(np.sum(mask * np.abs(final - initial)))


def preference_adjustment(
    previous: np.ndarray,
    current: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> float:
    """Absolute preference adjustment between two consecutive CRP states."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape:
        raise ValueError("preference profiles must have the same shape")
    if observation_mask is None:
        return float(np.sum(np.abs(current - previous)))
    mask = np.asarray(observation_mask, dtype=float)
    if mask.shape != previous.shape:
        raise ValueError("observation_mask must have the same shape as preferences")
    return float(np.sum(mask * np.abs(current - previous)))


def total_membership_adjustment(
    initial_memberships: np.ndarray, final_memberships: np.ndarray
) -> float:
    """Total absolute initial-to-final membership adjustment."""
    initial = np.asarray(initial_memberships, dtype=float)
    final = np.asarray(final_memberships, dtype=float)
    if initial.shape != final.shape:
        raise ValueError("initial and final membership profiles must have the same shape")
    return float(np.sum(np.abs(final - initial)))


def membership_adjustment(previous: np.ndarray, current: np.ndarray) -> float:
    """Absolute membership adjustment between two committed CRP states."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape:
        raise ValueError("membership profiles must have the same shape")
    return float(np.sum(np.abs(current - previous)))


def preference_movement(
    previous: np.ndarray,
    current: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> float:
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape:
        raise ValueError("preference profiles must have the same shape")
    if observation_mask is None:
        return float(np.mean((current - previous) ** 2))
    beta = _beta_weights(current, observation_mask)
    return float(np.sum(beta * (current - previous) ** 2) / current.shape[0])


def adjustment_per_dm(total_adjustment: float, n: int) -> float:
    """Normalize an aggregate L1 adjustment by the number of DMs."""
    if n <= 0:
        raise ValueError("n must be positive")
    return float(total_adjustment) / float(n)


def adjustment_per_cell(total_adjustment: float, n: int, l: int) -> float:
    """Normalize an aggregate L1 adjustment by the number of preference cells."""
    if n <= 0 or l <= 0:
        raise ValueError("n and l must be positive")
    return float(total_adjustment) / float(n * l)


def directed_soft_modularity(
    adjacency: sparse.spmatrix, memberships: np.ndarray
) -> float:
    """Compute directed Newman-Girvan modularity for soft memberships.

    For row-oriented trust edges ``A[i, j]``, this evaluates

    ``sum_ij (A_ij - d_out[i] d_in[j] / M) s_i^T s_j / M``.

    The observed term uses sparse matrix multiplication ``A @ S``. The
    null-model term is factorized by community dimension, so neither a dense
    ``N x N`` matrix nor an ``M x K`` edge-membership array is formed.
    """
    matrix = sparse.csr_matrix(adjacency, dtype=float)
    membership = np.asarray(memberships, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    if membership.ndim != 2 or membership.shape[0] != matrix.shape[0]:
        raise ValueError("memberships must have shape (N, K)")
    if np.any(membership < -1e-12):
        raise ValueError("memberships must be nonnegative")
    if not np.allclose(membership.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("membership rows must lie on the simplex")
    edge_count = float(matrix.sum())
    if edge_count <= 0.0:
        raise ValueError("adjacency must contain at least one positive edge")
    out_degree = np.asarray(matrix.sum(axis=1)).ravel()
    in_degree = np.asarray(matrix.sum(axis=0)).ravel()
    observed = float(np.sum(membership * (matrix @ membership)))
    out_mass = out_degree @ membership
    in_mass = in_degree @ membership
    expected = float(np.dot(out_mass, in_mass) / edge_count)
    return float((observed - expected) / edge_count)


def directed_hard_modularity(
    adjacency: sparse.spmatrix, labels: np.ndarray
) -> float:
    """Compute directed modularity for integer hard community labels."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional vector")
    unique = {label: column for column, label in enumerate(sorted(set(labels.tolist())))}
    memberships = np.zeros((labels.size, len(unique)), dtype=float)
    memberships[np.arange(labels.size), [unique[label] for label in labels]] = 1.0
    return directed_soft_modularity(adjacency, memberships)


def directed_overlapping_modularity(
    adjacency: sparse.spmatrix, labels: np.ndarray
) -> float:
    """Compute directed modularity for crisp overlapping communities.

    ``labels[i, k]`` is a binary indicator that DM ``i`` belongs to community
    ``k``.  Following the directed overlapping modularity definition of
    Nicosia et al. (2009), each node contributes its community indicators
    normalized by its number of memberships.  The implementation keeps the
    observed term sparse and factorizes the directed degree-preserving null
    model, so it does not construct an ``N x N`` matrix.
    """
    matrix = sparse.csr_matrix(adjacency, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    edge_count = float(matrix.sum())
    if edge_count <= 0.0:
        raise ValueError("adjacency must contain at least one positive edge")

    if sparse.issparse(labels):
        membership = sparse.csr_matrix(labels, dtype=float)
        if membership.shape[0] != matrix.shape[0] or membership.shape[1] < 1:
            raise ValueError("labels must have shape (N, K)")
        if membership.nnz and not np.all(
            np.isclose(membership.data, 0.0, atol=1e-12)
            | np.isclose(membership.data, 1.0, atol=1e-12)
        ):
            raise ValueError("labels must be binary crisp indicators")
        membership.data[:] = 1.0
        overlap_count = np.asarray(membership.sum(axis=1)).ravel()
        if np.any(overlap_count <= 0.0):
            raise ValueError("each node must belong to at least one community")
        normalized = membership.multiply(
            1.0 / overlap_count[:, None]
        ).tocsr()
        observed = float(normalized.multiply(matrix @ normalized).sum())
        out_degree = np.asarray(matrix.sum(axis=1)).ravel()
        in_degree = np.asarray(matrix.sum(axis=0)).ravel()
        out_mass = np.asarray(out_degree @ normalized).ravel()
        in_mass = np.asarray(in_degree @ normalized).ravel()
        expected = float(np.dot(out_mass, in_mass) / edge_count)
        return float((observed - expected) / edge_count)

    membership = np.asarray(labels, dtype=float)
    if membership.ndim != 2 or membership.shape[0] != matrix.shape[0]:
        raise ValueError("labels must have shape (N, K)")
    if membership.shape[1] < 1:
        raise ValueError("labels must contain at least one community")
    if not np.all(
        np.isclose(membership, 0.0, atol=1e-12)
        | np.isclose(membership, 1.0, atol=1e-12)
    ):
        raise ValueError("labels must be binary crisp indicators")
    membership = (membership > 0.5).astype(float)
    overlap_count = membership.sum(axis=1)
    if np.any(overlap_count <= 0.0):
        raise ValueError("each node must belong to at least one community")

    normalized = membership / overlap_count[:, None]
    out_degree = np.asarray(matrix.sum(axis=1)).ravel()
    in_degree = np.asarray(matrix.sum(axis=0)).ravel()
    observed = float(np.sum(normalized * (matrix @ normalized)))
    out_mass = out_degree @ normalized
    in_mass = in_degree @ normalized
    expected = float(np.dot(out_mass, in_mass) / edge_count)
    return float((observed - expected) / edge_count)
