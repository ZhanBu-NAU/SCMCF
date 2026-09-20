"""Influence matrices on the fixed directed trust support."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .backend import EdgeInfluence, NumpyBackend


def _edge_layout(adjacency: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    degrees = np.diff(adjacency.indptr)
    rows = np.repeat(np.arange(adjacency.shape[0], dtype=np.int64), degrees)
    return rows, adjacency.indices.astype(np.int64, copy=False)


def community_edge_influence(
    adjacency: sparse.csr_matrix,
    memberships: np.ndarray,
    epsilon_w: float,
    backend: NumpyBackend | None = None,
    observation_mask: np.ndarray | None = None,
) -> EdgeInfluence:
    """Construct the SCMCF matrix W(S) without materializing a dense matrix."""
    backend = backend or NumpyBackend()
    adjacency = sparse.csr_matrix(adjacency, dtype=float)
    rows, cols = _edge_layout(adjacency)
    degrees = np.diff(adjacency.indptr).astype(float)
    scores = np.einsum("ek,ek->e", memberships[rows], memberships[cols])
    edge_weights = (
        epsilon_w + (1.0 - 2.0 * epsilon_w) * scores
    ) / degrees[rows]
    social_mass = backend.segment_sum(adjacency.indptr, edge_weights)
    return EdgeInfluence(
        rows=rows,
        cols=cols,
        row_ptr=adjacency.indptr.astype(np.int64, copy=True),
        edge_weights=edge_weights,
        self_weights=1.0 - social_mass,
        n=adjacency.shape[0],
        backend=backend,
        observation_mask=(
            None if observation_mask is None else np.asarray(observation_mask, dtype=float)
        ),
    )


def uniform_edge_influence(
    adjacency: sparse.csr_matrix,
    backend: NumpyBackend | None = None,
    observation_mask: np.ndarray | None = None,
) -> EdgeInfluence:
    """Construct a neutral fixed-support DeGroot matrix with zero self-weight."""
    backend = backend or NumpyBackend()
    adjacency = sparse.csr_matrix(adjacency, dtype=float)
    rows, cols = _edge_layout(adjacency)
    degrees = np.diff(adjacency.indptr).astype(float)
    return EdgeInfluence(
        rows=rows,
        cols=cols,
        row_ptr=adjacency.indptr.astype(np.int64, copy=True),
        edge_weights=1.0 / degrees[rows],
        self_weights=np.zeros(adjacency.shape[0], dtype=float),
        n=adjacency.shape[0],
        backend=backend,
        observation_mask=(
            None if observation_mask is None else np.asarray(observation_mask, dtype=float)
        ),
    )


def assert_valid_influence_matrix(
    matrix: sparse.spmatrix,
    adjacency: sparse.spmatrix,
    tolerance: float = 1e-9,
) -> None:
    values = sparse.csr_matrix(matrix, dtype=float)
    if np.any(values.data < -tolerance):
        raise ValueError("influence weights must be nonnegative")
    if not np.allclose(np.asarray(values.sum(axis=1)).ravel(), 1.0, atol=tolerance):
        raise ValueError("influence matrix must be row stochastic")
    off_diagonal = values.copy()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    support = sparse.csr_matrix(adjacency, dtype=float)
    unsupported = off_diagonal.multiply(support == 0)
    if unsupported.nnz:
        raise ValueError("influence matrix uses an edge outside the trust support")
