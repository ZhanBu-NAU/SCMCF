"""Validated input model shared by SCMCF and external baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class CRPInstance:
    adjacency: sparse.csr_matrix
    preferences: np.ndarray
    observation_mask: np.ndarray
    memberships: np.ndarray
    epsilon_w: float = 1e-6
    consensus_threshold: float = 0.95
    max_rounds: int = 100
    community_labels: np.ndarray | None = None
    _adjacency_csc: sparse.csc_matrix = field(init=False, repr=False)
    _beta: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        adjacency = sparse.csr_matrix(self.adjacency, dtype=float)
        adjacency.sort_indices()
        preferences = np.asarray(self.preferences, dtype=float)
        mask = np.asarray(self.observation_mask, dtype=float)
        memberships = np.asarray(self.memberships, dtype=float)
        n = adjacency.shape[0]
        if adjacency.shape != (n, n):
            raise ValueError("adjacency must be square")
        if preferences.ndim != 2 or preferences.shape[0] != n:
            raise ValueError("preferences must have shape (N, L)")
        if mask.shape != preferences.shape or not np.all(np.isin(mask, [0.0, 1.0])):
            raise ValueError("observation_mask must be a binary (N, L) matrix")
        if np.any(mask.sum(axis=1) == 0) or np.any(mask.sum(axis=0) == 0):
            raise ValueError("every DM and alternative needs an observed entry")
        if np.any((preferences < 0.0) | (preferences > 1.0)):
            raise ValueError("preferences must lie in [0, 1]")
        if memberships.ndim != 2 or memberships.shape[0] != n:
            raise ValueError("memberships must have shape (N, K)")
        if memberships.shape[1] < 1 or np.any(memberships < 0.0):
            raise ValueError("memberships must be nonnegative and nonempty")
        if not np.allclose(memberships.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("each membership row must lie on the simplex")
        if adjacency.diagonal().any():
            raise ValueError("adjacency must not contain self-loops")
        if np.any(np.diff(adjacency.indptr) == 0):
            raise ValueError("every DM needs an outgoing trust neighbor")
        components, _ = connected_components(
            adjacency, directed=True, connection="strong"
        )
        if components != 1:
            raise ValueError("the retained directed topology must be strongly connected")
        if not 0.0 < self.epsilon_w < 0.5:
            raise ValueError("epsilon_w must lie in (0, 1/2)")
        if not 0.0 < self.consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must lie in (0, 1]")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        labels = None if self.community_labels is None else np.asarray(self.community_labels)
        if labels is not None and (labels.ndim != 2 or labels.shape[0] != n):
            raise ValueError("community_labels must have shape (N, K_c)")
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "_adjacency_csc", adjacency.tocsc())
        object.__setattr__(self, "preferences", preferences.copy())
        object.__setattr__(self, "observation_mask", mask.copy())
        object.__setattr__(self, "memberships", memberships.copy())
        object.__setattr__(self, "community_labels", None if labels is None else labels.copy())
        object.__setattr__(self, "_beta", mask / mask.sum(axis=1, keepdims=True))

    @property
    def n(self) -> int:
        return self.adjacency.shape[0]

    @property
    def l(self) -> int:
        return self.preferences.shape[1]

    @property
    def k(self) -> int:
        return self.memberships.shape[1]

    @property
    def beta(self) -> np.ndarray:
        return self._beta

    @property
    def evolution_mask(self) -> np.ndarray:
        return np.ones_like(self.observation_mask)

    @property
    def evolution_beta(self) -> np.ndarray:
        return np.full_like(self.observation_mask, 1.0 / self.l)

    @property
    def adjacency_csc(self) -> sparse.csc_matrix:
        return self._adjacency_csc

    def out_neighbors(self, i: int) -> np.ndarray:
        start, end = self.adjacency.indptr[i : i + 2]
        return self.adjacency.indices[start:end]

    def in_neighbors(self, i: int) -> np.ndarray:
        start, end = self._adjacency_csc.indptr[i : i + 2]
        return self._adjacency_csc.indices[start:end]
