"""NumPy/SciPy runtime primitives for the CPU-only package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class NumpyBackend:
    """Small array adapter retained by the common mechanism interface."""

    name = "numpy"
    device = "cpu"
    is_torch = False

    def array(self, value: Any, *, dtype=float) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    def index(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.int64)

    def boolean(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=bool)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def clone(self, value: Any) -> np.ndarray:
        return np.array(value, copy=True)

    def zeros(self, shape: Any) -> np.ndarray:
        return np.zeros(shape, dtype=float)

    def ones(self, shape: Any) -> np.ndarray:
        return np.ones(shape, dtype=float)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=float)

    def abs(self, value: Any) -> np.ndarray:
        return np.abs(value)

    def mean(self, value: Any, axis: int | None = None) -> np.ndarray:
        return np.mean(value, axis=axis)

    def min(self, value: Any, axis: int | None = None) -> np.ndarray:
        return np.min(value, axis=axis)

    def sum(self, value: Any, axis: int | None = None) -> np.ndarray:
        return np.sum(value, axis=axis)

    def maximum(self, left: Any, right: Any) -> np.ndarray:
        return np.maximum(left, right)

    def clip(self, value: Any, low: float, high: float) -> np.ndarray:
        return np.clip(value, low, high)

    def where(self, condition: Any, left: Any, right: Any) -> np.ndarray:
        return np.where(condition, left, right)

    def outer(self, left: Any, right: Any) -> np.ndarray:
        return np.outer(left, right)

    def stack(self, values: Any, axis: int = 0) -> np.ndarray:
        return np.stack(tuple(values), axis=axis)

    def segment_sum(self, row_ptr: Any, values: Any) -> np.ndarray:
        row_ptr = np.asarray(row_ptr, dtype=np.int64)
        values = np.asarray(values)
        result = np.zeros(
            (row_ptr.size - 1, *values.shape[1:]), dtype=values.dtype
        )
        lengths = np.diff(row_ptr)
        nonempty = np.flatnonzero(lengths)
        if nonempty.size:
            result[nonempty] = np.add.reduceat(
                values, row_ptr[nonempty], axis=0
            )[: nonempty.size]
        return result

    def scalar(self, value: Any) -> float:
        return float(np.asarray(value))

    def count_nonzero(self, value: Any) -> int:
        return int(np.count_nonzero(value))

    def evolve(
        self,
        row_ptr: Any,
        rows: Any,
        cols: Any,
        edge_weights: Any,
        self_weights: Any,
        preferences: Any,
        observation_mask: Any = None,
    ) -> np.ndarray:
        values = np.asarray(preferences, dtype=float)
        if observation_mask is not None:
            mask = np.asarray(observation_mask, dtype=float)
            edge_delta = edge_weights[:, None] * mask[cols] * (
                values[cols] - values[rows]
            )
            return values + mask * self.segment_sum(row_ptr, edge_delta)
        social = self.segment_sum(
            row_ptr, edge_weights[:, None] * values[cols]
        )
        return self_weights[:, None] * values + social


def resolve_backend(name: str = "numpy", device: str | None = None) -> NumpyBackend:
    """Return the package's only supported compute backend."""
    del device
    if name not in {"numpy", "cpu", "auto"}:
        raise ValueError("this package supports only the NumPy CPU backend")
    return NumpyBackend()


@dataclass
class EdgeInfluence:
    """Fixed-support row-stochastic influence in CSR edge layout."""

    rows: np.ndarray
    cols: np.ndarray
    row_ptr: np.ndarray
    edge_weights: np.ndarray
    self_weights: np.ndarray
    n: int
    backend: NumpyBackend
    observation_mask: np.ndarray | None = None

    def evolve(self, preferences: np.ndarray) -> np.ndarray:
        return self.backend.evolve(
            self.row_ptr,
            self.rows,
            self.cols,
            self.edge_weights,
            self.self_weights,
            preferences,
            self.observation_mask,
        )

    def clone(self) -> "EdgeInfluence":
        return EdgeInfluence(
            self.rows.copy(),
            self.cols.copy(),
            self.row_ptr.copy(),
            self.edge_weights.copy(),
            self.self_weights.copy(),
            self.n,
            self.backend,
            None if self.observation_mask is None else self.observation_mask.copy(),
        )

    def to_csr(self):
        from scipy import sparse

        off_diagonal = sparse.csr_matrix(
            (self.edge_weights, (self.rows, self.cols)), shape=(self.n, self.n)
        )
        return (off_diagonal + sparse.diags(self.self_weights)).tocsr()
