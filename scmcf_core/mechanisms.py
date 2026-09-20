"""Common mechanism contract and the paper-defined SCMCF implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .backend import EdgeInfluence, NumpyBackend
from .instance import CRPInstance
from .topology import (
    impact_set_first_fit_coloring,
    local_modularity_overlapping_communities,
    overlapping_memberships_from_communities,
)
from .weights import community_edge_influence, uniform_edge_influence


@dataclass(frozen=True)
class FeedbackDiagnostics:
    accepted_responses: int = 0
    direct_feedback_updates: int = 0
    sweeps: int = 0
    membership_change: float = 0.0
    membership_adjustment: float = 0.0
    potential_before: float | None = None
    potential_after: float | None = None
    active_edges: int | None = None
    terminated: bool = True


class InfluenceMechanism(ABC):
    """Stateful response rule consumed by the common CRP runner."""

    name = "mechanism"
    stability_termination_supported = False

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        self.instance = instance
        self.backend = backend or NumpyBackend()
        adjacency = instance.adjacency
        adjacency_csc = instance.adjacency_csc
        self._out_indptr = adjacency.indptr
        self._out_indices = adjacency.indices
        self._in_indptr = adjacency_csc.indptr
        self._in_indices = adjacency_csc.indices
        self._in_degree = np.diff(self._in_indptr)
        self._observation_mask_np = np.asarray(instance.observation_mask, dtype=float)
        self._observation_mask = self._observation_mask_np
        self._evolution_mask_np = np.ones_like(self._observation_mask_np)
        self._evolution_mask = self._evolution_mask_np
        self._beta_np = np.asarray(instance.evolution_beta, dtype=float)
        self._beta = self._beta_np
        self._initial_weights = uniform_edge_influence(
            adjacency, self.backend, observation_mask=self._evolution_mask
        )
        self._weights = self._initial_weights.clone()
        self._initial_memberships = None
        self._rows = self._initial_weights.rows
        self._cols = self._initial_weights.cols
        self._row_ptr = self._initial_weights.row_ptr
        self._out_degree_np = np.diff(self._out_indptr).astype(float)
        self._inverse_out_degree_np = 1.0 / self._out_degree_np
        self._out_degree = self._out_degree_np
        self._edge_rows_np = self._rows
        self._edge_cols_np = self._cols
        csc_rows = np.asarray(self._in_indices, dtype=np.int64)
        csc_cols = np.repeat(
            np.arange(instance.n, dtype=np.int64), self._in_degree.astype(np.int64)
        )
        csr_keys = self._edge_rows_np * instance.n + self._edge_cols_np
        csc_keys = csc_rows * instance.n + csc_cols
        incoming_positions = np.searchsorted(csr_keys, csc_keys)
        if not np.array_equal(csr_keys[incoming_positions], csc_keys):
            raise ValueError("CSR and CSC edge layouts are inconsistent")
        self._in_edge_positions = incoming_positions.astype(np.int64, copy=False)
        self.initialization_metadata = {
            "initial_weight_source": "uniform_fixed_support",
            "consensus_assessment_precedes_first_transition": True,
        }

    def _out_neighbors_of(self, i: int) -> np.ndarray:
        return self._out_indices[self._out_indptr[i] : self._out_indptr[i + 1]]

    def _in_neighbors_of(self, i: int) -> np.ndarray:
        return self._in_indices[self._in_indptr[i] : self._in_indptr[i + 1]]

    def _edge_mean_abs_difference(self, preferences: np.ndarray) -> np.ndarray:
        values = np.asarray(preferences, dtype=float)
        shared = self._evolution_mask_np[self._rows] * self._evolution_mask_np[self._cols]
        counts = shared.sum(axis=1)
        numerators = np.sum(
            shared * np.abs(values[self._rows] - values[self._cols]), axis=1
        )
        return np.divide(
            numerators,
            counts,
            out=np.zeros_like(numerators),
            where=counts > 0.0,
        )

    def _community_influence(self, memberships: np.ndarray) -> EdgeInfluence:
        return community_edge_influence(
            self.instance.adjacency,
            memberships,
            self.instance.epsilon_w,
            self.backend,
            observation_mask=self._evolution_mask,
        )

    @property
    def weights(self) -> EdgeInfluence:
        return self._weights

    @property
    def memberships(self) -> np.ndarray | None:
        return None

    @property
    def initial_memberships(self) -> np.ndarray | None:
        return self._initial_memberships

    def close(self) -> None:
        return None

    def feedback_state_is_stable(self, diagnostics: FeedbackDiagnostics) -> bool:
        del diagnostics
        return False

    def evolve_after_feedback(self, preferences: np.ndarray) -> np.ndarray:
        return self.weights.evolve(preferences)

    @abstractmethod
    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        pass


def project_simplex(vectors: np.ndarray) -> np.ndarray:
    """Euclidean projection of one vector or a batch onto the simplex."""
    values = np.asarray(vectors, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("vectors must have shape (K,) or (B, K)")
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=float)[None, :]
    positive = ordered - cumulative / indices > 0.0
    rho = np.sum(positive, axis=1).astype(int) - 1
    rows = np.arange(values.shape[0])
    theta = cumulative[rows, rho] / (rho + 1.0)
    projected = np.maximum(values - theta[:, None], 0.0)
    return projected[0] if one_dimensional else projected


def solve_simplex_qp_batch(
    hessians: np.ndarray,
    linears: np.ndarray,
    initials: np.ndarray,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> np.ndarray:
    """Deterministic projected-FISTA candidates for convex simplex QPs."""
    hessians = np.asarray(hessians, dtype=float)
    linears = np.asarray(linears, dtype=float)
    initials = np.asarray(initials, dtype=float)
    if hessians.ndim != 3 or hessians.shape[1] != hessians.shape[2]:
        raise ValueError("hessians must have shape (B, K, K)")
    if linears.shape != initials.shape or linears.shape != hessians.shape[:2]:
        raise ValueError("linears and initials must have shape (B, K)")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("QP tolerance and iteration limit must be positive")
    if hessians.shape[0] == 0:
        return initials.copy()
    hessians = 0.5 * (hessians + np.swapaxes(hessians, 1, 2))
    x = project_simplex(initials)
    flat = np.max(np.abs(hessians), axis=(1, 2)) <= 1e-14
    if np.any(flat):
        minimum = np.min(linears[flat], axis=1, keepdims=True)
        face = np.isclose(linears[flat], minimum, rtol=1e-12, atol=1e-14)
        retained = np.sum(np.where(face, x[flat], 0.0), axis=1, keepdims=True)
        x[flat] = np.where(
            face,
            x[flat] + (1.0 - retained) / np.sum(face, axis=1, keepdims=True),
            0.0,
        )
    y = x.copy()
    lipschitz = np.linalg.eigvalsh(hessians)[:, -1]
    steps = 1.0 / np.maximum(lipschitz, 1e-12)
    active = ~flat
    momentum = 1.0
    for _ in range(max_iterations):
        if not np.any(active):
            break
        gradients = np.matmul(hessians, y[..., None]).squeeze(-1) + linears
        proposed = project_simplex(y - steps[:, None] * gradients)
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = proposed + (
            (momentum - 1.0) / next_momentum
        ) * (proposed - x)
        next_x = np.where(active[:, None], proposed, x)
        active &= np.max(np.abs(next_x - x), axis=1) > tolerance
        x = next_x
        y = np.where(active[:, None], accelerated, y)
        momentum = next_momentum
    return project_simplex(x)


def _improvement_mask(
    candidates: np.ndarray,
    references: np.ndarray,
    hessians: np.ndarray,
    linears: np.ndarray,
    epsilon_qp: float,
) -> np.ndarray:
    candidate_values = 0.5 * np.einsum(
        "bi,bij,bj->b", candidates, hessians, candidates
    ) + np.einsum("bi,bi->b", linears, candidates)
    reference_values = 0.5 * np.einsum(
        "bi,bij,bj->b", references, hessians, references
    ) + np.einsum("bi,bi->b", linears, references)
    return candidate_values <= reference_values - epsilon_qp


def _degree_balanced_memberships(
    out_degree: np.ndarray,
    in_degree: np.ndarray,
    community_count: int,
) -> np.ndarray:
    """Build a deterministic community-agnostic assignment for a fixed K."""
    outgoing = np.asarray(out_degree, dtype=float).reshape(-1)
    incoming = np.asarray(in_degree, dtype=float).reshape(-1)
    if outgoing.shape != incoming.shape:
        raise ValueError("in-degree and out-degree vectors must have equal shape")
    n = outgoing.size
    if not 1 <= community_count <= n:
        raise ValueError("community_count must lie in [1, N]")
    node_ids = np.arange(n, dtype=np.int64)
    ordering = np.lexsort((node_ids, -(outgoing + incoming)))
    memberships = np.zeros((n, community_count), dtype=float)
    memberships[ordering, np.arange(n) % community_count] = 1.0
    return memberships


class SCMCFMechanism(InfluenceMechanism):
    """Soft-community-mediated consensus feedback from the current paper."""

    name = "SCMCF"

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        alpha: float = 0.5,
        feedback_scope: str = "observed",
        response_execution: str = "batched",
        reference_mode: str = "retained_forecast",
        local_cost_scope: str = "response_impact",
        membership_feedback: bool = True,
        response_until_equilibrium: bool = False,
        initialization_mode: str = "local_overlap",
        epsilon_qp: float = 1e-6,
        qp_max_iterations: int = 100,
    ) -> None:
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and nonnegative")
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if feedback_scope not in {"observed", "all_cells"}:
            raise ValueError("feedback_scope must be observed or all_cells")
        if response_execution not in {"batched", "serial"}:
            raise ValueError("response_execution must be batched or serial")
        if reference_mode not in {"retained_forecast", "current_collective"}:
            raise ValueError(
                "reference_mode must be retained_forecast or current_collective"
            )
        if local_cost_scope not in {"response_impact", "own"}:
            raise ValueError("local_cost_scope must be response_impact or own")
        if not isinstance(membership_feedback, bool):
            raise ValueError("membership_feedback must be boolean")
        if not isinstance(response_until_equilibrium, bool):
            raise ValueError("response_until_equilibrium must be boolean")
        if initialization_mode not in {"local_overlap", "degree_balanced"}:
            raise ValueError(
                "initialization_mode must be local_overlap or degree_balanced"
            )
        if epsilon_qp <= 0.0 or qp_max_iterations <= 0:
            raise ValueError("QP tolerance and iteration limit must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.feedback_scope = feedback_scope
        self.response_execution = response_execution
        self.reference_mode = reference_mode
        self.local_cost_scope = local_cost_scope
        self.membership_feedback = membership_feedback
        self.response_until_equilibrium = response_until_equilibrium
        self.initialization_mode = initialization_mode
        self.epsilon_qp = float(epsilon_qp)
        self.qp_max_iterations = int(qp_max_iterations)

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        edges = [
            (int(source), int(target))
            for source, target in zip(*instance.adjacency.nonzero())
        ]
        overlap = local_modularity_overlapping_communities(
            instance.n, edges, gamma_dir=self.gamma
        )
        topology_initial = overlapping_memberships_from_communities(
            overlap.communities, instance.n
        )
        initial = topology_initial
        if self.initialization_mode == "degree_balanced":
            initial = _degree_balanced_memberships(
                self._out_degree_np,
                self._in_degree,
                topology_initial.shape[1],
            )
        colors, classes, ordering = impact_set_first_fit_coloring(instance.n, edges)
        self._memberships = initial.copy()
        self._initial_memberships = initial.copy()
        self.colors = np.asarray(colors, dtype=int)
        self.color_classes = [np.asarray(group, dtype=int) for group in classes]
        self.coloring_order = np.asarray(ordering, dtype=int)
        self._weights = self._community_influence(self._memberships)
        self._initial_weights = self._weights.clone()
        self._feedback_beta = (
            instance.beta.copy()
            if self.feedback_scope == "observed"
            else np.full((instance.n, instance.l), 1.0 / instance.l)
        )
        self.initialization_metadata = {
            "initialization_method": (
                "directed_local_overlap_and_impact_first_fit"
                if self.initialization_mode == "local_overlap"
                else "degree_balanced_round_robin_and_impact_first_fit"
            ),
            "initialization_mode": self.initialization_mode,
            "gamma": self.gamma,
            "community_count": int(initial.shape[1]),
            "compatible_class_count": len(self.color_classes),
            "alpha": self.alpha,
            "feedback_scope": self.feedback_scope,
            "response_execution": self.response_execution,
            "reference_mode": self.reference_mode,
            "local_cost_scope": self.local_cost_scope,
            "membership_feedback": self.membership_feedback,
            "response_until_equilibrium": self.response_until_equilibrium,
            "epsilon_w": instance.epsilon_w,
            "epsilon_qp": self.epsilon_qp,
            "qp_max_iterations": self.qp_max_iterations,
        }

    @property
    def memberships(self) -> np.ndarray:
        return self._memberships

    def _reference(self, profile: np.ndarray) -> np.ndarray:
        masses = self._feedback_beta.sum(axis=0)
        return np.divide(
            np.sum(self._feedback_beta * profile, axis=0),
            masses,
            out=np.zeros(profile.shape[1], dtype=float),
            where=masses > 0.0,
        )

    def _potential(
        self,
        prospective: np.ndarray,
        current: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        consensus = np.sum(
            self._feedback_beta * (prospective - reference[None, :]) ** 2
        )
        movement = np.sum(self._feedback_beta * (prospective - current) ** 2)
        return float(self.alpha * consensus + (1.0 - self.alpha) * movement)

    def _qp_coefficients(
        self,
        i: int,
        preferences: np.ndarray,
        memberships: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        epsilon_w = self.instance.epsilon_w
        scale = 1.0 - 2.0 * epsilon_w
        k = memberships.shape[1]
        out_i = self._out_neighbors_of(i)
        d_i = float(out_i.size)
        differences_i = preferences[out_i] - preferences[i]
        constant_i = preferences[i] + epsilon_w / d_i * differences_i.sum(axis=0)
        coefficient_i = scale / d_i * (differences_i.T @ memberships[out_i])
        beta_i = self._feedback_beta[i]
        gram = coefficient_i.T @ (beta_i[:, None] * coefficient_i)
        linear = coefficient_i.T @ (beta_i * (constant_i - targets[i]))

        if self.local_cost_scope == "response_impact":
            for raw_h in self._in_neighbors_of(i):
                h = int(raw_h)
                out_h = self._out_neighbors_of(h)
                d_h = float(out_h.size)
                differences_h = preferences[out_h] - preferences[h]
                coefficient_hi = scale / d_h * np.outer(
                    preferences[i] - preferences[h], memberships[h]
                )
                constant_h = (
                    preferences[h] + epsilon_w / d_h * differences_h.sum(axis=0)
                )
                without_i = out_h != i
                if np.any(without_i):
                    neighbors = out_h[without_i]
                    scores = memberships[neighbors] @ memberships[h]
                    constant_h += scale / d_h * np.sum(
                        scores[:, None] * (preferences[neighbors] - preferences[h]),
                        axis=0,
                    )
                beta_h = self._feedback_beta[h]
                gram += coefficient_hi.T @ (beta_h[:, None] * coefficient_hi)
                linear += coefficient_hi.T @ (
                    beta_h * (constant_h - targets[h])
                )
        return 2.0 * gram, 2.0 * linear

    def _candidate_batch(
        self,
        players: np.ndarray,
        preferences: np.ndarray,
        snapshot: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = [
            self._qp_coefficients(int(i), preferences, snapshot, targets)
            for i in players
        ]
        hessians = np.stack([item[0] for item in coefficients])
        linears = np.stack([item[1] for item in coefficients])
        current = snapshot[players]
        candidates = solve_simplex_qp_batch(
            hessians,
            linears,
            current,
            tolerance=self.epsilon_qp,
            max_iterations=self.qp_max_iterations,
        )
        accepted = _improvement_mask(
            candidates,
            current,
            hessians,
            linears,
            self.epsilon_qp,
        )
        return candidates, accepted

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        anchor = self._memberships.copy()
        candidate = anchor.copy()
        status_quo = self._community_influence(anchor).evolve(current)
        reference_profile = (
            status_quo if self.reference_mode == "retained_forecast" else current
        )
        reference = self._reference(reference_profile)
        targets = self.alpha * reference[None, :] + (1.0 - self.alpha) * current
        potential_before = self._potential(status_quo, current, reference)
        accepted_total = 0
        response_sweeps = 0
        equilibrium_reached = not self.membership_feedback

        if self.membership_feedback:
            while True:
                accepted_this_sweep = 0
                for players in self.color_classes:
                    if self.response_execution == "batched":
                        snapshot = candidate.copy()
                        responses, accepted = self._candidate_batch(
                            players, current, snapshot, targets
                        )
                        if np.any(accepted):
                            candidate[players[accepted]] = responses[accepted]
                            accepted_this_sweep += int(np.count_nonzero(accepted))
                        continue
                    for raw_i in players:
                        i = int(raw_i)
                        responses, accepted = self._candidate_batch(
                            np.asarray([i]), current, candidate.copy(), targets
                        )
                        if accepted[0]:
                            candidate[i] = responses[0]
                            accepted_this_sweep += 1
                response_sweeps += 1
                accepted_total += accepted_this_sweep
                if not self.response_until_equilibrium:
                    break
                if accepted_this_sweep == 0:
                    equilibrium_reached = True
                    break

        candidate_weights = self._community_influence(candidate)
        prospective = candidate_weights.evolve(current)
        potential_after = self._potential(prospective, current, reference)
        if (
            self.membership_feedback
            and self.local_cost_scope == "response_impact"
            and potential_after > potential_before + 1e-7
        ):
            raise RuntimeError("the compatible response sweep increased the potential")
        membership_delta = candidate - anchor
        self._memberships = candidate
        self._weights = candidate_weights
        return FeedbackDiagnostics(
            accepted_responses=accepted_total,
            sweeps=response_sweeps,
            membership_change=float(np.sum(membership_delta**2) / (2.0 * self.instance.n)),
            membership_adjustment=float(np.sum(np.abs(membership_delta))),
            potential_before=potential_before,
            potential_after=potential_after,
            active_edges=int(self.instance.adjacency.nnz),
            terminated=equilibrium_reached or accepted_total == 0,
        )
