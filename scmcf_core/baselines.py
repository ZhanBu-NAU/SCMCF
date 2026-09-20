"""Canonical fixed-support baselines used by the clean comparison package."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .backend import EdgeInfluence
from .instance import CRPInstance
from .mechanisms import FeedbackDiagnostics, InfluenceMechanism


__all__ = ["SNDGMechanism", "TEDGMechanism", "DGBCMechanism"]


class SNDGMechanism(InfluenceMechanism):
    """Keep a fixed social-network DeGroot matrix.

    The source model uses one fixed self-confidence value on every diagonal
    entry and distributes the remaining mass uniformly over the outgoing
    support.  The clean comparison exposes only the scalar workbook parameter.
    """

    name = "SNDG"

    def __init__(self, self_weight: float = 0.5) -> None:
        if not np.isfinite(self_weight) or not 0.0 < self_weight < 1.0:
            raise ValueError("self_weight must lie strictly between 0 and 1")
        self.self_weight = float(self_weight)

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        beta = self.backend.ones((instance.n,)) * self.self_weight
        beta_source = "fixed_scalar"
        initialization_method = "fixed_topology_uniform_degroot"
        social_mass = 1.0 - beta
        edge_weights = social_mass[self._rows] / self._out_degree[self._rows]
        self._initial_weights = EdgeInfluence(
            self.backend.clone(self._rows),
            self.backend.clone(self._cols),
            self.backend.clone(self._row_ptr),
            edge_weights,
            beta,
            instance.n,
            self.backend,
            self.backend.clone(self._evolution_mask),
        )
        self._weights = self._initial_weights.clone()
        self.initialization_metadata = {
            "initialization_method": initialization_method,
            "beta_source": beta_source,
            "self_weight": self.self_weight,
            "initial_weight_source": initialization_method,
            "topology_mode": "fixed_directed_support",
            "consensus_assessment_precedes_first_transition": True,
            "assessment_order": "initial_consensus_assessment_before_transition",
            "feedback_timing": "assessment -> one_fixed_influence_opinion_evolution",
        }

    def feedback(self, preferences: np.ndarray, round_index: int) -> FeedbackDiagnostics:
        self._weights = self._initial_weights.clone()
        return FeedbackDiagnostics()


class TEDGMechanism(InfluenceMechanism):
    """Update edge influence from current preference similarity.

    The calibrated self-weights induced by the fixed initial memberships are
    retained. Defaults match the paper experiment configuration. The common
    runner performs the initial consensus assessment before this mechanism's
    first opinion-evolution transition; no pre-assessment evolution is done.
    """

    name = "TEDG"

    feedback_timing = (
        "assessment -> one_opinion_evolution -> exogenous_trust_feedback "
        "-> next_assessment"
    )

    def __init__(
        self,
        reward_rate: float = 0.20,
        punishment_rate: float = 0.20,
        sensitivity: float = 1.0,
    ) -> None:
        if not 0.0 <= reward_rate <= 1.0:
            raise ValueError("reward_rate must lie in [0, 1]")
        if not 0.0 <= punishment_rate < 1.0:
            raise ValueError("punishment_rate must lie in [0, 1)")
        if sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive")
        self.reward_rate = reward_rate
        self.punishment_rate = punishment_rate
        self.sensitivity = sensitivity

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._self_confidence = self.backend.clone(self._initial_weights.self_weights)
        trust = np.ones(self._rows.size, dtype=float)
        self._trust = self.backend.array(trust)
        self._weights = self._matrix_from_trust()
        self._pending_preferences = None
        self.initialization_metadata.update(
            {
                "assessment_order": "initial_consensus_assessment_before_transition",
                "feedback_timing": self.feedback_timing,
                "trust_update_order": "after_one_opinion_evolution",
                "initial_trust_source": "neutral_unit_edge_trust",
            }
        )

    def evolve_after_feedback(self, preferences):
        """Return the opinion step prepared before TEDG trust feedback."""
        pending = getattr(self, "_pending_preferences", None)
        if pending is None:
            return self.weights.evolve(preferences)
        self._pending_preferences = None
        return pending

    def _matrix_from_trust(self) -> EdgeInfluence:
        n = self.instance.n
        row_totals = self.backend.segment_sum(self._row_ptr, self._trust)
        if np.any(self.backend.to_numpy(row_totals) <= 0.0):
            raise RuntimeError("TEDG produced a row with no positive trust")
        social_mass = 1.0 - self._self_confidence
        edge_weights = (
            social_mass[self._rows] * self._trust / row_totals[self._rows]
        )
        return EdgeInfluence(
            self._rows,
            self._cols,
            self._row_ptr,
            edge_weights,
            self.backend.clone(self._self_confidence),
            n,
            self.backend,
            self._evolution_mask,
        )

    def feedback(self, preferences: np.ndarray, round_index: int) -> FeedbackDiagnostics:
        del round_index
        # The source TEDG ordering allows endogenous opinion dynamics after
        # the current consensus assessment and before exogenous trust update.
        evolved = self.weights.evolve(preferences)
        self._pending_preferences = self.backend.clone(evolved)
        distances = self._edge_mean_abs_difference(evolved)
        similarities = self.backend.clip(1.0 - distances, 0.0, 1.0)
        targets = similarities**self.sensitivity
        increases = self.reward_rate * self.backend.maximum(
            targets - self._trust, 0.0
        )
        decreases = self.punishment_rate * self.backend.maximum(
            self._trust - targets, 0.0
        )
        self._trust = self.backend.clip(
            self._trust + increases - decreases, 0.0, 1.0
        )
        self._weights = self._matrix_from_trust()
        return FeedbackDiagnostics(
            active_edges=self.backend.count_nonzero(self._trust),
            sweeps=1,
        )


class DGBCMechanism(InfluenceMechanism):
    """Apply mixed DeGroot/HK screening with OSM network updating."""

    name = "DGBC"
    feedback_timing = (
        "assessment -> one_mixed_opinion_evolution -> OSM_network_update "
        "-> next_assessment"
    )

    def __init__(
        self,
        confidence: float | np.ndarray = 0.175,
        network_update: str = "osm",
        selection_rate: float = 0.001,
    ) -> None:
        if network_update != "osm":
            raise ValueError("the clean DGBC implementation requires network_update='osm'")
        if not 0.0 < selection_rate <= 1.0:
            raise ValueError("selection_rate must lie in (0, 1]")
        self.confidence = confidence
        self.network_update = network_update
        self.selection_rate = float(selection_rate)

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        rows, cols = instance.adjacency.nonzero()
        confidence = np.asarray(self.confidence, dtype=float)
        confidence_is_scalar = confidence.ndim == 0
        if confidence_is_scalar:
            confidence = np.full(instance.n, float(confidence))
        if confidence.shape != (instance.n,) or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("confidence must be a scalar or an N-vector in [0, 1]")
        self._confidence = self.backend.array(confidence)
        self._base_self = self.backend.clone(self._initial_weights.self_weights)
        self._base_edge = self.backend.clone(self._initial_weights.edge_weights)
        self._support_rows_np = np.asarray(rows, dtype=np.int64)
        self._support_cols_np = np.asarray(cols, dtype=np.int64)
        self._rewired_edges = 0
        self.initialization_metadata.update({
            "network_update": self.network_update,
            "network_update_order": (
                "after_consensus_assessment -> opinion_evolution -> network_update"
            ),
            "assessment_order": "initial_consensus_assessment_before_transition",
            "feedback_timing": self.feedback_timing,
            "selection_rate": self.selection_rate,
            "consensus_assessment_precedes_first_transition": True,
            "confidence": (
                float(np.asarray(self.confidence))
                if confidence_is_scalar
                else "node_specific"
            ),
        })
        self._pending_preferences = None

    def evolve_after_feedback(self, preferences):
        """Return the opinion step already computed before OSM updating."""
        pending = getattr(self, "_pending_preferences", None)
        if pending is None:
            return self.weights.evolve(preferences)
        self._pending_preferences = None
        return pending

    @property
    def validation_adjacency(self):
        """Current support for optional row-stochasticity validation."""
        return sparse.csr_matrix(
            (
                np.ones(self._support_rows_np.size, dtype=float),
                (self._support_rows_np, self._support_cols_np),
            ),
            shape=(self.instance.n, self.instance.n),
        )

    def _relation_masks(self, distances: np.ndarray):
        support = sparse.csr_matrix(
            (
                np.ones(self._support_rows_np.size, dtype=float),
                (self._support_rows_np, self._support_cols_np),
            ),
            shape=(self.instance.n, self.instance.n),
        )
        mutual_matrix = support.multiply(support.T).tocsr()
        mutual = np.asarray(
            mutual_matrix[self._support_rows_np, self._support_cols_np]
        ).reshape(-1) > 0.0
        confidence = self.backend.to_numpy(self._confidence)
        active = mutual | (
            distances <= confidence[self._support_rows_np]
        )
        return mutual, active

    def _osm_support(self, preferences: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rewire dissatisfied weak relations toward closest unconnected DMs.

        For each source, strong and satisfied-weak relations are retained.  A
        fraction of the dissatisfied weak relations, at least one when such a
        relation exists, is replaced by the closest currently unconnected
        candidates.  This sparse numerical rule follows the source OSM
        direction while avoiding an all-pairs edge materialization.
        """
        n = self.instance.n
        confidence = self.backend.to_numpy(self._confidence)
        support_sets = [set() for _ in range(n)]
        for source, target in zip(self._support_rows_np, self._support_cols_np):
            support_sets[int(source)].add(int(target))
        support_matrix = sparse.csr_matrix(
            (
                np.ones(self._support_rows_np.size, dtype=float),
                (self._support_rows_np, self._support_cols_np),
            ),
            shape=(n, n),
        )
        mutual_matrix = support_matrix.multiply(support_matrix.T).tocsr()
        new_edges: list[tuple[int, int]] = []
        for source in range(n):
            start, end = support_matrix.indptr[source : source + 2]
            neighbors = support_matrix.indices[start:end]
            neighbor_distances = distances[start:end]
            mutual = np.asarray(mutual_matrix[source, neighbors].toarray()).ravel() > 0.0
            keep = mutual | (neighbor_distances <= confidence[source])
            kept = neighbors[keep]
            dissatisfied_count = int(np.count_nonzero(~keep))
            add_count = (
                0
                if dissatisfied_count == 0
                else max(1, int(np.ceil(self.selection_rate * dissatisfied_count)))
            )
            candidates = np.asarray(
                [node for node in range(n) if node != source and node not in support_sets[source]],
                dtype=np.int64,
            )
            if add_count and candidates.size:
                candidate_distances = np.mean(
                    np.abs(preferences[candidates] - preferences[source]),
                    axis=1,
                )
                order = np.lexsort((candidates, candidate_distances))
                additions = candidates[order[:add_count]]
            else:
                additions = np.empty(0, dtype=np.int64)
            for target in np.concatenate((kept, additions)):
                new_edges.append((source, int(target)))
        if new_edges:
            edge_array = np.asarray(new_edges, dtype=np.int64)
            order = np.lexsort((edge_array[:, 1], edge_array[:, 0]))
            edge_array = edge_array[order]
            return edge_array[:, 0], edge_array[:, 1]
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    def _install_support(self, rows_np: np.ndarray, cols_np: np.ndarray) -> None:
        rows_np = np.asarray(rows_np, dtype=np.int64)
        cols_np = np.asarray(cols_np, dtype=np.int64)
        self._support_rows_np = rows_np
        self._support_cols_np = cols_np
        self._rows = self.backend.index(rows_np)
        self._cols = self.backend.index(cols_np)
        row_ptr_np = np.concatenate(
            ([0], np.cumsum(np.bincount(rows_np, minlength=self.instance.n), dtype=np.int64))
        )
        self._row_ptr = self.backend.index(row_ptr_np)
        self._edge_rows_np = rows_np
        self._edge_cols_np = cols_np
        out_degree = np.bincount(rows_np, minlength=self.instance.n).astype(float)
        self._out_degree_np = out_degree
        self._inverse_out_degree_np = np.divide(
            1.0,
            out_degree,
            out=np.zeros_like(out_degree),
            where=out_degree > 0.0,
        )
        self._out_degree = self.backend.array(out_degree)
        social_mass = 1.0 - self._base_self
        edge = self.backend.where(
            self._out_degree[self._rows] > 0.0,
            social_mass[self._rows] / self._out_degree[self._rows],
            self.backend.zeros(self._rows.shape),
        )
        self._base_edge = edge
        self._rewired_edges += int(rows_np.size)

    def feedback(self, preferences: np.ndarray, round_index: int) -> FeedbackDiagnostics:
        del round_index
        # The source ordering is opinion evolution first and OSM rewiring
        # second.  The common runner calls this method after assessment and
        # then consumes ``_pending_preferences`` without evolving twice.
        evolved = self.weights.evolve(preferences)
        self._pending_preferences = self.backend.clone(evolved)
        distances = self._edge_mean_abs_difference(evolved)
        distances_np = self.backend.to_numpy(distances)
        previous_edges = self._support_rows_np.size
        rows_np, cols_np = self._osm_support(
            self.backend.to_numpy(evolved), distances_np
        )
        self._install_support(rows_np, cols_np)
        distances = self._edge_mean_abs_difference(evolved)
        distances_np = self.backend.to_numpy(distances)
        if self._support_rows_np.size:
            _, active_np = self._relation_masks(distances_np)
        else:
            active_np = np.empty(0, dtype=bool)
        self.initialization_metadata["last_rewired_edge_count"] = int(
            abs(self._support_rows_np.size - previous_edges)
        )
        active = self.backend.boolean(active_np)
        active_base = self._base_edge * active
        n = self.instance.n
        totals = self.backend.segment_sum(self._row_ptr, active_base)
        has_active = totals > 0.0
        social_mass = 1.0 - self._base_self
        safe_totals = self.backend.where(
            has_active, totals, self.backend.ones((n,))
        )
        edge_weights = (
            social_mass[self._rows] * active_base / safe_totals[self._rows]
        )
        self_weights = self.backend.where(
            has_active, self._base_self, self.backend.ones((n,))
        )
        self._weights = EdgeInfluence(
            self._rows,
            self._cols,
            self._row_ptr,
            edge_weights,
            self_weights,
            n,
            self.backend,
            self._evolution_mask,
        )
        return FeedbackDiagnostics(
            active_edges=self.backend.count_nonzero(active),
        )
