"""CPU implementations of the ten canonical external comparison adapters.

These classes deliberately do not implement the SCMCF membership game. They
keep the intervention object of the selected external mechanism: dynamic trust
and recommendation feedback for Guo, dynamic trust and limited compromise for
Zou, overlapping-community recommendation feedback for Ding, and
overlapping-community feedback with reverse trust updating for Teng.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as scipy_sparse
from scipy.optimize import linprog

from .backend import EdgeInfluence, NumpyBackend
from .community_methods import (
    cr_oca_overlapping_labels,
    lfm_overlapping_labels,
    symmetric_preference_support,
    wdslpa_overlapping_labels,
)
from .instance import CRPInstance
from .mechanisms import FeedbackDiagnostics, InfluenceMechanism
from .metrics import collective_preference, individual_consensus_levels


__all__ = [
    "DTRFMechanism", "DTLCMechanism", "OCRFMechanism", "DCRTFMechanism",
    "OBCFMechanism", "OCCFMechanism", "NGPFMechanism",
]

RESPONSE_STRENGTH_MODES = frozenset({"fixed_surrogate", "source_adaptive"})
ADAPTIVE_ETA_MODES = frozenset({"degree", "trust", "dispersion"})


def _validate_response_strength_mode(mode: str) -> str:
    if mode not in RESPONSE_STRENGTH_MODES:
        choices = ", ".join(sorted(RESPONSE_STRENGTH_MODES))
        raise ValueError(f"response_strength_mode must be one of: {choices}")
    return mode


def _validate_adaptive_eta_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    if mode not in ADAPTIVE_ETA_MODES:
        choices = ", ".join(sorted(ADAPTIVE_ETA_MODES))
        raise ValueError(f"adaptive_eta must be one of: {choices}, or None")
    return mode


def _pagerank_on_support(
    adjacency,
    nodes: np.ndarray | None = None,
    *,
    damping: float = 0.85,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> np.ndarray:
    """Compute a PageRank vector on a directed induced support.

    The source OBCF model applies the same PageRank recurrence globally and
    inside every detected subgroup.  This helper keeps that recurrence sparse
    and handles induced dangling rows by redistributing their mass uniformly.
    """
    if not 0.0 < damping < 1.0:
        raise ValueError("damping must lie in (0, 1)")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("PageRank tolerance and iterations must be positive")
    support = adjacency.tocsr()
    if nodes is None:
        node_ids = np.arange(support.shape[0], dtype=np.int64)
    else:
        node_ids = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if node_ids.size == 0:
        return np.empty(0, dtype=float)
    induced = support[node_ids][:, node_ids]
    induced = (induced > 0.0).astype(float).tocsr()
    induced.sort_indices()
    size = node_ids.size
    out_degree = np.diff(induced.indptr).astype(float)
    # The source recurrence is a row-stochastic random walk with uniform
    # dangling redistribution.  Build that sparse transition once so every
    # iteration uses a compiled sparse matrix-vector product instead of a
    # Python loop over all sources.
    inverse_out_degree = np.divide(
        1.0,
        out_degree,
        out=np.zeros_like(out_degree),
        where=out_degree > 0.0,
    )
    transition = induced.multiply(inverse_out_degree[:, None]).tocsr()
    rank = np.full(size, 1.0 / size, dtype=float)
    teleport = (1.0 - damping) / size
    dangling = out_degree == 0.0
    for _ in range(max_iterations):
        dangling_mass = float(rank[dangling].sum())
        next_rank = np.full(
            size,
            teleport + damping * dangling_mass / size,
            dtype=float,
        )
        next_rank += damping * np.asarray(transition.T @ rank).ravel()
        if np.max(np.abs(next_rank - rank)) <= tolerance:
            rank = next_rank
            break
        rank = next_rank
    total = float(rank.sum())
    return rank / total if total > 0.0 else np.full(size, 1.0 / size)


def _batched_local_pagerank(
    adjacency: scipy_sparse.spmatrix,
    labels: scipy_sparse.csr_matrix,
    *,
    damping: float,
    tolerance: float,
    max_iterations: int,
) -> scipy_sparse.csr_matrix:
    """Compute all induced-community PageRank vectors in sparse batches.

    For a community incidence matrix ``B``, ``A @ B`` gives each member's
    within-community out-degree for every community.  The recurrence can then
    be applied to all communities at once as ``A.T @ H`` with a sparse mask
    back onto ``B``.  This is algebraically the same induced-subgraph
    PageRank used by the source adapter, without one Python solver per
    community.
    """
    support = (scipy_sparse.csr_matrix(adjacency, dtype=float) > 0.0).astype(float)
    support.sort_indices()
    incidence = scipy_sparse.csr_matrix(labels, dtype=float)
    incidence.sort_indices()
    n, community_count = incidence.shape
    sizes = np.asarray(incidence.sum(axis=0)).ravel()
    sizes = np.maximum(sizes, 1.0)
    local_out = (support @ incidence).tocsr()
    local_out.sort_indices()
    inverse_out = local_out.copy()
    inverse_out.data = 1.0 / np.maximum(inverse_out.data, 1e-12)
    active_mask = local_out.copy()
    active_mask.data[:] = 1.0
    teleport = incidence.multiply((1.0 - damping) / sizes[None, :])
    state = incidence.multiply(1.0 / sizes[None, :]).tocsr()
    for _ in range(max_iterations):
        scaled = state.multiply(inverse_out)
        incoming = (support.T @ scaled).tocsr().multiply(incidence)
        dangling = state - state.multiply(active_mask)
        dangling_mass = np.asarray(dangling.sum(axis=0)).ravel()
        next_state = (
            teleport
            + damping * incoming
            + incidence.multiply(damping * dangling_mass[None, :] / sizes[None, :])
        ).tocsr()
        next_state.eliminate_zeros()
        difference = next_state - state
        max_change = float(
            np.max(np.abs(difference.data), initial=0.0)
        )
        state = next_state
        if max_change <= tolerance:
            break
    return state


def _solve_wang_subgroup_lp(
    current: np.ndarray,
    members: np.ndarray,
    collective: np.ndarray,
    local_weights: np.ndarray,
    confidence: np.ndarray,
    *,
    stage: str,
    target_disagreement: float | None = None,
    lp_tolerance: float = 1e-8,
) -> tuple[np.ndarray, float, float, bool, int]:
    """Solve one numerical adaptation of Wang's subgroup LPs.

    The source models use ELICIT/TrFN values and mean Manhattan distances.
    For the common numerical profile, each ELICIT distance is represented by
    the mean absolute distance over the L profile cells.  The collective
    opinion is the current global weighted opinion, as defined in the source
    models; it is held fixed while one subgroup recommendation is computed.
    """
    if stage not in {
        "maximum_consensus",
        "minimum_adjustment_unattainable",
        "minimum_adjustment_attainable",
    }:
        raise ValueError(f"unsupported Wang LP stage: {stage}")
    values = np.asarray(current, dtype=float)
    member_ids = np.asarray(members, dtype=np.int64).reshape(-1)
    if member_ids.size == 0:
        return np.empty((0, values.shape[1])), 1.0, 0.0, True, 0
    profile = values[member_ids]
    target = np.asarray(collective, dtype=float).reshape(-1)
    weights = np.asarray(local_weights, dtype=float).reshape(-1)
    eta = np.asarray(confidence, dtype=float).reshape(-1)
    if weights.shape != (member_ids.size,):
        raise ValueError("local_weights must match subgroup members")
    if eta.shape != (member_ids.size,):
        raise ValueError("confidence must match subgroup members")
    if target.shape != (values.shape[1],):
        raise ValueError("collective must match the profile dimension")
    weights = np.maximum(weights, 0.0)
    weights = weights / max(float(weights.sum()), 1e-12)
    eta = np.clip(eta, 0.0, 1.0)
    nq, dimension = profile.shape
    block = nq * dimension
    z_offset = block
    u_offset = 2 * block
    n_variables = 3 * block

    row_ids: list[int] = []
    col_ids: list[int] = []
    coefficients: list[float] = []
    rhs: list[float] = []

    def add_inequality(entries: list[tuple[int, float]], bound: float) -> None:
        row = len(rhs)
        for column, value in entries:
            row_ids.append(row)
            col_ids.append(column)
            coefficients.append(float(value))
        rhs.append(float(bound))

    # z_{i,l} >= |x_{i,l} - collective_l|.
    for i in range(nq):
        for ell in range(dimension):
            x_index = i * dimension + ell
            z_index = z_offset + x_index
            add_inequality(
                [(x_index, 1.0), (z_index, -1.0)], target[ell]
            )
            add_inequality(
                [(x_index, -1.0), (z_index, -1.0)], -target[ell]
            )

    # u_{i,l} >= |x_{i,l} - current_{i,l}|.
    for i in range(nq):
        for ell in range(dimension):
            x_index = i * dimension + ell
            u_index = u_offset + x_index
            add_inequality(
                [(x_index, 1.0), (u_index, -1.0)], profile[i, ell]
            )
            add_inequality(
                [(x_index, -1.0), (u_index, -1.0)], -profile[i, ell]
            )

    bounded_stage = stage in {
        "maximum_consensus",
        "minimum_adjustment_attainable",
    }
    if bounded_stage:
        for i in range(nq):
            add_inequality(
                [
                    (u_offset + i * dimension + ell, 1.0)
                    for ell in range(dimension)
                ],
                eta[i] * dimension,
            )

    if target_disagreement is not None:
        if not 0.0 <= target_disagreement <= 1.0 + lp_tolerance:
            raise ValueError("target_disagreement must lie in [0, 1]")
        add_inequality(
            [
                (z_offset + i * dimension + ell, weights[i])
                for i in range(nq)
                for ell in range(dimension)
            ],
            (float(target_disagreement) + lp_tolerance) * dimension,
        )

    objective = np.zeros(n_variables, dtype=float)
    if stage == "maximum_consensus":
        for i in range(nq):
            objective[z_offset + i * dimension : z_offset + (i + 1) * dimension] = weights[i]
    else:
        for i in range(nq):
            objective[u_offset + i * dimension : u_offset + (i + 1) * dimension] = weights[i]

    # Bounds keep both the modified profile and absolute-value auxiliaries
    # within the normalized numerical profile range.
    bounds = [(0.0, 1.0)] * block + [(0.0, 1.0)] * block + [(0.0, 1.0)] * block
    matrix = scipy_sparse.coo_matrix(
        (coefficients, (row_ids, col_ids)),
        shape=(len(rhs), n_variables),
    ).tocsr()
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.asarray(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        disagreement = float(
            np.sum(weights[:, None] * np.abs(profile - target[None, :]))
            / dimension
        )
        return profile.copy(), 1.0 - disagreement, 0.0, False, int(result.status)

    recommendation = np.clip(np.asarray(result.x[:block]), 0.0, 1.0).reshape(
        nq, dimension
    )
    disagreement = float(
        np.sum(weights[:, None] * np.abs(recommendation - target[None, :]))
        / dimension
    )
    adjustment = float(
        np.sum(weights[:, None] * np.abs(recommendation - profile))
        / dimension
    )
    return recommendation, 1.0 - disagreement, adjustment, True, int(result.status)


class _FeedbackAdapter(InfluenceMechanism):
    """Shared state and CPU-only transition hook for external mechanisms."""

    source_id = "unspecified"
    feedback_object = "direct_preference_feedback"
    native_constraints = "one_bounded_response_per_feedback_round"
    network_mode = "common_support"
    # Proxies may update weights, trust, or active-edge masks, but they do not
    # add or delete adjacency entries unless a class explicitly implements it.
    topology_mode = "fixed_candidate_support_dynamic_weights"
    native_representation = "numerical"
    source_community_representation = "none"
    implemented_community_representation = "none"
    community_label_source = "none"
    community_overlap_rule = "none"
    # Most source CRPs terminate a feedback phase with a revised preference
    # profile. A separate DeGroot step is only appropriate when the source
    # explicitly contains an opinion-dynamics stage.
    native_opinion_dynamics = False
    feedback_timing = "assessment -> direct_feedback -> next_assessment"
    assessment_order = "initial_consensus_assessment_before_transition"
    feedback_order = "direct_feedback"
    community_refresh_policy = "not_applicable"
    opinion_evolution_rule = "none"
    exclude_overlapping_from_opinion_dynamics = False
    # External mechanisms can use the runner's optional stability stop after
    # validating all state components that affect their next transition.
    stability_termination_supported = True

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        if backend is not None and backend.is_torch:
            raise ValueError("literature mechanisms currently require NumPy/CPU")
        super().initialize(instance, NumpyBackend())
        self._pending_preferences: np.ndarray | None = None
        self._last_trust_variation = 0.0
        self._last_trust_max_variation = 0.0
        self._last_community_state_changed = False
        self._last_community_overlap = 0.0
        self._last_direct_feedback_adjustment = 0.0
        self._last_opinion_dynamics_adjustment = 0.0
        self._last_response_control_rule = "not_recorded"
        self._last_response_rate_summary: dict[str, float | None] = {
            "mean": None,
            "min": None,
            "max": None,
        }
        self._last_response_radius_summary: dict[str, float | None] = {
            "mean": None,
            "min": None,
            "max": None,
        }
        self._last_response_acceptance_fraction: float | None = None

    def feedback_state_is_stable(self, diagnostics: FeedbackDiagnostics) -> bool:
        """Check response, trust, community, and opinion state stability."""
        if not diagnostics.terminated or self.native_opinion_dynamics:
            self._last_feedback_state_stable = False
            return False
        if self._last_trust_max_variation > self.change_tolerance:
            self._last_feedback_state_stable = False
            return False
        self._last_feedback_state_stable = not self._last_community_state_changed
        return self._last_feedback_state_stable

    @staticmethod
    def _validate_response_strength_mode(mode: str) -> str:
        return _validate_response_strength_mode(mode)

    def _record_response_controls(
        self,
        response_rate: np.ndarray,
        response_radius: np.ndarray | None,
        *,
        rule: str,
        accepted: np.ndarray | None = None,
    ) -> None:
        rates = np.asarray(response_rate, dtype=float).reshape(-1)
        self._last_response_control_rule = rule
        self._last_response_rate_summary = {
            "mean": float(np.mean(rates)) if rates.size else None,
            "min": float(np.min(rates)) if rates.size else None,
            "max": float(np.max(rates)) if rates.size else None,
        }
        if response_radius is None:
            self._last_response_radius_summary = {
                "mean": None,
                "min": None,
                "max": None,
            }
        else:
            radii = np.asarray(response_radius, dtype=float).reshape(-1)
            self._last_response_radius_summary = {
                "mean": float(np.mean(radii)) if radii.size else None,
                "min": float(np.min(radii)) if radii.size else None,
                "max": float(np.max(radii)) if radii.size else None,
            }
        self._last_response_acceptance_fraction = (
            float(np.mean(np.asarray(accepted, dtype=bool)))
            if accepted is not None and np.asarray(accepted).size
            else None
        )

    def evolve_after_feedback(self, preferences: np.ndarray) -> np.ndarray:
        """Apply the source-native transition after one feedback phase."""
        base = (
            preferences
            if self._pending_preferences is None
            else self._pending_preferences
        )
        self._pending_preferences = None
        current = np.asarray(preferences, dtype=float)
        base = np.asarray(base, dtype=float)
        base = self._mask_update(current, base)
        self._last_direct_feedback_adjustment = float(
            np.sum(np.abs(base - current))
        )
        if not self.native_opinion_dynamics:
            self._last_opinion_dynamics_adjustment = 0.0
            return base
        evolved = self.weights.evolve(base)
        if self.exclude_overlapping_from_opinion_dynamics:
            labels = getattr(self, "_community_label_csr", None)
            if labels is not None:
                overlapping = np.asarray(labels.sum(axis=1)).ravel() > 1
                evolved = np.where(overlapping[:, None], base, evolved)
        updated = self._mask_update(base, evolved)
        self._last_opinion_dynamics_adjustment = float(
            np.sum(np.abs(np.asarray(updated) - base))
        )
        # Gai's source-native transition includes an opinion-evolution step.
        # Refresh its preference-dependent communities after that step so the
        # next assessment and recorded community modularity use the same state.
        refresh = getattr(self, "_refresh_communities", None)
        if refresh is not None:
            refresh(np.asarray(updated, dtype=float), np.empty(0, dtype=float))
        return updated

    def _consensus_levels(self, preferences: np.ndarray) -> np.ndarray:
        """Return the consensus levels used by the source feedback rule.

        The evolution mask defines the cells visible to feedback.  In the
        main neutral-imputation setting it is all ones, so the response rule
        evaluates the complete profile; the separate observation mask is used
        only by the CRP runner's reported metrics.
        """
        return individual_consensus_levels(
            preferences, self.instance.evolution_mask
        )

    def _edge_distances(self, preferences: np.ndarray) -> np.ndarray:
        """Return mean absolute edge differences on shared evolving cells."""
        values = np.asarray(preferences, dtype=float)
        edge_count = self._rows.size
        distances = np.ones(edge_count, dtype=float)
        # Keep profile temporaries bounded for million-edge, 84-dimensional
        # Deezer instances.  The per-edge reduction and arithmetic are
        # unchanged; only the evaluation order across independent rows moves
        # into fixed-size chunks.
        edge_chunk = 65_536
        for start in range(0, edge_count, edge_chunk):
            end = min(start + edge_chunk, edge_count)
            source_mask = self._evolution_mask_np[self._rows[start:end]]
            target_mask = self._evolution_mask_np[self._cols[start:end]]
            shared = source_mask * target_mask
            counts = shared.sum(axis=1)
            numerators = np.sum(
                shared
                * np.abs(
                    values[self._rows[start:end]]
                    - values[self._cols[start:end]]
                ),
                axis=1,
            )
            distances[start:end] = np.divide(
                numerators,
                counts,
                # No shared evolving cell provides no similarity evidence;
                # distance one prevents sparse pairs from being twins.
                out=np.ones(end - start, dtype=float),
                where=counts > 0.0,
            )
        return distances

    def _mask_update(
        self, current: np.ndarray, updated: np.ndarray
    ) -> np.ndarray:
        """Freeze only cells excluded by the evolution mask.

        Under the main neutral-imputation setting the evolution mask is all
        ones, so both observed and initially missing cells are updated.
        """
        return np.where(self._evolution_mask_np > 0.0, updated, current)

    @staticmethod
    def _row_totals(values: np.ndarray, row_ptr: np.ndarray) -> np.ndarray:
        totals = np.add.reduceat(values, row_ptr[:-1])
        return np.maximum(totals, 1e-12)

    def _row_normalized_targets(
        self, preferences: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Return edge-score-weighted targets on shared evolving cells."""
        if not self.backend.is_torch and np.all(self._evolution_mask_np > 0.5):
            values = np.asarray(preferences, dtype=float)
            scores = np.asarray(scores, dtype=float)
            row_totals = np.asarray(
                np.add.reduceat(scores, self._row_ptr[:-1]), dtype=float
            )
            row_totals = np.maximum(row_totals, 1e-12)
            normalized = scores / row_totals[self._edge_rows_np]
            target_matrix = scipy_sparse.csr_matrix(
                (normalized, (self._edge_rows_np, self._edge_cols_np)),
                shape=(self.instance.n, self.instance.n),
            )
            target_matrix.sum_duplicates()
            weighted = np.asarray(target_matrix @ values, dtype=float)
            observed_mass = np.asarray(target_matrix.sum(axis=1)).ravel()[:, None]
            return np.divide(
                weighted,
                observed_mass,
                out=values.copy(),
                where=observed_mass > 0.0,
            )
        row_totals = self._row_totals(scores, self._row_ptr)
        repeated_totals = np.repeat(row_totals, np.diff(self._row_ptr))
        normalized = scores / repeated_totals
        weighted = self.backend.segment_sum(
            self._row_ptr,
            normalized[:, None]
            * self._evolution_mask_np[self._cols]
            * preferences[self._cols],
        )
        observed_mass = self.backend.segment_sum(
            self._row_ptr,
            normalized[:, None] * self._evolution_mask_np[self._cols],
        )
        return np.divide(
            weighted,
            observed_mass,
            out=np.asarray(preferences, dtype=float).copy(),
            where=observed_mass > 0.0,
        )

    def _matrix_from_trust(self, trust: np.ndarray) -> EdgeInfluence:
        row_totals = self._row_totals(trust, self._row_ptr)
        repeated_totals = np.repeat(row_totals, np.diff(self._row_ptr))
        social_mass = 1.0 - self._self_confidence
        edge_weights = (
            social_mass[self._rows] * trust / repeated_totals
        )
        return EdgeInfluence(
            self._rows,
            self._cols,
            self._row_ptr,
            self.backend.array(edge_weights),
            self.backend.clone(self._self_confidence),
            self.instance.n,
            self.backend,
            self._evolution_mask,
        )

    def _matrix_from_edge_scores(self, scores: np.ndarray) -> EdgeInfluence:
        """Build a row-stochastic matrix from nonnegative edge scores."""
        raw_totals = np.add.reduceat(scores, self._row_ptr[:-1])
        has_active = raw_totals > 1e-12
        row_totals = np.maximum(raw_totals, 1e-12)
        repeated_totals = np.repeat(row_totals, np.diff(self._row_ptr))
        social_mass = 1.0 - self._self_confidence
        edge_weights = social_mass[self._rows] * scores / repeated_totals
        self_weights = np.where(has_active, self._self_confidence, 1.0)
        return EdgeInfluence(
            self._rows,
            self._cols,
            self._row_ptr,
            self.backend.array(edge_weights),
            self.backend.array(self_weights),
            self.instance.n,
            self.backend,
            self._evolution_mask,
        )

    @property
    def memberships(self) -> None:
        """External adapters do not expose SCMCF membership adjustments."""
        return None

    @property
    def adapter_metadata(self) -> dict[str, object]:
        return {
            "mechanism_label": self.name,
            "mechanism_full_name": self.full_name,
            "mechanism_abbreviation": self.name,
            "source_id": self.source_id,
            "native_representation": self.native_representation,
            "source_community_representation": self.source_community_representation,
            "implemented_community_representation": self.implemented_community_representation,
            "community_label_source": self.community_label_source,
            "community_overlap_rule": self.community_overlap_rule,
            "community_detector": getattr(self, "community_detector", "none"),
            "community_preference_input": getattr(
                self, "community_preference_input", "none"
            ),
            "community_update_rule": getattr(
                self, "community_update_rule", "none"
            ),
            "community_count": getattr(self, "_last_community_count", None),
            "community_similarity_threshold": getattr(
                self, "similarity_threshold", None
            ),
            "lfm_alpha": getattr(self, "lfm_alpha", None),
            "confidence_bound": getattr(self, "confidence_bound", None),
            "adaptive_eta": getattr(self, "adaptive_eta", None),
            "eta_min": getattr(self, "eta_min", None),
            "eta_max": getattr(self, "eta_max", None),
            "eta_signal_source": getattr(self, "eta_signal_source", None),
            "last_eta_summary": getattr(
                self,
                "_last_eta_summary",
                {"mean": None, "min": None, "max": None, "std": None},
            ),
            "similarity_support_threshold": getattr(
                self, "similarity_support_threshold", None
            ),
            "network_mode": self.network_mode,
            "topology_mode": self.topology_mode,
            "feedback_object": self.feedback_object,
            "native_constraints": self.native_constraints,
            "native_opinion_dynamics": self.native_opinion_dynamics,
            "feedback_timing": self.feedback_timing,
            "assessment_order": self.assessment_order,
            "feedback_order": self.feedback_order,
            "trust_update_order": getattr(
                self, "trust_update_order", "not_applicable"
            ),
            "primary_trust_update_rule": getattr(
                self, "primary_trust_update_rule", "not_applicable"
            ),
            "community_refresh_policy": self.community_refresh_policy,
            "opinion_evolution_rule": self.opinion_evolution_rule,
            "exclude_overlapping_from_opinion_dynamics": (
                self.exclude_overlapping_from_opinion_dynamics
            ),
            "transition_mode": (
                "direct_feedback_then_opinion_dynamics"
                if self.native_opinion_dynamics
                else "direct_feedback_only"
            ),
            "stability_termination_supported": (
                self.stability_termination_supported
            ),
            "direct_feedback_adjustment": getattr(
                self, "_last_direct_feedback_adjustment", 0.0
            ),
            "opinion_dynamics_adjustment": getattr(
                self, "_last_opinion_dynamics_adjustment", 0.0
            ),
            "encoding_mae": 0.0,
            "encoding_max_abs_error": 0.0,
            "trust_variation": getattr(self, "_last_trust_variation", 0.0),
            "trust_max_variation": getattr(
                self, "_last_trust_max_variation", 0.0
            ),
            "feedback_state_stable": getattr(
                self, "_last_feedback_state_stable", False
            ),
            "community_overlap": getattr(self, "_last_community_overlap", 0.0),
            "response_strength_mode": getattr(
                self, "response_strength_mode", "not_applicable"
            ),
            "response_control_rule": getattr(
                self, "_last_response_control_rule", "not_recorded"
            ),
            "configured_response_rate": getattr(self, "response_rate", None),
            "configured_trust_rate": getattr(self, "trust_rate", None),
            "configured_role_threshold": getattr(
                self, "role_threshold_spec", getattr(self, "role_threshold", None)
            ),
            "effective_role_threshold": getattr(
                self, "_last_role_threshold", getattr(self, "role_threshold", None)
            ),
            "configured_response_radius": getattr(
                self, "compromise_radius",
                getattr(self, "willingness_radius", getattr(self, "feedback_radius", None)),
            ),
            "last_response_rate": getattr(
                self,
                "_last_response_rate_summary",
                {"mean": None, "min": None, "max": None},
            ),
            "last_response_radius": getattr(
                self,
                "_last_response_radius_summary",
                {"mean": None, "min": None, "max": None},
            ),
            "last_response_acceptance_fraction": getattr(
                self, "_last_response_acceptance_fraction", None
            ),
            "targeted_dm_count": int(
                getattr(self, "_last_targeted_dms", np.empty(0, dtype=np.int64)).size
            ),
            "target_selection_policy": getattr(
                self, "target_selection_policy", "not_applicable"
            ),
            "targeted_dm_indices": [
                int(value)
                for value in getattr(
                    self, "_last_targeted_dms", np.empty(0, dtype=np.int64)
                )
            ],
            "recommendation_mode": getattr(
                self, "_last_recommendation_mode", "not_applicable"
            ),
            "secondary_trust_source": getattr(
                self, "_last_secondary_trust_source", "not_applicable"
            ),
            "pagerank_damping": getattr(self, "pagerank_damping", None),
            "pagerank_tolerance": getattr(self, "pagerank_tolerance", None),
            "pagerank_max_iterations": getattr(
                self, "pagerank_max_iterations", None
            ),
            "global_local_weight": getattr(self, "global_local_weight", None),
            "elicitation_grid_size": getattr(
                self, "elicitation_grid_size", None
            ),
            "elicitation_stage": getattr(
                self,
                "_last_elicitation_stage",
                "not_applicable",
            ),
            "stage1_maximum_consensus": getattr(
                self, "_last_stage1_maximum_consensus", None
            ),
            "stage2_minimum_adjustment": getattr(
                self, "_last_stage2_minimum_adjustment", None
            ),
            "wang_lp_tolerance": getattr(self, "_wang_lp_tolerance", None),
            "wang_subgroup_consensus": (
                None
                if getattr(self, "_last_wang_subgroup_consensus", None) is None
                else [
                    float(value)
                    for value in self._last_wang_subgroup_consensus
                ]
            ),
            "wang_deficient_subgroups": [
                int(value)
                for value in getattr(
                    self,
                    "_last_wang_deficient_subgroups",
                    np.empty(0, dtype=np.int64),
                )
            ],
            "wang_stage2_modes": list(
                getattr(self, "_last_wang_stage2_modes", [])
            ),
            "wang_lp_statuses": [
                int(value)
                for value in getattr(self, "_last_wang_lp_statuses", [])
            ],
            "wang_lp_fallbacks": int(
                getattr(self, "_last_wang_lp_fallbacks", 0)
            ),
        }


class _DTLCImplementation(_FeedbackAdapter):
    """Trust evolution plus dynamic limited-compromise adapter.

    The mechanism keeps a historical edge-trust state, updates it from current
    preference similarity, and gives each low-consensus DM one bounded
    minimum-adjustment recommendation. It does not solve a membership QP.
    """

    name = "Zou-Trust-Compromise-proxy"
    full_name = "Dynamic Trust Limited-Compromise"
    source_id = "Zou2024INS"
    feedback_object = "minimum_adjustment_dynamic_limited_compromise"
    native_constraints = "dynamic_compromise_radius_and_response_threshold"
    native_representation = "probabilistic_linguistic_expected_value_surrogate"
    feedback_order = "trust_update_then_direct_limited_compromise_feedback"

    def __init__(
        self,
        trust_rate: float = 0.35,
        compromise_radius: float = 0.20,
        response_rate: float = 1.0,
        response_threshold: float = 0.95,
        change_tolerance: float = 1e-4,
        response_strength_mode: str = "source_adaptive",
    ) -> None:
        if not 0.0 < trust_rate <= 1.0:
            raise ValueError("trust_rate must lie in (0, 1]")
        if not 0.0 < compromise_radius <= 1.0:
            raise ValueError("compromise_radius must lie in (0, 1]")
        if not 0.0 < response_rate <= 1.0:
            raise ValueError("response_rate must lie in (0, 1]")
        if not 0.0 < response_threshold <= 1.0:
            raise ValueError("response_threshold must lie in (0, 1]")
        if change_tolerance <= 0.0:
            raise ValueError("change_tolerance must be positive")
        response_strength_mode = self._validate_response_strength_mode(
            response_strength_mode
        )
        # Kept for compatibility with the earlier similarity-smoothed proxy;
        # source DTNU carries primary trust forward without this rate.
        self.trust_rate = trust_rate
        self.compromise_radius = compromise_radius
        self.response_rate = response_rate
        self.response_threshold = response_threshold
        self.change_tolerance = change_tolerance
        self.response_strength_mode = response_strength_mode

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._self_confidence = self.backend.to_numpy(
            self._initial_weights.self_weights
        ).copy()
        # Neutral topology-only starting trust for DTLC.
        self._trust = np.ones(self._rows.size, dtype=float)
        self._weights = self._matrix_from_trust(self._trust)

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        levels = self._consensus_levels(current)
        distances = self._edge_distances(current)
        similarities = np.clip(1.0 - distances, 0.0, 1.0)
        previous_trust = self._trust.copy()
        self._trust = np.clip(
            (1.0 - self.trust_rate) * self._trust
            + self.trust_rate * similarities,
            1e-6,
            1.0,
        )
        self._last_trust_variation = float(
            np.mean(np.abs(self._trust - previous_trust))
        )
        self._last_trust_max_variation = float(
            np.max(np.abs(self._trust - previous_trust), initial=0.0)
        )
        self._weights = self._matrix_from_trust(self._trust)

        targets = self._row_normalized_targets(current, self._trust)
        eligible = levels < self.response_threshold
        if self.response_strength_mode == "source_adaptive":
            # Zou et al.'s limited-compromise threshold is
            # beta = (trust + 1 - confidence) / 2.  The source defines it at
            # subgroup level; this numerical adapter aggregates the outgoing
            # edge values for each DM and uses full acceptance inside beta.
            degree = np.diff(self._row_ptr).astype(float)
            confidence = np.divide(
                np.add.reduceat(similarities, self._row_ptr[:-1]),
                np.maximum(degree, 1.0),
            )
            beta_edges = 0.5 * (
                self._trust + 1.0 - confidence[self._rows]
            )
            beta_totals = self._row_totals(
                self._trust, self._row_ptr
            )
            radius_values = np.divide(
                np.add.reduceat(beta_edges * self._trust, self._row_ptr[:-1]),
                beta_totals,
            )
            response_rate = np.ones(current.shape[0], dtype=float)
            radius = np.clip(radius_values, 0.0, 1.0)[:, None]
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="source_dt_limited_compromise_beta=(trust+1-confidence)/2",
            )
        else:
            radius_scale = np.clip(
                (1.0 - levels) / max(1.0 - self.response_threshold, 1e-12),
                0.0,
                1.0,
            )
            response_rate = np.full(current.shape[0], self.response_rate)
            radius = self.compromise_radius * radius_scale[:, None]
            self._record_response_controls(
                response_rate,
                radius[:, 0],
                rule="fixed_surrogate_rate_times_clipped_radius",
            )
        delta = response_rate[:, None] * np.clip(targets - current, -radius, radius)
        updated = np.clip(
            np.where(eligible[:, None], current + delta, current), 0.0, 1.0
        )
        updated = self._mask_update(current, updated)
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=int(self._trust.size),
            terminated=not bool(np.any(changed)),
        )


class _OverlappingCommunityMixin:
    """Reusable source-native community state for overlapping proxies.

    ``crisp_multi_label`` keeps community affiliation as binary set membership.
    This is the only supported representation for the external mechanisms.
    """

    _COMMUNITY_MODES = {"crisp_multi_label"}
    community_detector = "method-specific crisp detector"
    community_update_rule = (
        "recompute from the current preference profile at initialization and "
        "after every feedback response"
    )
    community_preference_input = "full_profile_neutral_0.5_imputation"
    community_refresh_policy = "dynamic_after_each_feedback"

    def __init__(
        self,
        overlap_strength: float,
        response_rate: float,
        compromise_radius: float,
        response_threshold: float,
        role_threshold: float,
        community_refresh: float,
        change_tolerance: float,
        response_strength_mode: str = "source_adaptive",
        community_mode: str = "crisp_multi_label",
    ) -> None:
        if not 0.0 <= overlap_strength <= 1.0:
            raise ValueError("overlap_strength must lie in [0, 1]")
        if not 0.0 < response_rate <= 1.0:
            raise ValueError("response_rate must lie in (0, 1]")
        if not 0.0 < compromise_radius <= 1.0:
            raise ValueError("compromise_radius must lie in (0, 1]")
        if not 0.0 < response_threshold <= 1.0:
            raise ValueError("response_threshold must lie in (0, 1]")
        if not 0.0 <= role_threshold <= 1.0:
            raise ValueError("role_threshold must lie in [0, 1]")
        if not 0.0 < community_refresh <= 1.0:
            raise ValueError("community_refresh must lie in (0, 1]")
        if change_tolerance <= 0.0:
            raise ValueError("change_tolerance must be positive")
        response_strength_mode = _validate_response_strength_mode(
            response_strength_mode
        )
        if community_mode not in self._COMMUNITY_MODES:
            raise ValueError(
                "community_mode must be 'crisp_multi_label'"
            )
        self.overlap_strength = overlap_strength
        self.response_rate = response_rate
        self.compromise_radius = compromise_radius
        self.response_threshold = response_threshold
        self.role_threshold = role_threshold
        self.community_refresh = community_refresh
        self.change_tolerance = change_tolerance
        self.response_strength_mode = response_strength_mode
        self.community_mode = community_mode

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        totals = np.sum(values, axis=1, keepdims=True)
        return values / np.maximum(totals, 1e-12)

    def _detect_communities(self, preferences: np.ndarray) -> np.ndarray:
        """Return method-specific crisp multi-label communities."""
        raise NotImplementedError(
            f"{type(self).__name__} must define _detect_communities"
        )

    def _community_profile(self, preferences: np.ndarray) -> np.ndarray:
        """Return the complete profile used by an external community method."""
        values = np.asarray(preferences, dtype=float)
        if values.shape != self.instance.preferences.shape:
            raise ValueError("preferences have an unexpected shape")
        return values.copy()

    def _set_community_state(self, labels: np.ndarray) -> None:
        sparse_labels = scipy_sparse.issparse(labels)
        if sparse_labels:
            labels_csr = scipy_sparse.csr_matrix(labels, dtype=np.uint8)
            labels_csr.data[:] = 1
            labels_csr.eliminate_zeros()
            labels_csr.sort_indices()
            shape = labels_csr.shape
        else:
            labels = np.asarray(labels, dtype=bool)
            shape = labels.shape
        if len(shape) != 2 or shape[0] != self.instance.n:
            raise ValueError("community detector must return shape (N, K_c)")
        if shape[1] < 1:
            raise ValueError("community detector must return at least one community")
        # Every DM must have at least one crisp label.  Detectors normally
        # guarantee this; the fallback keeps the interface total for filtered
        # supports that isolate a node.
        if sparse_labels:
            empty_rows = np.flatnonzero(
                np.asarray(labels_csr.getnnz(axis=1)).ravel() == 0
            )
            if empty_rows.size:
                extra = scipy_sparse.csr_matrix(
                    (np.ones(empty_rows.size, dtype=np.uint8),
                     (empty_rows, np.arange(empty_rows.size))),
                    shape=(self.instance.n, empty_rows.size),
                )
                labels_csr = scipy_sparse.hstack(
                    (labels_csr, extra), format="csr"
                )
        else:
            empty_rows = ~labels.any(axis=1)
            if np.any(empty_rows):
                extra = np.zeros(
                    (self.instance.n, int(np.count_nonzero(empty_rows))),
                    dtype=bool,
                )
                extra[np.flatnonzero(empty_rows), np.arange(extra.shape[1])] = True
                labels = np.concatenate((labels, extra), axis=1)
        previous = getattr(self, "_community_labels", None)
        if sparse_labels:
            current_csr = labels_csr
        else:
            current_csr = scipy_sparse.csr_matrix(labels.astype(np.uint8))
        if previous is None:
            changed = True
        else:
            previous_csr = scipy_sparse.csr_matrix(previous, dtype=np.uint8)
            changed = bool(
                previous_csr.shape != current_csr.shape
                or (previous_csr != current_csr).nnz
            )
        self._last_community_state_changed = changed
        # The detector and cached edge index are sparse for every instance.
        # Retain a dense compatibility view only for genuinely small matrices
        # because older callers inspect ``_community_labels`` directly.
        self._community_labels = (
            current_csr.toarray().astype(bool)
            if self.instance.n * current_csr.shape[1] <= 10_000_000
            else current_csr.copy()
        )
        self._community_label_csr = current_csr
        self._community_label_csc = current_csr.tocsc()
        self._community_memberships = (
            self._community_labels.astype(float)
            if isinstance(self._community_labels, np.ndarray)
            else current_csr.astype(float)
        )
        self._last_community_count = int(current_csr.shape[1])
        self.implemented_community_representation = "crisp_multi_label"
        self.community_label_source = self.community_detector
        self.community_overlap_rule = "shared_label_indicator"

    @property
    def community_labels(self) -> np.ndarray | scipy_sparse.csr_matrix | None:
        """Return the current source-native crisp overlapping labels."""
        labels = getattr(self, "_community_labels", None)
        if labels is None:
            return None
        # Preserve the historical ndarray-facing API for small callers while
        # keeping runner and mechanism internals sparse for every instance.
        if scipy_sparse.issparse(labels) and labels.shape[0] * labels.shape[1] <= 10_000_000:
            return labels.toarray().astype(bool)
        return labels.copy()

    def _overlap_scores(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        shared = np.sum(
            np.asarray(left, dtype=bool) & np.asarray(right, dtype=bool), axis=-1
        )
        return (shared > 0).astype(float)

    def _community_members(self, community: int) -> np.ndarray:
        """Return member ids using the cached CSC incidence index."""
        labels = self._community_label_csc
        start, end = labels.indptr[community : community + 2]
        return labels.indices[start:end]

    def _dm_communities(self, dm: int) -> np.ndarray:
        """Return community ids for one DM using the cached CSR index."""
        labels = self._community_label_csr
        start, end = labels.indptr[dm : dm + 2]
        return labels.indices[start:end]

    def _local_weights(self, members: np.ndarray, community: int) -> np.ndarray:
        """Read local PageRank weights without densifying a sparse column."""
        values = self._local_pagerank[np.asarray(members), int(community)]
        if scipy_sparse.issparse(values):
            values = values.toarray()
        return np.asarray(values, dtype=float).reshape(-1)

    def _local_weights_for_dm(
        self, dm: int, communities: np.ndarray
    ) -> np.ndarray:
        values = self._local_pagerank[int(dm), np.asarray(communities)]
        if scipy_sparse.issparse(values):
            values = values.toarray()
        return np.asarray(values, dtype=float).reshape(-1)

    def _edge_overlap_scores(
        self, sources: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Return edge-wise shared-community indicators without dense expansion."""
        labels = getattr(self, "_community_label_csr", None)
        if labels is None:
            return self._overlap_scores(
                self._community_labels[np.asarray(sources)],
                self._community_labels[np.asarray(targets)],
            )
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        if sources.shape != targets.shape:
            raise ValueError("edge source and target arrays must have equal shape")
        shared = labels[sources].multiply(labels[targets])
        return (np.asarray(shared.sum(axis=1)).ravel() > 0.0).astype(float)

    def _node_overlap_scores(
        self, sources: np.ndarray, target: int | np.ndarray
    ) -> np.ndarray:
        """Return shared-community indicators for one node against many nodes."""
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        targets = np.full(sources.size, int(target), dtype=np.int64) if np.isscalar(target) else np.asarray(target, dtype=np.int64).reshape(-1)
        return self._edge_overlap_scores(sources, targets)

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._self_confidence = self.backend.to_numpy(
            self._initial_weights.self_weights
        ).copy()
        self.source_community_representation = "crisp_multi_label_set"
        self._set_community_state(self._detect_communities(instance.preferences))
        # Initialization is not a feedback transition and must not count as a
        # community change for the first optional stability check.
        self._last_community_state_changed = False
        self._last_community_overlap = self._overlap_rate()
        self._on_community_state_updated()

    def _overlap_rate(self) -> float:
        return float(
            np.mean(
                np.asarray(self._community_label_csr.sum(axis=1)).ravel()
                > 1
            )
        )

    def _refresh_communities(
        self, preferences: np.ndarray, similarities: np.ndarray
    ) -> None:
        del similarities
        if self.community_refresh_policy == "static_after_initialization":
            self._last_community_state_changed = False
            return
        refreshed = self._detect_communities(np.asarray(preferences, dtype=float))
        self._set_community_state(refreshed)
        self._last_community_overlap = self._overlap_rate()
        self._on_community_state_updated()

    def _on_community_state_updated(self) -> None:
        """Hook for proxies whose influence matrix depends on community state."""


class _OBCFImplementation(_OverlappingCommunityMixin, _FeedbackAdapter):
    """Overlapping-community bounded-confidence feedback adapter."""

    name = "Wang-Overlapping-BC-proxy"
    full_name = "Overlapping-Community Bounded-Confidence Feedback"
    source_id = "Wang2024INSoverlap"
    feedback_object = "overlapping_community_bounded_confidence_elicit"
    native_constraints = (
        "confidence_bound_pagerank_weights_and_two_stage_elicit"
    )
    native_representation = "elicited_linguistic_expected_value_surrogate"
    community_detector = "Wang-LFM(trust-topology-support)"
    community_refresh_policy = "static_after_initialization"
    community_preference_input = "none_topology_trust_only"
    community_update_rule = (
        "initialize from the symmetrized trust topology and retain across rounds"
    )
    feedback_order = (
        "assessment -> deficient_subgroups -> "
        "stage1_maximum_consensus -> stage2_minimum_adjustment -> "
        "overlapping_recommendation_aggregation -> bounded_acceptance"
    )
    network_mode = "source_native_bounded_confidence_common_support"

    def __init__(
        self,
        confidence: float | None = None,
        overlap_strength: float = 0.35,
        response_rate: float = 1.0,
        feedback_radius: float = 0.20,
        response_threshold: float = 0.95,
        community_refresh: float = 0.30,
        change_tolerance: float = 1e-4,
        response_strength_mode: str = "source_adaptive",
        community_mode: str = "crisp_multi_label",
        pagerank_damping: float = 0.85,
        pagerank_tolerance: float = 1e-8,
        pagerank_max_iterations: int = 100,
        global_local_weight: float = 0.50,
        elicitation_grid_size: int = 17,
        lfm_alpha: float = 1.0,
        confidence_bound: float | None = None,
        similarity_support_threshold: float | None = None,
        adaptive_eta: str | None = "dispersion",
        eta_min: float = 0.0,
        eta_max: float = 1.0,
        trust_scores: np.ndarray | None = None,
    ) -> None:
        if confidence is not None and confidence_bound is not None and not np.isclose(
            confidence, confidence_bound
        ):
            raise ValueError(
                "confidence and confidence_bound must agree when both are provided"
            )
        resolved_confidence_bound = (
            confidence_bound
            if confidence_bound is not None
            else (0.50 if confidence is None else confidence)
        )
        super().__init__(
            overlap_strength=overlap_strength,
            response_rate=response_rate,
            compromise_radius=feedback_radius,
            response_threshold=response_threshold,
            role_threshold=resolved_confidence_bound,
            community_refresh=community_refresh,
            change_tolerance=change_tolerance,
            response_strength_mode=response_strength_mode,
            community_mode=community_mode,
        )
        if not 0.0 <= resolved_confidence_bound <= 1.0:
            raise ValueError("confidence_bound must lie in [0, 1]")
        if similarity_support_threshold is not None and not (
            0.0 <= similarity_support_threshold <= 1.0
        ):
            raise ValueError("similarity_support_threshold must lie in [0, 1]")
        if not 0.0 < pagerank_damping < 1.0:
            raise ValueError("pagerank_damping must lie in (0, 1)")
        if pagerank_tolerance <= 0.0 or pagerank_max_iterations <= 0:
            raise ValueError(
                "pagerank_tolerance and pagerank_max_iterations must be positive"
            )
        if not 0.0 <= global_local_weight <= 1.0:
            raise ValueError("global_local_weight must lie in [0, 1]")
        if elicitation_grid_size < 2:
            raise ValueError("elicitation_grid_size must be at least 2")
        if not np.isfinite(lfm_alpha) or lfm_alpha <= 0.0:
            raise ValueError("lfm_alpha must be finite and positive")
        adaptive_eta = _validate_adaptive_eta_mode(adaptive_eta)
        if not 0.0 <= eta_min <= eta_max <= 1.0:
            raise ValueError(
                "eta_min and eta_max must satisfy 0 <= eta_min <= eta_max <= 1"
            )
        if trust_scores is not None:
            trust_scores = np.asarray(trust_scores, dtype=float).reshape(-1)
            if trust_scores.size == 0 or not np.all(np.isfinite(trust_scores)):
                raise ValueError("trust_scores must be a non-empty finite vector")
        self.confidence_bound = float(resolved_confidence_bound)
        # Preserve the old attribute for callers that inspect it directly.
        self.confidence = self.confidence_bound
        self.similarity_support_threshold = (
            None
            if similarity_support_threshold is None
            else float(similarity_support_threshold)
        )
        self.pagerank_damping = float(pagerank_damping)
        self.pagerank_tolerance = float(pagerank_tolerance)
        self.pagerank_max_iterations = int(pagerank_max_iterations)
        self.global_local_weight = float(global_local_weight)
        self.elicitation_grid_size = int(elicitation_grid_size)
        self.lfm_alpha = float(lfm_alpha)
        self.adaptive_eta = adaptive_eta
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.trust_scores = (
            None if trust_scores is None else trust_scores.copy()
        )
        self.eta_signal_source = (
            "common_scalar"
            if adaptive_eta is None
            else (
                "global_pagerank_proxy"
                if adaptive_eta == "trust" and trust_scores is None
                else adaptive_eta
            )
        )
        self._last_eta_summary = {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }
        self._wang_lp_tolerance = 1e-8

    @staticmethod
    def _normalize_eta_signal(signal: np.ndarray) -> np.ndarray:
        values = np.asarray(signal, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(
                "adaptive eta signal must be a non-empty finite vector"
            )
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper - lower <= 1e-12:
            return np.full(values.size, 0.5, dtype=float)
        return (values - lower) / (upper - lower)

    def _adaptive_eta_vector(self, current: np.ndarray) -> np.ndarray:
        """Resolve per-DM bounded confidence for the current feedback round.

        These policies are transparent numerical extensions. Wang et al. treat
        DM-specific confidence levels as external inputs and do not define an
        adaptive formula based on topology, trust, or dispersion.
        """
        n = self.instance.n
        if self.adaptive_eta is None:
            values = np.full(n, self.confidence_bound, dtype=float)
            self.eta_signal_source = "common_scalar"
        elif self.adaptive_eta == "degree":
            degree = np.diff(self._row_ptr).astype(float)
            values = self.eta_min + (
                self.eta_max - self.eta_min
            ) * self._normalize_eta_signal(np.log1p(degree))
            self.eta_signal_source = "log_out_degree"
        elif self.adaptive_eta == "trust":
            if self.trust_scores is None:
                signal = np.asarray(self._global_pagerank, dtype=float)
                self.eta_signal_source = "global_pagerank_proxy"
            else:
                if self.trust_scores.shape != (n,):
                    raise ValueError(
                        "trust_scores must have one value for each DM"
                    )
                signal = self.trust_scores
                self.eta_signal_source = "provided_dm_trust"
            values = self.eta_min + (
                self.eta_max - self.eta_min
            ) * self._normalize_eta_signal(signal)
        elif self.adaptive_eta == "dispersion":
            profile = self._community_profile(current)
            dispersion = np.mean(
                np.abs(profile - np.mean(profile, axis=1, keepdims=True)),
                axis=1,
            )
            values = self.eta_min + (
                self.eta_max - self.eta_min
            ) * self._normalize_eta_signal(dispersion)
            self.eta_signal_source = "within_profile_mean_absolute_dispersion"
        else:
            raise RuntimeError(
                f"unsupported adaptive eta mode: {self.adaptive_eta}"
            )
        values = np.clip(values, 0.0, 1.0)
        self._last_eta_summary = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }
        self._last_confidence_bounds = values.copy()
        return values

    def _detect_communities(self, preferences: np.ndarray) -> np.ndarray:
        del preferences
        # Wang et al. apply LFM to the social trust relationship network.
        # Community membership is therefore initialized from topology and is
        # not recomputed from each subsequently revised preference profile.
        return lfm_overlapping_labels(
            self.instance.adjacency,
            alpha=self.lfm_alpha,
        )

    def _initialize_pagerank_state(self) -> None:
        """Compute the source global and subgroup PageRank weights once."""
        labels = self._community_label_csr
        self._global_pagerank = _pagerank_on_support(
            self.instance.adjacency,
            damping=self.pagerank_damping,
            tolerance=self.pagerank_tolerance,
            max_iterations=self.pagerank_max_iterations,
        )
        self._local_pagerank = _batched_local_pagerank(
            self.instance.adjacency,
            labels,
            damping=self.pagerank_damping,
            tolerance=self.pagerank_tolerance,
            max_iterations=self.pagerank_max_iterations,
        )
        self._last_stage1_maximum_consensus = None
        self._last_stage2_minimum_adjustment = None
        self._last_elicitation_stage = "not_run"

    @staticmethod
    def _weighted_profile_mean(
        preferences: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(preferences, dtype=float)
        coefficients = np.asarray(weights, dtype=float).reshape(-1)
        total = float(coefficients.sum())
        if total <= 0.0:
            return np.mean(values, axis=0)
        return np.sum(coefficients[:, None] * values, axis=0) / total

    def _pagerank_targets(self, preferences: np.ndarray) -> np.ndarray:
        """Combine global and local PageRank-weighted subgroup opinions."""
        current = np.asarray(preferences, dtype=float)
        global_target = self._weighted_profile_mean(
            current, self._global_pagerank
        )
        if scipy_sparse.issparse(self._local_pagerank):
            local_weight = self._local_pagerank
            community_totals = np.asarray(local_weight.sum(axis=0)).ravel()
            local_targets = np.asarray(local_weight.T @ current, dtype=float)
            local_targets = local_targets / np.maximum(
                community_totals[:, None], 1e-12
            )
            local_per_dm = np.asarray(local_weight @ local_targets, dtype=float)
            dm_totals = np.asarray(local_weight.sum(axis=1)).ravel()
            local_per_dm = local_per_dm / np.maximum(dm_totals[:, None], 1e-12)
            missing = dm_totals <= 1e-12
            if np.any(missing):
                local_per_dm[missing] = global_target
            return (
                self.global_local_weight * global_target[None, :]
                + (1.0 - self.global_local_weight) * local_per_dm
            )
        local_targets = []
        for community in range(self._community_labels.shape[1]):
            members = self._community_members(community)
            local_targets.append(
                self._weighted_profile_mean(
                    current[members], self._local_weights(members, community)
                )
            )
        local_targets = np.asarray(local_targets, dtype=float)
        local_per_dm = np.empty_like(current)
        for dm in range(self.instance.n):
            communities = self._dm_communities(dm)
            if communities.size == 0:
                local_per_dm[dm] = global_target
                continue
            weights = self._local_weights_for_dm(dm, communities)
            if float(weights.sum()) <= 1e-12:
                weights = np.ones(communities.size, dtype=float)
            local_per_dm[dm] = self._weighted_profile_mean(
                local_targets[communities], weights
            )
        return (
            self.global_local_weight * global_target[None, :]
            + (1.0 - self.global_local_weight) * local_per_dm
        )

    def _two_stage_elicitation(
        self, preferences: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Numerically realize ELICIT stage 1 and stage 2 on each subgroup."""
        current = np.asarray(preferences, dtype=float)
        global_target = self._weighted_profile_mean(
            current, self._global_pagerank
        )
        candidate_sum = np.zeros_like(current)
        candidate_weight = np.zeros(self.instance.n, dtype=float)
        stage1_scores = []
        stage2_coefficients = []
        grid = np.linspace(0.0, 1.0, self.elicitation_grid_size)
        for community in range(self._community_labels.shape[1]):
            members = self._community_members(community)
            if members.size == 0:
                continue
            local_weights = self._local_weights(members, community)
            if float(local_weights.sum()) <= 1e-12:
                local_weights = np.ones(members.size, dtype=float)
            bounded_gap = np.clip(
                targets[members] - current[members],
                -self.confidence_bound,
                self.confidence_bound,
            )
            scores = np.empty(grid.size, dtype=float)
            proposals = []
            for position, coefficient in enumerate(grid):
                proposal = np.clip(
                    current[members] + coefficient * bounded_gap,
                    0.0,
                    1.0,
                )
                proposals.append(proposal)
                disagreement = np.mean(
                    np.abs(proposal - global_target[None, :]), axis=1
                )
                scores[position] = 1.0 - float(
                    np.sum(local_weights * disagreement)
                    / max(float(local_weights.sum()), 1e-12)
                )
            best_position = int(np.argmax(scores))
            feasible = np.flatnonzero(
                scores >= self.response_threshold - 1e-12
            )
            selected_position = (
                int(feasible[0]) if feasible.size else best_position
            )
            chosen = proposals[selected_position]
            candidate_sum[members] += local_weights[:, None] * chosen
            candidate_weight[members] += local_weights
            stage1_scores.append(float(scores[best_position]))
            stage2_coefficients.append(float(grid[selected_position]))
        updated = np.divide(
            candidate_sum,
            np.maximum(candidate_weight[:, None], 1e-12),
            out=current.copy(),
            where=candidate_weight[:, None] > 1e-12,
        )
        self._last_stage1_maximum_consensus = (
            float(max(stage1_scores)) if stage1_scores else None
        )
        self._last_stage2_minimum_adjustment = (
            float(np.mean(stage2_coefficients))
            if stage2_coefficients
            else None
        )
        return updated

    def _active_scores(
        self, preferences: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        similarities = np.clip(1.0 - self._edge_distances(preferences), 0.0, 1.0)
        if self.similarity_support_threshold is None:
            active = np.ones_like(similarities, dtype=bool)
        else:
            active = similarities >= self.similarity_support_threshold
        overlap = self._edge_overlap_scores(self._rows, self._cols)
        return similarities, active, overlap * active

    def _wang_subgroup_consensus(
        self, preferences: np.ndarray, collective: np.ndarray
    ) -> np.ndarray:
        """Compute Wang's current subgroup consensus levels."""
        current = np.asarray(preferences, dtype=float)
        levels = np.ones(self._community_labels.shape[1], dtype=float)
        dimension = current.shape[1]
        for community in range(self._community_labels.shape[1]):
            members = self._community_members(community)
            if members.size == 0:
                continue
            weights = self._local_weights(members, community)
            weights = weights / max(float(weights.sum()), 1e-12)
            disagreement = float(
                np.sum(
                    weights[:, None]
                    * np.abs(current[members] - collective[None, :])
                )
                / dimension
            )
            levels[community] = 1.0 - disagreement
        return levels

    def _wang_two_stage_feedback(
        self, current: np.ndarray
    ) -> FeedbackDiagnostics:
        """Apply Wang et al.'s two-stage recommendation logic.

        Each deficient subgroup is solved independently against the current
        global collective opinion.  Recommendations for an overlapping DM
        are combined using its local PageRank weights and then screened once
        using that DM's bounded-confidence distance.
        """
        # The generic source LP is retained for small instances.  On a large
        # sparse graph, thousands of independent subgroup LPs dominate the
        # run time; use the equivalent bounded direction and scalar line
        # search below to preserve the two-stage acceptance semantics without
        # constructing one sparse LP matrix per subgroup.
        if current.shape[0] * current.shape[1] > 100_000:
            return self._wang_two_stage_feedback_fast(current)

        labels = self._community_label_csr
        n, dimension = current.shape
        collective = self._weighted_profile_mean(current, self._global_pagerank)
        subgroup_levels = self._wang_subgroup_consensus(current, collective)
        deficient = np.flatnonzero(
            subgroup_levels < self.response_threshold - self._wang_lp_tolerance
        )
        candidate_sum = np.zeros_like(current)
        candidate_weight = np.zeros(n, dtype=float)
        stage1_values: list[float] = []
        stage2_values: list[float] = []
        stage2_modes: list[str] = []
        lp_statuses: list[int] = []
        lp_fallbacks = 0

        confidence_for_dm = self._adaptive_eta_vector(current)
        for community in deficient:
            members = self._community_members(int(community))
            local_weights = self._local_weights(members, community)
            local_weights = local_weights / max(float(local_weights.sum()), 1e-12)
            stage1, stage1_cl, _, stage1_ok, stage1_status = (
                _solve_wang_subgroup_lp(
                    current,
                    members,
                    collective,
                    local_weights,
                    confidence_for_dm[members],
                    stage="maximum_consensus",
                    lp_tolerance=self._wang_lp_tolerance,
                )
            )
            stage1_values.append(float(stage1_cl))
            lp_statuses.append(stage1_status)
            if not stage1_ok:
                lp_fallbacks += 1

            if stage1_cl >= self.response_threshold - self._wang_lp_tolerance:
                stage2_name = "minimum_adjustment_attainable"
                target_disagreement = 1.0 - self.response_threshold
            else:
                stage2_name = "minimum_adjustment_unattainable"
                target_disagreement = 1.0 - stage1_cl
            recommendation, _, stage2_adjustment, stage2_ok, stage2_status = (
                _solve_wang_subgroup_lp(
                    current,
                    members,
                    collective,
                    local_weights,
                    confidence_for_dm[members],
                    stage=stage2_name,
                    target_disagreement=target_disagreement,
                    lp_tolerance=self._wang_lp_tolerance,
                )
            )
            stage2_values.append(float(stage2_adjustment))
            stage2_modes.append(stage2_name)
            lp_statuses.append(stage2_status)
            if not stage2_ok:
                lp_fallbacks += 1
            candidate_sum[members] += local_weights[:, None] * recommendation
            candidate_weight[members] += local_weights

        candidate_mask = candidate_weight > 1e-12
        candidate = np.divide(
            candidate_sum,
            np.maximum(candidate_weight[:, None], 1e-12),
            out=current.copy(),
            where=candidate_mask[:, None],
        )
        recommendation_distance = np.mean(np.abs(candidate - current), axis=1)
        accepted = candidate_mask & (
            recommendation_distance <= confidence_for_dm + self._wang_lp_tolerance
        )
        updated = np.where(accepted[:, None], candidate, current)
        updated = self._mask_update(current, updated)
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance

        self._last_targeted_dms = np.flatnonzero(candidate_mask).astype(np.int64)
        self._last_active_edges = int(self.instance.adjacency.nnz)
        # Wang's CRP has no post-feedback opinion dynamics.  Keep the
        # topology-derived influence state available to the common runner,
        # but do not use similarity support to gate the recommendation.
        self._weights = self._initial_weights.clone()
        self._last_stage1_maximum_consensus = (
            float(max(stage1_values)) if stage1_values else None
        )
        self._last_stage2_minimum_adjustment = (
            float(np.mean(stage2_values)) if stage2_values else None
        )
        self._last_wang_subgroup_consensus = subgroup_levels
        self._last_wang_deficient_subgroups = deficient.astype(np.int64)
        self._last_wang_stage2_modes = stage2_modes
        self._last_wang_lp_statuses = lp_statuses
        self._last_wang_lp_fallbacks = int(lp_fallbacks)
        self._last_elicitation_stage = "stage1_then_stage2"
        accepted_candidates = accepted[candidate_mask]
        self._record_response_controls(
            np.ones(n, dtype=float),
            None,
            rule="source_wang_two_stage_lp_bounded_acceptance",
            accepted=accepted_candidates,
        )
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=self._last_active_edges,
            terminated=not bool(np.any(changed)),
        )

    def _wang_two_stage_feedback_fast(
        self, current: np.ndarray
    ) -> FeedbackDiagnostics:
        """Scalable numerical realization of Wang's two-stage response.

        Each subgroup first moves toward the global collective opinion under
        its DM-specific L1 confidence budget.  If the response is attainable,
        a short bisection finds the smallest common interpolation coefficient
        reaching the subgroup threshold.  The final per-DM distance screen,
        overlap aggregation, and diagnostics are identical to the LP path.
        """
        n, dimension = current.shape
        collective = self._weighted_profile_mean(current, self._global_pagerank)
        subgroup_levels = self._wang_subgroup_consensus(current, collective)
        deficient = np.flatnonzero(
            subgroup_levels < self.response_threshold - self._wang_lp_tolerance
        )
        confidence_for_dm = self._adaptive_eta_vector(current)
        candidate_sum = np.zeros_like(current)
        candidate_weight = np.zeros(n, dtype=float)
        stage1_values: list[float] = []
        stage2_values: list[float] = []
        stage2_modes: list[str] = []

        for community in deficient:
            members = self._community_members(int(community))
            if members.size == 0:
                continue
            local_weights = self._local_weights(members, int(community))
            local_weights = local_weights / max(float(local_weights.sum()), 1e-12)
            gap = collective[None, :] - current[members]
            gap_l1 = np.sum(np.abs(gap), axis=1)
            budget = confidence_for_dm[members] * dimension
            scale = np.minimum(
                1.0,
                np.divide(
                    budget,
                    np.maximum(gap_l1, 1e-12),
                ),
            )
            stage1 = np.clip(current[members] + scale[:, None] * gap, 0.0, 1.0)
            stage1_disagreement = float(
                np.sum(
                    local_weights[:, None]
                    * np.abs(stage1 - collective[None, :])
                )
                / dimension
            )
            stage1_cl = 1.0 - stage1_disagreement
            stage1_values.append(stage1_cl)

            if stage1_cl >= self.response_threshold - self._wang_lp_tolerance:
                target_disagreement = 1.0 - self.response_threshold
                low, high = 0.0, 1.0
                direction = stage1 - current[members]
                for _ in range(24):
                    middle = 0.5 * (low + high)
                    proposal = current[members] + middle * direction
                    disagreement = float(
                        np.sum(
                            local_weights[:, None]
                            * np.abs(proposal - collective[None, :])
                        )
                        / dimension
                    )
                    if disagreement <= target_disagreement:
                        high = middle
                    else:
                        low = middle
                recommendation = np.clip(
                    current[members] + high * direction, 0.0, 1.0
                )
                stage2_modes.append("minimum_adjustment_attainable")
                stage2_values.append(
                    float(
                        np.sum(
                            local_weights[:, None]
                            * np.abs(recommendation - current[members])
                        )
                        / dimension
                    )
                )
            else:
                recommendation = stage1
                stage2_modes.append("minimum_adjustment_unattainable")
                stage2_values.append(
                    float(
                        np.sum(
                            local_weights[:, None]
                            * np.abs(recommendation - current[members])
                        )
                        / dimension
                    )
                )
            candidate_sum[members] += local_weights[:, None] * recommendation
            candidate_weight[members] += local_weights

        candidate_mask = candidate_weight > 1e-12
        candidate = np.divide(
            candidate_sum,
            np.maximum(candidate_weight[:, None], 1e-12),
            out=current.copy(),
            where=candidate_mask[:, None],
        )
        recommendation_distance = np.mean(np.abs(candidate - current), axis=1)
        accepted = candidate_mask & (
            recommendation_distance <= confidence_for_dm + self._wang_lp_tolerance
        )
        updated = self._mask_update(
            current, np.where(accepted[:, None], candidate, current)
        )
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance

        self._last_targeted_dms = np.flatnonzero(candidate_mask).astype(np.int64)
        self._last_active_edges = int(self.instance.adjacency.nnz)
        self._weights = self._initial_weights.clone()
        self._last_stage1_maximum_consensus = (
            float(max(stage1_values)) if stage1_values else None
        )
        self._last_stage2_minimum_adjustment = (
            float(np.mean(stage2_values)) if stage2_values else None
        )
        self._last_wang_subgroup_consensus = subgroup_levels
        self._last_wang_deficient_subgroups = deficient.astype(np.int64)
        self._last_wang_stage2_modes = stage2_modes
        self._last_wang_lp_statuses = []
        self._last_wang_lp_fallbacks = 0
        self._last_elicitation_stage = "stage1_then_stage2_vectorized"
        self._record_response_controls(
            np.ones(n, dtype=float),
            None,
            rule="source_wang_two_stage_bounded_projection",
            accepted=accepted[candidate_mask],
        )
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=self._last_active_edges,
            terminated=not bool(np.any(changed)),
        )

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._initialize_pagerank_state()
        self._last_wang_subgroup_consensus = np.empty(
            self._community_labels.shape[1], dtype=float
        )
        self._last_wang_deficient_subgroups = np.empty(0, dtype=np.int64)
        self._last_wang_stage2_modes: list[str] = []
        self._last_wang_lp_statuses: list[int] = []
        self._last_wang_lp_fallbacks = 0
        if self.response_strength_mode == "source_adaptive":
            self._last_active_edges = int(instance.adjacency.nnz)
            self._weights = self._initial_weights.clone()
        else:
            similarities, active, scores = self._active_scores(instance.preferences)
            del similarities
            self._last_active_edges = int(np.count_nonzero(active))
            self._weights = self._matrix_from_edge_scores(scores)

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        if self.response_strength_mode == "source_adaptive":
            return self._wang_two_stage_feedback(current)
        levels = self._consensus_levels(current)
        similarities, active, scores = self._active_scores(current)
        self._last_active_edges = int(np.count_nonzero(active))
        # PageRank supplies the source global/local influence weights; the
        # bounded-confidence edge support remains the fallback for isolated
        # rows in the numerical adapter.
        pagerank_targets = self._pagerank_targets(current)
        confidence_targets = self._row_normalized_targets(current, scores)
        fallback = self._row_normalized_targets(current, active.astype(float))
        row_has_active = self._row_totals(scores, self._row_ptr) > 1e-12
        confidence_targets = np.where(
            row_has_active[:, None], confidence_targets, fallback
        )
        confidence_targets = np.where(
            row_has_active[:, None], confidence_targets, current
        )
        targets = np.where(
            np.isfinite(pagerank_targets), pagerank_targets, confidence_targets
        )
        eligible = levels < self.response_threshold
        if self.response_strength_mode == "source_adaptive":
            # Stage 1 maximizes each subgroup's attainable consensus under the
            # confidence bound; stage 2 selects the smallest grid coefficient
            # reaching the threshold, or the stage-1 maximizer if infeasible.
            elicited = self._two_stage_elicitation(current, targets)
            self._last_elicitation_stage = "stage1_then_stage2"
            response_rate = np.ones(current.shape[0], dtype=float)
            radius_values = np.ones(current.shape[0], dtype=float)
            accepted = np.max(np.abs(elicited - current), axis=1) > 1e-12
            eligible = eligible & accepted
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="source_pagerank_two_stage_elicit",
                accepted=accepted,
            )
            targets = elicited
        else:
            self._last_elicitation_stage = "fixed_surrogate_bypass"
            radius_scale = np.clip(
                (1.0 - levels) / max(1.0 - self.response_threshold, 1e-12),
                0.0,
                1.0,
            )
            radius_values = self.compromise_radius * radius_scale
            response_rate = np.full(current.shape[0], self.response_rate)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="fixed_surrogate_rate_times_clipped_radius",
            )
        delta = response_rate[:, None] * np.clip(
            targets - current,
            -radius_values[:, None],
            radius_values[:, None],
        )
        updated = np.clip(
            np.where(eligible[:, None], current + delta, current), 0.0, 1.0
        )
        updated = self._mask_update(current, updated)
        self._refresh_communities(updated, similarities)
        _, active_after, scores_after = self._active_scores(updated)
        self._last_active_edges = int(np.count_nonzero(active_after))
        self._weights = self._matrix_from_edge_scores(scores_after)
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=self._last_active_edges,
            terminated=not bool(np.any(changed)),
        )


class _OCCFImplementation(_OverlappingCommunityMixin, _FeedbackAdapter):
    """Bidirectional overlapping-community compromise adapter."""

    name = "Gai-Overlapping-Compromise-proxy"
    full_name = "Overlapping-Community Compromise Feedback"
    source_id = "Gai2024TSMCS"
    feedback_object = "overlapping_community_bidirectional_compromise"
    native_constraints = "prospect_loss_aversion_and_dynamic_compromise_radius"
    native_representation = "numerical_prospect_value_adapter"
    community_detector = "Gai-LFM(similarity-thresholded-trust-network)"
    community_preference_input = "full_profile_neutral_0.5_imputation"
    community_update_rule = (
        "refresh after the within-community opinion step for the next assessment"
    )
    feedback_order = "bidirectional_feedback_then_within_community_evolution"
    opinion_evolution_rule = "one_step_within_community_only"
    exclude_overlapping_from_opinion_dynamics = True
    network_mode = "source_native_bidirectional_community_common_support"
    native_opinion_dynamics = True
    feedback_timing = (
        "assessment -> direct_bidirectional_feedback -> "
        "one_community_opinion_dynamics_step -> next_assessment"
    )

    def __init__(
        self,
        overlap_strength: float = 0.35,
        response_rate: float = 1.0,
        compromise_radius: float = 0.20,
        response_threshold: float = 0.95,
        role_threshold: float = 0.50,
        similarity_threshold: float = 0.70,
        lfm_alpha: float = 1.0,
        loss_aversion: float = 2.25,
        bidirectional_weight: float = 0.50,
        community_refresh: float = 0.30,
        change_tolerance: float = 1e-4,
        response_strength_mode: str = "source_adaptive",
        community_mode: str = "crisp_multi_label",
    ) -> None:
        super().__init__(
            overlap_strength=overlap_strength,
            response_rate=response_rate,
            compromise_radius=compromise_radius,
            response_threshold=response_threshold,
            role_threshold=role_threshold,
            community_refresh=community_refresh,
            change_tolerance=change_tolerance,
            response_strength_mode=response_strength_mode,
            community_mode=community_mode,
        )
        if loss_aversion < 1.0:
            raise ValueError("loss_aversion must be at least one")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must lie in [0, 1]")
        if not np.isfinite(lfm_alpha) or lfm_alpha <= 0.0:
            raise ValueError("lfm_alpha must be finite and positive")
        if not 0.0 <= bidirectional_weight <= 1.0:
            raise ValueError("bidirectional_weight must lie in [0, 1]")
        self.loss_aversion = loss_aversion
        self.similarity_threshold = similarity_threshold
        self.lfm_alpha = float(lfm_alpha)
        self.bidirectional_weight = bidirectional_weight

    def _source_adaptive_willingness(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        collective = collective_preference(
            current, self.instance.evolution_mask
        )
        gap = target - current
        mask = self._evolution_mask_np
        denominators = np.maximum(mask.sum(axis=1), 1.0)
        grid = np.linspace(0.0, 1.0, 9)
        best = np.zeros(current.shape[0], dtype=float)
        best_score = np.full(current.shape[0], -np.inf, dtype=float)
        for coefficient in grid:
            candidate = self._mask_update(
                current,
                np.clip(current + coefficient * gap, 0.0, 1.0),
            )
            score = 1.0 - np.sum(
                mask * np.abs(candidate - collective), axis=1
            ) / denominators
            improved = score > best_score + 1e-12
            best[improved] = coefficient
            best_score[improved] = score[improved]
        return best

    def _detect_communities(self, preferences: np.ndarray) -> np.ndarray:
        profile = self._community_profile(preferences)
        support, similarity = symmetric_preference_support(
            self.instance.adjacency,
            profile,
            threshold=self.similarity_threshold,
        )
        return lfm_overlapping_labels(
            support,
            alpha=self.lfm_alpha,
            edge_weights=similarity,
        )

    def _on_community_state_updated(self) -> None:
        overlap = self._edge_overlap_scores(self._rows, self._cols)
        self._weights = self._matrix_from_edge_scores(overlap)

    def _incoming_targets(
        self, preferences: np.ndarray, similarities: np.ndarray | None = None
    ) -> np.ndarray:
        if np.all(self._evolution_mask_np > 0.5):
            values = np.asarray(preferences, dtype=float)
            overlap_edges = self._edge_overlap_scores(
                self._rows, self._cols
            )
            incoming_overlap = overlap_edges[self._in_edge_positions]
            incoming_rows = np.asarray(self._in_indices, dtype=np.int64)
            incoming_targets = np.repeat(
                np.arange(self.instance.n, dtype=np.int64), self._in_degree
            )
            if similarities is None:
                similarities = np.clip(
                    1.0 - self._edge_distances(values), 0.0, 1.0
                )
            similarities = np.asarray(similarities, dtype=float)[
                self._in_edge_positions
            ]
            scores = incoming_overlap * (0.5 + 0.5 * similarities)
            matrix = scipy_sparse.csr_matrix(
                (scores, (incoming_targets, incoming_rows)),
                shape=(self.instance.n, self.instance.n),
            )
            matrix.sum_duplicates()
            totals = np.asarray(matrix.sum(axis=1)).ravel()[:, None]
            return np.divide(
                np.asarray(matrix @ values, dtype=float),
                totals,
                out=values.copy(),
                where=totals > 1e-12,
            )
        targets = np.array(preferences, copy=True)
        for i in range(self.instance.n):
            neighbors = self.instance.in_neighbors(i)
            if neighbors.size == 0:
                continue
            overlap = self._node_overlap_scores(neighbors, i)
            shared = (
                self._evolution_mask_np[neighbors]
                * self._evolution_mask_np[i]
            )
            counts = shared.sum(axis=1)
            numerators = np.sum(
                shared * np.abs(preferences[neighbors] - preferences[i]), axis=1
            )
            similarity = np.clip(
                1.0
                - np.divide(
                    numerators,
                    counts,
                    out=np.ones_like(numerators),
                    where=counts > 0.0,
                ),
                0.0,
                1.0,
            )
            scores = overlap * (0.5 + 0.5 * similarity)
            total = float(scores.sum())
            if total > 1e-12:
                targets[i] = (scores[:, None] * preferences[neighbors]).sum(axis=0) / total
        return targets

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        levels = self._consensus_levels(current)
        similarities = np.clip(1.0 - self._edge_distances(current), 0.0, 1.0)
        overlap = self._edge_overlap_scores(self._rows, self._cols)
        forward = self._row_normalized_targets(
            current, overlap * (0.5 + 0.5 * similarities)
        )
        backward = self._incoming_targets(current, similarities)
        target = (
            (1.0 - self.bidirectional_weight) * forward
            + self.bidirectional_weight * backward
        )
        direction = target - current
        loss_factor = np.where(direction < 0.0, self.loss_aversion, 1.0)
        eligible = levels < self.response_threshold
        if self.response_strength_mode == "source_adaptive":
            response_rate = self._source_adaptive_willingness(current, target)
            loss_penalty = np.sum(
                self._evolution_mask_np * loss_factor, axis=1
            ) / np.maximum(self._evolution_mask_np.sum(axis=1), 1.0)
            response_rate = np.clip(response_rate / loss_penalty, 0.0, 1.0)
            radius_values = np.ones(current.shape[0], dtype=float)
            radius_matrix = radius_values[:, None]
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="source_bidirectional_prospect_willingness_line_search",
            )
        else:
            radius_scale = np.clip(
                (1.0 - levels) / max(1.0 - self.response_threshold, 1e-12),
                0.0,
                1.0,
            )
            radius_matrix = (
                self.compromise_radius * radius_scale[:, None] / loss_factor
            )
            radius_values = np.sum(
                self._evolution_mask_np * radius_matrix, axis=1
            ) / np.maximum(self._evolution_mask_np.sum(axis=1), 1.0)
            response_rate = np.full(current.shape[0], self.response_rate)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="fixed_surrogate_rate_times_clipped_radius",
            )
        delta = response_rate[:, None] * np.clip(
            direction,
            -radius_matrix,
            radius_matrix,
        )
        updated = np.clip(
            np.where(eligible[:, None], current + delta, current), 0.0, 1.0
        )
        updated = self._mask_update(current, updated)
        # Keep the source community partition fixed during this feedback and
        # its following within-community opinion step.  The partition is
        # refreshed by ``evolve_after_feedback`` after that step, so the next
        # assessment sees communities generated from the new profile.
        self._weights = self._matrix_from_edge_scores(overlap)
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=int(self._rows.size),
            terminated=not bool(np.any(changed)),
        )


class _NGPFImplementation(_FeedbackAdapter):
    """Scalar-action network-game and incentive adapter.

    Each DM chooses one scalar adjustment magnitude under a per-round budget.
    That scalar action is decoded into a vector displacement in the direction
    of the peer target; no membership or vector QP is used.
    """

    name = "Lang-Network-Game-proxy"
    full_name = "Network-Game Peer-Effect Feedback"
    source_id = "Lang2026EJOR"
    feedback_object = "scalar_network_game_peer_effect_incentive"
    native_constraints = "per_dm_scalar_budget_and_incentive_cap"
    native_representation = "scalar_action_to_preference_vector_adapter"
    # The published Lang et al. framework also studies an optional
    # interaction-link intervention problem. This proxy implements only the
    # scalar action/peer-effect component on the supplied support A0.
    network_mode = "source_native_interaction_fixed_support"
    topology_mode = "fixed_candidate_support_similarity_reweighted"
    feedback_order = "scalar_network_game_action_then_next_assessment"

    def __init__(
        self,
        peer_effect: float = 0.35,
        incentive_rate: float = 0.35,
        response_rate: float = 0.60,
        action_budget: float = 0.12,
        response_threshold: float = 0.95,
        change_tolerance: float = 1e-4,
    ) -> None:
        if not 0.0 <= peer_effect <= 1.0:
            raise ValueError("peer_effect must lie in [0, 1]")
        if not 0.0 < incentive_rate <= 1.0:
            raise ValueError("incentive_rate must lie in (0, 1]")
        if not 0.0 < response_rate <= 1.0:
            raise ValueError("response_rate must lie in (0, 1]")
        if not 0.0 < action_budget <= 1.0:
            raise ValueError("action_budget must lie in (0, 1]")
        if not 0.0 < response_threshold <= 1.0:
            raise ValueError("response_threshold must lie in (0, 1]")
        if change_tolerance <= 0.0:
            raise ValueError("change_tolerance must be positive")
        self.peer_effect = peer_effect
        self.incentive_rate = incentive_rate
        self.response_rate = response_rate
        self.action_budget = action_budget
        self.response_threshold = response_threshold
        self.change_tolerance = change_tolerance

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._self_confidence = self.backend.to_numpy(
            self._initial_weights.self_weights
        ).copy()
        self._base_interaction = self.backend.to_numpy(
            self._initial_weights.edge_weights
        ).copy()
        self._actions = np.zeros(instance.n, dtype=float)
        self._incentives = np.zeros(instance.n, dtype=float)
        self._last_action_total = 0.0
        self._last_incentive_mean = 0.0

    @property
    def adapter_metadata(self) -> dict[str, object]:
        metadata = super().adapter_metadata
        metadata.update(
            {
                "scalar_action_total": self._last_action_total,
                "incentive_mean": self._last_incentive_mean,
            }
        )
        return metadata

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        levels = self._consensus_levels(current)
        similarities = np.clip(1.0 - self._edge_distances(current), 0.0, 1.0)
        interaction = self._base_interaction * (0.5 + 0.5 * similarities)
        self._weights = self._matrix_from_edge_scores(interaction)
        neighbor_targets = self._row_normalized_targets(current, interaction)
        global_target = collective_preference(
            current, self.instance.evolution_mask
        )
        target = (
            (1.0 - self.peer_effect) * neighbor_targets
            + self.peer_effect * global_target
        )
        gap = target - current
        scalar_gap = np.sum(
            self._evolution_mask_np * np.abs(gap), axis=1
        ) / self._evolution_mask_np.sum(axis=1)
        self._incentives = np.clip(
            (1.0 - self.incentive_rate) * self._incentives
            + self.incentive_rate * np.clip(scalar_gap / self.action_budget, 0.0, 1.0),
            0.0,
            1.0,
        )
        cap = self.action_budget * (1.0 + self._incentives)
        actions = self.response_rate * np.minimum(scalar_gap, cap)
        eligible = levels < self.response_threshold
        actions = np.where(eligible, actions, 0.0)
        scales = actions / np.maximum(scalar_gap, 1e-12)
        updated = self._mask_update(
            current, np.clip(current + scales[:, None] * gap, 0.0, 1.0)
        )
        self._actions = actions
        self._last_action_total = float(np.sum(actions))
        self._last_incentive_mean = float(np.mean(self._incentives))
        self._last_trust_variation = float(np.mean(np.abs(interaction - self._base_interaction)))
        self._last_trust_max_variation = float(
            np.max(np.abs(interaction - self._base_interaction), initial=0.0)
        )
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=int(np.count_nonzero(interaction > 0.0)),
            terminated=not bool(np.any(changed)),
        )


class _DTRFImplementation(_FeedbackAdapter):
    """Interactive dynamic-trust recommendation-feedback adapter.

    The mechanism keeps a primary edge-trust state and a secondary trust signal
    induced by each recipient's observed modification behavior. Each feedback
    phase first identifies a low-contribution target and gives it a trust-lead
    recommendation. Only after the response is obtained is secondary trust
    updated and combined with primary trust for the next assessment.
    """

    name = "Guo-Trust-Feedback-proxy"
    full_name = "Dynamic Trust Recommendation Feedback"
    source_id = "Guo2024ESWA"
    feedback_object = "trust_lead_recommendation_feedback"
    native_constraints = (
        "targeted_dm_trust_lead_or_efficiency_recommendation_and_response_threshold"
    )
    native_representation = "numerical_secondary_trust_adapter"
    network_mode = "source_native_dynamic_trust_common_support"
    feedback_order = (
        "assessment -> targeted_dm_detection -> trust_lead_recommendation -> "
        "direct_feedback -> behavior_secondary_trust_update -> "
        "comprehensive_trust_for_next_assessment"
    )
    trust_update_order = "after_direct_feedback_from_modification_behavior"
    primary_trust_update_rule = (
        "carry_forward_then_average_with_secondary; trust_rate_compatibility_only"
    )
    all_eligible_per_round = False
    target_selection_policy = "single_lowest_contribution_per_feedback_round"
    default_recommendation_mode = "trust_lead"

    def __init__(
        self,
        trust_rate: float = 0.30,
        secondary_trust_rate: float = 0.40,
        response_rate: float = 1.0,
        willingness_radius: float = 0.50,
        response_threshold: float = 0.95,
        change_tolerance: float = 1e-4,
        response_strength_mode: str = "source_adaptive",
        recommendation_mode: str | None = None,
    ) -> None:
        if not 0.0 < trust_rate <= 1.0:
            raise ValueError("trust_rate must lie in (0, 1]")
        if not 0.0 < secondary_trust_rate <= 1.0:
            raise ValueError("secondary_trust_rate must lie in (0, 1]")
        if not 0.0 < response_rate <= 1.0:
            raise ValueError("response_rate must lie in (0, 1]")
        if not 0.0 < willingness_radius <= 1.0:
            raise ValueError("willingness_radius must lie in (0, 1]")
        if not 0.0 < response_threshold <= 1.0:
            raise ValueError("response_threshold must lie in (0, 1]")
        if change_tolerance <= 0.0:
            raise ValueError("change_tolerance must be positive")
        response_strength_mode = self._validate_response_strength_mode(
            response_strength_mode
        )
        if recommendation_mode is None:
            recommendation_mode = self.default_recommendation_mode
        if recommendation_mode not in {"trust_lead", "efficiency_driven"}:
            raise ValueError(
                "recommendation_mode must be 'trust_lead' or 'efficiency_driven'"
            )
        self.trust_rate = trust_rate
        self.secondary_trust_rate = secondary_trust_rate
        self.response_rate = response_rate
        self.willingness_radius = willingness_radius
        self.response_threshold = response_threshold
        self.change_tolerance = change_tolerance
        self.response_strength_mode = response_strength_mode
        self.recommendation_mode = recommendation_mode

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._self_confidence = self.backend.to_numpy(
            self._initial_weights.self_weights
        ).copy()
        # Neutral topology-only starting trust for DTRF.
        self._trust = np.ones(self._rows.size, dtype=float)
        self._secondary_trust = self._trust.copy()
        self._recipient_behavior = np.zeros(instance.n, dtype=float)
        self._weights = self._matrix_from_trust(self._trust)
        self._last_targeted_dms = np.empty(0, dtype=np.int64)
        self._last_recommendation_mode = self.recommendation_mode
        self._last_secondary_trust_source = "relative_modification_behavior"

    def _decision_weights(self, trust: np.ndarray) -> np.ndarray:
        """Return normalized receiver importance induced by an edge trust state."""
        matrix = self._matrix_from_trust(trust)
        edge_weights = self.backend.to_numpy(matrix.edge_weights)
        self_weights = self.backend.to_numpy(matrix.self_weights)
        values = self_weights + np.bincount(
            self._cols, weights=edge_weights, minlength=self.instance.n
        )
        total = float(values.sum())
        return (
            values / total
            if total > 0.0
            else np.full(self.instance.n, 1.0 / self.instance.n)
        )

    def _targeted_dm_mask(
        self, levels: np.ndarray, comprehensive_trust: np.ndarray
    ) -> np.ndarray:
        """Select the response set while preserving the per-DM rule."""
        eligible = levels < self.response_threshold
        selected = np.zeros(self.instance.n, dtype=bool)
        if not np.any(eligible):
            self._last_targeted_dms = np.empty(0, dtype=np.int64)
            return selected
        if self.all_eligible_per_round:
            selected = eligible.copy()
            self._last_targeted_dms = np.flatnonzero(selected).astype(np.int64)
            return selected
        contribution = self._decision_weights(comprehensive_trust) * levels
        candidates = np.flatnonzero(eligible)
        order = np.lexsort((candidates, contribution[candidates]))
        target = int(candidates[order[0]])
        selected[target] = True
        self._last_targeted_dms = np.asarray([target], dtype=np.int64)
        return selected

    def _select_recommendation_peer(
        self,
        target: int,
        preferences: np.ndarray,
        comprehensive_trust: np.ndarray,
        collective: np.ndarray | None = None,
    ) -> int | None:
        """Select the trust-lead or efficiency-driven peer for one target."""
        start, end = self._row_ptr[target : target + 2]
        if start == end:
            return None
        candidates = self._cols[start:end]
        edge_trust = comprehensive_trust[start:end]
        if self.recommendation_mode == "trust_lead":
            return int(candidates[int(np.argmax(edge_trust))])
        if collective is None:
            collective = collective_preference(
                preferences, self.instance.evolution_mask
            )
        distances = np.mean(
            np.abs(preferences[candidates] - collective[None, :]), axis=1
        )
        # Maximize consensus efficiency first and use trust as a deterministic
        # tie-break, matching the source ER scenario's lexicographic intent.
        order = np.lexsort((-edge_trust, distances))
        return int(candidates[int(order[0])])

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        levels = self._consensus_levels(current)
        # DTNU/TERF uses the trust state available at the assessment. Primary
        # trust is carried forward; the previous secondary signal is combined
        # with it only to form the current recommendation state.
        previous_secondary = self._secondary_trust.copy()
        comprehensive_trust = 0.5 * (self._trust + previous_secondary)
        recommendation_scores = np.maximum(comprehensive_trust, 1e-6)
        targets = self._row_normalized_targets(current, recommendation_scores)
        targeted = self._targeted_dm_mask(levels, comprehensive_trust)
        collective = None
        if self.recommendation_mode == "efficiency_driven":
            # The collective target is shared by every per-DM peer-selection
            # call.  Compute it once instead of once per eligible DM.
            collective = collective_preference(
                current, self.instance.evolution_mask
            )

        # Each selected DM follows its most trusted outgoing neighbour (or the
        # efficiency-driven peer).  The canonical DTRF schedule selects every
        # eligible DM.
        for target in self._last_targeted_dms:
            best = self._select_recommendation_peer(
                int(target), current, comprehensive_trust, collective
            )
            if best is not None:
                targets[target] = current[best]

        if self.response_strength_mode == "source_adaptive":
            # The source TERF procedure is acceptance based. The numerical
            # adapter uses the same full accepted recommendation for each
            # selected DM; non-eligible DMs retain their current profile.
            response_rate = np.ones(current.shape[0], dtype=float)
            radius_values = np.ones(current.shape[0], dtype=float)
            control_rule = (
                "source_terf_all_eligible_dms_all_accept"
                if self.all_eligible_per_round
                else "source_terf_targeted_trust_lead_all_accept"
            )
            self._record_response_controls(
                response_rate,
                radius_values,
                rule=control_rule,
                accepted=targeted,
            )
        else:
            willingness = np.clip(
                (1.0 - levels) / max(1.0 - self.response_threshold, 1e-12),
                0.0,
                1.0,
            )
            radius_values = self.willingness_radius * willingness
            response_rate = np.full(current.shape[0], self.response_rate)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="fixed_surrogate_rate_times_clipped_radius",
            )
        delta = response_rate[:, None] * np.clip(
            targets - current,
            -radius_values[:, None],
            radius_values[:, None],
        )
        updated = np.clip(
            np.where(targeted[:, None], current + delta, current), 0.0, 1.0
        )
        updated = self._mask_update(current, updated)
        response_magnitude = np.sum(
            self._evolution_mask_np * np.abs(updated - current), axis=1
        ) / self._evolution_mask_np.sum(axis=1)
        self._recipient_behavior = np.clip(response_magnitude, 0.0, 1.0)

        # Secondary trust is generated from modification behaviour, not from
        # current opinion similarity. Relative modification magnitudes provide
        # the directed numerical signal on each supported trust edge.
        source_change = response_magnitude[self._rows]
        recipient_change = response_magnitude[self._cols]
        behavior_total = source_change + recipient_change
        secondary_targets = np.divide(
            recipient_change,
            behavior_total,
            out=np.full_like(recipient_change, 0.5),
            where=behavior_total > 1e-12,
        )
        self._secondary_trust = np.clip(
            (1.0 - self.secondary_trust_rate) * previous_secondary
            + self.secondary_trust_rate * secondary_targets,
            1e-6,
            1.0,
        )
        comprehensive_next = 0.5 * (self._trust + self._secondary_trust)
        self._weights = self._matrix_from_trust(comprehensive_next)
        self._last_trust_variation = float(
            np.mean(np.abs(self._secondary_trust - previous_secondary))
        )
        self._last_trust_max_variation = float(
            np.max(
                np.abs(self._secondary_trust - previous_secondary), initial=0.0
            )
        )
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=int(self._trust.size),
            terminated=not bool(np.any(changed)),
        )


class _OCRFImplementation(_OverlappingCommunityMixin, _FeedbackAdapter):
    """Overlapping-community recommendation feedback adapter.

    The default state is the source-native crisp multi-label community matrix.
    Each low-consensus DM receives one community recommendation whose positive
    and negative role contributions are kept separate.
    """

    name = "Ding-Overlapping-CRP-proxy"
    full_name = "Overlapping-Community Recommendation Feedback"
    source_id = "Ding2026EJOR"
    feedback_object = "overlapping_community_positive_negative_recommendation"
    native_constraints = "overlap_role_threshold_and_compromise_radius"
    native_representation = "numerical"
    community_detector = "Ding-CR-OCA(preference-collaboration+trust-support)"
    network_mode = "source_native_community_common_support"
    feedback_order = "overlapping_community_direct_feedback"
    community_refresh_policy = "dynamic_after_each_feedback"

    def __init__(
        self,
        overlap_strength: float = 0.35,
        response_rate: float = 1.0,
        compromise_radius: float = 0.20,
        response_threshold: float = 0.95,
        role_threshold: float = 0.50,
        community_refresh: float = 0.30,
        change_tolerance: float = 1e-4,
        response_strength_mode: str = "source_adaptive",
        community_mode: str = "crisp_multi_label",
    ) -> None:
        super().__init__(
            overlap_strength=overlap_strength,
            response_rate=response_rate,
            compromise_radius=compromise_radius,
            response_threshold=response_threshold,
            role_threshold=role_threshold,
            community_refresh=community_refresh,
            change_tolerance=change_tolerance,
            response_strength_mode=response_strength_mode,
            community_mode=community_mode,
        )

    def _source_adaptive_willingness(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Numerically emulate the source ELICIT willingness optimization.

        The source models optimize a willingness coefficient for each
        subgroup/overlapping DM.  In the common numerical profile, a declared
        one-dimensional line search chooses the smallest coefficient attaining
        the best individual consensus against the current collective profile.
        """
        collective = collective_preference(
            current, self.instance.evolution_mask
        )
        gap = target - current
        mask = self._evolution_mask_np
        denominators = np.maximum(mask.sum(axis=1), 1.0)
        grid = np.linspace(0.0, 1.0, 9)
        best = np.zeros(current.shape[0], dtype=float)
        best_score = np.full(current.shape[0], -np.inf, dtype=float)
        for coefficient in grid:
            candidate = self._mask_update(
                current,
                np.clip(current + coefficient * gap, 0.0, 1.0),
            )
            score = 1.0 - np.sum(
                mask * np.abs(candidate - collective), axis=1
            ) / denominators
            improved = score > best_score + 1e-12
            best[improved] = coefficient
            best_score[improved] = score[improved]
        return best

    def _detect_communities(self, preferences: np.ndarray) -> np.ndarray:
        profile = self._community_profile(preferences)
        return cr_oca_overlapping_labels(
            profile,
            adjacency=self.instance.adjacency,
        )

    def _on_community_state_updated(self) -> None:
        overlap = self._edge_overlap_scores(self._rows, self._cols)
        self._weights = self._matrix_from_edge_scores(overlap)

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        levels = self._consensus_levels(current)
        similarities = np.clip(1.0 - self._edge_distances(current), 0.0, 1.0)
        overlap_scores = self._edge_overlap_scores(self._rows, self._cols)
        positive_scores = overlap_scores * np.maximum(
            similarities - self.role_threshold, 0.0
        )
        negative_scores = overlap_scores * np.maximum(
            self.role_threshold - similarities, 0.0
        )
        positive_targets = self._row_normalized_targets(current, positive_scores)
        fallback_targets = self._row_normalized_targets(current, overlap_scores)
        positive_totals = self._row_totals(positive_scores, self._row_ptr)
        negative_totals = self._row_totals(negative_scores, self._row_ptr)
        positive_available = positive_totals > 1e-12
        positive_targets = np.where(
            positive_available[:, None], positive_targets, fallback_targets
        )
        negative_targets = self._row_normalized_targets(current, negative_scores)
        negative_available = negative_totals > 1e-12
        negative_targets = np.where(
            negative_available[:, None], negative_targets, current
        )
        negative_fraction = negative_totals / np.maximum(
            positive_totals + negative_totals, 1e-12
        )
        signed_target = np.clip(
            positive_targets
            + negative_fraction[:, None]
            * (positive_targets - negative_targets),
            0.0,
            1.0,
        )
        eligible = levels < self.response_threshold
        if self.response_strength_mode == "source_adaptive":
            response_rate = self._source_adaptive_willingness(
                current, signed_target
            )
            radius_values = np.ones(current.shape[0], dtype=float)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="source_elicit_numeric_willingness_line_search",
            )
        else:
            radius_scale = np.clip(
                (1.0 - levels) / max(1.0 - self.response_threshold, 1e-12),
                0.0,
                1.0,
            )
            radius_values = self.compromise_radius * radius_scale
            response_rate = np.full(current.shape[0], self.response_rate)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="fixed_surrogate_rate_times_clipped_radius",
            )
        delta = response_rate[:, None] * np.clip(
            signed_target - current,
            -radius_values[:, None],
            radius_values[:, None],
        )
        updated = np.clip(
            np.where(eligible[:, None], current + delta, current), 0.0, 1.0
        )
        updated = self._mask_update(current, updated)
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        self._refresh_communities(updated, similarities)
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=int(self._rows.size),
            terminated=not bool(np.any(changed)),
        )


class _DCRTFImplementation(_OCRFImplementation):
    """Dynamic overlapping-community CRP with reverse trust updating.

    Numerical preferences are encoded as the degenerate intuitionistic-fuzzy
    triple ``(mu, nu, h)=(p, 1-p, 0)``. Community feedback is computed in the
    decoded numerical space, while opinion adjustment feeds back into the
    directed trust state used by the next influence update.
    """

    name = "Teng-Dynamic-Community-CRP-proxy"
    full_name = "Dynamic Community Reverse-Trust Feedback"
    source_id = "Teng2024INS"
    feedback_object = "overlapping_community_feedback_reverse_trust_update"
    native_constraints = "intuitionistic_fuzzy_encoding_and_compromise_radius"
    native_representation = "intuitionistic_fuzzy_mu_nu_h_zero"
    community_detector = "Teng-WDSLPA(preference-similarity+dynamic-trust)"
    network_mode = "source_native_dynamic_trust_overlapping_community"
    feedback_order = "feedback_then_reverse_trust_update_then_regroup"
    community_refresh_policy = "dynamic_after_each_feedback"

    def __init__(
        self,
        overlap_strength: float = 0.35,
        trust_rate: float = 0.30,
        opinion_trust_rate: float = 0.45,
        response_rate: float = 1.0,
        compromise_radius: float = 0.20,
        response_threshold: float = 0.95,
        role_threshold: float | str = 0.50,
        community_refresh: float = 0.30,
        change_tolerance: float = 1e-4,
        response_strength_mode: str = "source_adaptive",
        community_mode: str = "crisp_multi_label",
    ) -> None:
        role_threshold_mode: str | None = None
        role_threshold_spec: float | str
        if isinstance(role_threshold, str):
            role_threshold_mode = role_threshold.strip().lower()
            if role_threshold_mode not in {"mean", "median"}:
                raise ValueError(
                    "role_threshold must be a number in [0, 1], 'mean', or 'median'"
                )
            # The mixin validates a numeric fallback; the effective value is
            # replaced at the beginning of every feedback round below.
            resolved_role_threshold = 0.50
            role_threshold_spec = role_threshold_mode
        else:
            resolved_role_threshold = float(role_threshold)
            role_threshold_spec = resolved_role_threshold
        super().__init__(
            overlap_strength=overlap_strength,
            response_rate=response_rate,
            compromise_radius=compromise_radius,
            response_threshold=response_threshold,
            role_threshold=resolved_role_threshold,
            community_refresh=community_refresh,
            change_tolerance=change_tolerance,
            response_strength_mode=response_strength_mode,
            community_mode=community_mode,
        )
        if not 0.0 < trust_rate <= 1.0:
            raise ValueError("trust_rate must lie in (0, 1]")
        if not 0.0 < opinion_trust_rate <= 1.0:
            raise ValueError("opinion_trust_rate must lie in (0, 1]")
        self.trust_rate = trust_rate
        self.opinion_trust_rate = opinion_trust_rate
        self.role_threshold_mode = role_threshold_mode
        self.role_threshold_spec = role_threshold_spec
        self._last_role_threshold = resolved_role_threshold

    def _resolve_role_threshold(self, similarities: np.ndarray) -> float:
        """Resolve the role split from the current directed edge similarities."""
        if self.role_threshold_mode is None:
            threshold = float(self.role_threshold)
        else:
            values = np.asarray(similarities, dtype=float).reshape(-1)
            if values.size == 0:
                threshold = 0.50
            elif self.role_threshold_mode == "mean":
                threshold = float(np.mean(values))
            else:
                threshold = float(np.median(values))
            threshold = float(np.clip(threshold, 0.0, 1.0))
        self._last_role_threshold = threshold
        return threshold

    def initialize(self, instance: CRPInstance, backend=None) -> None:
        super().initialize(instance, backend)
        self._native_mu = np.asarray(instance.preferences, dtype=float).copy()
        self._native_nu = 1.0 - self._native_mu
        self._native_hesitation = np.zeros_like(self._native_mu)
        self._self_confidence = self.backend.to_numpy(
            self._initial_weights.self_weights
        ).copy()
        overlap = self._edge_overlap_scores(self._rows, self._cols)
        self._trust = np.clip(
            instance.epsilon_w + (1.0 - 2.0 * instance.epsilon_w) * overlap,
            1e-6,
            1.0,
        )
        # The source dynamic-community process uses the current trust state
        # when forming labels. Re-run the detector once after that state is
        # initialized so the first recorded community profile is consistent
        # with the first influence matrix.
        similarities = np.clip(
            1.0 - self._edge_distances(instance.preferences), 0.0, 1.0
        )
        self._refresh_communities(instance.preferences, similarities)
        refreshed_overlap = self._edge_overlap_scores(self._rows, self._cols)
        self._trust = np.clip(
            instance.epsilon_w
            + (1.0 - 2.0 * instance.epsilon_w) * refreshed_overlap,
            1e-6,
            1.0,
        )
        self._weights = self._combined_weights(self._trust, refreshed_overlap)

    def _detect_communities(self, preferences: np.ndarray) -> np.ndarray:
        profile = self._community_profile(preferences)
        rows = self._rows
        cols = self._cols
        similarities = np.empty(rows.size, dtype=float)
        edge_chunk = 65_536
        for start in range(0, rows.size, edge_chunk):
            end = min(start + edge_chunk, rows.size)
            similarities[start:end] = np.clip(
                1.0
                - np.mean(
                    np.abs(profile[rows[start:end]] - profile[cols[start:end]]),
                    axis=1,
                ),
                0.0,
                1.0,
            )
        trust = getattr(self, "_trust", None)
        if trust is None:
            trust = np.ones(self._rows.size, dtype=float)
        return wdslpa_overlapping_labels(
            self.instance.adjacency,
            preferences,
            edge_trust=np.asarray(trust, dtype=float),
            edge_similarity=similarities,
            iterations=10,
            frequency_threshold=0.10,
            max_labels=32,
        )

    def _encode_native(self, preferences: np.ndarray) -> None:
        self._native_mu = np.clip(np.asarray(preferences, dtype=float), 0.0, 1.0)
        self._native_nu = 1.0 - self._native_mu
        self._native_hesitation = np.zeros_like(self._native_mu)

    def _combined_weights(
        self, trust: np.ndarray, overlap: np.ndarray | None = None
    ) -> EdgeInfluence:
        if overlap is None:
            overlap = self._edge_overlap_scores(self._rows, self._cols)
        effective = trust * (0.5 + 0.5 * np.clip(overlap, 0.0, 1.0))
        row_totals = self._row_totals(effective, self._row_ptr)
        repeated_totals = np.repeat(row_totals, np.diff(self._row_ptr))
        edge_weights = (
            (1.0 - self._self_confidence)[self._rows]
            * effective
            / repeated_totals
        )
        return EdgeInfluence(
            self._rows,
            self._cols,
            self._row_ptr,
            self.backend.array(edge_weights),
            self.backend.clone(self._self_confidence),
            self.instance.n,
            self.backend,
            self._evolution_mask,
        )

    def feedback(
        self, preferences: np.ndarray, round_index: int
    ) -> FeedbackDiagnostics:
        del round_index
        current = np.asarray(preferences, dtype=float)
        levels = self._consensus_levels(current)
        similarities = np.clip(1.0 - self._edge_distances(current), 0.0, 1.0)
        overlap_scores = self._edge_overlap_scores(self._rows, self._cols)
        role_threshold = self._resolve_role_threshold(similarities)
        previous_trust = self._trust.copy()
        # The source sequence uses the current trust/community state to issue
        # feedback, then revises trust from the post-response opinions before
        # regrouping.  Trust is therefore updated below, after ``updated`` is
        # available, rather than before the recommendation is formed.

        positive_scores = self._trust * overlap_scores * np.maximum(
            similarities - role_threshold, 0.0
        )
        negative_scores = self._trust * overlap_scores * np.maximum(
            role_threshold - similarities, 0.0
        )
        positive_targets = self._row_normalized_targets(current, positive_scores)
        fallback_targets = self._row_normalized_targets(current, self._trust)
        positive_totals = self._row_totals(positive_scores, self._row_ptr)
        negative_totals = self._row_totals(negative_scores, self._row_ptr)
        positive_targets = np.where(
            (positive_totals > 1e-12)[:, None], positive_targets, fallback_targets
        )
        negative_targets = self._row_normalized_targets(current, negative_scores)
        negative_targets = np.where(
            (negative_totals > 1e-12)[:, None], negative_targets, current
        )
        negative_fraction = negative_totals / np.maximum(
            positive_totals + negative_totals, 1e-12
        )
        target = np.clip(
            positive_targets
            + negative_fraction[:, None] * (positive_targets - negative_targets),
            0.0,
            1.0,
        )
        eligible = levels < self.response_threshold
        if self.response_strength_mode == "source_adaptive":
            # Teng et al. use directed trust as the weight of the source
            # opinion in the recipient's update.  Aggregate the effective
            # trust-community edge weights to obtain one convex-combination
            # coefficient per recipient DM.
            control_scores = self._trust * overlap_scores
            control_totals = self._row_totals(control_scores, self._row_ptr)
            use_fallback = control_totals <= 1e-12
            control_scores = np.where(
                np.repeat(use_fallback, np.diff(self._row_ptr)),
                self._trust,
                control_scores,
            )
            control_totals = self._row_totals(control_scores, self._row_ptr)
            response_rate = np.clip(
                np.add.reduceat(
                    control_scores * self._trust, self._row_ptr[:-1]
                )
                / control_totals,
                0.0,
                1.0,
            )
            radius_values = np.ones(current.shape[0], dtype=float)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="source_dynamic_trust_weighted_convex_adjustment",
            )
        else:
            radius_scale = np.clip(
                (1.0 - levels) / max(1.0 - self.response_threshold, 1e-12),
                0.0,
                1.0,
            )
            radius_values = self.compromise_radius * radius_scale
            response_rate = np.full(current.shape[0], self.response_rate)
            self._record_response_controls(
                response_rate,
                radius_values,
                rule="fixed_surrogate_rate_times_clipped_radius",
            )
        delta = response_rate[:, None] * np.clip(
            target - current,
            -radius_values[:, None],
            radius_values[:, None],
        )
        updated = np.clip(
            np.where(eligible[:, None], current + delta, current), 0.0, 1.0
        )
        updated = self._mask_update(current, updated)
        self._encode_native(updated)

        post_similarity = np.clip(
            1.0
            - np.mean(
                np.abs(updated[self._rows] - updated[self._cols]), axis=1
            ),
            0.0,
            1.0,
        )
        reverse_targets = 0.60 * post_similarity + 0.40 * overlap_scores
        self._trust = np.clip(
            (1.0 - self.opinion_trust_rate) * self._trust
            + self.opinion_trust_rate * reverse_targets,
            1e-6,
            1.0,
        )
        self._last_trust_variation = float(
            np.mean(np.abs(self._trust - previous_trust))
        )
        self._last_trust_max_variation = float(
            np.max(np.abs(self._trust - previous_trust), initial=0.0)
        )
        self._refresh_communities(updated, post_similarity)
        final_overlap = self._edge_overlap_scores(self._rows, self._cols)
        self._weights = self._combined_weights(self._trust, final_overlap)
        changed = np.max(np.abs(updated - current), axis=1) > self.change_tolerance
        self._pending_preferences = updated
        return FeedbackDiagnostics(
            direct_feedback_updates=int(np.count_nonzero(changed)),
            sweeps=1,
            active_edges=int(self._trust.size),
            terminated=not bool(np.any(changed)),
        )


# Canonical mechanism API names used by the unified comparison runner.
class DTRFMechanism(_DTRFImplementation):
    """Canonical DTRF schedule used in the mechanism comparison.

    Every below-threshold DM receives one response opportunity per round. The
    per-DM response and trust-update equations are inherited unchanged; only
    the canonical peer-selection default is the efficiency-driven branch.
    """

    name = "DTRF"
    full_name = "Dynamic Trust Recommendation Feedback"
    def __init__(
        self,
        response_rate: float = 1.0,
        willingness_radius: float = 0.50,
        recommendation_mode: str = "efficiency_driven",
    ) -> None:
        if recommendation_mode != "efficiency_driven":
            raise ValueError(
                "the clean DTRF implementation requires "
                "recommendation_mode='efficiency_driven'"
            )
        super().__init__(
            trust_rate=0.30,
            secondary_trust_rate=0.40,
            response_rate=response_rate,
            willingness_radius=willingness_radius,
            response_threshold=1.0,
            response_strength_mode="fixed_surrogate",
            recommendation_mode=recommendation_mode,
        )
    # The canonical comparison mechanism gives every below-threshold DM one
    # independent response opportunity per round.  Recommendation selection,
    # acceptance, and secondary-trust equations remain unchanged per DM.
    all_eligible_per_round = True
    target_selection_policy = "all_eligible_dms_once_per_feedback_round"
    # Use the existing efficiency-driven branch for the canonical all-eligible
    # schedule; trust-lead remains available as an explicit sensitivity option.
    default_recommendation_mode = "efficiency_driven"
    native_constraints = (
        "per_dm_trust_lead_or_efficiency_recommendation_and_response_threshold"
    )
    feedback_order = (
        "assessment -> all_eligible_dm_detection -> per_dm_recommendation -> "
        "direct_feedback -> behavior_secondary_trust_update -> "
        "comprehensive_trust_for_next_assessment"
    )


class DTLCMechanism(_DTLCImplementation):
    name = "DTLC"
    full_name = "Dynamic Trust Limited-Compromise Feedback"
    def __init__(
        self,
        response_rate: float = 1.0,
        compromise_radius: float = 0.10,
    ) -> None:
        super().__init__(
            trust_rate=0.35,
            compromise_radius=compromise_radius,
            response_rate=response_rate,
            response_threshold=1.0,
            response_strength_mode="fixed_surrogate",
        )


class OCRFMechanism(_OCRFImplementation):
    name = "OCRF"
    full_name = "Overlapping-Community Recommendation Feedback"
    def __init__(
        self,
        role_threshold: float = 0.90,
    ) -> None:
        super().__init__(
            overlap_strength=0.35,
            response_rate=1.0,
            compromise_radius=0.20,
            response_threshold=1.0,
            role_threshold=role_threshold,
            response_strength_mode="source_adaptive",
        )


class DCRTFMechanism(_DCRTFImplementation):
    name = "DCRTF"
    full_name = "Dynamic Community Reverse-Trust Feedback"
    def __init__(
        self,
        role_threshold: float = 0.50,
        opinion_trust_rate: float = 0.50,
    ) -> None:
        super().__init__(
            overlap_strength=0.35,
            trust_rate=0.30,
            opinion_trust_rate=opinion_trust_rate,
            response_rate=1.0,
            compromise_radius=0.20,
            response_threshold=1.0,
            role_threshold=role_threshold,
            response_strength_mode="source_adaptive",
        )


class OBCFMechanism(_OBCFImplementation):
    name = "OBCF"
    full_name = "Overlapping-Community Bounded-Confidence Feedback"
    def __init__(
        self,
        lfm_alpha: float = 1.0,
        response_rate: float = 1.0,
        feedback_radius: float = 0.10,
        global_local_weight: float = 0.10,
    ) -> None:
        super().__init__(
            overlap_strength=0.35,
            response_rate=response_rate,
            feedback_radius=feedback_radius,
            response_threshold=1.0,
            response_strength_mode="fixed_surrogate",
            global_local_weight=global_local_weight,
            lfm_alpha=lfm_alpha,
        )


class OCCFMechanism(_OCCFImplementation):
    name = "OCCF"
    full_name = "Overlapping-Community Compromise Feedback"
    def __init__(
        self,
        similarity_threshold: float = 0.50,
        loss_aversion: float = 1.0,
        bidirectional_weight: float = 0.50,
        lfm_alpha: float = 1.0,
    ) -> None:
        super().__init__(
            overlap_strength=0.35,
            response_rate=1.0,
            compromise_radius=0.20,
            response_threshold=1.0,
            role_threshold=0.50,
            similarity_threshold=similarity_threshold,
            lfm_alpha=lfm_alpha,
            loss_aversion=loss_aversion,
            bidirectional_weight=bidirectional_weight,
            response_strength_mode="source_adaptive",
        )


class NGPFMechanism(_NGPFImplementation):
    name = "NGPF"
    full_name = "Network-Game Peer-Effect Feedback"

    def __init__(
        self,
        peer_effect: float = 0.05,
        incentive_rate: float = 0.15,
        response_rate: float = 1.0,
        action_budget: float = 1.0,
    ) -> None:
        super().__init__(
            peer_effect=peer_effect,
            incentive_rate=incentive_rate,
            response_rate=response_rate,
            action_budget=action_budget,
            response_threshold=1.0,
            change_tolerance=1e-4,
        )
