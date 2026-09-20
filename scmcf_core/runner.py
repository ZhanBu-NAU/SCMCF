"""Common consensus-reaching loop and mask-based endpoint measures."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .backend import NumpyBackend
from .instance import CRPInstance
from .mechanisms import FeedbackDiagnostics, InfluenceMechanism


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    gcl: float
    minimum_ci: float
    cumulative_preference_adjustment: float
    terminal_preference_adjustment: float
    accepted_responses: int = 0
    direct_feedback_updates: int = 0
    response_sweeps: int = 0
    membership_adjustment: float = 0.0
    cumulative_membership_adjustment: float = 0.0
    potential_before: float | None = None
    potential_after: float | None = None
    feedback_seconds: float = 0.0


@dataclass(frozen=True)
class RunResult:
    mechanism: str
    attained: bool
    threshold_round: int | None
    report_round: int
    termination_reason: str
    records: tuple[RoundRecord, ...]
    final_preferences: np.ndarray
    final_memberships: np.ndarray | None
    mechanism_seconds: float


def consensus_levels(
    preferences: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    counts = mask.sum(axis=1)
    beta = mask / counts[:, None]
    masses = beta.sum(axis=0)
    collective = np.divide(
        np.sum(beta * preferences, axis=0),
        masses,
        out=np.zeros(preferences.shape[1], dtype=float),
        where=masses > 0.0,
    )
    individual = 1.0 - np.sum(beta * np.abs(preferences - collective), axis=1)
    return float(np.mean(individual)), float(np.min(individual))


class CRPRunner:
    def __init__(
        self,
        *,
        evaluation_scope: str = "observed",
        stopping_criterion: str = "mean_gcl",
        stop_on_stable_feedback: bool = False,
    ) -> None:
        if evaluation_scope not in {"observed", "all_cells"}:
            raise ValueError("evaluation_scope must be observed or all_cells")
        if stopping_criterion not in {"mean_gcl", "minimum_ci"}:
            raise ValueError("stopping_criterion must be mean_gcl or minimum_ci")
        self.evaluation_scope = evaluation_scope
        self.stopping_criterion = stopping_criterion
        self.stop_on_stable_feedback = bool(stop_on_stable_feedback)
        self.backend = NumpyBackend()

    def _attained(self, gcl: float, minimum_ci: float, threshold: float) -> bool:
        value = gcl if self.stopping_criterion == "mean_gcl" else minimum_ci
        return value >= threshold

    def run(self, instance: CRPInstance, mechanism: InfluenceMechanism) -> RunResult:
        started = perf_counter()
        mechanism.initialize(instance, self.backend)
        try:
            preferences = instance.preferences.copy()
            initial = preferences.copy()
            mask = (
                instance.observation_mask
                if self.evaluation_scope == "observed"
                else np.ones_like(instance.observation_mask)
            )
            evaluated_cells = int(mask.sum())
            gcl, minimum_ci = consensus_levels(preferences, mask)
            records = [RoundRecord(0, gcl, minimum_ci, 0.0, 0.0)]
            if self._attained(gcl, minimum_ci, instance.consensus_threshold):
                return RunResult(
                    mechanism.name,
                    True,
                    0,
                    0,
                    "consensus",
                    tuple(records),
                    preferences,
                    None if mechanism.memberships is None else mechanism.memberships.copy(),
                    perf_counter() - started,
                )

            cumulative_adjustment = 0.0
            cumulative_membership = 0.0
            pending_started = perf_counter()
            pending = mechanism.feedback(preferences, 0)
            pending_seconds = perf_counter() - pending_started
            attained = False
            threshold_round: int | None = None
            termination = "max_rounds"
            for round_index in range(1, instance.max_rounds + 1):
                updated = np.asarray(mechanism.evolve_after_feedback(preferences), dtype=float)
                cumulative_adjustment += float(np.sum(mask * np.abs(updated - preferences)))
                cumulative_membership += pending.membership_adjustment
                gcl, minimum_ci = consensus_levels(updated, mask)
                records.append(
                    RoundRecord(
                        round_index=round_index,
                        gcl=gcl,
                        minimum_ci=minimum_ci,
                        cumulative_preference_adjustment=(
                            cumulative_adjustment / evaluated_cells
                        ),
                        terminal_preference_adjustment=(
                            float(np.sum(mask * np.abs(updated - initial))) / evaluated_cells
                        ),
                        accepted_responses=pending.accepted_responses,
                        direct_feedback_updates=pending.direct_feedback_updates,
                        response_sweeps=pending.sweeps,
                        membership_adjustment=pending.membership_adjustment,
                        cumulative_membership_adjustment=cumulative_membership,
                        potential_before=pending.potential_before,
                        potential_after=pending.potential_after,
                        feedback_seconds=pending_seconds,
                    )
                )
                preferences = updated
                if self._attained(gcl, minimum_ci, instance.consensus_threshold):
                    attained = True
                    threshold_round = round_index
                    termination = "consensus"
                    break
                if (
                    self.stop_on_stable_feedback
                    and mechanism.stability_termination_supported
                    and mechanism.feedback_state_is_stable(pending)
                ):
                    termination = "feedback_state_stable"
                    break
                if round_index == instance.max_rounds:
                    break
                pending_started = perf_counter()
                pending = mechanism.feedback(preferences, round_index)
                pending_seconds = perf_counter() - pending_started
            return RunResult(
                mechanism=mechanism.name,
                attained=attained,
                threshold_round=threshold_round,
                report_round=records[-1].round_index,
                termination_reason=termination,
                records=tuple(records),
                final_preferences=preferences.copy(),
                final_memberships=(
                    None if mechanism.memberships is None else mechanism.memberships.copy()
                ),
                mechanism_seconds=perf_counter() - started,
            )
        finally:
            mechanism.close()
