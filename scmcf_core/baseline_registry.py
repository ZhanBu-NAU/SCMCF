"""Single public registry for the ten external comparison mechanisms."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .baselines import DGBCMechanism, SNDGMechanism, TEDGMechanism
from .feedback_baselines import (
    DCRTFMechanism,
    DTLCMechanism,
    DTRFMechanism,
    NGPFMechanism,
    OBCFMechanism,
    OCCFMechanism,
    OCRFMechanism,
)


BASELINE_NAMES = (
    "SNDG",
    "TEDG",
    "DGBC",
    "DTRF",
    "DTLC",
    "OCRF",
    "DCRTF",
    "OBCF",
    "OCCF",
    "NGPF",
)

BASELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "SNDG": {"self_weight": 0.5},
    "TEDG": {
        "reward_rate": 0.20,
        "punishment_rate": 0.20,
        "sensitivity": 1.0,
    },
    "DGBC": {
        "confidence": 0.175,
        "network_update": "osm",
        "selection_rate": 0.001,
    },
    "DTRF": {
        "response_rate": 1.0,
        "willingness_radius": 0.50,
        "recommendation_mode": "efficiency_driven",
    },
    "DTLC": {
        "response_rate": 1.0,
        "compromise_radius": 0.10,
    },
    "OCRF": {
        "role_threshold": 0.90,
    },
    "DCRTF": {
        "role_threshold": 0.50,
        "opinion_trust_rate": 0.50,
    },
    "OBCF": {
        "lfm_alpha": 1.0,
        "response_rate": 1.0,
        "feedback_radius": 0.10,
        "global_local_weight": 0.10,
    },
    "OCCF": {
        "similarity_threshold": 0.50,
        "loss_aversion": 1.0,
        "bidirectional_weight": 0.50,
        "lfm_alpha": 1.0,
    },
    "NGPF": {
        "peer_effect": 0.05,
        "incentive_rate": 0.15,
        "response_rate": 1.0,
        "action_budget": 1.0,
    },
}

# These choices are part of the frozen comparison implementation and are not
# public sensitivity switches in this clean package.
_FIXED_BRANCHES: dict[str, dict[str, Any]] = {
    # Response eligibility is fixed at CI < 1 for these seven adapters. Keep
    # it in protocol metadata rather than exposing it as a tunable parameter.
    "DTRF": {
        "response_strength_mode": "fixed_surrogate",
        "response_threshold": 1.0,
    },
    "DTLC": {
        "response_strength_mode": "fixed_surrogate",
        "response_threshold": 1.0,
    },
    "OCRF": {
        "response_strength_mode": "source_adaptive",
        "response_threshold": 1.0,
    },
    "DCRTF": {
        "response_strength_mode": "source_adaptive",
        "response_threshold": 1.0,
    },
    "OBCF": {
        "response_strength_mode": "fixed_surrogate",
        "response_threshold": 1.0,
    },
    "OCCF": {
        "response_strength_mode": "source_adaptive",
        "response_threshold": 1.0,
    },
    "NGPF": {"response_threshold": 1.0},
}

_CLASSES = {
    "SNDG": SNDGMechanism,
    "TEDG": TEDGMechanism,
    "DGBC": DGBCMechanism,
    "DTRF": DTRFMechanism,
    "DTLC": DTLCMechanism,
    "OCRF": OCRFMechanism,
    "DCRTF": DCRTFMechanism,
    "OBCF": OBCFMechanism,
    "OCCF": OCCFMechanism,
    "NGPF": NGPFMechanism,
}


def baseline_defaults(name: str) -> dict[str, Any]:
    if name not in BASELINE_DEFAULTS:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
    return deepcopy(BASELINE_DEFAULTS[name])


def create_baseline(name: str, overrides: dict[str, Any] | None = None):
    """Build one canonical adapter with only its declared public parameters."""
    if name not in BASELINE_DEFAULTS:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides) - set(BASELINE_DEFAULTS[name])
    if unknown:
        allowed = ", ".join(BASELINE_DEFAULTS[name])
        raise ValueError(
            f"unsupported {name} parameter(s): {sorted(unknown)}; allowed: {allowed}"
        )
    parameters = baseline_defaults(name)
    parameters.update(overrides)
    if name == "SNDG" and (
        parameters.get("self_weight") is None
        or not np.isscalar(parameters.get("self_weight"))
    ):
        raise ValueError("SNDG self_weight must be one scalar in (0, 1)")
    if name == "DGBC" and parameters["network_update"] != "osm":
        raise ValueError("the clean DGBC comparison implements only network_update='osm'")
    if name == "DTRF" and parameters["recommendation_mode"] != "efficiency_driven":
        raise ValueError(
            "the clean DTRF comparison implements only "
            "recommendation_mode='efficiency_driven'"
        )
    # Branch choices are frozen comparison protocol metadata.  Canonical
    # adapter classes set them internally; they are deliberately not exposed
    # as public constructor/CLI switches.
    mechanism = _CLASSES[name](**parameters)
    mechanism.comparison_parameters = parameters
    mechanism.comparison_branch = deepcopy(_FIXED_BRANCHES.get(name, {}))
    return mechanism
