"""Dataset loading, fixed-mask construction, and missing-cell completion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .instance import CRPInstance


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PACKAGE_ROOT / "Dataset"
DATASET_NAMES = ("FilmTrust", "Ciao", "Deezer_RO", "Deezer_HU", "Deezer_HR")
COMPLETION_POLICIES = ("alternative_mean", "neutral_0.5", "constant", "random_uniform")


def available_profiles(dataset: str) -> tuple[int, ...]:
    source = DATASET_ROOT / dataset
    if dataset not in DATASET_NAMES or not source.is_dir():
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASET_NAMES}")
    roots = [source / "instances", source / "profiles"]
    values = {
        int(path.name[1:])
        for root in roots
        if root.is_dir()
        for path in root.glob("L[0-9][0-9][0-9]")
        if path.is_dir()
    }
    return tuple(sorted(values))


def randomize_observation_mask(
    observation_mask: np.ndarray,
    *,
    drop_rate: float,
    seed: int,
) -> np.ndarray:
    """Drop observed cells while retaining one per DM and alternative."""
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError("mask drop_rate must lie in [0, 1)")
    mask = (np.asarray(observation_mask, dtype=float) > 0.5).astype(float)
    if drop_rate == 0.0:
        return mask
    rng = np.random.default_rng(seed)
    observed = np.argwhere(mask > 0.5)
    target = int(np.floor(drop_rate * observed.shape[0]))
    row_counts = mask.sum(axis=1).astype(int)
    column_counts = mask.sum(axis=0).astype(int)
    dropped = 0
    for position in rng.permutation(observed.shape[0]):
        i, ell = map(int, observed[position])
        if row_counts[i] <= 1 or column_counts[ell] <= 1:
            continue
        mask[i, ell] = 0.0
        row_counts[i] -= 1
        column_counts[ell] -= 1
        dropped += 1
        if dropped == target:
            break
    return mask


def complete_preferences(
    reported_values: np.ndarray,
    observation_mask: np.ndarray,
    *,
    policy: str = "alternative_mean",
    constant: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Complete missing cells without changing any currently observed entry."""
    values = np.asarray(reported_values, dtype=float)
    mask = np.asarray(observation_mask, dtype=float)
    if values.shape != mask.shape:
        raise ValueError("reported_values and observation_mask must have equal shape")
    if policy not in COMPLETION_POLICIES:
        raise ValueError(f"completion policy must be one of {COMPLETION_POLICIES}")
    missing = mask < 0.5
    completed = values.copy()
    if not np.any(missing):
        return completed
    if policy == "alternative_mean":
        counts = mask.sum(axis=0)
        if np.any(counts == 0.0):
            raise ValueError("alternative_mean requires one observation per alternative")
        fill = np.sum(mask * values, axis=0) / counts
        completed[missing] = np.broadcast_to(fill, values.shape)[missing]
    elif policy == "neutral_0.5":
        completed[missing] = 0.5
    elif policy == "constant":
        if not 0.0 <= constant <= 1.0:
            raise ValueError("completion constant must lie in [0, 1]")
        completed[missing] = constant
    else:
        rng = np.random.default_rng(seed)
        completed[missing] = rng.uniform(0.0, 1.0, size=int(np.count_nonzero(missing)))
    return completed


def _load_edges(path: Path, n: int) -> sparse.csr_matrix:
    edges = np.loadtxt(path, delimiter=",", dtype=np.int64, skiprows=1)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    if edges.shape[1] != 2 or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError(f"invalid edge list: {path}")
    adjacency = sparse.csr_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def _load_base_node_ids(path: Path, n: int) -> np.ndarray:
    if not path.is_file():
        return np.arange(n, dtype=np.int64)
    result = np.full(n, -1, dtype=np.int64)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result[int(row["node_id"])] = int(row["base_node_id"])
    if np.any(result < 0):
        raise ValueError(f"incomplete node mapping: {path}")
    return result


def _load_memberships(path: Path, base_nodes: np.ndarray) -> np.ndarray:
    labels: dict[int, int] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[int(row["node_id"])] = int(row["z0"])
    retained = np.asarray([labels[int(node)] for node in base_nodes], dtype=int)
    unique = {label: column for column, label in enumerate(sorted(set(retained)))}
    memberships = np.zeros((retained.size, len(unique)), dtype=float)
    memberships[np.arange(retained.size), [unique[x] for x in retained]] = 1.0
    return memberships


def load_instance(
    dataset: str,
    profile_l: int,
    *,
    completion_policy: str = "alternative_mean",
    completion_value: float = 0.5,
    completion_seed: int = 0,
    mask_drop_rate: float = 0.0,
    mask_seed: int = 0,
    epsilon_w: float = 1e-6,
    threshold: float = 0.95,
    max_rounds: int = 100,
) -> tuple[CRPInstance, dict[str, object]]:
    """Load one retained network-preference profile and apply runtime hooks."""
    profiles = available_profiles(dataset)
    if profile_l not in profiles:
        raise ValueError(
            f"{dataset} has no L={profile_l} profile; available values are {profiles}"
        )
    source = DATASET_ROOT / dataset
    materialized = source / "instances" / f"L{profile_l:03d}"
    profile = source / "profiles" / f"L{profile_l:03d}"
    directory = materialized if (materialized / "metadata.json").is_file() else profile
    with np.load(directory / "preferences.npz") as archive:
        stored = np.asarray(archive["P"], dtype=float)
        original_mask = (
            np.asarray(archive["O"], dtype=float)
            if "O" in archive.files
            else np.ones_like(stored)
        )
    n = stored.shape[0]
    adjacency_path = directory / "edges.csv"
    if not adjacency_path.is_file():
        adjacency_path = source / "edges.csv"
    adjacency = _load_edges(adjacency_path, n)
    base_nodes = _load_base_node_ids(directory / "nodes.csv", n)
    memberships = _load_memberships(source / "labels.csv", base_nodes)
    mask = randomize_observation_mask(
        original_mask, drop_rate=mask_drop_rate, seed=mask_seed
    )
    # Cells dropped by a randomized mask retain their original values only as
    # source data; the selected completion policy supplies their model input.
    preferences = complete_preferences(
        stored,
        mask,
        policy=completion_policy,
        constant=completion_value,
        seed=completion_seed,
    )
    instance = CRPInstance(
        adjacency=adjacency,
        preferences=preferences,
        observation_mask=mask,
        memberships=memberships,
        epsilon_w=epsilon_w,
        consensus_threshold=threshold,
        max_rounds=max_rounds,
    )
    source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata: dict[str, object] = {
        "dataset": dataset,
        "profile_l": profile_l,
        "n": instance.n,
        "m": int(instance.adjacency.nnz),
        "l": instance.l,
        "source_protocol": source_metadata.get("protocol_version"),
        "profile_directory": str(directory),
        "materialized_profile": directory == materialized,
        "original_observed_cells": int(np.count_nonzero(original_mask)),
        "observed_cells": int(np.count_nonzero(mask)),
        "mask_drop_rate_requested": float(mask_drop_rate),
        "mask_drop_rate_realized": 1.0 - float(mask.sum() / original_mask.sum()),
        "mask_seed": int(mask_seed),
        "completion_policy": completion_policy,
        "completion_value": completion_value if completion_policy == "constant" else None,
        "completion_seed": completion_seed if completion_policy == "random_uniform" else None,
    }
    return instance, metadata
