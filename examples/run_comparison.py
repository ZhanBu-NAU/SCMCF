"""Reproduce the cross-mechanism endpoint comparison on one profile.

Runs SCMCF and the ten external baselines with the parameters fixed in the
paper and prints the four threshold-attainment metrics per mechanism:

    python examples/run_comparison.py --dataset FilmTrust --profile-l 10

Non-attaining runs keep their run-end CI/TPA/CPA values and report "--" for
the threshold stage, mirroring how they are marked in the paper's tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Make the repository root importable when this file is executed directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scmcf_core import (
    BASELINE_NAMES,
    CRPRunner,
    SCMCFMechanism,
    create_baseline,
    load_instance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="FilmTrust")
    parser.add_argument("--profile-l", type=int, default=10)
    args = parser.parse_args()

    instance, metadata = load_instance(args.dataset, args.profile_l)
    print(
        f"{args.dataset}-{args.profile_l}: N={metadata['n']}, "
        f"M={metadata['m']}, L={metadata['l']}, "
        f"reported cells={metadata['observed_cells']}"
    )
    header = (
        f"{'mechanism':<10}{'attained':>9}{'T_mu':>6}{'CI_min':>9}"
        f"{'TPA':>9}{'CPA':>9}{'time(s)':>9}"
    )
    print("\n" + header)
    print("-" * len(header))

    mechanisms = [SCMCFMechanism(), *(create_baseline(n) for n in BASELINE_NAMES)]
    for mechanism in mechanisms:
        started = perf_counter()
        result = CRPRunner().run(instance, mechanism)
        elapsed = perf_counter() - started
        final = result.records[-1]
        stage = f"{result.threshold_round}" if result.attained else "--"
        print(
            f"{mechanism.name:<10}{str(result.attained):>9}{stage:>6}"
            f"{final.minimum_ci:>9.4f}"
            f"{final.terminal_preference_adjustment:>9.4f}"
            f"{final.cumulative_preference_adjustment:>9.4f}"
            f"{elapsed:>9.1f}"
        )


if __name__ == "__main__":
    main()
