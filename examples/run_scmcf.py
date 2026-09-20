"""Run SCMCF on one network-preference profile and print its endpoints.

Uses the paper defaults: gamma_dir = 1.0, alpha = 0.5, consensus threshold
mu = 0.95, and stage limit T_max = 100.  Example:

    python examples/run_scmcf.py --dataset FilmTrust --profile-l 10
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

from scmcf_core import CRPRunner, SCMCFMechanism, load_instance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="FilmTrust")
    parser.add_argument("--profile-l", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    instance, metadata = load_instance(args.dataset, args.profile_l)
    print(
        f"{args.dataset}-{args.profile_l}: N={metadata['n']}, "
        f"M={metadata['m']}, L={metadata['l']}, "
        f"reported cells={metadata['observed_cells']}"
    )

    mechanism = SCMCFMechanism(gamma=args.gamma, alpha=args.alpha)
    started = perf_counter()
    result = CRPRunner().run(instance, mechanism)
    elapsed = perf_counter() - started

    final = result.records[-1]
    print(
        f"attained={result.attained}  "
        f"T_mu={result.threshold_round if result.attained else '--'}  "
        f"CI_min={final.minimum_ci:.4f}  "
        f"TPA={final.terminal_preference_adjustment:.4f}  "
        f"CPA={final.cumulative_preference_adjustment:.4f}  "
        f"time={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
