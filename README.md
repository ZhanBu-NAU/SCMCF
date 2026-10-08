# SCMCF: Prospective Feedback Games for Consensus Reaching in Social Network Group Decision Making with Incomplete Preferences

This repository contains the reference implementation of **SCMCF**
(soft-community-mediated consensus feedback), the consensus mechanism introduced
in the paper *Prospective Feedback Games for Consensus Reaching in Social
Network Group Decision Making with Incomplete Preferences*, together with the
nine network–preference profiles used in its experiments and numerical adapters
for the ten external consensus baselines compared in the paper.

## Overview

SCMCF models consensus feedback as a **prospective feedback game (PFG)**: after
an unsuccessful consensus assessment, each decision maker (DM) strategically
adjusts a simplex-constrained soft-community membership vector that recalibrates
peer influence and residual self-retention on the fixed trust topology before
the next preference-evolution step. The implementation provides

- the general PFG construction (response-impact sets, impact-local costs,
  exact-potential incentives, compatible response classes),
- the SCMCF specialization with topology-driven overlapping-community
  initialization, compatible-class batched response sweeps, and deterministic
  simplex-QP responses,
- ten external CRP baselines---SNDG [4], TEDG [5], DGBC [6], DTRF [7],
  DTLC [8], OCRF [9], DCRTF [10], OBCF [11], OCCF [12], and NGPF [13]---through
  a common adapter interface,
- the four threshold-attainment metrics used in the paper: $T_\mu$, $CI_{\min}$,
  TPA, and CPA,
- the nine retained network–preference profiles (FilmTrust, Ciao, and three
  Deezer networks).

All mechanisms execute deterministically given the retained inputs.

## Installation

Requires Python 3.10 or later with only NumPy and SciPy:

```bash
python -m pip install -r requirements.txt
```

## Data

`Dataset/` holds one directory per source network with the fixed trust edges
(`edges.csv`), the base community labels used by the adapters (`labels.csv`),
and the retained profile directories used in the paper. Each profile directory
stores the numerical preference matrix and its fixed availability mask
(`preferences.npz`, keys `P` and `O`), the node mapping (`nodes.csv`, when the
retained subnetwork is indexed), and the retained rating table
(`alternatives.csv`).

The nine paper profiles are:

| Profile | N | M | L | Observed cells | Observed share |
|---|---:|---:|---:|---:|---:|
| FilmTrust-10 | 211 | 832 | 10 | 1,255 | 59.48% |
| FilmTrust-30 | 221 | 890 | 30 | 2,961 | 44.66% |
| FilmTrust-50 | 223 | 927 | 50 | 3,869 | 34.70% |
| Ciao-10 | 2,092 | 43,161 | 10 | 5,110 | 24.43% |
| Ciao-30 | 2,802 | 55,856 | 30 | 10,530 | 12.53% |
| Ciao-50 | 3,168 | 62,640 | 50 | 13,857 | 8.75% |
| Deezer-RO | 41,773 | 251,652 | 84 | 3,508,932 | 100% |
| Deezer-HU | 47,538 | 445,774 | 84 | 3,993,192 | 100% |
| Deezer-HR | 54,573 | 996,404 | 84 | 4,584,132 | 100% |

FilmTrust and Ciao provide numerical ratings with directed trust relations and
are incomplete; the Deezer profiles provide 84 binary genre indicators on
undirected friendship networks (each friendship is represented by two
reciprocal directed arcs). Missing entries are completed by the
alternative-wise observed mean, the completion participates in preference
evolution, while consensus assessment and the adjustment metrics use the
reported entries only (the fixed availability mask).

The datasets derive from public sources: FilmTrust [1], Ciao (Epinions-style
trust network) [2], and the Deezer user-network collection [3].

## Quickstart

Run SCMCF with the paper defaults on the sparsest complete demonstration
profile:

```bash
python examples/run_scmcf.py --dataset FilmTrust --profile-l 10
```

Reproduce the cross-mechanism endpoint comparison on one profile (SCMCF plus
the ten external baselines):

```bash
python examples/run_comparison.py --dataset FilmTrust --profile-l 10
```

Both scripts print the threshold-attainment endpoints of every mechanism:
attainment flag, threshold stage $T_\mu$, weakest-DM consensus $CI_{\min}$,
terminal and cumulative preference-adjustment cost per reported cell (TPA,
CPA), and wall-clock time. Use `--dataset {FilmTrust,Ciao,Deezer_RO,Deezer_HU,Deezer_HR}`
and `--profile-l {10,30,50}` (`84` for the Deezer profiles) to select any of
the nine paper profiles.

## Parameters

SCMCF uses the defaults fixed before the paper comparison: community resolution
`gamma=1.0` ($\gamma_{\mathrm{dir}}$), consensus–movement trade-off
`alpha=0.5`, influence safeguard `epsilon_w=1e-6` (set at instance loading),
QP tolerance `epsilon_qp=1e-6`, per-QP iteration cap `qp_max_iterations=100`,
consensus threshold $\mu=0.95$, and stage limit $T_{\max}=100`. The example
scripts expose `--gamma` and `--alpha` for the parameter-sensitivity slices.

## Repository structure

```
SCMCF/
├── scmcf_core/            # core library
│   ├── data.py            # dataset loading, availability masks, completion
│   ├── instance.py        # CRP instance container
│   ├── mechanisms.py      # SCMCF mechanism
│   ├── feedback_baselines.py, baselines.py, community_methods.py
│   │                      # external baseline adapters
│   ├── baseline_registry.py  # canonical baseline names and parameters
│   ├── topology.py        # impact sets, compatible-class scheduling
│   ├── weights.py         # membership-calibrated influence construction
│   ├── runner.py          # consensus-reaching loop and endpoint measures
│   ├── metrics.py, backend.py, instance.py
├── Dataset/               # nine network–preference profiles
├── examples/
│   ├── run_scmcf.py       # single-mechanism demonstration
│   └── run_comparison.py  # cross-mechanism endpoint comparison
├── Supplementary_Material.pdf  # supplementary material of the paper
├── requirements.txt
└── LICENSE
```

## Citation

If you use this code, please cite the accompanying article:

```bibtex
@article{bu2026scmcf,
  title   = {Prospective Feedback Games for Consensus Reaching in Social
             Network Group Decision Making with Incomplete Preferences},
  author  = {Bu, Zhan and Zhao, Ziyi and Zhang, Shanfan and Wang, Yuyao},
  journal = {(to appear)},
  year    = {2026}
}
```

## References

Datasets:

[1] G. Guo, J. Zhang, and D. Thalmann, "Merging trust in collaborative
filtering to alleviate data sparsity and cold start," *Knowl.-Based Syst.*,
2013 (FilmTrust).

[2] J. Tang, H. Gao, and X. Hu, "Exploiting homophily effect for trust
prediction," *Proc. WSDM*, 2012 (Ciao).

[3] B. Rozemberczki, R. Davies, R. Sarkar, and C. Sutton, "GEMSEC: Graph
embedding with self-clustering," *Proc. ASONAM*, 2019 (Deezer networks).

External baselines:

[4] Z. Ding, X. Chen, Y. Dong, and F. Herrera, "Consensus reaching in social
network DeGroot model: The roles of the self-confidence and node degree,"
*Inf. Sci.*, vol. 486, pp. 62–72, 2019 (SNDG).

[5] Y. Zhang, X. Chen, L. Gao, Y. Dong, and W. Pedrycz, "Consensus reaching
with trust evolution in social network group decision making," *Expert Syst.
Appl.*, vol. 188, 116022, 2022 (TEDG).

[6] Z. Wu, Q. Zhou, Y. Dong, J. Xu, A. H. Altalhi, and F. Herrera, "Mixed
opinion dynamics based on DeGroot model and Hegselmann–Krause model in social
networks," *IEEE Trans. Syst., Man, Cybern., Syst.*, vol. 53, no. 1,
pp. 296–308, 2023 (DGBC).

[7] J. Wu, S. Wang, F. Chiclana, and E. Herrera-Viedma, "Two-fold
personalized feedback mechanism for social network consensus by uninorm
interval trust propagation," *IEEE Trans. Cybern.*, vol. 52, no. 10,
pp. 11081-11092, 2022 (DTRF).

[8] P. Liu, Y. Li, and P. Wang, "Opinion dynamics and minimum
adjustment-driven consensus model for multi-criteria large-scale group
decision making under a novel social trust propagation mechanism,"
*IEEE Trans. Fuzzy Syst.*, vol. 31, no. 1, pp. 307-321, 2023 (DTLC).

[9] R.-X. Ding, B. Yang, Y. Huang, Y. Zhang, and F. Chiclana, "Social
network-based overlapping community clustering and feedback mechanism for
large-scale group decision making," *Eur. J. Oper. Res.*, vol. 329, no. 2,
pp. 518–535, 2026 (OCRF).

[10] Z. Hua, S. Xu, J. Wang, J. Liu, and L. Martinez, "Bilevel consensus
in large-scale group decision making: Integrating structural holes and
community dynamics," *IEEE Trans. Fuzzy Syst.*, vol. 34, no. 4,
pp. 1282-1294, 2026 (DCRTF).

[11] Y.-M. Wang, H.-H. Song, B. Dutta, D. García-Zamora, and L. Martínez,
"Consensus reaching in LSGDM: Overlapping community detection and bounded
confidence-driven feedback mechanism," *Inf. Sci.*, vol. 679, 121104,
2024 (OBCF).

[12] T. Gai, J. Wu, F. Chiclana, M. Cao, and R. R. Yager, "Dynamic compromise
behavior driven bidirectional feedback mechanism for group consensus with
overlapping communities in social network," *IEEE Trans. Syst., Man, Cybern.,
Syst.*, vol. 54, no. 10, pp. 6149–6161, 2024 (OCCF).

[13] N. Lang, L. Wang, and Q. Zha, "Network game in group decision making:
Managing consensus with incentive and interaction interventions," *Eur. J.
Oper. Res.*, vol. 329, pp. 950–965, 2026 (NGPF).

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
