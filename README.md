# Multiscale Entropy Analysis of Complex Networks

Sebastián Brzovic, Cristóbal Rojas, Andrés Abeliuk

**Preprint:** [arXiv:2510.11524](https://arxiv.org/abs/2510.11524)

Code and data for the paper of the same name. It extends compression-based graph entropy
to multiple scales via spectral coarsening, and asks whether the resulting multiscale
description is a useful characterization of structural complexity — assessed by how well
it predicts link-prediction accuracy across 558 real-world networks.

Everything the paper reports is reproducible from the raw corpus upward. This README says
which command produces which figure and table, and what each costs.

> The arXiv preprint is the earlier version of this work, posted under its original
> title *Networks Multiscale Entropy Analysis*. This repository tracks the current
> revision, which expands the corpus to 558 networks, adds AUC alongside prediction
> entropy, and adds the coarsening-robustness and runtime analyses. Numbers here will
> not all match the preprint.

## Layout

```
data/
  CommunityFitNet_updated.pickle   raw corpus, 572 networks (see THIRD_PARTY.md)
  all_networks.pkl                 the 558 networks used here, built from the above
algorithm/                         library: coarsening, compression entropy, link prediction
experiments/                       the pipeline scripts
results/                           intermediate CSVs (committed, so stage 3 runs alone)
figures/  tables/                  what the paper includes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # stage 3 only — figures and tables
pip install -r requirements-compute.txt    # add stage 2 — re-derives the numbers
```

Stage 2 additionally needs PyTorch and PyTorch Geometric; install torch first, matched to
your CUDA version.

## The pipeline

```
data/CommunityFitNet_updated.pickle
  │
  └─(1) build_full_corpus.py  ──────────────►  data/all_networks.pkl        seconds
        │
        ├─(2a) run_real_networks_experiment.py ►  real_networks_results_all.csv
        │      entropy at 5 scales + SEAL and Adamic-Adar link prediction   GPU, hours
        │
        ├─(2b) coarsening_robustness.py ───────►  coarsening_robustness.csv
        │      the same trajectories under a second coarsening algorithm    hours
        │
        └─(2c) runtime_benchmark.py ───────────►  runtime_benchmark.csv
               per-stage wall-clock and peak memory                         ~1 hour
                 │
                 └─(3) rebuild_regression.py, clustering_analysis.py,
                       trajectory_figures.py, runtime_benchmark.py --analyze
                       ────────────────────────►  figures/*.pdf, tables/*.tex   seconds
```

Run it with `make`:

```bash
make figures    # stage 3: every figure and table, from the committed CSVs (seconds)
make compute    # stage 2: re-derive the CSVs from the corpus (GPU, many hours)
make corpus     # stage 1: rebuild data/all_networks.pkl
make all        # all three
make sync       # copy the generated PDFs into the manuscript's imagenes/ tree
```

`make figures` is the usual entry point and needs neither torch nor a GPU. `results/` is
committed precisely so the paper's artifacts can be regenerated without stage 2.

## What produces what

Figure and table numbers refer to the manuscript.

| Paper artifact | Produced by | Reads |
|---|---|---|
| Fig. 2 — entropy trajectories by domain | `trajectory_figures.py` | `coarsening_robustness.csv` |
| Fig. 3 — k-means / PCA clusters | `clustering_analysis.py` | `coarsening_robustness.csv` |
| Table 2 — cluster composition | `clustering_analysis.py` | `coarsening_robustness.csv` |
| Fig. 4 — predicted vs actual, SEAL AUC | `rebuild_regression.py` | `real_networks_results_all.csv` |
| Table 3 — SEAL AUC regression ladder | `rebuild_regression.py` | ″ |
| Table 4 — pipeline cost by level | `runtime_benchmark.py --analyze` | `runtime_benchmark.csv` |
| Fig. 5 — runtime, memory and speedup | `runtime_benchmark.py --analyze` | ″ |
| Figs. 7–9, Tables 8–10 — appendix regressions | `rebuild_regression.py` | `real_networks_results_all.csv` |
| Table 11, Fig. 11 — scaling fits, cost by level | `runtime_benchmark.py --analyze` | `runtime_benchmark.csv` |
| Tables 5–6 — coarsening robustness | `coarsening_robustness.py --analyze` | `coarsening_robustness.csv` |

Figures are copied into the manuscript by `make sync`. **Table bodies are pasted into
`body.tex` / `appendix.tex` by hand** — the files in `tables/` are drop-in `tabular`
environments, so after regenerating them you must re-paste. This is the one manual link
in the chain, and a stale paste is how one table once drifted out of step with the
numbers behind it.

### Figures the code does not generate

Twelve raster figures in the paper are static assets with no generator in this
repository, and no command above regenerates them:

- the six per-domain coarsening panels (Fig. A3) — the network selection was made by hand
  and survives only as a note beside the images;
- the two residual panels (Fig. 10) — plotted in a notebook that never called `savefig`,
  from a 431-network merge that has since been superseded;
- `experimento2500.png`, `Adamic.png`, `Aritmeticopng.png` and the Figure 1 synthetic
  panel — produced by notebooks belonging to the earlier version of this project.

Those notebooks and their data were removed when the repository was reduced to the
reproduction path. They remain in git history at commit `bb806bf` if the earlier analyses
are ever needed.

## Method, in brief

**Coarsening.** Loukas local variation coarsening (`algorithm/coarsening_utils.py`),
applied repeatedly to reach 80/60/40/20% of the original node count. The default
candidate family is closed neighborhoods; `coarsening_robustness.py` compares it against
the edge-based family, and `--methods all` additionally offers three non-spectral
criteria that the paper does not use.

**Compression entropy.** SZIP structural encoding followed by arithmetic coding
(`algorithm/calculo_entropia.py`), normalized against Erdős–Rényi graphs matched in node
and edge count, averaged over ten draws.

**Link prediction.** Adamic–Adar and SEAL on a shared train/test split
(`algorithm/entropia_link_prediction.py`), each reporting AUC and a prediction entropy.
Ranks break ties uniformly at random: since Adamic–Adar scores zero for every pair with
no common neighbor, an optimistic or pessimistic convention changes the answer by an
order of magnitude on sparse graphs. Prediction entropy is normalized analytically as
`H* = H / (log₂N − 1)` after Sun et al. (2020), which is bounded in [0,1] by
construction. A simulated Erdős–Rényi baseline is not usable here: Adamic–Adar's ranks on
a sparse random graph collapse into the first bin, the baseline entropy tends to zero,
and the ratio diverges — a structural failure that more draws do not fix.

## Notes

- `results/real_networks_results_all.csv` is the consolidated experiment output and the
  only one the analysis reads. `run_real_networks_experiment.py --domain X` writes a
  per-domain file instead, which is useful for resuming a long run.
- `experiments/common.py` holds the shared constants, paths, figure style and loaders.
  Anything used by more than one script belongs there rather than being copied.
- Third-party code and data are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
