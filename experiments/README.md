# Experiments for the TKDD revision

Scripts added while preparing the ACM TKDD submission, in response to the Nature
Communications referee reports (NCOMMS-25-78191-T, Jan 2026). Run them in the order
below — step 1 gates everything else, because two defects in the existing analysis
must be fixed before any number goes into the manuscript.

All scripts assume the project venv:

```bash
source ~/venvs/research/bin/activate
```

They are resumable: each saves incrementally and skips work already on disk, so
interrupting and restarting is safe. Pass `--overwrite` to force a clean recompute.

---

## Step 1 — Fix the analysis (required first)

### 1a. Repair the Adamic–Adar baseline

The existing pipeline normalises AA prediction entropy by a **single** Erdős–Rényi
draw, while the manuscript specifies ten (which the compression path already does).
When that draw yields near-zero baseline entropy the ratio explodes: 210 of 602 values
exceed 2.0, maximum 102, against a median of 1.02. SEAL is unaffected.

```bash
python experiments/rebuild_regression.py --fix-aa-baseline
```

Writes `results/aa_baseline_fixed.csv`. Slow but CPU-only — no GPU needed. Check the
printed summary: the count above 2.0 should be near zero afterwards.

### 1b. Regression and figures — run the notebook

`algorithm/Entropy_Experiments/multiscale_entropy_regression.ipynb` runs locally and
is the primary path. Its merge is sound: against the current result files it yields
**431 networks across all six domains**. (Its *stored* outputs show n = 17 because it
was last executed on 26 Jun, before the per-domain CSVs completed on 30 Jun – 6 Jul.
Re-running it is all that is needed.)

Export the final cell's predicted-vs-actual panel — the one with `y = df['SEAL_auc']` —
as PDF to `imagenes/correlacion/seal_auc_predicted_vs_actual.pdf`. That is Figure 6 in
the manuscript. AUC is the primary target throughout the revision, being the metric
TKDD readers expect; prediction entropy is reported as secondary.

Then run the batch companion for the pieces the notebook does not cover:

```bash
python experiments/rebuild_regression.py
```

It reproduces the same merge (with a guard that warns if n < 400) and adds: drop-in
LaTeX tables in `tables/` for all three targets, the k=3 clustering behind Table 2, and
the coarsened-scale-only models.

One number the manuscript needs specifically: the printed line
`H_40 alone  R2=...  retains X% of Model 5 R2`. For SEAL AUC this is **R² = 0.566,
91% of Model 5 and above the full graph's 0.471** — the central claim of `sec:cost`,
which pairs it with the speedup from step 3.

---

## Step 2 — Coarsening robustness (Reviewer 2, point 1)

Tests whether the three entropy regimes are a property of the networks or an artifact
of Loukas' spectral coarsening. Recomputes the pipeline under five reduction
strategies and compares regime assignments by Adjusted Rand Index.

```bash
# ~120 networks x 5 methods x 4 levels; parallelise by domain across terminals
python experiments/coarsening_robustness.py --per-domain 20

# or:
for d in Biological Social Economic Transportation Technological Informational; do
    python experiments/coarsening_robustness.py --domain "$d" &
done; wait

python experiments/coarsening_robustness.py --analyze
```

The informative comparison is against `heavy_edge`, which optimises no spectral
criterion — agreement with it cannot be explained by shared low-frequency bias.

**Report whatever it shows.** If the minimum ARI against `variation_neighborhood` is
≥ 0.70, the regimes are method-invariant and the objection is closed. If it is lower,
the manuscript scopes the claim to spectral coarsening explicitly. The analysis step
prints which case applies. Do not omit an unfavourable result — a referee who runs
this will find it, and the scoped claim is still publishable.

Fills the placeholder in `body.tex`, section `sec:robustness`.

---

## Step 3 — Runtime and memory (Reviewer 2, point 4)

The manuscript asserts a computational benefit in four places and never measured it.

```bash
pip install psutil            # for real peak RSS; falls back to tracemalloc without it
python experiments/runtime_benchmark.py --n-networks 50 --repeats 3
python experiments/runtime_benchmark.py --analyze
```

Times coarsening, SZIP encoding, arithmetic coding and link prediction **separately**,
because they scale differently and an end-to-end number would hide which one limits
applicability. Reports per-stage exponents in `t ~ n^α`, cost per reduction level, the
median 100%-vs-40% speedup, and an extrapolated feasibility limit.

Report the speedup **paired with** the R² retained by `H_40` alone from step 1b. A
speedup without its accuracy cost is not a result. Flag the extrapolated limit as an
extrapolation, not a measurement.

Fills the placeholder in `body.tex`, section `sec:cost`.

---

## Step 4 — Figures

Steps 1–3 emit vector PDF directly into `figures/`. The figures still carried over from
the original submission are raster PNG (Reviewer 2, point 7) and need regenerating from
their source notebooks with `savefig(..., format='pdf', bbox_inches='tight')`:

| Manuscript figure | Source |
|---|---|
| `experimento2500` | `Network_Families/Analysis_Synthetic_Networks.ipynb` |
| six `grafos_Reales/*` domain panels | `Real_World_Networks/Analysis_Synthetic_Networks.ipynb` |
| `Kmeans3english` | now produced by `rebuild_regression.py` as `kmeans_pca.pdf` |
| `predictedvsactual_model{1,5}` | now produced by `rebuild_regression.py` |
| `model{1,5}_residuals` | `multiscale_entropy_regression.ipynb` |
| `entropiaLCyLP/{Aritmeticopng,Adamic,Jaccard}` | `LinkPrediction_Experiments/linkPredictionEntropy.ipynb` |

Regenerate the clustering and regression figures **after** step 1, since their numbers
change once the merge is fixed.

---

## Outputs consumed by the manuscript

| File | Used in |
|---|---|
| `tables/table_regression_*.tex` | Table 3 and its SEAL/AUC counterparts |
| `tables/table_cluster_composition.tex` | Table 2 |
| `results/coarsening_robustness_ari.csv` | `sec:robustness` |
| `results/runtime_benchmark_summary.csv`, `results/runtime_scaling_fits.csv` | `sec:cost` |
| `figures/*.pdf` | Figures 4–8 |
