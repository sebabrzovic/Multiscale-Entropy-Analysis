"""
Experiment B — Cost and scaling of the multiscale entropy pipeline.

Reviewer 2 (Nature Communications, NCOMMS-25-78191-T) noted that the manuscript
asserts a computational benefit from coarsening in four places without ever measuring
it: no runtime, no memory footprint, no complexity analysis. This script instruments
each stage of the pipeline separately and produces the numbers the claim needs.

Stages timed independently
--------------------------
    coarsen   spectral coarsening to a given reduction level
    encode    SZIP encoding of the (possibly coarsened) graph into B1/B2
    compress  arithmetic coding of the binary string
    linkpred  leave-one-out / held-out link prediction (Adamic-Adar)

The headline number the manuscript needs is the ratio
    (cost of the full pipeline at 100%) / (cost at 40%)
reported together with the R^2 retained when predictability is estimated from the
40% graph. Those two numbers together are the paper's practical claim.

Usage
-----
    source ~/venvs/research/bin/activate

    python experiments/runtime_benchmark.py --n-networks 50 --repeats 3
    python experiments/runtime_benchmark.py --analyze

Outputs
-------
    results/runtime_benchmark.csv          — tidy per (network, level, stage, repeat)
    results/runtime_benchmark_summary.csv  — mean cost per level and stage
    results/runtime_scaling_fits.csv       — fitted log-log exponents per stage
    figures/runtime_scaling.pdf            — cost vs n, log-log, one panel per stage
    figures/runtime_by_level.pdf           — cost per reduction level
"""

import argparse
import gc
import os
import sys
import threading
import time
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# The graph stack is needed only to *run* the benchmark. --analyze reads the CSV it
# produced and needs nothing beyond numpy/pandas/matplotlib, so a missing networkx or
# pygsp must not stop the analysis from being reproduced on another machine.
try:
    import networkx as nx
    from pygsp import graphs as pygsp_graphs
    from tqdm import tqdm
    _HAVE_GRAPH_STACK = True
except ImportError as _exc:                            # pragma: no cover
    nx = pygsp_graphs = tqdm = None
    _GRAPH_STACK_ERROR = _exc
    _HAVE_GRAPH_STACK = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments import common                             # noqa: E402
from experiments.common import PAPER_STYLE, build_graph    # noqa: E402

if _HAVE_GRAPH_STACK:
    from algorithm.coarsening_utils import coarsen
    from algorithm.calculo_entropia import (M_adyacencia, Encoder,
                                            get_optimized_compression_length)

# entropia_link_prediction imports torch, which the coarsen/encode/compress stages do
# not need. Imported lazily so --no-linkpred runs without the deep-learning stack.
elp = None


def _load_elp():
    global elp
    if elp is None:
        import algorithm.entropia_link_prediction as _elp
        elp = _elp
    return elp

try:
    import psutil
    _PROC = psutil.Process()
    _HAVE_PSUTIL = True
except ImportError:                                    # pragma: no cover
    _PROC = None
    _HAVE_PSUTIL = False

PKL_PATH = common.CORPUS_PKL
MIN_NODES = common.MIN_NODES
MIN_EDGES = common.MIN_EDGES
LEVELS = common.LEVELS
VALID_DOMAINS = common.DOMAINS
OUT_CSV = os.path.join(PROJECT_ROOT, 'results', 'runtime_benchmark.csv')


# ── measurement ──────────────────────────────────────────────────────────────

class _RSSSampler(threading.Thread):
    """Poll process RSS in the background to capture a stage's true peak.

    tracemalloc only sees allocations made through the CPython allocator, which
    misses most of what numpy and scipy allocate. Sampling RSS catches those, at
    the cost of missing peaks shorter than the polling interval — acceptable here
    because the stages we time run for milliseconds to seconds, not microseconds.
    """

    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        # NOT `self._stop`: threading.Thread already defines _stop() as a method and
        # calls it internally when the thread finishes. Shadowing it with an Event
        # makes that internal call raise "'Event' object is not callable".
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak = max(self.peak, _PROC.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)
        return self.peak


def measure(fn):
    """Run fn(); return (result, seconds, peak_bytes).

    peak_bytes is the increase in peak process RSS over the baseline taken just
    before the call (psutil path), or peak traced Python allocation (fallback).
    """
    gc.collect()
    if _HAVE_PSUTIL:
        base = _PROC.memory_info().rss
        sampler = _RSSSampler()
        sampler.start()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        peak = sampler.stop()
        return result, elapsed, max(0, peak - base)

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


# ── graph handling ───────────────────────────────────────────────────────────

def size_stratified_sample(networks_df, n_networks, seed):
    """Pick networks spread log-uniformly over the node-count range.

    Log-uniform spacing is what makes the log-log scaling fit meaningful: a sample
    dominated by 100-node graphs cannot constrain the exponent.
    """
    df = networks_df.copy()
    df['_n'] = df['nodes_id'].apply(len)
    df = df[df['_n'] >= MIN_NODES].sort_values('_n').reset_index(drop=True)
    if len(df) <= n_networks:
        return df.drop(columns='_n')

    targets = np.logspace(np.log10(df['_n'].iloc[0]),
                          np.log10(df['_n'].iloc[-1]), n_networks)
    idx = sorted({int(np.abs(df['_n'].values - t).argmin()) for t in targets})
    return df.iloc[idx].drop(columns='_n').reset_index(drop=True)


# ── the benchmark ────────────────────────────────────────────────────────────

def bench_network(G, name, domain, repeats, test_ratio, seed, do_linkpred):
    """Return a list of tidy rows for one network, across all reduction levels."""
    rows = []
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    base = dict(domain=domain, name=name, n_nodes_original=n0, n_edges_original=e0)

    W = nx.to_scipy_sparse_array(G)
    Gp = pygsp_graphs.Graph(W)

    for level in LEVELS:
        for rep in range(repeats):
            # ── stage 1: coarsen ────────────────────────────────────────────
            if level == 100:
                G_lvl, t_coarsen, m_coarsen = G, 0.0, 0
            else:
                try:
                    (out, t_coarsen, m_coarsen) = measure(
                        lambda: coarsen(Gp, K=10, r=1 - level / 100))
                    G_lvl = nx.from_scipy_sparse_array(out[1].W)
                except Exception as exc:
                    tqdm.write(f'    coarsen failed at {level}%: {exc}')
                    continue

            n_lvl, e_lvl = G_lvl.number_of_nodes(), G_lvl.number_of_edges()
            row_base = {**base, 'level': level, 'repeat': rep,
                        'n_nodes': n_lvl, 'n_edges': e_lvl}
            rows.append({**row_base, 'stage': 'coarsen',
                         'seconds': t_coarsen, 'peak_bytes': m_coarsen})

            # ── stage 2: SZIP encoding ──────────────────────────────────────
            try:
                (cod, t_enc, m_enc) = measure(lambda: Encoder(M_adyacencia(G_lvl)))
                B1, B2, _ = cod
                binary = B1 + B2
            except Exception as exc:
                tqdm.write(f'    encode failed at {level}%: {exc}')
                continue
            rows.append({**row_base, 'stage': 'encode',
                         'seconds': t_enc, 'peak_bytes': m_enc,
                         'bits': len(binary)})

            # ── stage 3: arithmetic coding ──────────────────────────────────
            try:
                (_, t_cmp, m_cmp) = measure(
                    lambda: get_optimized_compression_length(binary))
            except Exception as exc:
                tqdm.write(f'    compress failed at {level}%: {exc}')
                continue
            rows.append({**row_base, 'stage': 'compress',
                         'seconds': t_cmp, 'peak_bytes': m_cmp})

            # ── stage 4: link prediction ────────────────────────────────────
            if do_linkpred and e_lvl >= MIN_EDGES:
                _load_elp()
                try:
                    def _lp():
                        _, train, test = elp.split_edges(G_lvl, test_ratio=test_ratio,
                                                         seed=seed)
                        return elp.evaluate_link_prediction_heuristic(
                            G_lvl, train, test, predictor='adamic_adar', seed=seed)
                    (_, t_lp, m_lp) = measure(_lp)
                    rows.append({**row_base, 'stage': 'linkpred',
                                 'seconds': t_lp, 'peak_bytes': m_lp})
                except Exception as exc:
                    tqdm.write(f'    linkpred failed at {level}%: {exc}')
    return rows


def run_bench(args):
    if not _HAVE_GRAPH_STACK:
        raise SystemExit(f'Running the benchmark needs networkx/pygsp: {_GRAPH_STACK_ERROR}')
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if not _HAVE_PSUTIL:
        print('psutil not installed — memory figures fall back to tracemalloc and will\n'
              'undercount numpy/scipy allocations. `pip install psutil` for real RSS.\n')

    networks_df = pd.read_pickle(args.corpus or PKL_PATH)
    sample = size_stratified_sample(networks_df, args.n_networks, args.seed)
    sizes = sample['nodes_id'].apply(len)
    print(f'Benchmarking {len(sample)} networks, '
          f'{sizes.min()} – {sizes.max()} nodes, {args.repeats} repeat(s) each')
    print(f'Levels: {LEVELS}\n')

    done, rows = set(), []
    if os.path.exists(OUT_CSV) and not args.overwrite:
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict('records')
        done = set(prev['name'])
        print(f'Resuming: {len(done)} networks already benchmarked.\n')

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc='networks'):
        name = row.get('network_name', row.get('title', 'unknown'))
        if name in done:
            continue
        domain = row['networkDomain']
        G = build_graph(row)
        if G.number_of_nodes() < MIN_NODES or G.number_of_edges() < MIN_EDGES:
            continue
        tqdm.write(f'\n[{domain}] {name}  '
                   f'({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
        rows += bench_network(G, name, domain, args.repeats,
                              args.test_ratio, args.seed, not args.no_linkpred)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f'\nSaved {len(rows)} rows → {OUT_CSV}')
    print('Now run:  python experiments/runtime_benchmark.py --analyze')


# ── analysis ─────────────────────────────────────────────────────────────────


STAGE_ORDER = ['coarsen', 'encode', 'compress', 'linkpred']
STAGE_LABEL = {'coarsen': 'Coarsening', 'encode': 'SZIP encoding',
               'compress': 'Arithmetic coding', 'linkpred': 'Link prediction'}
STAGE_COLOR = {'coarsen': '#568f8b', 'encode': '#1d4a60',
               'compress': '#cd7e59', 'linkpred': '#d15252'}

# Coarsening costs nothing at 100% by definition, so its scaling is fitted at the
# 40% level, where the measured time is the full cost of reducing the original
# graph to that size. Every other stage is fitted on the uncoarsened graph.
FIT_LEVEL = {'coarsen': 40, 'encode': 100, 'compress': 100, 'linkpred': 100}


def _loglog_fit(x, y):
    """Fit log10(y) = alpha*log10(x) + c. Returns (alpha, c, r2)."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    alpha, c = np.polyfit(lx, ly, 1)
    resid = ly - (alpha * lx + c)
    r2 = 1 - resid.var() / ly.var() if ly.var() > 0 else np.nan
    return alpha, c, r2


def _invert(alpha, c, budget):
    """Largest n for which the fitted power law stays under `budget`."""
    return 10 ** ((np.log10(budget) - c) / alpha)


def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)

    df = pd.read_csv(OUT_CSV)
    stages = [s for s in STAGE_ORDER if s in set(df['stage'])]
    figdir = os.path.join(PROJECT_ROOT, 'figures')
    tabdir = os.path.join(PROJECT_ROOT, 'tables')
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    n_net = df['name'].nunique()
    n_lo, n_hi = df['n_nodes_original'].min(), df['n_nodes_original'].max()
    print(f'{n_net} networks, {n_lo}-{n_hi} nodes, '
          f'{df["repeat"].nunique()} repeats per configuration.')

    # ── mean cost per level ──────────────────────────────────────────────────
    summary = (df.groupby(['level', 'stage'])
                 .agg(seconds_mean=('seconds', 'mean'),
                      seconds_std=('seconds', 'std'),
                      peak_mb_mean=('peak_bytes', lambda s: s.mean() / 2**20),
                      n=('seconds', 'size'))
                 .round(4).reset_index())
    summary_path = OUT_CSV.replace('.csv', '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('\nMean cost per reduction level and stage:')
    print(summary.to_string(index=False))
    print(f'-> {summary_path}')

    # Per (network, level): total pipeline seconds, and the peak of any one stage.
    per_net = (df.groupby(['name', 'level', 'repeat'])
                 .agg(sec=('seconds', 'sum'), pk=('peak_bytes', 'max'))
                 .groupby(['name', 'level']).mean())
    tot = per_net['sec'].unstack('level')
    pk = per_net['pk'].unstack('level')

    # ── scaling exponents ────────────────────────────────────────────────────
    fits = []
    for stage in stages:
        lvl = FIT_LEVEL[stage]
        sub = df[(df['stage'] == stage) & (df['level'] == lvl) & (df['seconds'] > 0)]
        g = sub.groupby('n_nodes_original')['seconds'].mean()
        if len(g) < 5:
            continue
        a_t, c_t, r2_t = _loglog_fit(g.index, g.values)
        gm = (df[(df['stage'] == stage) & (df['level'] == lvl) & (df['peak_bytes'] > 0)]
              .groupby('n_nodes_original')['peak_bytes'].max())
        a_m, c_m, r2_m = _loglog_fit(gm.index, gm.values / 2**20) if len(gm) >= 5 \
            else (np.nan, np.nan, np.nan)
        fits.append({'stage': stage, 'fit_level': lvl,
                     'time_exponent': round(a_t, 3), 'time_intercept_log10': round(c_t, 3),
                     'time_r2': round(r2_t, 3),
                     'n_max_1h': round(_invert(a_t, c_t, 3600)),
                     'mem_exponent': round(a_m, 3), 'mem_intercept_log10': round(c_m, 3),
                     'mem_r2': round(r2_m, 3),
                     'n_max_8gb': round(_invert(a_m, c_m, 8 * 1024)),
                     'n_points': len(g)})
    fits_df = pd.DataFrame(fits)
    fits_path = os.path.join(PROJECT_ROOT, 'results', 'runtime_scaling_fits.csv')
    fits_df.to_csv(fits_path, index=False)
    print('\nEmpirical scaling (time ~ n^a, peak memory ~ n^b):')
    print(fits_df.to_string(index=False))
    print(f'-> {fits_path}')

    # ── headline numbers ─────────────────────────────────────────────────────
    both = tot.dropna(subset=[100, 40])
    sp = both[100] / both[40]
    sizes = df.groupby('name')['n_nodes_original'].first().reindex(both.index)
    a_s, c_s, r2_s = _loglog_fit(sizes.values, sp.values)
    crossover = 10 ** (-c_s / a_s)

    pk_both = pk.dropna(subset=[100, 40])
    mem_ratio = (pk_both[100] / pk_both[40].replace(0, np.nan)).dropna()

    # The substitution the paper actually proposes: compute entropy on the 40%
    # graph instead of running the predictor on the full graph.
    ent = (df[df['stage'] != 'linkpred']
           .groupby(['name', 'level', 'repeat'])['seconds'].sum()
           .groupby(['name', 'level']).mean().unstack('level'))
    lp100 = (df[(df['stage'] == 'linkpred') & (df['level'] == 100)]
             .groupby('name')['seconds'].mean())
    sub_idx = ent.index.intersection(lp100.index)
    subst = (lp100.reindex(sub_idx) / ent[40].reindex(sub_idx)).dropna()

    tercile = {}
    for lo, hi, lab in [(0, 100, '<100'), (100, 500, '100-500'), (500, 10**9, '>=500')]:
        m = (sizes >= lo) & (sizes < hi)
        if m.sum():
            tercile[lab] = (int(m.sum()), float(sp[m].median()))

    head = {
        'n_networks': n_net, 'n_nodes_min': int(n_lo), 'n_nodes_max': int(n_hi),
        'speedup_median': round(float(sp.median()), 3),
        'speedup_q25': round(float(sp.quantile(.25)), 3),
        'speedup_q75': round(float(sp.quantile(.75)), 3),
        'speedup_aggregate': round(float(both[100].sum() / both[40].sum()), 3),
        'speedup_exponent': round(a_s, 3), 'speedup_fit_r2': round(r2_s, 3),
        'crossover_nodes': round(crossover),
        'n_for_2x': round(_invert(a_s, c_s, 2)),
        'mem_ratio_median': round(float(mem_ratio.median()), 2),
        'peak_mb_median_100': round(float(pk_both[100].median()) / 2**20, 3),
        'peak_mb_median_40': round(float(pk_both[40].median()) / 2**20, 3),
        'substitution_median': round(float(subst.median()), 2),
        'linkpred_share_at_100': round(float(
            df[(df.level == 100) & (df.stage == 'linkpred')]['seconds'].mean()
            / df[df.level == 100].groupby('stage')['seconds'].mean().sum()), 3),
    }
    for lab, (k, med) in tercile.items():
        head[f'speedup_median_n{lab}'] = round(med, 3)
        head[f'k_n{lab}'] = k
    # The substitution ratio also grows with size; report it on the large graphs,
    # where the choice actually matters.
    big = sizes.reindex(subst.index) >= 500
    if big.sum():
        head['substitution_median_n>=500'] = round(float(subst[big].median()), 2)
    head_path = os.path.join(PROJECT_ROOT, 'results', 'runtime_headline.csv')
    pd.Series(head).rename('value').to_csv(head_path, index_label='quantity')

    print('\n' + '=' * 72)
    print(f'End-to-end pipeline cost, 100% vs 40% ({len(sp)} networks):')
    print(f'  median speedup {sp.median():.2f}x  (IQR {sp.quantile(.25):.2f}-'
          f'{sp.quantile(.75):.2f}), aggregate {both[100].sum()/both[40].sum():.2f}x')
    print(f'  speedup ~ n^{a_s:.2f} (R2={r2_s:.2f}); breaks even at n ~ {crossover:.0f} nodes')
    for lab, (k, med) in tercile.items():
        print(f'    n {lab:8s} (k={k:2d}): median {med:.2f}x')
    print(f'  median peak-memory reduction {mem_ratio.median():.1f}x '
          f'({pk_both[100].median()/2**20:.2f} MB -> {pk_both[40].median()/2**20:.3f} MB)')
    print(f'  entropy at 40% vs running the predictor at 100%: '
          f'{subst.median():.1f}x cheaper overall, '
          f'{subst[big].median():.1f}x on the {int(big.sum())} networks with n >= 500')
    print(f'  link prediction is {head["linkpred_share_at_100"]:.0%} of pipeline time at 100%')
    print(f'-> {head_path}')
    print('=' * 72)

    # ── LaTeX tables ─────────────────────────────────────────────────────────
    piv_t = summary.pivot(index='level', columns='stage', values='seconds_mean').reindex(LEVELS)
    tot_mean = tot.mean().reindex(LEVELS)
    pk_mean = (pk.mean() / 2**20).reindex(LEVELS)
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule',
             r'\textit{Retained} & \multicolumn{4}{c}{\textit{Mean seconds per stage}} '
             r'& \textit{Total} & \textit{Peak} \\',
             r'\cmidrule(lr){2-5}',
             r'(\%) & ' + ' & '.join(STAGE_LABEL[s] for s in stages) +
             r' & (s) & (MB) \\', r'\midrule']
    for lv in LEVELS:
        cells = [f'{piv_t.loc[lv, s]:.3f}' if s in piv_t.columns and
                 not pd.isna(piv_t.loc[lv, s]) else '---' for s in stages]
        lines.append(f'{lv} & ' + ' & '.join(cells) +
                     f' & {tot_mean[lv]:.3f} & {pk_mean[lv]:.2f} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    common.write_table('table_runtime_cost.tex', lines)

    lines = [r'\begin{tabular}{lccccc}', r'\toprule',
             r'\textit{Stage} & \textit{Time} $\alpha$ & $R^2$ & '
             r'\textit{Memory} $\beta$ & $R^2$ & \textit{Nodes in 8\,GB} \\',
             r'\midrule']
    for _, r in fits_df.iterrows():
        nm = f"{r['n_max_8gb']:,.0f}" if np.isfinite(r['mem_exponent']) and \
            r['mem_r2'] > 0.5 else '---'
        beta = f"{r['mem_exponent']:.2f}" if r['mem_r2'] > 0.5 else '---'
        br2 = f"{r['mem_r2']:.2f}" if r['mem_r2'] > 0.5 else '---'
        lines.append(f"{STAGE_LABEL[r['stage']]} & {r['time_exponent']:.2f} & "
                     f"{r['time_r2']:.2f} & {beta} & {br2} & {nm} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    common.write_table('table_runtime_scaling.tex', lines)
    print(f'-> {tabdir}/table_runtime_{{cost,scaling}}.tex')

    # ── figure: scaling + speedup ────────────────────────────────────────────
    # Panels 1-2 share the stage colours, so the stage legend is drawn once for the
    # figure. The fitted exponents live in table_runtime_scaling.tex rather than in
    # per-panel legends, which at this width cover the data.
    # 5.48 in is the acmsmall text width, so the figure is placed at 1:1 and the
    # 8 pt labels below render at 8 pt rather than being scaled down by LaTeX.
    fig, axes = plt.subplots(1, 3, figsize=(5.48, 2.15))

    ax = axes[0]
    handles = []
    for stage in stages:
        lvl = FIT_LEVEL[stage]
        g = (df[(df['stage'] == stage) & (df['level'] == lvl) & (df['seconds'] > 0)]
             .groupby('n_nodes_original')['seconds'].mean())
        ax.loglog(g.index, g.values, 'o', ms=2.6, alpha=0.5,
                  color=STAGE_COLOR[stage], mec='none')
        f = fits_df[fits_df['stage'] == stage]
        if len(f):
            xs = np.array([g.index.min(), g.index.max()], float)
            ln, = ax.loglog(xs, 10 ** f['time_intercept_log10'].iloc[0] *
                            xs ** f['time_exponent'].iloc[0], '-', lw=1.1,
                            color=STAGE_COLOR[stage], label=STAGE_LABEL[stage])
            handles.append(ln)
    ax.set_xlabel('Nodes'); ax.set_ylabel('Seconds')
    ax.set_title('Runtime by stage')

    ax = axes[1]
    for stage in stages:
        lvl = FIT_LEVEL[stage]
        g = (df[(df['stage'] == stage) & (df['level'] == lvl) & (df['peak_bytes'] > 0)]
             .groupby('n_nodes_original')['peak_bytes'].max())
        ax.loglog(g.index, g.values / 2**20, 'o', ms=2.6, alpha=0.5,
                  color=STAGE_COLOR[stage], mec='none')
        f = fits_df[(fits_df['stage'] == stage) & (fits_df['mem_r2'] > 0.5)]
        if len(f):
            xs = np.array([g.index.min(), g.index.max()], float)
            ax.loglog(xs, 10 ** f['mem_intercept_log10'].iloc[0] *
                      xs ** f['mem_exponent'].iloc[0], '-', lw=1.1,
                      color=STAGE_COLOR[stage])
    ax.set_xlabel('Nodes'); ax.set_ylabel('Peak memory (MB)')
    ax.set_title('Memory by stage')

    ax = axes[2]
    ax.scatter(sizes.values, sp.values, s=9, alpha=0.6, color='#1d4a60',
               edgecolors='none')
    xs = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 50)
    ax.plot(xs, 10 ** c_s * xs ** a_s, '-', lw=1.1, color='#cd7e59')
    ax.axhline(1.0, color='0.35', lw=0.7, ls='--')
    ax.axvline(crossover, color='0.35', lw=0.7, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Nodes'); ax.set_ylabel(r'Speedup ($100\%\!\rightarrow\!40\%$)')
    # No legend: at this panel width it covers the fit whichever corner it goes in.
    # The exponent and break-even size are stated in the caption and the text.
    ax.set_title('End-to-end speedup')

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(w_pad=1.1, rect=(0, 0.10, 1, 1))
    fig.legend(handles=handles, loc='lower center', ncol=2,
               frameon=False, handlelength=1.4, columnspacing=1.8,
               bbox_to_anchor=(0.5, -0.05))
    fig.savefig(os.path.join(figdir, 'runtime_scaling.pdf'), format='pdf',
                bbox_inches='tight')
    plt.close(fig)

    # ── figure: cost composition by level ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    piv = piv_t[[s for s in stages if s in piv_t.columns]]
    bottom = np.zeros(len(LEVELS))
    xs = np.arange(len(LEVELS))
    for stage in stages:
        if stage not in piv.columns:
            continue
        vals = piv[stage].values
        ax.bar(xs, vals, 0.62, bottom=bottom, color=STAGE_COLOR[stage],
               label=STAGE_LABEL[stage], edgecolor='white', linewidth=0.4)
        bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in LEVELS])
    ax.set_xlabel('Nodes retained (\\%)' if plt.rcParams['text.usetex']
                  else 'Nodes retained (%)')
    ax.set_ylabel('Mean seconds per network')
    ax.legend(frameon=False, handlelength=1.0, borderpad=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'runtime_by_level.pdf'), format='pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Figures -> {figdir}/runtime_{{scaling,by_level}}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-networks', type=int, default=50,
                   help='Networks in the log-uniform size-stratified sample')
    p.add_argument('--repeats', type=int, default=3, help='Timing repetitions')
    p.add_argument('--test-ratio', type=float, default=0.2)
    p.add_argument('--corpus', default=None,
                   help='Path to a corpus pickle. Defaults to the undirected-only'
                        ' corpus; pass all_networks.pkl (see build_full_corpus.py)'
                        ' to include symmetrized directed networks.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-linkpred', action='store_true',
                   help='Skip the link-prediction stage (much faster)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()

    analyze(args) if args.analyze else run_bench(args)
