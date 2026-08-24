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
import networkx as nx
from pygsp import graphs as pygsp_graphs
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

PKL_PATH = os.path.join(PROJECT_ROOT, 'algorithm', 'Entropy_Experiments',
                        'Real_World_Networks', 'all_networks.pkl')
MIN_NODES = 20
MIN_EDGES = 30
LEVELS = [100, 80, 60, 40, 20]
VALID_DOMAINS = ['Biological', 'Social', 'Economic',
                 'Transportation', 'Technological', 'Informational']
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

def build_graph(row):
    is_directed = 'Directed' in row['graphProperties']
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(np.array(row['nodes_id']))
    G.add_edges_from(np.array(row['edges_id']))
    G = nx.to_undirected(G)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


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

def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_CSV)
    stages = ['coarsen', 'encode', 'compress', 'linkpred']
    stages = [s for s in stages if s in set(df['stage'])]

    # ── mean cost per level ──────────────────────────────────────────────────
    summary = (df.groupby(['level', 'stage'])
                 .agg(seconds_mean=('seconds', 'mean'),
                      seconds_std=('seconds', 'std'),
                      peak_mb_mean=('peak_bytes', lambda s: s.mean() / 2**20),
                      n=('seconds', 'size'))
                 .round(4).reset_index())
    summary_path = OUT_CSV.replace('.csv', '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('Mean cost per reduction level and stage:')
    print(summary.to_string(index=False))
    print(f'\n→ {summary_path}')

    # ── scaling exponents: seconds ~ n^alpha, fitted on the 100% graphs ──────
    fits = []
    at100 = df[df['level'] == 100]
    for stage in stages:
        sub = at100[(at100['stage'] == stage) & (at100['seconds'] > 0)]
        sub = sub.groupby('n_nodes')['seconds'].mean().reset_index()
        if len(sub) < 5:
            continue
        x, y = np.log10(sub['n_nodes']), np.log10(sub['seconds'])
        alpha, intercept = np.polyfit(x, y, 1)
        resid = y - (alpha * x + intercept)
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
        fits.append({'stage': stage, 'exponent': round(alpha, 3),
                     'intercept_log10': round(intercept, 3),
                     'r2': round(r2, 3), 'n_points': len(sub)})
    fits_df = pd.DataFrame(fits)
    fits_path = OUT_CSV.replace('.csv', '').replace('runtime_benchmark',
                                                    'runtime_scaling_fits') + '.csv'
    fits_df.to_csv(fits_path, index=False)
    print('\nEmpirical scaling  seconds ~ n^exponent  (fitted at 100%):')
    print(fits_df.to_string(index=False))
    print(f'→ {fits_path}')

    # ── the headline number ──────────────────────────────────────────────────
    per_net = (df.groupby(['name', 'level'])['seconds'].sum()
                 .reset_index()
                 .pivot(index='name', columns='level', values='seconds')
                 .dropna(subset=[100, 40]))
    speedup = (per_net[100] / per_net[40]).replace([np.inf, -np.inf], np.nan).dropna()
    peak_net = (df.groupby(['name', 'level'])['peak_bytes'].max()
                  .reset_index()
                  .pivot(index='name', columns='level', values='peak_bytes')
                  .dropna(subset=[100, 40]))
    mem_ratio = (peak_net[100] / peak_net[40].replace(0, np.nan)).dropna()

    print('\n' + '=' * 72)
    print(f'End-to-end pipeline cost, 100% vs 40% ({len(speedup)} networks):')
    print(f'  median speedup : {speedup.median():.2f}x')
    print(f'  mean speedup   : {speedup.mean():.2f}x')
    print(f'  IQR            : {speedup.quantile(.25):.2f}x – {speedup.quantile(.75):.2f}x')
    if len(mem_ratio):
        print(f'  median peak-memory reduction : {mem_ratio.median():.2f}x')
    print('\nQuote this alongside the R^2 that Model 5 retains when predictability is')
    print('estimated from the 40% graph (see experiments/rebuild_regression.py).')
    print('=' * 72)

    # ── extrapolated feasibility limit ───────────────────────────────────────
    if len(fits_df):
        total_alpha = fits_df.loc[fits_df['stage'] != 'linkpred', 'exponent'].max()
        row = fits_df[fits_df['exponent'] == total_alpha].iloc[0]
        for budget_hours in (1, 24):
            budget_s = budget_hours * 3600
            n_max = 10 ** ((np.log10(budget_s) - row['intercept_log10']) / row['exponent'])
            print(f'Extrapolated limit for stage "{row["stage"]}" '
                  f'(alpha={row["exponent"]}): ~{n_max:,.0f} nodes in {budget_hours} h')
        print('Extrapolation assumes the fitted power law holds beyond the measured range;\n'
              'state it as an estimate in the manuscript, not a measurement.')

    # ── figures ──────────────────────────────────────────────────────────────
    figdir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 3.4), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        sub = at100[(at100['stage'] == stage) & (at100['seconds'] > 0)]
        g = sub.groupby('n_nodes')['seconds'].mean()
        ax.loglog(g.index, g.values, 'o', markersize=4, alpha=0.7)
        f = fits_df[fits_df['stage'] == stage]
        if len(f):
            xs = np.array([g.index.min(), g.index.max()])
            ax.loglog(xs, 10 ** f['intercept_log10'].iloc[0] * xs ** f['exponent'].iloc[0],
                      '-', linewidth=1.4,
                      label=f"$\\alpha$={f['exponent'].iloc[0]:.2f}")
            ax.legend(fontsize=8)
        ax.set_title(stage, fontsize=10)
        ax.set_xlabel('nodes')
        ax.set_ylabel('seconds')
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'runtime_scaling.pdf'),
                format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6.5, 4))
    piv = summary.pivot(index='level', columns='stage', values='seconds_mean')
    piv = piv.reindex(LEVELS)
    piv.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('nodes retained (%)')
    ax.set_ylabel('mean seconds per network')
    ax.set_title('Pipeline cost by reduction level', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'runtime_by_level.pdf'),
                format='pdf', bbox_inches='tight')
    print(f'\nFigures → {figdir}/runtime_{{scaling,by_level}}.pdf')


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
