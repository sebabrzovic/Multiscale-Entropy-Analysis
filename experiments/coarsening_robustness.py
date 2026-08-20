"""
Experiment A — Robustness of multiscale entropy to the choice of graph reduction.

Reviewer 2 (Nature Communications, NCOMMS-25-78191-T) objected that all multiscale
representations in the paper come from Loukas' spectral coarsening, which is designed
to preserve low-frequency Laplacian components. The reported entropy regimes may
therefore be an artefact of that particular algorithm rather than a property of the
networks. This script recomputes the whole multiscale entropy pipeline under more than
one reduction strategy, on the real-world corpus, and measures whether the k=3 regime
assignment survives.

Scope of the default run
------------------------
By default it compares the two LOCAL VARIATION algorithms:

    variation_neighborhood   candidate sets = closed neighborhoods {v} u N(v)
    variation_edges          candidate sets = individual edges

These differ only in their candidate family; both minimize the same restricted
spectral similarity objective. Agreement between them therefore shows the regimes do
not depend on which vertex sets are offered to the optimizer -- a real test, since the
two families produce very different contraction granularity and per-level reduction
ratios -- but it does NOT show the regimes are independent of spectral coarsening as
such. For the stronger claim, add the non-spectral baselines:

    python experiments/coarsening_robustness.py --methods all

which additionally runs heavy_edge, algebraic_JC and affinity_GS. Do not describe the
default run as establishing method-invariance in general; the honest claim is
invariance to the candidate family.

The decisive statistic is the Adjusted Rand Index between the k=3 regime clustering
obtained under each method and the one obtained under `variation_neighborhood` (the
method used in the paper). High ARI => the regimes survive the change.

Usage
-----
    source ~/venvs/research/bin/activate

    # 1. Run the sweep (resumable; safe to interrupt and restart)
    python experiments/coarsening_robustness.py --per-domain 20

    # parallelise across terminals by domain
    python experiments/coarsening_robustness.py --domain Biological &
    python experiments/coarsening_robustness.py --domain Social     &

    # wider comparison including non-spectral baselines
    python experiments/coarsening_robustness.py --methods all

    # 2. Analyse once the sweep is complete
    python experiments/coarsening_robustness.py --analyze

Outputs
-------
    results/coarsening_robustness.csv        — tidy per (network, method, level) entropy
    results/coarsening_robustness_ari.csv    — Adjusted Rand Index matrix
    results/coarsening_robustness_rho.csv    — mean Spearman rho between trajectories
    figures/coarsening_robustness_ari.pdf    — ARI heatmap (omitted for a 2-method run)
    figures/coarsening_robustness_traj.pdf   — mean trajectory per domain, one line per method
"""

import argparse
import os
import sys
import time
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

from algorithm.coarsening_utils import coarsen, get_entropy_metadata_aritmethicEncoding

# ── constants ────────────────────────────────────────────────────────────────
PKL_PATH = os.path.join(PROJECT_ROOT, 'algorithm', 'Entropy_Experiments',
                        'Real_World_Networks', 'undirected_networks.pkl')

# Same filtering as run_real_networks_experiment.py, so the samples are comparable.
MIN_NODES = 20
MIN_EDGES = 30

REDUCTION_LEVELS = [80, 60, 40, 20]
VALID_DOMAINS = ['Biological', 'Social', 'Economic',
                 'Transportation', 'Technological', 'Informational']

# The reference method is listed first: every ARI is computed against it.
#
# Default is the two local variation algorithms, which differ only in their
# candidate family (closed neighborhoods vs. individual edges) while minimizing
# the same restricted spectral similarity objective. This tests sensitivity to
# the candidate family, NOT to the spectral criterion itself -- see the note in
# the module docstring.
DEFAULT_METHODS = [
    'variation_neighborhood',   # used in the paper (Loukas, JMLR 2019)
    'variation_edges',          # Loukas, edge-based local variation
]

# Available for a wider comparison via --methods; these optimize non-spectral
# criteria and would test the stronger claim.
NON_SPECTRAL_METHODS = [
    'heavy_edge',               # classical heavy-edge matching
    'algebraic_JC',             # algebraic distance, Jacobi relaxation
    'affinity_GS',              # affinity, Gauss-Seidel
]

METHODS = list(DEFAULT_METHODS)          # rebound from --methods at startup
REFERENCE_METHOD = DEFAULT_METHODS[0]

OUT_CSV = os.path.join(PROJECT_ROOT, 'results', 'coarsening_robustness.csv')


# ── graph construction (mirrors run_real_networks_experiment.py) ─────────────

def build_graph(row):
    """Largest connected component of the undirected version, relabelled 0..n-1."""
    is_directed = 'Directed' in row['graphProperties']
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(np.array(row['nodes_id']))
    G.add_edges_from(np.array(row['edges_id']))
    G = nx.to_undirected(G)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def stratified_sample(networks_df, per_domain, seed):
    """Pick `per_domain` networks per domain, spread evenly across the size range.

    Even spacing over the size-sorted list (rather than a uniform random draw) keeps
    small and large networks represented in every domain, so a method that degrades
    only on large graphs cannot hide behind a small-graph-heavy sample.
    """
    rng = np.random.default_rng(seed)
    picked = []
    for domain in VALID_DOMAINS:
        sub = networks_df[networks_df['networkDomain'] == domain]
        if len(sub) == 0:
            continue
        sizes = sub['nodes_id'].apply(len)
        sub = sub.assign(_n=sizes).sort_values('_n')
        if len(sub) <= per_domain:
            picked.append(sub)
            continue
        idx = np.linspace(0, len(sub) - 1, per_domain).round().astype(int)
        idx = np.unique(idx)
        # jitter within the size-sorted list so repeated runs are not identical
        jitter = rng.integers(-1, 2, size=len(idx))
        idx = np.clip(idx + jitter, 0, len(sub) - 1)
        picked.append(sub.iloc[np.unique(idx)])
    return pd.concat(picked).drop(columns='_n', errors='ignore').reset_index(drop=True)


# ── the sweep ────────────────────────────────────────────────────────────────

def entropy_at_levels(G, method):
    """Normalised compression entropy at 100% and each reduction level, for one method.

    Returns (dict {level: entropy_or_None}, dict {level: coarsen_seconds}).
    The 100% entropy does not depend on the method but is recorded per method anyway
    so each row is self-contained; it is cheap relative to the coarsening itself.
    """
    ent, secs = {}, {}
    try:
        ent[100] = get_entropy_metadata_aritmethicEncoding(G)['Entropy Normalizado']
    except Exception as exc:
        tqdm.write(f'      entropy failed at 100%: {exc}')
        ent[100] = None
    secs[100] = 0.0

    W = nx.to_scipy_sparse_array(G)
    Gp = pygsp_graphs.Graph(W)

    for pct in REDUCTION_LEVELS:
        t0 = time.perf_counter()
        try:
            _, Gc, _, _ = coarsen(Gp, K=10, r=1 - pct / 100, method=method)
            G_red = nx.from_scipy_sparse_array(Gc.W)
            ent[pct] = get_entropy_metadata_aritmethicEncoding(G_red)['Entropy Normalizado']
        except Exception as exc:
            tqdm.write(f'      {method} failed at {pct}%: {exc}')
            ent[pct] = None
        secs[pct] = time.perf_counter() - t0
    return ent, secs


def run_sweep(args):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    networks_df = pd.read_pickle(PKL_PATH)
    if args.domain:
        networks_df = networks_df[networks_df['networkDomain'] == args.domain]
        networks_df = networks_df.reset_index(drop=True)
        if len(networks_df) == 0:
            sys.exit(f'No networks for domain "{args.domain}". Valid: {VALID_DOMAINS}')

    sample = stratified_sample(networks_df, args.per_domain, args.seed)
    print(f'Sampled {len(sample)} networks: '
          f'{sample["networkDomain"].value_counts().to_dict()}')
    print(f'Methods: {METHODS}')
    print(f'Levels : {REDUCTION_LEVELS}\n')

    # Resume: skip (name, method) pairs already present on disk.
    done = set()
    rows = []
    if os.path.exists(OUT_CSV) and not args.overwrite:
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict('records')
        done = set(zip(prev['name'], prev['method']))
        print(f'Resuming: {len(done)} (network, method) pairs already computed.\n')

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc='networks'):
        domain = row['networkDomain']
        name = row.get('network_name', row.get('title', 'unknown'))

        G = build_graph(row)
        n, e = G.number_of_nodes(), G.number_of_edges()
        if n < MIN_NODES or e < MIN_EDGES:
            tqdm.write(f'[{domain}] {name} — skipped ({n} nodes, {e} edges)')
            continue

        tqdm.write(f'\n[{domain}] {name}  ({n} nodes, {e} edges)')
        base = dict(domain=domain, name=name, n_nodes=n, n_edges=e)

        for method in METHODS:
            if (name, method) in done:
                tqdm.write(f'    {method:24s} — cached')
                continue
            tqdm.write(f'    {method:24s} ...')
            ent, secs = entropy_at_levels(G, method)
            traj = '  '.join(f'H_{lv}={ent[lv]:.3f}' if ent[lv] is not None else f'H_{lv}=None'
                             for lv in sorted(ent, reverse=True))
            tqdm.write(f'      {traj}')
            for lv in sorted(ent, reverse=True):
                rows.append({**base, 'method': method, 'level': lv,
                             'value': ent[lv], 'coarsen_seconds': secs[lv]})
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # incremental save

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f'\nSaved {len(rows)} rows → {OUT_CSV}')
    print('Now run:  python experiments/coarsening_robustness.py --analyze')


# ── analysis ─────────────────────────────────────────────────────────────────

def analyze(args):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_CSV)
    levels = [100, 80, 60, 40, 20]

    # Analyse whatever methods are actually present, ordered with the reference
    # first, so this works for a 2-method run and a 5-method run alike.
    present = list(dict.fromkeys(df['method'].tolist()))
    methods = ([REFERENCE_METHOD] if REFERENCE_METHOD in present else []) + \
              [m for m in present if m != REFERENCE_METHOD]
    if len(methods) < 2:
        sys.exit(f'Need at least two methods in {OUT_CSV}; found {methods}.')
    print(f'Methods present: {methods}')

    # Wide table: one row per (name, method), one column per level.
    wide = df.pivot_table(index=['domain', 'name', 'method'],
                          columns='level', values='value').reset_index()
    wide = wide.dropna(subset=levels)
    print(f'Complete trajectories: {len(wide)} (network, method) pairs')

    # Only networks that succeeded under *every* method can enter the comparison,
    # otherwise the clusterings would be computed on different sets of graphs.
    counts = wide.groupby('name')['method'].nunique()
    common = set(counts[counts == len(methods)].index)
    wide = wide[wide['name'].isin(common)]
    print(f'Networks with all {len(methods)} methods: {len(common)}')
    if len(common) < 20:
        print('WARNING: fewer than 20 networks common to all methods — ARI will be noisy.')

    # ── regime clustering per method ─────────────────────────────────────────
    labels = {}
    for method in methods:
        sub = wide[wide['method'] == method].sort_values('name')
        X = StandardScaler().fit_transform(sub[levels].values)
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        labels[method] = pd.Series(km.labels_, index=sub['name'].values)

    ari = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            common_names = labels[a].index.intersection(labels[b].index)
            ari.loc[a, b] = adjusted_rand_score(labels[a][common_names],
                                                labels[b][common_names])
    ari_path = OUT_CSV.replace('.csv', '_ari.csv')
    ari.round(4).to_csv(ari_path)
    print(f'\nAdjusted Rand Index vs {REFERENCE_METHOD}:')
    print(ari[REFERENCE_METHOD].round(3).to_string())
    print(f'\nFull ARI matrix → {ari_path}')

    # ── trajectory agreement ─────────────────────────────────────────────────
    rho = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        wa = wide[wide['method'] == a].set_index('name')[levels]
        for b in methods:
            wb = wide[wide['method'] == b].set_index('name')[levels]
            idx = wa.index.intersection(wb.index)
            per_network = [spearmanr(wa.loc[i].values, wb.loc[i].values).correlation
                           for i in idx]
            rho.loc[a, b] = np.nanmean(per_network)
    rho_path = OUT_CSV.replace('.csv', '_rho.csv')
    rho.round(4).to_csv(rho_path)
    print(f'Mean per-network Spearman rho matrix → {rho_path}')

    # ── figures (vector PDF — Reviewer 2 point 7) ────────────────────────────
    figdir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(ari.values.astype(float), vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(methods, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = ari.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if v < 0.6 else 'black', fontsize=8)
    ax.set_title('Regime agreement across reduction methods\n(Adjusted Rand Index, k=3)',
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'coarsening_robustness_ari.pdf'),
                format='pdf', bbox_inches='tight')

    domains = sorted(wide['domain'].unique())
    ncols = 3
    nrows = int(np.ceil(len(domains) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, domain in zip(axes, domains):
        sub = wide[wide['domain'] == domain]
        for method in methods:
            m = sub[sub['method'] == method][levels].mean()
            ax.plot(levels, m.values, marker='o', label=method, linewidth=1.4)
        ax.set_title(domain, fontsize=10)
        ax.invert_xaxis()
        ax.set_xlabel('nodes retained (%)')
        ax.set_ylabel('normalised entropy')
    for ax in axes[len(domains):]:
        ax.axis('off')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'coarsening_robustness_traj.pdf'),
                format='pdf', bbox_inches='tight')
    print(f'Figures → {figdir}/coarsening_robustness_{{ari,traj}}.pdf')

    # ── verdict ──────────────────────────────────────────────────────────────
    others = [m for m in methods if m != REFERENCE_METHOD]
    min_ari = ari.loc[others, REFERENCE_METHOD].min()
    has_non_spectral = any(m in NON_SPECTRAL_METHODS for m in methods)

    print('\n' + '=' * 72)
    print(f'Minimum ARI vs {REFERENCE_METHOD} = {min_ari:.3f}')
    if min_ari >= 0.7:
        print('The regime assignment survives the change of reduction method.')
    else:
        print('The regime assignment does NOT survive. Report this value as measured')
        print('and scope the regime claim accordingly — do not omit or soften it.')

    if has_non_spectral:
        print('\nNon-spectral baselines were included, so this speaks to method-')
        print('invariance in general (Reviewer 2 point 1).')
    else:
        print('\nNOTE: only local variation methods were run. They differ in candidate')
        print('family but minimize the same spectral objective, so this establishes')
        print('invariance to the CANDIDATE FAMILY, not to spectral coarsening as such.')
        print('Do not write it up as the latter. For the stronger claim, rerun with')
        print('  python experiments/coarsening_robustness.py --methods all')
    print('=' * 72)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--per-domain', type=int, default=20,
                   help='Networks sampled per domain (default 20)')
    p.add_argument('--domain', default=None, help=f'Restrict to one domain: {VALID_DOMAINS}')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true',
                   help='Ignore existing CSV and recompute from scratch')
    p.add_argument('--analyze', action='store_true',
                   help='Skip the sweep; compute ARI/Spearman/figures from the CSV')
    p.add_argument('--methods', default='local-variation',
                   help="'local-variation' (default: the two Loukas algorithms), "
                        "'all' (adds heavy_edge, algebraic_JC, affinity_GS), or a "
                        "comma-separated list of method names")
    args = p.parse_args()

    if args.methods == 'local-variation':
        METHODS = list(DEFAULT_METHODS)
    elif args.methods == 'all':
        METHODS = DEFAULT_METHODS + NON_SPECTRAL_METHODS
    else:
        METHODS = [m.strip() for m in args.methods.split(',') if m.strip()]
        unknown = set(METHODS) - set(DEFAULT_METHODS + NON_SPECTRAL_METHODS)
        if unknown:
            sys.exit(f'Unknown method(s): {sorted(unknown)}. '
                     f'Known: {DEFAULT_METHODS + NON_SPECTRAL_METHODS}')
    if REFERENCE_METHOD not in METHODS:
        sys.exit(f'The reference method {REFERENCE_METHOD!r} must be included '
                 f'so ARI has a baseline to compare against.')

    if args.analyze:
        analyze(args)
    else:
        run_sweep(args)
