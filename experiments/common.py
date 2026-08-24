"""Shared constants, paths, and loaders for the analysis scripts.

Everything here was previously duplicated across two or more scripts, in a few cases
with the copies silently disagreeing: `PAPER_STYLE` existed twice with different tick
directions, the six-domain list existed four times in two different orders, and the
corpus path was rebuilt by hand in five places. Import from here instead.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

# networkx is imported inside build_graph rather than here: the figure/table scripts
# import this module and must keep working in an environment that has only
# numpy/pandas/matplotlib installed.

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, 'data')
RESULTS = os.path.join(PROJECT_ROOT, 'results')
TABLES = os.path.join(PROJECT_ROOT, 'tables')
FIGURES = os.path.join(PROJECT_ROOT, 'figures')

# The corpus of 558 networks built by build_full_corpus.py from the raw
# CommunityFitNet pickle. Every script that reads networks starts here.
CORPUS_PKL = os.path.join(DATA, 'all_networks.pkl')
SOURCE_PICKLE = os.path.join(DATA, 'CommunityFitNet_updated.pickle')

# ── corpus conventions ───────────────────────────────────────────────────────
LEVELS = [100, 80, 60, 40, 20]      # percent of nodes retained
REDUCTION_LEVELS = LEVELS[1:]       # the levels that require actual coarsening
MIN_NODES = 20
MIN_EDGES = 30
SEED = 42
K_CLUSTERS = 3

# One canonical order, used for both validation and plotting. The two orderings
# that previously coexisted differed only in where Technological sat.
DOMAINS = ['Biological', 'Social', 'Economic',
           'Technological', 'Transportation', 'Informational']

# Colour-blind-safe palette used across every figure in the paper.
PALETTE = ['#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252', '#b4d2b1']

# ── figure style ─────────────────────────────────────────────────────────────
PAPER_STYLE = {
    # Type 42 embeds TrueType outlines rather than bitmaps. ACM (and most
    # publishers) reject Type 3 fonts, which is what matplotlib emits by default.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Libertine is the acmart body face; the fallbacks keep this working on
    # machines without it rather than silently substituting a sans-serif.
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Libertinus Serif', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.grid': False,
    'savefig.dpi': 600,
}


def apply_paper_style():
    """Set the publication rcParams. Call before creating any figure."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)


# ── graphs ───────────────────────────────────────────────────────────────────
def build_graph(row):
    """Largest connected component of the undirected version, relabelled 0..n-1.

    Corpus edge lists are stored as unordered pairs (u <= v), so a row flagged
    Directed carries no orientation to discard and symmetrizing loses nothing.
    Coarsening and the spectral quantities downstream need a connected graph.

    Note the relabelling happens only when the graph was disconnected, matching the
    behaviour every result in results/ was computed under. Do not "tidy" this into an
    unconditional relabel: it would renumber the nodes of already-connected graphs
    and change their encodings.
    """
    import networkx as nx

    is_directed = 'Directed' in row['graphProperties']
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(np.array(row['nodes_id']))
    G.add_edges_from(np.array(row['edges_id']))
    G = nx.to_undirected(G)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def load_corpus(path=None):
    """Load the network corpus, with a message that says how to build it."""
    path = path or CORPUS_PKL
    if not os.path.exists(path):
        sys.exit(f'{path} not found — run:\n'
                 f'    python experiments/build_full_corpus.py')
    return pd.read_pickle(path)


# ── result loaders ───────────────────────────────────────────────────────────
def load_trajectories(source='robustness'):
    """Entropy trajectories: one row per network, columns 100/80/60/40/20 + domain.

    `robustness` reads coarsening_robustness.csv restricted to the neighborhood-based
    algorithm, which covers the full corpus and is the default. `real` reads the
    per-domain experiment files, which carry the same quantity as `spectral_entropy`.
    """
    if source == 'robustness':
        path = os.path.join(RESULTS, 'coarsening_robustness.csv')
        if not os.path.exists(path):
            sys.exit(f'{path} not found — run coarsening_robustness.py first.')
        df = pd.read_csv(path)
        df = df[df['method'] == 'variation_neighborhood']
    else:
        # Deliberately NOT a glob over real_networks_results_*.csv: that pattern also
        # matches per-domain files from earlier runs, and concatenating them
        # double-counts every network present in more than one. Use the consolidated
        # file, which is what rebuild_regression.py reads too.
        path = os.path.join(RESULTS, 'real_networks_results_all.csv')
        if not os.path.exists(path):
            files = sorted(glob.glob(os.path.join(
                RESULTS, 'real_networks_results_[A-Z]*.csv')))
            if not files:
                sys.exit(f'No real_networks_results_all.csv in {RESULTS}')
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            df = df.drop_duplicates(subset=['domain', 'name', 'level', 'measure'])
        else:
            df = pd.read_csv(path)
        df = df[df['measure'] == 'spectral_entropy']

    wide = df.pivot_table(index=['domain', 'name'], columns='level', values='value')
    return wide.dropna(subset=LEVELS).reset_index()


# ── LaTeX ────────────────────────────────────────────────────────────────────
def write_table(filename, lines):
    """Write a LaTeX fragment atomically.

    Formatting the body before opening the file matters: the previous version opened
    with 'w' (truncating at once) and formatted afterwards, so any exception left a
    zero-byte table on disk that LaTeX happily included as nothing.
    """
    body = '\n'.join(lines).rstrip('\n') + '\n'
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, filename)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(body)
    os.replace(tmp, path)
    return path


# ── prediction entropy ───────────────────────────────────────────────────────
def analytic_entropy_from_ranks(ranks, N):
    """Normalized prediction entropy H* = H / (log2(N) - 1), after Sun et al. (2020).

    Ranks are binned at width N, giving ceil(C(N,2)/N) ~= (N-1)/2 bins; H is the
    Shannon entropy of that histogram. The denominator is the entropy of a uniform
    spread over those bins, i.e. what an uninformative predictor achieves, so H*
    lands in [0, 1] by construction.

    This replaces an Erdos-Renyi baseline estimated by simulation. The sampled
    version is unusable for Adamic-Adar on sparse graphs: AA scores zero for any
    pair without a common neighbor, so its ranks on a sparse random graph collapse
    into the first bin, the baseline entropy tends to zero, and the ratio diverges.
    Averaging over more draws does not help -- the failure is structural, not
    sampling variance.

    Returns (H, denominator, H*).
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    n_bins = int(np.ceil((N * (N - 1) / 2) / N))
    bins = np.minimum((ranks - 1) // N, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())
    denom = float(np.log2(N) - 1.0)
    H_star = H / denom
    if not (-1e-9 <= H_star <= 1.0 + 1e-9):
        raise AssertionError(f'H*={H_star:.4f} outside [0,1] (N={N}, H={H:.4f})')
    return H, denom, float(np.clip(H_star, 0.0, 1.0))
