"""Figure 1 — multiscale entropy across synthetic network families.

Four families with known structure are coarsened through the same ladder used on the
real corpus, so that the entropy trajectories of real networks can be read against
references whose regularity is understood by construction:

    ring       a cycle plus random chords -- the additive (non-rewiring) small-world
               construction; the manuscript calls this family "small-world"
    barabasi   preferential attachment    -- heavy-tailed degrees, hub-dominated
    regular    5-regular random graph     -- homogeneous degrees, no hubs
    grid       2-D lattice                -- maximally regular, no long-range structure

Provenance
----------
This reimplements an analysis that previously lived in two Jupyter notebooks
(`algorithm/src/entropiaGrafosReales.ipynb` cell 35 built the graphs into a 27 MB
pickle; `Entropy_Experiments/Network_Families/Analysis_Synthetic_Networks.ipynb`
computed and plotted). Both were removed when the repository was reduced to the code
needed to reproduce the paper; they remain in git history at commit bb806bf. The
construction parameters below are taken from that history verbatim.

Two things differ deliberately from the original:

  * The graphs are generated from a seed rather than loaded from a committed pickle.
    They are synthetic, so the pickle was never data -- only a cache. This makes the
    figure reproducible without shipping 27 MB, at the cost that the curves will not
    match the published figure edge-for-edge: the original run was unseeded.
  * The ring chords are drawn in one vectorised pass instead of a Python double loop
    over all O(N^2) pairs with sparse element assignment, which at N = 2500 dominated
    the entire experiment. The distribution is identical.

Cost
----
A stage-2 script, but a cheap one: it needs no GPU and no corpus. At N = 2500 a graph
takes 90-160 s for the whole five-level ladder, depending on family -- the ring carries
~18k edges once its chords are drawn and costs nearly twice what the grid does, and the
ten Erdos-Renyi reference draws each normalisation needs dominate throughout. Budget
about 1.5-2 hours for the 4 x 10 default on one core. Use --instances 2 --sizes 500 for a smoke test. Results are
checkpointed after every graph and completed (family, size, instance) triples are
skipped, so an interrupted run resumes.

Usage
-----
    python experiments/synthetic_families.py                 # compute (slow)
    python experiments/synthetic_families.py --analyze       # figure only (instant)

Outputs
-------
    results/synthetic_families.csv        — one row per (family, size, instance, level)
    figures/synthetic_families.pdf        — Figure 1
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (      # noqa: E402
    LEVELS, REDUCTION_LEVELS, RESULTS, FIGURES, SEED, PALETTE,
    PAPER_STYLE, spectral_entropy_at_levels,
)

OUT_CSV = os.path.join(RESULTS, 'synthetic_families.csv')

# Taken from entropiaGrafosReales.ipynb cell 35 (commit bb806bf).
RING_CHORD_P = 0.005
REGULAR_DEGREE = 5
DEFAULT_SIZES = [2500]
DEFAULT_INSTANCES = 10

FAMILY_ORDER = ['ring', 'barabasi', 'regular', 'grid']
FAMILY_LABEL = {'ring': 'Small-world', 'barabasi': 'Barabási–Albert',
                'regular': 'Random regular ($d=5$)', 'grid': '2-D grid'}
FAMILY_COLOR = dict(zip(FAMILY_ORDER, PALETTE))
FAMILY_MARKER = {'ring': 'o', 'barabasi': 's', 'regular': '^', 'grid': 'D'}


# ── graph construction ───────────────────────────────────────────────────────

def grid_dims(N):
    """Factor N into m x n as squarely as possible, matching the original search."""
    m = int(np.sqrt(N))
    while m > 1 and m * (N // m) != N:
        m -= 1
    return m, N // m


def make_graph(family, N, rng):
    """Build one instance of `family` on N nodes."""
    import networkx as nx

    if family == 'ring':
        # Cycle, plus a chord between each non-adjacent pair with probability p.
        # The original walked every (u, v) pair in Python and assigned into a sparse
        # matrix; this draws the same Bernoulli field in one pass.
        G = nx.cycle_graph(N)
        iu, iv = np.triu_indices(N, k=2)
        keep = rng.random(iu.size) < RING_CHORD_P
        G.add_edges_from(zip(iu[keep].tolist(), iv[keep].tolist()))
        return G

    if family == 'barabasi':
        # pygsp's own generator, as the original used (defaults m0 = m = 1). Not
        # interchangeable with nx.barabasi_albert_graph: pygsp adds 1 to every degree
        # before normalising the attachment probabilities, which softens preferential
        # attachment and yields a less hub-dominated tree. Since hub structure is
        # exactly what this family is here to exhibit, the difference matters.
        from pygsp import graphs as pygsp_graphs
        Gp = pygsp_graphs.BarabasiAlbert(N=N, seed=int(rng.integers(2**31)))
        return nx.from_scipy_sparse_array(Gp.W.tocsr())

    if family == 'regular':
        return nx.random_regular_graph(REGULAR_DEGREE, N,
                                       seed=int(rng.integers(2**31)))

    if family == 'grid':
        m, n = grid_dims(N)
        if m * n != N:
            raise ValueError(f'cannot factor N={N} into a grid')
        return nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, n))

    raise ValueError(f'unknown family: {family}')


# ── compute ──────────────────────────────────────────────────────────────────

def run(args):
    from tqdm import tqdm

    args.sizes = args.sizes or DEFAULT_SIZES

    os.makedirs(RESULTS, exist_ok=True)
    done = set()
    rows = []
    if os.path.exists(OUT_CSV) and not args.overwrite:
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict('records')
        done = set(map(tuple, prev[['family', 'size', 'instance']]
                       .drop_duplicates().values.tolist()))
        print(f'Resuming: {len(done)} (family, size, instance) triples already done.')

    # Instance is the OUTER loop so that one instance of every family completes before
    # a second instance of any: the figure is then meaningful (if noisy) at any point in
    # a two-hour run, instead of showing one family until the very end.
    todo = [(f, N, i) for N in args.sizes
            for i in range(args.instances)
            for f in args.families
            if (f, N, i) not in done]
    if not todo:
        print('Nothing to compute.')
        return
    print(f'{len(todo)} graphs to process '
          f'({len(args.families)} families x {len(args.sizes)} sizes '
          f'x {args.instances} instances).')

    for family, N, i in tqdm(todo, desc='graphs'):
        # Seed per (family, size, instance) so any one graph can be regenerated
        # on its own, and adding instances never disturbs the existing ones.
        rng = np.random.default_rng([args.seed, hash(family) % 2**31, N, i])
        try:
            G = make_graph(family, N, rng)
        except Exception as exc:
            tqdm.write(f'  build failed {family} N={N} #{i}: {exc}')
            continue

        ent = spectral_entropy_at_levels(G, REDUCTION_LEVELS, warn=tqdm.write)
        for level, value in ent.items():
            rows.append({'family': family, 'size': N, 'instance': i,
                         'level': level, 'value': value,
                         'n_nodes': G.number_of_nodes(),
                         'n_edges': G.number_of_edges()})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # checkpoint every graph

    print(f'\n-> {OUT_CSV}')


# ── figure ───────────────────────────────────────────────────────────────────

def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)

    if not os.path.exists(OUT_CSV):
        sys.exit(f'{OUT_CSV} not found — run without --analyze first.')
    df = pd.read_csv(OUT_CSV).dropna(subset=['value'])
    sizes = args.sizes if args.sizes else sorted(df['size'].unique(), reverse=True)
    sizes = [s for s in sizes if s in set(df['size'])]
    if not sizes:
        sys.exit('No requested size present in the results.')

    for N in sizes:
        have = set(df[df['size'] == N]['family'])
        missing = [f for f in FAMILY_ORDER if f not in have]
        if missing:
            print(f'  WARNING: N={N} has no data for {", ".join(missing)} — '
                  f'the figure below is from an incomplete run.')

    os.makedirs(FIGURES, exist_ok=True)
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 2.55),
                             squeeze=False, sharey=True)

    for ax, N in zip(axes[0], sizes):
        sub = df[df['size'] == N]
        for fam in FAMILY_ORDER:
            g = sub[sub['family'] == fam]
            if g.empty:
                continue
            stat = g.groupby('level')['value'].agg(['mean', 'std', 'count'])
            stat = stat.reindex([l for l in LEVELS if l in stat.index])
            # Standard error across instances; with one instance there is no band.
            se = (stat['std'] / np.sqrt(stat['count'])).fillna(0.0)
            ax.plot(stat.index, stat['mean'], '-', linewidth=1.6,
                    marker=FAMILY_MARKER[fam], markersize=4, zorder=3,
                    color=FAMILY_COLOR[fam], label=FAMILY_LABEL[fam])
            if (se > 0).any():
                ax.fill_between(stat.index, stat['mean'] - se, stat['mean'] + se,
                                color=FAMILY_COLOR[fam], alpha=0.18, linewidth=0,
                                zorder=2)
        # L* = 1 is the Erdos-Renyi reference, not decoration: it is the line the
        # small-world and random-regular families sit on.
        ax.axhline(1.0, color='0.45', linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
        ax.set_xticks(LEVELS)
        ax.set_xlabel('nodes retained (%)')
        # No panel title: with one panel it only repeats the caption. N and the
        # instance count belong there, as in the other trajectory figures.
        ax.spines[['top', 'right']].set_visible(False)

    axes[0][0].set_ylabel('normalized entropy $L^*$')
    # x runs from the full graph to the most aggressive reduction, matching the
    # reading order of the trajectory figures elsewhere in the paper.
    axes[0][0].invert_xaxis()
    # Two columns: a four-entry single column is tall enough to cover the gap the
    # curves leave between the near-incompressible families and the compressible ones,
    # which is the only clear space on the panel.
    # Lifted off dead centre so the second row clears the Barabasi-Albert peak at 40%.
    axes[0][-1].legend(fontsize=7, frameon=False, loc='center left', ncol=2,
                       bbox_to_anchor=(0.0, 0.63), columnspacing=1.0, handlelength=1.6)

    fig.tight_layout()
    path = os.path.join(FIGURES, 'synthetic_families.pdf')
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Figure -> {path}')

    print('\nMean normalized entropy by family and level:')
    for N in sizes:
        piv = (df[df['size'] == N].pivot_table(index='family', columns='level',
                                               values='value', aggfunc='mean')
               .reindex(index=[f for f in FAMILY_ORDER if f in set(df['family'])],
                        columns=[l for l in LEVELS]))
        print(f'\n  N = {N}')
        print(piv.round(4).to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', type=int, nargs='*', default=None,
                   help=f'Node counts (compute default {DEFAULT_SIZES}; '
                        f'--analyze defaults to whatever is in the CSV)')
    p.add_argument('--instances', type=int, default=DEFAULT_INSTANCES,
                   help=f'Graphs per family per size (default {DEFAULT_INSTANCES})')
    p.add_argument('--families', nargs='*', default=FAMILY_ORDER, choices=FAMILY_ORDER)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute from scratch instead of resuming')
    p.add_argument('--analyze', action='store_true',
                   help='Draw the figure from the existing CSV and exit')
    args = p.parse_args()
    analyze(args) if args.analyze else run(args)
