"""
Reproduce Figure 2: entropy trajectories by domain, stratified by node count.

One panel per network domain. Within each panel, one line per size stratum showing
how normalized compression entropy L*(G_r) evolves as the graph is coarsened to 80%,
60%, 40% and 20% of its original size.

Reading the plot
----------------
L* = 1 marks an Erdos-Renyi graph of the same size and density, so the dashed line at
1.0 is the reference the trajectories are measured against, not an arbitrary gridline.
A trajectory rising toward it means coarsening is destroying the structure that made
the graph compressible; a flat trajectory means the reduction is removing redundancy
without touching that structure.

Bands are bootstrap 95% confidence intervals on the mean, not the spread of the
underlying networks. With strata as small as a handful of graphs the mean alone would
imply more precision than the data supports.

Usage
-----
    python experiments/trajectory_figures.py
    python experiments/trajectory_figures.py --source real     # 431-network results
    python experiments/trajectory_figures.py --spread iqr      # show dispersion instead

Outputs
-------
    figures/entropy_trajectories_by_domain.pdf
    results/trajectory_summary.csv    — mean and CI per (domain, stratum, level)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import (load_trajectories, LEVELS, FIGURES,   # noqa: E402
                                RESULTS, SEED, PAPER_STYLE,
                                DOMAINS as DOMAIN_ORDER)

# Strata follow the manuscript: small 0-200, medium 200-600, large 600+.
STRATA = [('small (≤200)', -1, 200),
          ('medium (200–600)', 200, 600),
          ('large (>600)', 600, np.inf)]

OKABE_ITO = ['#0072B2', '#D55E00', '#009E73']


def add_stratum(wide, source):
    """Attach a size stratum using node count at the uncoarsened scale."""
    if 'n_nodes' in wide.columns:
        n = wide['n_nodes']
    else:
        # The trajectory table carries entropies only; recover sizes from the raw
        # results file rather than inferring them.
        path = os.path.join(RESULTS, 'coarsening_robustness.csv')
        raw = pd.read_csv(path)
        sizes = raw.drop_duplicates('name').set_index('name')['n_nodes']
        n = wide['name'].map(sizes)
    def which(v):
        if pd.isna(v):
            return None
        for name, lo, hi in STRATA:
            if lo < v <= hi:
                return name
        return None

    labels = [which(v) for v in n]
    return wide.assign(n_nodes=np.asarray(n), stratum=labels)


def bootstrap_ci(values, n_boot=2000, seed=SEED):
    """Percentile bootstrap CI for the mean. Returns (lo, hi); NaN if too few points."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def resolve_cmap(name, plt):
    """Resolve a colormap name through seaborn first, then matplotlib.

    Seaborn's sequential maps (flare, crest, rocket, mako) are not registered with
    matplotlib, so `plt.get_cmap` cannot see them. Seaborn also accepts matplotlib
    names, so trying it first costs nothing and keeps both namespaces available.
    Falls back cleanly if seaborn is not installed.
    """
    try:
        import seaborn as sns
        return sns.color_palette(name, as_cmap=True)
    except Exception:
        return plt.get_cmap(name)




def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)

    wide = load_trajectories(args.source)
    wide = add_stratum(wide, args.source)
    wide = wide.dropna(subset=['stratum'])
    print(f'{len(wide)} networks')
    print(pd.crosstab(wide['domain'], wide['stratum']).to_string())

    os.makedirs(FIGURES, exist_ok=True)
    domains = [d for d in DOMAIN_ORDER if d in set(wide['domain'])]
    ncols = args.ncols
    nrows = int(np.ceil(len(domains) / ncols))
    # Sized to the acmart acmsmall text width (~6.8 in) so the figure is placed at
    # 100% and the type in it matches the body text. Scaling a figure in LaTeX is
    # what makes axis labels come out at a different size from the caption.
    fig, axes = plt.subplots(nrows, ncols, figsize=(args.width, 2.55 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = LEVELS
    rows = []

    # Continuous size encoding on a linear node-count scale.
    norm = Normalize(vmin=wide['n_nodes'].min(), vmax=wide['n_nodes'].max())
    
    size_cmap = resolve_cmap(args.cmap, plt)

    for ax, dom in zip(axes, domains):
        sub = wide[wide['domain'] == dom]

        if args.color_by == 'size':
            # Draw small graphs first so the large ones, which are the interesting
            # end of the scale, are not buried under them.
            for _, row in sub.sort_values('n_nodes').iterrows():
                ax.plot(x, [row[l] for l in x],
                        color=size_cmap(norm(row['n_nodes'])),
                        linewidth=0.8, alpha=args.alpha, zorder=2,
                        solid_capstyle='round')
            ax.axhline(1.0, color='0.45', linestyle=(0, (4, 3)), linewidth=0.7,
                       zorder=5)
            ax.set_title(f'{dom} ($n={len(sub)}$)', pad=4)
            ax.set_xticks(x)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for sname, _, _ in STRATA:
                s_ = sub[sub['stratum'] == sname]
                for l in x:
                    if len(s_):
                        lo, hi = bootstrap_ci(s_[l])
                        rows.append(dict(domain=dom, stratum=sname, level=l,
                                         mean=s_[l].mean(), lo=lo, hi=hi, n=len(s_)))
            continue

        for i, (sname, _, _) in enumerate(STRATA):
            s_ = sub[sub['stratum'] == sname]
            if len(s_) == 0:
                continue

            if args.style == 'lines':
                # One line per network. Alpha is what makes 183 overlapping
                # trajectories legible: individually faint, collectively the dense
                # regions darken, so the distribution is visible without averaging
                # it away. Bands would hide exactly the crossing and fanning that
                # this view is meant to show.
                for _, row in s_.iterrows():
                    ax.plot(x, [row[l] for l in x], color=OKABE_ITO[i],
                            linewidth=0.7, alpha=args.alpha, zorder=2,
                            solid_capstyle='round')
                if args.overlay_mean:
                    ax.plot(x, [s_[l].mean() for l in x], color=OKABE_ITO[i],
                            linewidth=2.2, alpha=1.0, zorder=4,
                            path_effects=[pe.Stroke(linewidth=3.4, foreground='white'),
                                          pe.Normal()])
            else:
                means = [s_[l].mean() for l in x]
                los, his = zip(*[bootstrap_ci(s_[l]) for l in x])
                if args.spread == 'iqr':
                    los = [s_[l].quantile(.25) for l in x]
                    his = [s_[l].quantile(.75) for l in x]
                ax.plot(x, means, marker='o', markersize=4, linewidth=1.6,
                        color=OKABE_ITO[i], zorder=3)
                if not all(np.isnan(los)):
                    ax.fill_between(x, los, his, color=OKABE_ITO[i], alpha=0.18,
                                    linewidth=0, zorder=2)

            for l in x:
                lo, hi = bootstrap_ci(s_[l])
                rows.append(dict(domain=dom, stratum=sname, level=l,
                                 mean=s_[l].mean(), lo=lo, hi=hi, n=len(s_)))

        # L* = 1 is the Erdos-Renyi reference, not decoration.
        ax.axhline(1.0, color='0.45', linestyle=(0, (4, 3)), linewidth=0.7, zorder=5)
        ax.set_title(f'{dom} ($n={len(sub)}$)', pad=4)
        ax.set_xticks(x)

        # Proxy handles: the real lines are too faint to read in a legend.
        handles = [Line2D([], [], color=OKABE_ITO[i], linewidth=2,
                          label=f'{sname}, n={int((sub["stratum"] == sname).sum())}')
                   for i, (sname, _, _) in enumerate(STRATA)
                   if (sub['stratum'] == sname).any()]
        ax.legend(handles=handles, fontsize=7, frameon=False, loc='lower left')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # Invert once, not per-axis: with sharex=True every call flips the shared axis,
    # so six calls cancel out and the panels read 20% -> 100%, backwards.
    axes[0].invert_xaxis()

    for ax in axes[len(domains):]:
        ax.axis('off')
    # Label only the outer axes: repeating them on every panel is noise when the
    # axes are shared and identical.
    for ax in axes[len(domains) - ncols:len(domains)]:
        ax.set_xlabel('nodes retained (%)')
    for r in range(nrows):
        axes[r * ncols].set_ylabel('normalized entropy $L^*$')

    fig.tight_layout()
    if args.color_by == 'size':
        sm = plt.cm.ScalarMappable(norm=norm, cmap=size_cmap)
        sm.set_array([])
        fig.subplots_adjust(right=0.885)
        cax = fig.add_axes([0.905, 0.30, 0.015, 0.40])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('nodes in original graph', labelpad=6)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(width=0.6, length=2.5)

    path = os.path.join(FIGURES, 'entropy_trajectories_by_domain.pdf')
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure → {path}')

    out = os.path.join(RESULTS, 'trajectory_summary.csv')
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f'Summary → {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', choices=['robustness', 'real'], default='robustness')
    p.add_argument('--width', type=float, default=7.1,
                   help='Figure width in inches; default matches acmsmall text width')
    p.add_argument('--ncols', type=int, default=3,
                   help='Panels per row')
    p.add_argument('--cmap', default='flare',
                   help='Colormap for the size scale; flare is the most '
                        'colour-vision-deficiency-safe of the uniform maps')
    p.add_argument('--color-by', choices=['size', 'stratum'], default='size',
                   help="'size' = continuous colour scale on node count with a "
                        "colorbar (default); 'stratum' = three discrete size bins")
    p.add_argument('--style', choices=['lines', 'aggregate'], default='lines',
                   help="'lines' = one trajectory per network (default); "
                        "'aggregate' = mean with a band")
    p.add_argument('--alpha', type=float, default=0.28,
                   help='Opacity of individual trajectories (style=lines)')
    p.add_argument('--overlay-mean', action='store_true',
                   help='Draw the stratum mean on top of the individual lines')
    p.add_argument('--spread', choices=['ci', 'iqr'], default='ci',
                   help="'ci' = bootstrap CI on the mean (default); "
                        "'iqr' = interquartile range of the networks")
    main(p.parse_args())
