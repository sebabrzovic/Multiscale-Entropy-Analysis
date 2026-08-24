"""
Reproduce the entropy-regime clustering: Table 2 and the PCA cluster figure.

Produces, from the multiscale entropy trajectories:

  * Table 2  — cluster composition by network family, K-means (k=3), as LaTeX
  * Figure   — PCA projection of the five-dimensional entropy vectors, coloured
               by cluster assignment, as vector PDF
  * optional — the cluster-validity diagnostics behind the choice of k

Cluster labelling
-----------------
K-means numbers its clusters by initialization order, so the same data can produce
the same partition under different labels from run to run. Clusters are therefore
relabelled canonically, ordered by mean entropy of the uncoarsened graph:

    Cluster 1 = lowest H_100 ... Cluster 3 = highest H_100

This makes the table reproducible and the numbering meaningful, but it means the
indices here need not match a raw KMeans `labels_` array.

On the choice of k
------------------
k=3 is an interpretability choice, not one selected by an internal criterion:
silhouette, Calinski-Harabasz, Davies-Bouldin and bootstrap stability all prefer
k=2, and the gap statistic selects no k at all. Run --diagnostics to reproduce
those numbers. See Appendix "Choice of the Number of Clusters" in the paper.

Usage
-----
    python experiments/clustering_analysis.py
    python experiments/clustering_analysis.py --diagnostics
    python experiments/clustering_analysis.py --source real   # 431-network results

Outputs
-------
    tables/table_cluster_composition.tex
    figures/kmeans_pca_clusters.pdf
    results/cluster_assignments.csv
"""

import argparse
import glob
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, 'results')
TABLES = os.path.join(PROJECT_ROOT, 'tables')
FIGURES = os.path.join(PROJECT_ROOT, 'figures')

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

LEVELS = [100, 80, 60, 40, 20]
K = 3
SEED = 42

DOMAIN_ORDER = ['Biological', 'Social', 'Economic',
                'Technological', 'Transportation', 'Informational']


def load_trajectories(source):
    """Return a frame indexed by network with columns 100, 80, 60, 40, 20 and `domain`.

    `robustness` uses coarsening_robustness.csv restricted to the neighborhood-based
    algorithm, which covers the full corpus. `real` uses the per-domain result files,
    which carry the same measure under the name `spectral_entropy`.
    """
    if source == 'robustness':
        path = os.path.join(RESULTS, 'coarsening_robustness.csv')
        if not os.path.exists(path):
            sys.exit(f'{path} not found — run coarsening_robustness.py first.')
        df = pd.read_csv(path)
        df = df[df['method'] == 'variation_neighborhood']
        wide = df.pivot_table(index=['domain', 'name'], columns='level', values='value')
    else:
        files = sorted(glob.glob(os.path.join(RESULTS, 'real_networks_results_*.csv')))
        if not files:
            sys.exit(f'No real_networks_results_*.csv in {RESULTS}')
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[df['measure'] == 'spectral_entropy']
        wide = df.pivot_table(index=['domain', 'name'], columns='level', values='value')

    wide = wide.dropna(subset=LEVELS).reset_index()
    return wide


def cluster(wide, k=K, seed=SEED):
    """K-means on standardized trajectories, with clusters relabelled by mean H_100."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(wide[LEVELS].values)
    raw = KMeans(k, random_state=seed, n_init=10).fit_predict(X)

    order = (pd.Series(wide[100].values).groupby(raw).mean()
               .sort_values().index.tolist())
    remap = {old: new + 1 for new, old in enumerate(order)}
    labels = np.array([remap[r] for r in raw])
    return X, labels


def emit_table(wide, labels):
    os.makedirs(TABLES, exist_ok=True)
    wide = wide.assign(cluster=labels)
    comp = pd.crosstab(wide['domain'], wide['cluster'])
    comp = comp.reindex([d for d in DOMAIN_ORDER if d in comp.index])
    for c in range(1, K + 1):
        if c not in comp.columns:
            comp[c] = 0
    comp = comp[sorted(comp.columns)]

    profiles = wide.groupby('cluster')[LEVELS].mean().round(3)
    print('\nCluster profiles (mean trajectory, ordered by H_100):')
    print(profiles.to_string())
    print('\nComposition by domain:')
    print(comp.to_string())

    lines = [r'\begin{table}[t]', r'    \centering',
             r'    \caption{Cluster composition by network family using K-means ($k=3$) '
             r'on the %d-network corpus. Mean trajectories: ' % len(wide) +
             '; '.join(r'cluster~%d $%.2f \to %.2f$' %
                       (c, profiles.loc[c, 100], profiles.loc[c, 20])
                       for c in sorted(profiles.index)) + '.}',
             r'    \label{tab:kmeans3}',
             r'    \begin{tabular}{l' + 'c' * K + '}', r'        \toprule',
             r'        \textbf{Network Family} & ' +
             ' & '.join(r'\textbf{Cluster %d}' % c for c in sorted(comp.columns)) + r' \\',
             r'        \midrule']
    for dom, row in comp.iterrows():
        lines.append(f'        {dom:14s} & ' +
                     ' & '.join(f'{int(v):3d}' for v in row.values) + r' \\')
    lines += [r'        \midrule',
              r'        \textbf{Total} & ' +
              ' & '.join(r'\textbf{%d}' % int(v) for v in comp.sum().values) + r' \\',
              r'        \bottomrule', r'    \end{tabular}', r'\end{table}']

    path = os.path.join(TABLES, 'table_cluster_composition.tex')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nTable → {path}')
    return comp


def emit_figure(wide, X, labels, args):
    """PCA projection shown twice: coloured by cluster, and by domain.

    Both panels use the same coordinates, so the reader can compare the two
    colourings directly. That comparison is the point: the clustering never sees
    the domain labels, so any correspondence between the panels is the evidence
    for the claim that entropy regimes track network domain.
    """
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)

    os.makedirs(FIGURES, exist_ok=True)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    domains = wide['domain'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # ── left: cluster ────────────────────────────────────────────────────────
    cl_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in sorted(set(labels)):
        m = labels == c
        axes[0].scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.75,
                        color=cl_colors[c - 1], edgecolors='none',
                        label=f'Cluster {c} (n={int(m.sum())})')
    axes[0].set_title('Coloured by cluster ($k=3$)', fontsize=11)
    axes[0].legend(fontsize=8, frameon=False, loc='best')

    # ── right: domain ────────────────────────────────────────────────────────
    present = [d for d in DOMAIN_ORDER if d in set(domains)]
    dom_colors = plt.cm.Dark2(np.linspace(0, 1, max(len(present), 3)))
    for i, d in enumerate(present):
        m = domains == d
        axes[1].scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.75,
                        color=dom_colors[i], edgecolors='none',
                        label=f'{d} (n={int(m.sum())})')
    axes[1].set_title('Coloured by network domain', fontsize=11)
    axes[1].legend(fontsize=8, frameon=False, loc='best')

    for ax in axes:
        ax.set_xlabel(f'PC1 ({100 * var[0]:.0f}% of variance)')
    axes[0].set_ylabel(f'PC2 ({100 * var[1]:.0f}% of variance)')

    fig.tight_layout()
    path = os.path.join(FIGURES, 'kmeans_pca_clusters.pdf')
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Figure → {path}   (PC1+PC2 = {100 * var[:2].sum():.0f}% of variance)')

    # ── single-panel variant: colour = domain, marker = cluster ──────────────
    #
    # Two variables on one set of axes, so both channels have to stay legible with
    # 558 overlapping points:
    #   * Okabe-Ito colours, which stay distinguishable under the common forms of
    #     colour blindness and do not rely on red/green contrast;
    #   * marker silhouettes chosen to differ in outline (circle / triangle /
    #     square) rather than in fill, since fill is carrying the domain;
    #   * thin white edges, which separate overlapping points without adding a
    #     third visual variable;
    #   * large domains drawn first, so the small ones (Informational, n=19) sit on
    #     top instead of being buried under Biological.
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba
    import matplotlib.patheffects as pe

    # Domain palette. Assigned in DOMAIN_ORDER, so the mapping is stable across
    # runs and matches the order used in Table 2.
    OKABE_ITO = ['#568f8b',  # Biological     (183 — largest group, needs weight)
                 '#1d4a60',  # Social          (tight, distinct cluster)
                 '#cd7e59',  # Economic
                 '#ddb247',  # Technological
                 '#d15252',  # Transportation
                 '#b4d2b1']  # Informational   (19 — palest, fewest points)
    # Cluster identity is carried by labelled centroids rather than marker shape.
    # With 558 overlapping points the shape channel was hard to read anyway, and
    # freeing it lets every point use the same clean circle so that colour — the
    # domain — is the only thing the reader has to decode from the marks.
    POINT_SIZE = 28
    # Face and edge carry separate alphas rather than one scatter-wide alpha, so
    # the balance between them stays under control: the edge is kept a little more
    # opaque than the fill, which preserves point definition in dense regions while
    # letting overlap read as density.
    FACE_ALPHA = 0.60
    EDGE_ALPHA = 0.70

    dom_color = {d: OKABE_ITO[i % len(OKABE_ITO)] for i, d in enumerate(present)}

    def darken(hex_color, factor=0.55):
        """Border colour: the fill darkened, rather than a neutral outline.

        A white or grey stroke would introduce a colour the reader has to ignore,
        and at small marker sizes it visually thins the fill. Darkening the fill
        keeps the domain encoding intact while giving each point a definite edge
        against its neighbours.
        """
        c = hex_color.lstrip('#')
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
        return '#%02x%02x%02x' % (int(r * factor), int(g * factor), int(b * factor))

    dom_edge = {d: darken(c) for d, c in dom_color.items()}
    draw_order = sorted(present, key=lambda d: -(domains == d).sum())

    # Text width for acmart acmsmall, so the figure goes in at 100% and its
    # type matches the body text rather than being scaled by LaTeX.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for d in draw_order:
        m = domains == d
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=POINT_SIZE, marker='o',
                   facecolor=to_rgba(dom_color[d], FACE_ALPHA),
                   edgecolors=to_rgba(dom_edge[d], EDGE_ALPHA), linewidths=0.6,
                   zorder=3 + draw_order.index(d))

    # Cluster boundaries — the k-means Voronoi partition.
    #
    # K-means assigns each network to its nearest centroid, so the partition is a
    # Voronoi diagram. It lives in the standardized five-dimensional space, though,
    # and a Voronoi diagram drawn from the *projected* centroids is not the same
    # thing: the 5-D cell walls are hyperplanes whose images under projection are
    # not the 2-D bisectors of the projected centres.
    #
    # We therefore draw the true partition restricted to the plotted plane. A grid
    # over the PC1-PC2 axes is mapped back into five dimensions with
    # `pca.inverse_transform` (which sets the three discarded components to their
    # mean), each grid point is assigned to the nearest 5-D centroid, and the
    # region borders are contoured. Every boundary shown is therefore a genuine
    # k-means decision boundary for the slice of the space being displayed, and
    # points can legitimately sit across a line from their own centroid — that is
    # the discarded 5% of variance, not an error.
    if args.boundary != 'none':
        centroids_5d = np.vstack([X[labels == c].mean(axis=0)
                                  for c in sorted(set(labels))])
        pad_x = 0.06 * (coords[:, 0].max() - coords[:, 0].min())
        pad_y = 0.06 * (coords[:, 1].max() - coords[:, 1].min())
        gx = np.linspace(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x, 400)
        gy = np.linspace(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y, 400)
        XX, YY = np.meshgrid(gx, gy)
        grid_2d = np.column_stack([XX.ravel(), YY.ravel()])
        grid_5d = pca.inverse_transform(grid_2d)
        d2 = ((grid_5d[:, None, :] - centroids_5d[None, :, :]) ** 2).sum(axis=2)
        Z = d2.argmin(axis=1).reshape(XX.shape)

        if args.boundary == 'voronoi-filled':
            # rasterized: a 400x400 mesh as vector cells would bloat the PDF and
            # slow every viewer, with no visible gain at print resolution.
            ax.pcolormesh(XX, YY, Z, cmap='Greys', alpha=0.25, vmin=-0.6,
                          vmax=len(centroids_5d) - 0.4, shading='auto',
                          zorder=0, rasterized=True)
        ax.contour(XX, YY, Z, levels=np.arange(len(centroids_5d)) + 0.5,
                   colors='0.4', linewidths=0.8, linestyles='--', zorder=1)

    # Centroids: the mean position of each cluster in the projection. Drawn as a
    # filled disc with the cluster number, above every data point, so they read as
    # annotation rather than as unusually large observations.
    for c in sorted(set(labels)):
        m = labels == c
        cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
        # A small cross with an offset label, rather than a filled disc: the
        # centroid is an annotation, and a heavy glyph competes with the data for
        # attention in exactly the dense regions where the data matters most. The
        # thin white stroke keeps it legible over dark markers.
        ax.scatter([cx], [cy], s=70, marker='X', facecolor='0.15',
                   edgecolors='white', linewidths=0.8, zorder=20)
        ax.annotate(str(c), (cx, cy), textcoords='offset points',
                    xytext=(7, 5), fontsize=8, color='0.15', zorder=21,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    dom_handles = [Line2D([], [], marker='o', linestyle='', markersize=7,
                          markerfacecolor=dom_color[d],
                          markeredgecolor=dom_edge[d], markeredgewidth=0.7,
                          label=f'{d} ({int((domains == d).sum())})')
                   for d in present]
    cl_handles = [Line2D([], [], marker='X', linestyle='', markersize=7,
                         markerfacecolor='0.15', markeredgecolor='white',
                         markeredgewidth=0.7,
                         label=f'Cluster {c} center ({int((labels == c).sum())})')
                  for c in sorted(set(labels))]

    leg1 = ax.legend(handles=dom_handles, title='Domain', fontsize=8,
                     title_fontsize=8.5, frameon=False, loc='upper left',
                     bbox_to_anchor=(1.01, 1.0), handletextpad=0.4)
    leg1._legend_box.align = 'left'
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=cl_handles, title='Cluster', fontsize=8,
                     title_fontsize=8.5, frameon=False, loc='upper left',
                     bbox_to_anchor=(1.01, 0.55), handletextpad=0.4)
    leg2._legend_box.align = 'left'

    ax.set_xlabel(f'PC1 ({100 * var[0]:.0f}% of variance)')
    ax.set_ylabel(f'PC2 ({100 * var[1]:.0f}% of variance)')
    # ax.set_title('Multiscale entropy trajectories in principal-component space',
    #              fontsize=11)
    ax.grid(alpha=0.15, linewidth=0.5, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # tight_layout does not account for legends anchored outside the axes, and
    # would clip the longer domain labels. Reserve the right margin explicitly and
    # pass the legends as extra artists so the saved bounding box includes them.
    fig.subplots_adjust(right=0.72)
    alt = os.path.join(FIGURES, 'kmeans_pca_clusters_single.pdf')
    fig.savefig(alt, format='pdf', bbox_inches='tight',
                bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f'Single-panel variant → {alt}')


def diagnostics(X):
    """Cluster-validity indices behind the choice of k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score, adjusted_rand_score)
    rng = np.random.default_rng(SEED)

    def km(k, data, seed=SEED):
        return KMeans(k, random_state=seed, n_init=10).fit(data)

    print(f'\n{"k":>3s} {"silhouette":>11s} {"Calinski-H":>11s} {"Davies-B":>9s} '
          f'{"gap":>7s} {"stability":>10s}')
    print('-' * 56)
    mins, maxs = X.min(0), X.max(0)
    for k in range(2, 11):
        m = km(k, X)
        lab = m.labels_
        refs = [np.log(km(k, rng.uniform(mins, maxs, size=X.shape), seed=b).inertia_)
                for b in range(25)]
        gap = np.mean(refs) - np.log(m.inertia_)
        stab = np.mean([
            adjusted_rand_score(lab[idx], km(k, X[idx], seed=b).labels_)
            for b, idx in enumerate(
                rng.choice(X.shape[0], size=int(0.8 * X.shape[0]), replace=False)
                for _ in range(25))])
        print(f'{k:3d} {silhouette_score(X, lab):11.3f} '
              f'{calinski_harabasz_score(X, lab):11.1f} '
              f'{davies_bouldin_score(X, lab):9.3f} {gap:7.3f} {stab:10.3f}')
    print('\nEvery index prefers k=2; the gap statistic increases monotonically and')
    print('selects no k. k=3 is an interpretability choice — see the paper appendix.')


def main(args):
    wide = load_trajectories(args.source)
    print(f'Loaded {len(wide)} networks from the "{args.source}" source')
    print(f'Domains: {wide["domain"].value_counts().to_dict()}')

    X, labels = cluster(wide, k=args.k)
    emit_table(wide, labels)
    emit_figure(wide, X, labels, args)

    out = wide[['domain', 'name']].assign(cluster=labels)
    path = os.path.join(RESULTS, 'cluster_assignments.csv')
    out.to_csv(path, index=False)
    print(f'Assignments → {path}')

    if args.diagnostics:
        diagnostics(X)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', choices=['robustness', 'real'], default='robustness',
                   help='Which results file supplies the entropy trajectories')
    p.add_argument('--k', type=int, default=K)
    p.add_argument('--boundary',
                   choices=['voronoi-filled', 'voronoi', 'none'],
                   default='voronoi-filled',
                   help="'voronoi' = the k-means decision boundaries, computed in "
                        "5-D and restricted to the plotted plane (default); "
                        "'voronoi-filled' adds light region shading; "
                        "'none' = centroids only")
    p.add_argument('--diagnostics', action='store_true',
                   help='Also print the cluster-validity indices across k')
    main(p.parse_args())
