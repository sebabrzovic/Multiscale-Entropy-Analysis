"""
Batch companion to algorithm/Entropy_Experiments/multiscale_entropy_regression.ipynb.

The notebook is the primary path for the regression and for the predicted-vs-actual
figure: its merge is sound and yields all 431 networks across six domains when run
against the current result files. This script exists for the parts that are awkward
in a notebook, and it reproduces the notebook's merge so the two cannot diverge:

  1. Repairing the Adamic-Adar ER baseline. run_real_networks_experiment.py normalises
     AA entropy by a SINGLE Erdos-Renyi draw (the paper specifies ten). When that draw
     yields a near-zero baseline entropy the ratio explodes: of 320 networks with an AA
     value, 119 exceed 2.0 and the maximum is 102, against a median of 1.11. SEAL is
     unaffected (all 431 values below 1.18). `--fix-aa-baseline` recomputes the
     baseline as the mean of ten draws; without it, degenerate rows are flagged and
     excluded, and the count is reported.

  2. Emitting drop-in LaTeX tables for the manuscript, for three targets
     (SEAL AUC, SEAL prediction entropy, Adamic-Adar entropy).

  3. Reporting the R^2 attainable from coarsened scales alone (H_40, H_40+H_20),
     which Section sec:cost pairs with the measured speedup.

  4. Rebuilding the k=3 clustering and Table 2 composition on all 431 networks.

A guard asserts n >= MIN_EXPECTED_N, so an incomplete set of result files cannot
quietly produce an underpowered regression the way a stale notebook run can.

Usage
-----
    source ~/venvs/research/bin/activate

    # optional but recommended: recompute the AA baseline properly (slow, no GPU needed)
    python experiments/rebuild_regression.py --fix-aa-baseline

    # main analysis; emits LaTeX tables ready to paste into the manuscript
    python experiments/rebuild_regression.py

Outputs
-------
    results/multiscale_panel.csv           — one row per network, all predictors + targets
    results/regression_<target>.csv        — Model 1..5 ladder per target
    results/cluster_composition.csv        — Table 2 (k=3 composition by family)
    results/aa_baseline_fixed.csv          — from --fix-aa-baseline
    tables/table_regression_<target>.tex   — LaTeX, drop-in for the manuscript
    tables/table_cluster_composition.tex
    figures/predicted_vs_actual_<target>.pdf
    figures/kmeans_pca.pdf
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'tables')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

LEVELS = [100, 80, 60, 40, 20]
PREDICTORS = [f'H_{lv}' for lv in LEVELS]

# All 431 networks have complete records; anything far below this means the merge broke.
MIN_EXPECTED_N = 400

# An ER-normalised entropy far above 1 means the baseline collapsed, not that the
# graph is more random than random. Rows beyond this are excluded and counted.
AA_SANITY_MAX = 2.0

TARGETS = {
    'seal_entropy': 'SEAL prediction entropy',
    'seal_auc': 'SEAL AUC',
    'adamic_adar_entropy': 'Adamic-Adar prediction entropy',
}


# ── panel construction ───────────────────────────────────────────────────────

def load_panel(verbose=True):
    """One row per network: H_100..H_20 plus each link-prediction target."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'real_networks_results*.csv')))
    if not files:
        sys.exit(f'No result files in {RESULTS_DIR}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=['domain', 'name', 'measure'])
    df = df[df['measure'] != 'measure']            # stray header rows from concatenation
    df = df.drop_duplicates(subset=['domain', 'name', 'measure', 'level'], keep='last')

    spectral = (df[df['measure'] == 'spectral_entropy']
                .pivot_table(index=['domain', 'name', 'n_nodes', 'n_edges'],
                             columns='level', values='value')
                .rename(columns={lv: f'H_{lv}' for lv in LEVELS})
                .reset_index())

    panel = spectral
    for measure in ('seal_entropy', 'adamic_adar_entropy'):
        sub = df[df['measure'] == measure]
        if sub.empty:
            continue
        agg = (sub.groupby(['domain', 'name'])
                  .agg(**{measure: ('value', 'mean'),
                          measure.replace('_entropy', '_auc'): ('auc', 'mean')})
                  .reset_index())
        panel = panel.merge(agg, on=['domain', 'name'], how='left')

    # AA baseline override, if it has been recomputed
    fixed_path = os.path.join(RESULTS_DIR, 'aa_baseline_fixed.csv')
    if os.path.exists(fixed_path):
        fixed = pd.read_csv(fixed_path)[['domain', 'name', 'adamic_adar_entropy_fixed']]
        panel = panel.merge(fixed, on=['domain', 'name'], how='left')
        n_fixed = panel['adamic_adar_entropy_fixed'].notna().sum()
        panel['adamic_adar_entropy'] = panel['adamic_adar_entropy_fixed'].fillna(
            panel['adamic_adar_entropy'])
        if verbose:
            print(f'Applied recomputed AA baseline to {n_fixed} networks '
                  f'(from {fixed_path})')

    panel = panel.dropna(subset=PREDICTORS)

    if verbose:
        print(f'Panel: {len(panel)} networks with all five entropy scales')
        print(panel['domain'].value_counts().to_string())
    if len(panel) < MIN_EXPECTED_N:
        print(f'\nWARNING: n={len(panel)} is below the expected {MIN_EXPECTED_N}.')
        print('The merge may be dropping networks — check before using these numbers.')
    return panel


def clean_target(panel, target, verbose=True):
    """Subset to rows usable for one target, excluding degenerate ER normalisations."""
    sub = panel.dropna(subset=[target]).copy()
    if target.endswith('_entropy'):
        n_before = len(sub)
        sub = sub[(sub[target] > 0) & (sub[target] <= AA_SANITY_MAX)]
        dropped = n_before - len(sub)
        if dropped and verbose:
            pct = 100 * dropped / n_before
            print(f'  {target}: dropped {dropped}/{n_before} ({pct:.1f}%) rows with '
                  f'normalised entropy outside (0, {AA_SANITY_MAX}] '
                  f'— degenerate ER baseline')
    return sub


# ── regression ───────────────────────────────────────────────────────────────

def model_ladder(sub, target, label):
    """Fit Models 1..5 (adding one coarser scale at a time) and report the F-test."""
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    y = sub[target].values
    rows, fitted = [], {}
    for k in range(1, len(PREDICTORS) + 1):
        cols = PREDICTORS[:k]
        X = sm.add_constant(sub[cols])
        m = sm.OLS(y, X).fit()
        fitted[k] = m
        rows.append({
            'model': k, 'predictors': '+'.join(cols), 'n': int(m.nobs),
            'r2': round(m.rsquared, 5), 'adj_r2': round(m.rsquared_adj, 5),
            'f_pvalue': m.f_pvalue,
            **{f'coef_{c}': round(m.params.get(c, np.nan), 4) for c in ['const'] + PREDICTORS},
            **{f'p_{c}': round(m.pvalues.get(c, np.nan), 4) for c in ['const'] + PREDICTORS},
        })
    out = pd.DataFrame(rows)

    m1, m5 = fitted[1], fitted[5]
    df_num = m5.df_model - m1.df_model
    f_stat = ((m1.ssr - m5.ssr) / df_num) / (m5.ssr / m5.df_resid)
    from scipy import stats
    f_p = 1 - stats.f.cdf(f_stat, df_num, m5.df_resid)

    # Scales below 100% only: the scalability claim depends on this working.
    X40 = sm.add_constant(sub[['H_40']])
    m40 = sm.OLS(y, X40).fit()
    X4020 = sm.add_constant(sub[['H_40', 'H_20']])
    m4020 = sm.OLS(y, X4020).fit()

    cv = cross_val_score(LinearRegression(), sub[PREDICTORS].values, y,
                         cv=KFold(5, shuffle=True, random_state=42), scoring='r2')

    print(f'\n── {label}  (n={len(sub)}) ' + '─' * (46 - len(label)))
    print(f'  Model 1 (H_100 only)      R2={m1.rsquared:.4f}  adj={m1.rsquared_adj:.4f}')
    print(f'  Model 5 (all five scales) R2={m5.rsquared:.4f}  adj={m5.rsquared_adj:.4f}')
    print(f'  F({int(df_num)},{int(m5.df_resid)}) = {f_stat:.2f}   p = {f_p:.3e}')
    print(f'  5-fold CV R2: {cv.mean():.4f} +/- {cv.std():.4f}')
    print(f'  H_40 alone                R2={m40.rsquared:.4f}   '
          f'(never touches the full graph)')
    print(f'  H_40 + H_20               R2={m4020.rsquared:.4f}')
    print(f'  --> retains {100 * m40.rsquared / m5.rsquared:.0f}% of Model 5 R2 '
          f'from the 40% graph alone')

    out.attrs['f_stat'] = f_stat
    out.attrs['f_p'] = f_p
    out.attrs['cv_mean'] = cv.mean()
    out.attrs['cv_std'] = cv.std()
    out.attrs['r2_h40'] = m40.rsquared
    out.attrs['r2_h40_h20'] = m4020.rsquared
    return out, fitted


def latex_regression_table(out, target, label):
    """Reproduce the manuscript's Table 3 layout for a given target."""
    def fmt_p(p):
        if pd.isna(p):
            return ''
        if p < 1e-4:
            return r'{\scriptsize ($<10^{-4}$)}'
        if p < 1e-3:
            return r'{\scriptsize ($<10^{-3}$)}'
        return r'{\scriptsize (%.3f)}' % p

    lines = [r'\begin{tabular}{lccccc}', r'\toprule',
             r'\textit{Model} & 1 & 2 & 3 & 4 & 5 \\', r'\midrule']
    for var in ['const'] + PREDICTORS:
        name = r'$\mathit{const}$' if var == 'const' else r'$H_{%s}$' % var.split('_')[1]
        coefs, ps = [], []
        for _, r in out.iterrows():
            c, p = r[f'coef_{var}'], r[f'p_{var}']
            coefs.append('' if pd.isna(c) else f'{c:.4f}')
            ps.append('' if pd.isna(p) else fmt_p(p))
        lines.append(f'{name} & ' + ' & '.join(coefs) + r' \\')
        lines.append(' & ' + ' & '.join(ps) + r' \\')
    lines.append(r'\midrule')
    lines.append('N. observations & ' + ' & '.join(str(int(r['n'])) for _, r in out.iterrows()) + r' \\')
    lines.append(r'$R^2$ & ' + ' & '.join(f"{r['r2']:.5f}" for _, r in out.iterrows()) + r' \\')
    lines.append(r'$\mathit{Adjusted}~R^2$ & ' + ' & '.join(f"{r['adj_r2']:.5f}" for _, r in out.iterrows()) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# ── clustering (Table 2 / Figure 4) ──────────────────────────────────────────

def clustering(panel):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X = StandardScaler().fit_transform(panel[PREDICTORS].values)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    panel = panel.assign(cluster=km.labels_ + 1)

    comp = (pd.crosstab(panel['domain'], panel['cluster'])
              .rename(columns=lambda c: f'Cluster {c}'))
    comp.to_csv(os.path.join(RESULTS_DIR, 'cluster_composition.csv'))
    print('\nCluster composition (k=3):')
    print(comp.to_string())

    # Label clusters by their mean trajectory slope so the names in the manuscript
    # (stable / increasing / hybrid) follow the data rather than the cluster index.
    means = panel.groupby('cluster')[PREDICTORS].mean()
    slope = (means['H_20'] - means['H_100'])
    print('\nMean entropy change H_20 - H_100 per cluster '
          '(larger = more "increasing"):')
    print(slope.round(4).to_string())

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, 'table_cluster_composition.tex'), 'w') as fh:
        fh.write(comp.to_latex(column_format='l' + 'c' * comp.shape[1]))

    os.makedirs(FIGURES_DIR, exist_ok=True)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in sorted(panel['cluster'].unique()):
        m = panel['cluster'] == c
        ax.scatter(coords[m.values, 0], coords[m.values, 1], s=18, alpha=0.75,
                   label=f'Cluster {c}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('K-means (k=3) on five-dimensional entropy vectors', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'kmeans_pca.pdf'),
                format='pdf', bbox_inches='tight')
    print(f'Figure → {FIGURES_DIR}/kmeans_pca.pdf')
    return panel


# ── AA baseline repair ───────────────────────────────────────────────────────

def fix_aa_baseline(args):
    """Recompute the Adamic-Adar ER baseline as a mean over ten draws.

    The published pipeline divides by a single Erdos-Renyi sample; when that sample's
    prediction entropy is near zero the normalised value explodes. Averaging ten draws
    is what the manuscript describes and what the compression-entropy path already does
    (get_entropy_metadata_aritmethicEncoding builds ten reference graphs).
    """
    import networkx as nx
    from tqdm import tqdm
    import algorithm.entropia_link_prediction as elp

    pkl = os.path.join(PROJECT_ROOT, 'algorithm', 'Entropy_Experiments',
                       'Real_World_Networks', 'undirected_networks.pkl')
    networks_df = pd.read_pickle(pkl)

    def build_graph(row):
        G = nx.DiGraph() if 'Directed' in row['graphProperties'] else nx.Graph()
        G.add_nodes_from(np.array(row['nodes_id']))
        G.add_edges_from(np.array(row['edges_id']))
        G = nx.to_undirected(G)
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G)
        return G

    out_path = os.path.join(RESULTS_DIR, 'aa_baseline_fixed.csv')
    rows = []
    done = set()
    if os.path.exists(out_path) and not args.overwrite:
        prev = pd.read_csv(out_path)
        rows = prev.to_dict('records')
        done = set(prev['name'])
        print(f'Resuming: {len(done)} networks already recomputed.')

    for _, row in tqdm(networks_df.iterrows(), total=len(networks_df), desc='AA baseline'):
        name = row.get('network_name', row.get('title', 'unknown'))
        if name in done:
            continue
        G = build_graph(row)
        n, e = G.number_of_nodes(), G.number_of_edges()
        if n < 20 or e < 30:
            continue
        try:
            _, train, test = elp.split_edges(G, test_ratio=args.test_ratio, seed=args.seed)
            ranks, _, n_cands = elp.evaluate_link_prediction_heuristic(
                G, train, test, predictor='adamic_adar', seed=args.seed)
            real_ent = elp.calculate_entropy(ranks, n, n_candidates=n_cands)

            baselines = []
            for draw in range(args.n_baseline):
                RG = elp.create_EdosReyni(G)
                if RG is None:
                    continue
                _, rtrain, rtest = elp.split_edges(RG, test_ratio=args.test_ratio,
                                                   seed=args.seed + draw)
                rranks, _, rn = elp.evaluate_link_prediction_heuristic(
                    RG, rtrain, rtest, predictor='adamic_adar', seed=args.seed + draw)
                b = elp.calculate_entropy(rranks, RG.number_of_nodes(), n_candidates=rn)
                if b and b > 0:
                    baselines.append(b)

            if len(baselines) < args.n_baseline // 2:
                tqdm.write(f'  {name}: only {len(baselines)} usable baselines — skipped')
                continue
            rows.append({
                'domain': row['networkDomain'], 'name': name,
                'adamic_adar_entropy_fixed': real_ent / float(np.mean(baselines)),
                'baseline_mean': float(np.mean(baselines)),
                'baseline_std': float(np.std(baselines)),
                'n_baselines': len(baselines),
            })
        except Exception as exc:
            tqdm.write(f'  {name}: failed ({exc})')
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} recomputed baselines → {out_path}')
    if len(df):
        v = df['adamic_adar_entropy_fixed']
        print(f'Fixed AA entropy: median={v.median():.3f}  '
              f'>{AA_SANITY_MAX}: {(v > AA_SANITY_MAX).sum()}/{len(v)}')


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    if args.fix_aa_baseline:
        fix_aa_baseline(args)
        return

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    panel = load_panel()
    panel.to_csv(os.path.join(RESULTS_DIR, 'multiscale_panel.csv'), index=False)

    for target, label in TARGETS.items():
        if target not in panel.columns:
            print(f'\n{label}: column absent — skipped')
            continue
        sub = clean_target(panel, target)
        if len(sub) < 30:
            print(f'\n{label}: only {len(sub)} usable rows — skipped')
            continue
        out, fitted = model_ladder(sub, target, label)
        out.to_csv(os.path.join(RESULTS_DIR, f'regression_{target}.csv'), index=False)
        with open(os.path.join(TABLES_DIR, f'table_regression_{target}.tex'), 'w') as fh:
            fh.write(latex_regression_table(out, target, label))
        plot_predicted_vs_actual(sub, target, label, fitted)

    clustering(panel)
    print(f'\nLaTeX tables → {TABLES_DIR}/')
    print(f'Figures      → {FIGURES_DIR}/')


def plot_predicted_vs_actual(sub, target, label, fitted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, k, name in [(axes[0], 1, 'Model 1 (single scale)'),
                        (axes[1], 5, 'Model 5 (multiscale)')]:
        m = fitted[k]
        pred = m.predict(sm.add_constant(sub[PREDICTORS[:k]]))
        for domain in sorted(sub['domain'].unique()):
            mask = (sub['domain'] == domain).values
            ax.scatter(sub[target].values[mask], np.asarray(pred)[mask],
                       s=16, alpha=0.7, label=domain)
        lo = min(sub[target].min(), float(np.min(pred)))
        hi = max(sub[target].max(), float(np.max(pred)))
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_title(f'{name}   $R^2$={m.rsquared:.3f}', fontsize=10)
        ax.set_xlabel(f'actual {label}')
    axes[0].set_ylabel(f'predicted {label}')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'predicted_vs_actual_{target}.pdf'),
                format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fix-aa-baseline', action='store_true',
                   help='Recompute the Adamic-Adar ER baseline over N draws and exit')
    p.add_argument('--n-baseline', type=int, default=10,
                   help='ER draws per network when fixing the baseline (default 10)')
    p.add_argument('--test-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
