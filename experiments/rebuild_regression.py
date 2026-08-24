"""
Regression analysis behind Table 3 and the predictability claims of Section sec:seal.

Rebuilds, from results/real_networks_results_all.csv, the Model 1..5 ladder for four
targets -- SEAL AUC, SEAL prediction entropy, Adamic-Adar AUC and Adamic-Adar prediction
entropy -- on the full 558-network corpus.
Every number quoted in the manuscript for these analyses should be reproducible by
running this script. The k=3 clustering behind Table 2 and Figure 4 is NOT duplicated
here -- experiments/clustering_analysis.py owns it, and labels clusters canonically by
increasing mean H_100. Two implementations would give two versions of the same table.

What it reports beyond the plain ladder
---------------------------------------
  * Coarse-scale-only models. H_40 and H_60 alone, and H_100+H_80+H_60 — the scales at
    which the two coarsening algorithms agree (see coarsening_robustness.py). These
    support the cost argument in sec:cost: a coarse scale alone outperforms the
    uncoarsened graph, and the three shallowest scales recover almost all of the
    five-scale fit.

  * The stratum split. Networks flagged Directed in the source metadata were absent
    from the original corpus and have flatter, higher-entropy trajectories. The
    entropy-predictability relationship is markedly weaker among them, so the ladder is
    also fitted separately per stratum: a relationship holding in only one stratum is
    not a corpus-wide result.

A guard warns if n < MIN_EXPECTED_N, so an incomplete link-prediction run cannot quietly
produce an underpowered regression.

Usage
-----
    python experiments/rebuild_regression.py

Outputs
-------
    results/multiscale_panel.csv           — one row per network, predictors + targets
    results/regression_<target>.csv        — Model 1..5 ladder per target
    tables/table_regression_<target>.tex   — LaTeX, drop-in for the manuscript
    figures/predicted_vs_actual_<target>.pdf
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

from experiments.common import LEVELS, CORPUS_PKL, write_table   # noqa: E402

PREDICTORS = [f'H_{lv}' for lv in LEVELS]

# The full corpus is 558 networks (all_networks.pkl); anything far below this means the
# merge broke or the link-prediction run is incomplete.
MIN_EXPECTED_N = 550

# Both predictors are reported against both measures of predictability: the
# distributional one (prediction entropy) and the accuracy one (AUC). Reporting AUC
# for SEAL but not for Adamic-Adar would leave the obvious comparison unmade.
TARGETS = {
    'seal_entropy': 'SEAL prediction entropy',
    'seal_auc': 'SEAL AUC',
    'adamic_adar_entropy': 'Adamic-Adar prediction entropy',
    'adamic_adar_auc': 'Adamic-Adar AUC',
}


# ── panel construction ───────────────────────────────────────────────────────

def load_panel(verbose=True):
    """One row per network: H_100..H_20 plus each link-prediction target."""
    # Prefer the consolidated all-domain file. Globbing real_networks_results*.csv
    # also picks up the per-domain files, whose link-prediction rows still carry the
    # OLD simulated-ER normalisation; deduplication then depends on filename sort
    # order to keep the right ones, which is not something to rely on.
    consolidated = os.path.join(RESULTS_DIR, 'real_networks_results_all.csv')
    if os.path.exists(consolidated):
        files = [consolidated]
        print(f'Reading {os.path.basename(consolidated)} (analytic H* normalisation)')
    else:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              'real_networks_results*.csv')))
        print(f'{os.path.basename(consolidated)} absent; falling back to '
              f'{len(files)} per-domain file(s) -- check which normalisation these use')
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

    # Stratum flag: networks flagged Directed in the source metadata were absent from
    # the original corpus. They behave differently enough (flat, high-entropy
    # trajectories) that pooling them without checking would hide a real effect.
    full_pkl = CORPUS_PKL
    if os.path.exists(full_pkl):
        allnet = pd.read_pickle(full_pkl)
        ncol = 'network_name' if 'network_name' in allnet.columns else 'title'
        wd = {str(r[ncol]): bool(r.get('was_directed', False))
              for _, r in allnet.iterrows()}
        panel['newly_included'] = [wd.get(str(n), False) for n in panel['name']]
    else:
        panel['newly_included'] = False

    panel = panel.dropna(subset=PREDICTORS)

    if verbose:
        print(f'Panel: {len(panel)} networks with all five entropy scales')
        print(panel['domain'].value_counts().to_string())
    if len(panel) < MIN_EXPECTED_N:
        print(f'\nWARNING: n={len(panel)} is below the expected {MIN_EXPECTED_N}.')
        print('The merge may be dropping networks — check before using these numbers.')
    return panel


def clean_target(panel, target, verbose=True):
    """Rows usable for one target.

    Under the analytic normalisation H* = H/(log2 N - 1) every entropy target is in
    [0, 1] by construction, so there is nothing to filter -- an out-of-range value
    would be a bug and is raised rather than quietly dropped. The old simulated-ER
    ratio needed a sanity bound because its denominator could collapse toward zero;
    that normalisation is gone.
    """
    sub = panel.dropna(subset=[target]).copy()
    if target.endswith('_entropy') and len(sub):
        bad = sub[(sub[target] < -1e-9) | (sub[target] > 1 + 1e-9)]
        if len(bad):
            raise AssertionError(
                f'{target}: {len(bad)} value(s) outside [0,1] — '
                f'range {sub[target].min():.3f} to {sub[target].max():.3f}. '
                f'Are these still ER-normalised?')
    return sub


# ── regression ───────────────────────────────────────────────────────────────

def model_ladder(sub, target, label):
    """Fit Models 1..5 (adding one coarser scale at a time) and report the F-test."""
    import statsmodels.api as sm
    from scipy import stats
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
    f_p = 1 - stats.f.cdf(f_stat, df_num, m5.df_resid)

    # Subsets that matter for the manuscript's claims:
    #   H_40 / H_60 alone  -> the scalability argument in sec:cost
    #   H_100+H_80+H_60    -> the scales where the two coarsening algorithms agree
    subsets = {'H_40 alone': ['H_40'],
               'H_60 alone': ['H_60'],
               'H_40 + H_20': ['H_40', 'H_20'],
               'H_100+H_80+H_60': ['H_100', 'H_80', 'H_60']}
    sub_r2 = {k: sm.OLS(y, sm.add_constant(sub[c])).fit().rsquared
              for k, c in subsets.items()}
    m40 = sm.OLS(y, sm.add_constant(sub[['H_40']])).fit()

    cv = cross_val_score(LinearRegression(), sub[PREDICTORS].values, y,
                         cv=KFold(5, shuffle=True, random_state=42), scoring='r2')

    print(f'\n── {label}  (n={len(sub)}) ' + '─' * (46 - len(label)))
    print(f'  Model 1 (H_100 only)      R2={m1.rsquared:.4f}  adj={m1.rsquared_adj:.4f}')
    print(f'  Model 5 (all five scales) R2={m5.rsquared:.4f}  adj={m5.rsquared_adj:.4f}')
    print(f'  F({int(df_num)},{int(m5.df_resid)}) = {f_stat:.2f}   p = {f_p:.3e}')
    print(f'  5-fold CV R2: {cv.mean():.4f} +/- {cv.std():.4f}')
    for k, v in sub_r2.items():
        note = '  (never touches the full graph)' if 'H_100' not in k else ''
        print(f'  {k:22s} R2={v:.4f}{note}')
    print(f'  --> H_40 alone retains {100 * m40.rsquared / m5.rsquared:.0f}% of Model 5 '
          f'R2, against {m1.rsquared:.4f} from the uncoarsened graph')

    # Stratum split. Reported whenever both strata are large enough to fit; a
    # relationship that holds only in one of them is not a corpus-wide result.
    if 'newly_included' in sub.columns and sub['newly_included'].nunique() > 1:
        print('  by stratum:')
        for stratum, mask in [('originally included', ~sub['newly_included']),
                              ('newly included', sub['newly_included'])]:
            g = sub[mask.values]
            if len(g) < 20:
                continue
            a1 = sm.OLS(g[target], sm.add_constant(g[PREDICTORS[:1]])).fit()
            a5 = sm.OLS(g[target], sm.add_constant(g[PREDICTORS])).fit()
            dfn = a5.df_model - a1.df_model
            f = ((a1.ssr - a5.ssr) / dfn) / (a5.ssr / a5.df_resid)
            pv = 1 - stats.f.cdf(f, dfn, a5.df_resid)
            print(f'    {stratum:22s} n={len(g):4d}  M1={a1.rsquared:.4f} '
                  f'M5={a5.rsquared:.4f}  F={f:.1f} p={pv:.2e}')

    out.attrs['f_stat'] = f_stat
    out.attrs['f_p'] = f_p
    out.attrs['cv_mean'] = cv.mean()
    out.attrs['cv_std'] = cv.std()
    out.attrs['r2_h40'] = m40.rsquared
    out.attrs.update({f'r2_{k}': v for k, v in sub_r2.items()})
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


# ── AA baseline repair ───────────────────────────────────────────────────────

def main(args):
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
        write_table(f'table_regression_{target}.tex',
                    latex_regression_table(out, target, label).split('\n'))
        plot_predicted_vs_actual(sub, target, label, fitted)

    print(f'\nLaTeX tables → {TABLES_DIR}/')
    print(f'Figures      → {FIGURES_DIR}/')


def plot_predicted_vs_actual(sub, target, label, fitted):
    """Predicted vs actual, one panel per model.

    Axis limits come from a Tukey far fence (3x IQR) on the observed values rather than
    from their extremes. A single network can otherwise stretch the axis far past the
    bulk of the data and compress the region where all the structure is -- for SEAL AUC
    one graph sits at 0.286 while the rest begin around 0.55. Outlying points are drawn,
    not dropped: matplotlib clips them at the spine, and the count is reported in the
    axis label so the reader knows they exist.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    preds = {k: fitted[k].predict(sm.add_constant(sub[PREDICTORS[:k]]))
             for k in (1, 5)}
    # Tukey far fence (3 x IQR) on the ACTUAL values. A quantile cut on the pooled
    # actual-and-predicted values is too blunt: predictions occupy a narrower band, so
    # pooling pulls the lower quantile up and clips genuine low observations along with
    # the true outlier. For SEAL AUC that cost eight points in the legitimate 0.51-0.55
    # tail to exclude the single network at 0.286. The far fence flags only points that
    # are extreme relative to the spread of the data itself.
    a = np.asarray(sub[target].values, dtype=float)
    q1, q3 = np.quantile(a, [0.25, 0.75])
    iqr = q3 - q1
    inl = a[(a >= q1 - 3 * iqr) & (a <= q3 + 3 * iqr)]
    allp = np.concatenate([np.asarray(v, dtype=float) for v in preds.values()])
    lo = min(inl.min(), allp.min())
    hi = max(inl.max(), allp.max())
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    n_out = int(((sub[target] < lo) | (sub[target] > hi)).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, k, name in [(axes[0], 1, 'Model 1 (single scale)'),
                        (axes[1], 5, 'Model 5 (multiscale)')]:
        m = fitted[k]
        pred = preds[k]
        for domain in sorted(sub['domain'].unique()):
            mask = (sub['domain'] == domain).values
            ax.scatter(sub[target].values[mask], np.asarray(pred)[mask],
                       s=20, alpha=0.6, edgecolors='none', label=domain,
                       clip_on=True)
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f'{name}   $R^2$={m.rsquared:.3f}', fontsize=10)
        xl = f'actual {label}'
        if n_out:
            xl += f'  ({n_out} network{"s" if n_out > 1 else ""} outside axis range)'
        ax.set_xlabel(xl)
    axes[0].set_ylabel(f'predicted {label}')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'predicted_vs_actual_{target}.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  figure: axis [{lo:.3f}, {hi:.3f}], {n_out} point(s) outside')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    main(p.parse_args())
