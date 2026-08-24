"""
Real-networks experiment: multiscale spectral entropy + link prediction (AA & SEAL).

Usage
-----
    source ~/venvs/research/bin/activate
    python run_real_networks_experiment.py [--output results/] [--seal-epochs 50]
                                           [--test-ratio 0.2] [--seed 42]
                                           [--domain Biological]

Run one instance per domain to parallelise across terminals:
    python run_real_networks_experiment.py --domain Biological   &
    python run_real_networks_experiment.py --domain Social       &
    python run_real_networks_experiment.py --domain Economic     &
    python run_real_networks_experiment.py --domain Transportation &
    python run_real_networks_experiment.py --domain Technological &
    python run_real_networks_experiment.py --domain Informational &

Outputs
-------
    <output>/real_networks_results_<domain>.csv   — one domain (with --domain)
    <output>/real_networks_results_all.csv        — whole corpus (without --domain);
                                                    this is what the analysis reads
    <output>/real_networks_summary_<tag>.csv      — mean/std per domain
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from pygsp import graphs as pygsp_graphs
from tqdm import tqdm


# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments import common
from experiments.common import analytic_entropy_from_ranks, build_graph
from algorithm.coarsening_utils import coarsen, get_entropy_metadata_aritmethicEncoding
import algorithm.entropia_link_prediction as elp

# ── constants ────────────────────────────────────────────────────────────────
PKL_PATH = common.CORPUS_PKL
REDUCTION_LEVELS = common.REDUCTION_LEVELS
MIN_NODES = common.MIN_NODES
MIN_EDGES = common.MIN_EDGES

VALID_DOMAINS = common.DOMAINS


# ── analytic entropy normalisation (Sun et al. 2020) ─────────────────────────

def spectral_entropy_at_levels(G, reduction_levels):
    """Return dict {level_pct: normalised_entropy} for each reduction level (100 = original)."""
    W  = nx.to_scipy_sparse_array(G)
    Gp = pygsp_graphs.Graph(W)
    out = {}
    try:
        ent = get_entropy_metadata_aritmethicEncoding(G)
        out[100] = ent['Entropy Normalizado']
    except Exception as exc:
        tqdm.write(f'    entropy failed at 100%: {exc}')
        out[100] = None
    for r_pct in reduction_levels:
        try:
            _, Gc, _, __ = coarsen(Gp, K=10, r=1 - r_pct / 100)
            G_red = nx.from_scipy_sparse_array(Gc.W)
            ent   = get_entropy_metadata_aritmethicEncoding(G_red)
            out[r_pct] = ent['Entropy Normalizado']
        except Exception as exc:
            tqdm.write(f'    coarsen failed at {r_pct}%: {exc}')
            out[r_pct] = None
    return out


def link_prediction_entropy(G, method, res):
    """Analytic (Sun et al. 2020) normalisation of link-prediction entropy.

        H* = H / (log2(N) - 1)

    replacing the previous H(G)/H(G_R) with a simulated Erdos-Renyi denominator. That
    denominator is structurally broken for Adamic-Adar -- AA scores zero for any pair
    with no common neighbour, so on a sparse ER graph nearly every candidate ties at
    zero, the ranks collapse into one bin and H(G_R) -> 0 by construction. It is small
    and stable, not noisy, so averaging draws does not repair it: of 440 networks, 172
    gave a ratio above 2 and 98 gave zero or less, with a maximum of 67.

    log2(N) - 1 = log2(N/2) is the entropy of a uniform distribution over the N/2 rank
    bins -- the value a predictor carrying no information would produce. It is
    deterministic, bounded away from zero, and puts H* in [0, 1] by construction.

    Returns (H_star, H_raw, denom); (None, None, None) if ranks are unavailable.
    """
    if not res:
        return None, None, None
    ranks = res.get('ranks')
    if ranks is None or len(ranks) == 0:
        return None, None, None
    try:
        H_raw, denom, H_star = analytic_entropy_from_ranks(ranks, G.number_of_nodes())
        return H_star, H_raw, denom
    except Exception as exc:
        tqdm.write(f'    analytic normalisation failed ({method}): {exc}')
        return None, None, None


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output, exist_ok=True)

    networks_df = pd.read_pickle(args.corpus or PKL_PATH)

    if args.domain:
        networks_df = networks_df[networks_df['networkDomain'] == args.domain].reset_index(drop=True)
        if len(networks_df) == 0:
            print(f'No networks found for domain "{args.domain}". Valid domains: {VALID_DOMAINS}')
            sys.exit(1)

    if args.max_networks:
        networks_df = networks_df.head(args.max_networks)

    n_total = len(networks_df)
    domain_tag = args.domain or 'all'
    print(f'Loaded {n_total} networks  [domain={domain_tag}]')
    print(f'Domain counts: {networks_df["networkDomain"].value_counts().to_dict()}')
    print(f'SEAL epochs={args.seal_epochs}  test_ratio={args.test_ratio}  seed={args.seed}\n')

    # per-domain output file so concurrent runs don't collide
    domain_csv = os.path.join(args.output, f'real_networks_results_{domain_tag}.csv')

    seal_kw = dict(
        epochs=args.seal_epochs,
        hidden_channels=32,
        num_layers=3,
        num_hops=2,
        k=0.6,
        batch_size=32,
        max_neg_samples=args.max_neg_samples,
    )


    # ── resume ────────────────────────────────────────────────────────────────
    # Re-running must not recompute networks that are already done: SEAL dominates
    # the cost of this script. A network counts as complete only if it has all five
    # spectral entropy levels AND a SEAL row, all non-null -- a partial or failed
    # network is recomputed rather than treated as finished. Rows for complete
    # networks are carried forward, so an interrupted run cannot truncate the file.
    rows = []
    done = set()
    if os.path.exists(domain_csv) and not args.overwrite and not args.linkpred_only:
        prev = pd.read_csv(domain_csv)
        ok = prev.dropna(subset=['value'])
        spec_ok = (ok[ok['measure'] == 'spectral_entropy']
                   .groupby('name')['level'].nunique() >= 5)
        seal_ok = ok[ok['measure'] == 'seal_entropy'].groupby('name').size() >= 1
        complete = {n for n in spec_ok[spec_ok].index if seal_ok.get(n, False)}
        done = complete
        rows = [r for r in prev.to_dict('records') if r['name'] in done]
        n_partial = prev['name'].nunique() - len(done)
        print(f'Resuming from {domain_csv}')
        print(f'  complete networks kept   : {len(done)}')
        print(f'  incomplete, will redo    : {n_partial}')
        print(f'  rows carried forward     : {len(rows)} of {len(prev)}\n')
    elif args.overwrite and os.path.exists(domain_csv):
        print(f'--overwrite: discarding existing {domain_csv}\n')

    n_processed = 0
    n_skipped   = 0
    n_cached    = 0

    # Spectral rows to reuse under --linkpred-only, keyed by network name.
    prior_spectral = {}
    if args.linkpred_only:
        # Spectral rows may sit in the consolidated file or in the per-domain files,
        # depending on how the run that produced them was invoked. Reading only
        # domain_csv silently recomputed a fraction of the corpus when --domain was
        # omitted, so gather from every result file present and dedupe.
        candidates = [domain_csv] + [
            os.path.join(args.output, f'real_networks_results_{d}.csv')
            for d in VALID_DOMAINS]
        found = [f for f in dict.fromkeys(candidates) if os.path.exists(f)]
        if not found:
            sys.exit(f'--linkpred-only needs existing spectral rows, but no '
                     f'real_networks_results*.csv exists in {args.output}.')
        _prev = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
        _prev = _prev[_prev['measure'] == 'spectral_entropy']
        _prev = _prev.drop_duplicates(subset=['name', 'level'], keep='last')
        for _n, _g in _prev.groupby('name'):
            prior_spectral[_n] = _g.to_dict('records')
        print(f'--linkpred-only: reusing spectral entropy for '
              f'{len(prior_spectral)} network(s); only link prediction is recomputed.\n')

    for _, row in tqdm(networks_df.iterrows(), total=n_total, desc=domain_tag):
        domain = row['networkDomain']
        name   = row.get('network_name', row.get('title', 'unknown'))

        G = build_graph(row)
        n = G.number_of_nodes()
        e = G.number_of_edges()

        if name in done:
            n_cached += 1
            continue

        if n < MIN_NODES or e < MIN_EDGES:
            n_skipped += 1
            tqdm.write(f'[{domain}] {name}  — skipped (too small: {n} nodes, {e} edges)')
            continue


        n_processed += 1
        remaining = n_total - n_processed - n_skipped
        tqdm.write(f'\n--- [{n_processed} processed | {n_skipped} skipped | {remaining} remaining] '
                   f'[{domain}] {name}  ({n} nodes, {e} edges)')

        base = dict(domain=domain, name=name, n_nodes=n, n_edges=e)

        # 1. Spectral entropy
        if args.linkpred_only:
            # Carry the existing spectral rows through untouched.
            for r in prior_spectral.get(name, []):
                rows.append(r)
            tqdm.write(f'  [1/3] spectral entropy: reused '
                       f'{len(prior_spectral.get(name, []))} cached row(s)')
        else:
            tqdm.write('  [1/3] spectral entropy (original + 4 coarsening levels)...')
            spec = spectral_entropy_at_levels(G, REDUCTION_LEVELS)
            level_summary = '  '.join(
                f'H_{r}={v:.3f}' if v is not None else f'H_{r}=None'
                for r, v in sorted(spec.items(), reverse=True))
            tqdm.write(f'        {level_summary}')
            for r_pct, val in spec.items():
                rows.append({**base, 'level': r_pct, 'measure': 'spectral_entropy',
                             'value': val, 'auc': None})

        # 2. Link prediction

        tqdm.write(f'  [2/3] link prediction (AA + SEAL, epochs={args.seal_epochs})...')
        try:
            lp = elp.compare_methods_shared_split(
                G,
                heuristics=('adamic_adar',),
                test_ratio=args.test_ratio,
                seed=args.seed,
                seal_kwargs=seal_kw,
            )
        except Exception as exc:
            tqdm.write(f'        failed: {exc}')
            lp = {}

        aa_res   = lp.get('adamic_adar', {})
        seal_res = lp.get('seal', {})
        tqdm.write(f'        AA   auc={aa_res.get("auc", "n/a")}')
        tqdm.write(f'        SEAL auc={seal_res.get("auc", "n/a")}')

        tqdm.write('  [3/3] analytic normalisation  H* = H / (log2(N) - 1)...')
        for method, label in [('adamic_adar', 'adamic_adar_entropy'),
                               ('seal',        'seal_entropy')]:
            res = lp.get(method, {})
            H_star, H_raw, denom = link_prediction_entropy(G, method, res)
            # H_raw and denom are stored alongside the ratio. Persisting only the
            # quotient is what made the previous normalisation failure impossible to
            # diagnose from the result files -- neither the numerator nor the
            # denominator could be recovered without re-running the whole pipeline.
            rows.append({**base, 'level': 100, 'measure': label,
                         'value': H_star, 'auc': res.get('auc'),
                         'H_raw': H_raw, 'denom': denom})
            tqdm.write(f'        {method}: H*={H_star}  (H_raw={H_raw}, '
                       f'denom={denom})')

        # incremental save to domain file
        pd.DataFrame(rows).to_csv(domain_csv, index=False)

    # ── Save domain results ───────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(domain_csv, index=False)
    tqdm.write(f'\nSaved {len(df)} rows → {domain_csv}'
               f'  ({n_processed} computed, {n_cached} reused, {n_skipped} too small)')

    # No merge step: a run without --domain already writes the consolidated
    # real_networks_results_all.csv, which is what rebuild_regression.py reads. The
    # old merge combined only the per-domain files, excluding _all.csv, so the merged
    # file and the file every downstream script reads drifted apart.

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_parts = []
    spec_df = df[df['measure'] == 'spectral_entropy']
    for level in [100] + REDUCTION_LEVELS:
        s = (spec_df[spec_df['level'] == level]
             .groupby('domain')['value']
             .agg(['mean', 'std'])
             .rename(columns={'mean': f'spectral_{level}pct_mean',
                              'std':  f'spectral_{level}pct_std'}))
        summary_parts.append(s)

    for method_label in ['adamic_adar_entropy', 'seal_entropy']:
        sub = df[df['measure'] == method_label]
        prefix = method_label.replace('_entropy', '')
        summary_parts.append(sub.groupby('domain')['value'].agg(['mean', 'std'])
                               .rename(columns={'mean': f'{prefix}_entropy_mean',
                                                'std':  f'{prefix}_entropy_std'}))
        summary_parts.append(sub.groupby('domain')['auc'].agg(['mean', 'std'])
                               .rename(columns={'mean': f'{prefix}_auc_mean',
                                                'std':  f'{prefix}_auc_std'}))

    if summary_parts:
        summary = pd.concat(summary_parts, axis=1).round(4)
        summary_path = os.path.join(args.output, f'real_networks_summary_{domain_tag}.csv')
        summary.to_csv(summary_path)
        tqdm.write(f'Saved summary → {summary_path}')
        print(summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real networks entropy + link prediction experiment')
    parser.add_argument('--output',       default='results',  help='Output directory')
    parser.add_argument('--domain',       default=None,       help=f'Run only this domain: {VALID_DOMAINS}')
    parser.add_argument('--seal-epochs',  type=int,   default=50,   help='SEAL training epochs')
    parser.add_argument('--test-ratio',   type=float, default=0.2,  help='Fraction of edges for test set (SEAL paper uses 0.1)')
    parser.add_argument('--corpus', default=None,
                   help='Path to a corpus pickle. Defaults to the undirected-only'
                        ' corpus; pass all_networks.pkl (see build_full_corpus.py)'
                        ' to include symmetrized directed networks.')
    parser.add_argument('--seed',         type=int,   default=42,   help='Random seed')
    parser.add_argument('--linkpred-only', action='store_true',
                        help='Recompute only the link-prediction entropies, reusing '
                             'the cached spectral entropy rows. The analytic '
                             'normalisation affects only link prediction, and '
                             'recomputing compression entropy at five levels is the '
                             'expensive part of a full run.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Ignore existing results and recompute everything. Without '
                             'this, complete networks are kept and skipped.')
    parser.add_argument('--max-networks',    type=int, default=None,  help='Limit to first N networks (for test runs)')
    parser.add_argument('--max-neg-samples', type=int, default=100000, help='Max negative pairs scored by SEAL (small graphs use fewer naturally)')
    main(parser.parse_args())
