"""
Build the full analysis corpus, including the networks flagged "Directed".

The corpus used previously kept only graphs recorded as undirected in the CommunityFitNet
metadata, dropping 129 of 572. The loss was very uneven -- technological networks retained
16 of 74 and informational 11 of 22 -- because circuits, software graphs, citations and web
graphs are predominantly directed at source.

That exclusion turns out to be unnecessary. CommunityFitNet stores every edge list in
canonical unordered form: across all 572 networks, directed and undirected alike, every
edge satisfies u <= v. Consequently the stored graphs carry no orientation information,
every "directed" network reads as acyclic with zero reciprocity, and building an
undirected graph from its edge list neither discards nor invents structure. The
graphProperties flag describes the original source network, not the representation in
the corpus.

This script therefore builds the corpus from all 572 networks. It still computes
reciprocity per network as a guard: if a future version of the source data does carry
orientation, reciprocity will be non-zero and the assumption above must be revisited
before pooling. The `was_directed` flag is retained on every row for the same reason --
it allows testing whether the newly included networks sit in different entropy regimes
than the originally included ones.

Usage
-----
    python experiments/build_full_corpus.py --source data/CommunityFitNet_updated.pickle

Output
------
    algorithm/Entropy_Experiments/Real_World_Networks/all_networks.pkl
        Same schema as undirected_networks.pkl, plus:
          was_directed    bool  — flagged Directed in the source metadata
          n_reciprocal    int   — reciprocated pairs found (expected 0, see above)
          reciprocity     float — n_reciprocal / edge count (expected 0)
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PKL = os.path.join(PROJECT_ROOT, 'algorithm', 'Entropy_Experiments',
                       'Real_World_Networks', 'all_networks.pkl')

MIN_NODES = 20
MIN_EDGES = 30


def symmetrize(row):
    """Return (nodes, undirected edges, was_directed, n_reciprocal, reciprocity).

    Symmetrization collapses each reciprocated pair into one edge, so the undirected
    edge count is generally lower than the directed one. Reciprocity is reported per
    network because it measures how much information the collapse discards: a graph
    with high reciprocity loses little, one with near-zero reciprocity loses the entire
    orientation structure.
    """
    directed = 'Directed' in str(row['graphProperties'])
    nodes = np.array(row['nodes_id'])
    edges = np.array(row['edges_id'])

    if edges.ndim != 2 or edges.shape[0] == 0:
        return nodes, edges, directed, 0, 0.0

    pairs = {(int(u), int(v)) for u, v in edges[:, :2]}
    if not directed:
        undirected = {tuple(sorted(p)) for p in pairs if p[0] != p[1]}
        return nodes, np.array(sorted(undirected)), False, 0, 0.0

    n_recip = sum(1 for (u, v) in pairs if u != v and (v, u) in pairs) // 2
    undirected = {tuple(sorted((u, v))) for (u, v) in pairs if u != v}
    recip = (2 * n_recip / len(pairs)) if pairs else 0.0
    return nodes, np.array(sorted(undirected)), True, n_recip, recip


def main(args):
    if not os.path.exists(args.source):
        sys.exit(
            f'Source corpus not found: {args.source}\n\n'
            'CommunityFitNet_updated.pickle is not in this repository. Download it from\n'
            '  https://github.com/Aghasemian/CommunityFitNet\n'
            'and pass its path with --source.'
        )

    df = pd.read_pickle(args.source)
    print(f'Loaded {len(df)} networks from {os.path.basename(args.source)}')
    print(f'Domains: {df["networkDomain"].value_counts().to_dict()}\n')

    rows, skipped = [], []
    for _, row in df.iterrows():
        nodes, edges, was_directed, n_recip, recip = symmetrize(row)
        if len(edges) == 0:
            skipped.append((row.get('network_name', '?'), 'no edges'))
            continue

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        if G.number_of_nodes() == 0:
            skipped.append((row.get('network_name', '?'), 'empty'))
            continue
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        n, e = G.number_of_nodes(), G.number_of_edges()
        if n < MIN_NODES or e < MIN_EDGES:
            skipped.append((row.get('network_name', '?'), f'too small ({n}n/{e}e)'))
            continue

        r = row.copy()
        # Store the symmetrized graph and mark it Undirected, so every downstream
        # consumer builds an undirected graph without special-casing.
        r['nodes_id'] = np.array(sorted(G.nodes()))
        r['edges_id'] = np.array(sorted(G.edges()))
        r['number_nodes'] = n
        r['number_edges'] = e
        r['ave_degree'] = 2 * e / n
        r['graphProperties'] = str(row['graphProperties']).replace('Directed', 'Undirected')
        r['was_directed'] = was_directed
        r['n_reciprocal'] = n_recip
        r['reciprocity'] = recip
        rows.append(r)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out.to_pickle(OUT_PKL, protocol=4)

    print(f'Retained {len(out)} networks  ({len(skipped)} dropped)')
    print(f'  natively undirected : {(~out.was_directed).sum()}')
    print(f'  symmetrized         : {out.was_directed.sum()}')
    print('\nPer domain (kept / symmetrized):')
    for dom, g in out.groupby('networkDomain'):
        print(f'  {dom:16s} {len(g):4d}   ({int(g.was_directed.sum())} symmetrized)')
    if out.was_directed.any():
        rr = out[out.was_directed]['reciprocity']
        print(f'\nReciprocity of symmetrized graphs: '
              f'median={rr.median():.3f}  mean={rr.mean():.3f}  '
              f'min={rr.min():.3f}  max={rr.max():.3f}')
        if rr.max() == 0:
            print('  All zero, as expected: the source stores edges as unordered pairs')
            print('  (u <= v) for every network, so no orientation was present to lose.')
        else:
            print('  NON-ZERO reciprocity: this source DOES carry orientation, contrary to')
            print('  the assumption documented above. Revisit before pooling these graphs')
            print('  with the natively undirected ones.')
    print(f'\nSaved → {OUT_PKL}')
    print('\nPoint the experiment scripts at this file with --corpus to use it.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True,
                   help='Path to CommunityFitNet_updated.pickle (572 networks)')
    main(p.parse_args())
