"""Link prediction and prediction entropy.

Two predictors are evaluated on a shared train/test split: the Adamic-Adar
neighborhood heuristic and SEAL, a graph neural network over enclosing subgraphs.
Both report an AUC and a prediction entropy, the Shannon entropy of the held-out
edges' rank distribution.

The SEAL implementation (_drnl_node_labeling, _extract_enclosing_subgraph, _DGCNN,
evaluate_link_prediction_seal) is adapted from the PyTorch Geometric example at
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py
See THIRD_PARTY.md.
"""

import math
from collections import Counter

import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, MLP, SortAggregation
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score


# Seeded generator for tie-breaking; see tiebroken_rank below.
_TIE_RNG = np.random.default_rng(20260821)


def tiebroken_rank(pair_scores_arr, target, rng=None):
    """Rank a held-out edge among the candidate scores, breaking ties at random.

        rank = n_greater + U{1, ..., n_ties}

    Counting only strictly-greater scores would assign the BEST rank in the tied
    block, and taking n_greater + n_ties the worst. Either makes the answer a
    property of the convention rather than of the graph wherever a scorer produces
    many ties, as Adamic-Adar does at zero: on a sparse ER graph with 93% of held-out
    edges scoring exactly zero, the same data yields H* = 0.083 (best), 0.081 (worst)
    and 0.969 (random). Only the last correctly reports that AA cannot predict a
    random graph. SEAL's continuous scores tie rarely, so the convention barely moves
    its value -- but the two predictors should be measured the same way.
    """
    rng = _TIE_RNG if rng is None else rng
    n_greater = int(np.sum(pair_scores_arr > target))
    n_ties = int(np.sum(pair_scores_arr == target)) + 1   # +1: the edge itself
    return n_greater + int(rng.integers(1, n_ties + 1))

def split_edges(G, test_ratio=0.1, seed=42):
    """
    Split graph edges into train and test sets with a fixed random seed.

    Returns
    -------
    node_list : list
        Ordered node list (index i → original node label).
    train_edges : list of (int, int)
        Training edges using re-indexed node ids 0..N-1.
    test_edges : list of (int, int)
        Test edges using re-indexed node ids 0..N-1.
    """
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    node_list = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}
    all_edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_edges))
    all_edges = [all_edges[i] for i in perm]
    n_test = max(1, int(len(all_edges) * test_ratio))
    return node_list, all_edges[n_test:], all_edges[:n_test]


def _global_ranks(pair_scores_arr, candidate_pairs, test_edges, train_set, score_fn):
    """
    Rank each test edge globally against all non-training pairs.

    pair_scores_arr : 1-D array of scores for candidate_pairs (same order).
    score_fn        : callable((u, v)) → float, used to score the test edge itself.
    """
    ranks = []
    for (u, v) in test_edges:
        target = score_fn(u, v)
        ranks.append(tiebroken_rank(pair_scores_arr, target))
    return ranks


# ---------------------------------------------------------------------------
# Heuristic link prediction on a pre-split train/test partition
# ---------------------------------------------------------------------------

def evaluate_link_prediction_heuristic(G, train_edges, test_edges,
                                        predictor='adamic_adar', seed=42):
    """
    Evaluate a heuristic link predictor on a pre-defined train/test split.

    Uses the same global ranking approach as the GCN evaluator: each test
    edge is ranked against ALL non-training pairs so entropy values are
    directly comparable.  AUC is computed by sampling an equal number of
    random non-edges as negatives.

    Parameters
    ----------
    G : networkx.Graph
        Original graph (used only to determine node set).
    train_edges : list of (int, int)
    test_edges  : list of (int, int)
    predictor   : 'jaccard' or 'adamic_adar'
    seed        : random seed for negative sampling (AUC)

    Returns
    -------
    ranks : list of int
    auc   : float
    """
    if predictor == 'jaccard':
        pred_func = nx.jaccard_coefficient
    elif predictor == 'adamic_adar':
        pred_func = nx.adamic_adar_index
    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    n_nodes = G.number_of_nodes()
    train_set = set(map(tuple, train_edges)) | {(v, u) for u, v in train_edges}
    test_set  = set(map(tuple, test_edges))  | {(v, u) for u, v in test_edges}

    # Build train graph (re-indexed nodes 0..N-1)
    G_train = nx.Graph()
    G_train.add_nodes_from(range(n_nodes))
    G_train.add_edges_from(train_edges)

    # All non-training upper-triangle pairs
    triu_i, triu_j = np.triu_indices(n_nodes, k=1)
    candidate_pairs = [
        (int(i), int(j)) for i, j in zip(triu_i, triu_j)
        if (int(i), int(j)) not in train_set
    ]

    # Score all candidate pairs at once
    scored = list(pred_func(G_train, candidate_pairs))
    pair_scores_arr = np.array([s for _, _, s in scored])
    pair_to_score = {(u, v): s for u, v, s in scored}

    def score_fn(u, v):
        return pair_to_score.get((u, v), pair_to_score.get((v, u), 0.0))

    # Ranks
    ranks = []
    for (u, v) in test_edges:
        target = score_fn(u, v)
        ranks.append(tiebroken_rank(pair_scores_arr, target))

    # AUC — positives: test edges; negatives: random sample of non-edges
    pos_scores = np.array([score_fn(u, v) for u, v in test_edges])

    rng = np.random.default_rng(seed)
    neg_pairs = [
        p for p in candidate_pairs
        if p not in test_set and (p[1], p[0]) not in test_set
    ]
    n_neg = min(len(pos_scores), len(neg_pairs))
    neg_sample = [neg_pairs[i] for i in rng.choice(len(neg_pairs), n_neg, replace=False)]
    neg_scores = np.array([score_fn(u, v) for u, v in neg_sample])

    y_true  = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_score)

    return ranks, auc, len(candidate_pairs)


# ---------------------------------------------------------------------------
# GCN link prediction (accepts external train/test split)
# ---------------------------------------------------------------------------

def _drnl_node_labeling(edge_index, src, dst, num_nodes):
    """
    Double Radius Node Labeling (DRNL) — PyG official implementation.

    Uses scipy shortest_path on the graph with src (resp. dst) removed to get
    distances that correctly handle the two-endpoint boundary conditions.
    Breaks structural symmetry in regular graphs (e.g. Watts-Strogatz).
    """
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.
    return z.to(torch.long)


def _extract_enclosing_subgraph(train_edge_index, src, dst, n_nodes, num_hops):
    """Extract h-hop enclosing subgraph around (src, dst) with DRNL labels."""
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        [src, dst], num_hops, train_edge_index,
        relabel_nodes=True, num_nodes=n_nodes,
    )
    new_src, new_dst = mapping[0].item(), mapping[1].item()

    mask1 = (sub_edge_index[0] != new_src) | (sub_edge_index[1] != new_dst)
    mask2 = (sub_edge_index[0] != new_dst) | (sub_edge_index[1] != new_src)
    sub_edge_index = sub_edge_index[:, mask1 & mask2]

    z = _drnl_node_labeling(sub_edge_index, new_src, new_dst,
                             num_nodes=subset.size(0))
    return Data(z=z, edge_index=sub_edge_index, num_nodes=subset.size(0))


class _DGCNN(torch.nn.Module):
    """
    DGCNN model from the PyG SEAL example.
    Architecture: GCN layers → SortAggregation → Conv1d x 2 → MLP.
    """

    def __init__(self, hidden_channels, num_layers, k, in_channels):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))

        self.pool = SortAggregation(k)

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, 128, 1], dropout=0.5, norm=None)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        x = torch.cat(xs[1:], dim=-1)
        x = self.pool(x, batch)
        x = x.unsqueeze(1)
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)
        return self.mlp(x)


def evaluate_link_prediction_seal(G, train_edges, test_edges,
                                   hidden_channels=32, num_layers=3,
                                   num_hops=2, k=0.6, epochs=50,
                                   lr=0.0001, batch_size=32, seed=42,
                                   max_neg_samples=50000):
    """
    SEAL link prediction (Zhang et al. 2018) using the official PyG DGCNN.

    Uses DRNL-labeled enclosing subgraphs.  Works on structurally regular
    graphs (e.g. Watts-Strogatz) where plain GCN produces uniform embeddings.

    Parameters
    ----------
    k : float or int
        If float < 1, treated as a percentile of subgraph sizes (PyG default
        0.6 = 60th percentile).  If int, used directly as the sort-pool size.

    Returns
    -------
    ranks : list of int
    auc   : float
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    n_nodes = G.number_of_nodes()

    train_ei = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    train_ei_undir = torch.cat([train_ei, train_ei.flip(0)], dim=1)

    train_set = set(map(tuple, train_edges)) | {(v, u) for u, v in train_edges}
    test_set  = set(map(tuple, test_edges))  | {(v, u) for u, v in test_edges}

    triu_i, triu_j = np.triu_indices(n_nodes, k=1)

    # ── Build training subgraph dataset ──────────────────────────────────
    all_non_edges = [(int(i), int(j))
                     for i, j in zip(triu_i, triu_j)
                     if (int(i), int(j)) not in train_set]
    neg_idx = rng.choice(len(all_non_edges),
                         size=min(len(train_edges), len(all_non_edges)),
                         replace=False)
    neg_train_pairs = [all_non_edges[i] for i in neg_idx]

    def build_dataset(pos_pairs, neg_pairs):
        graphs = []
        for label, pairs in [(1, pos_pairs), (0, neg_pairs)]:
            for (u, v) in pairs:
                try:
                    d = _extract_enclosing_subgraph(
                        train_ei_undir, u, v, n_nodes, num_hops)
                    d.y = torch.tensor([float(label)])
                    graphs.append(d)
                except Exception:
                    pass
        return graphs

    train_dataset = build_dataset(train_edges, neg_train_pairs)
    if not train_dataset:
        return [n_nodes] * len(test_edges), 0.5

    # One-hot encode DRNL labels (same as PyG example)
    all_z = torch.cat([d.z for d in train_dataset])
    max_z = int(all_z.max().item()) + 1
    for d in train_dataset:
        d.x = F.one_hot(d.z, max_z).to(torch.float)

    # Resolve k percentile → absolute pool size
    if isinstance(k, float) and k < 1:
        sizes = sorted(d.num_nodes for d in train_dataset)
        k_int = sizes[int(math.ceil(k * len(sizes))) - 1]
        k_int = int(max(10, k_int))
    else:
        k_int = int(k)

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = _DGCNN(hidden_channels=hidden_channels, num_layers=num_layers,
                   k=k_int, in_channels=max_z).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    # ── Training ──────────────────────────────────────────────────────────
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()

    # ── Score non-training pairs for global ranking ───────────────────────
    model.eval()
    non_train_pairs = [(int(i), int(j))
                       for i, j in zip(triu_i, triu_j)
                       if (int(i), int(j)) not in train_set]

    # For large graphs, cap the negative sample pool (test edges always kept).
    non_test = [p for p in non_train_pairs if p not in test_set and (p[1], p[0]) not in test_set]
    if len(non_test) > max_neg_samples:
        non_test = [non_test[i] for i in rng.choice(len(non_test), max_neg_samples, replace=False)]
    non_train_pairs = list(test_edges) + non_test

    def score_pairs(pairs):
        graphs, valid_idx = [], []
        for idx2, (u, v) in enumerate(pairs):
            try:
                d = _extract_enclosing_subgraph(
                    train_ei_undir, u, v, n_nodes, num_hops)
                d.x = F.one_hot(d.z, max_z).to(torch.float)
                d.y = torch.zeros(1)
                graphs.append(d)
                valid_idx.append(idx2)
            except Exception:
                pass
        if not graphs:
            return {}
        loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
        scores = []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                scores.extend(
                    torch.sigmoid(model(b.x, b.edge_index, b.batch)).view(-1).tolist())
        return {pairs[i]: scores[j] for j, i in enumerate(valid_idx)}

    pair_score_map = score_pairs(non_train_pairs)
    pair_scores_arr = np.array([pair_score_map.get(p, 0.0) for p in non_train_pairs])

    ranks = []
    for (u, v) in test_edges:
        target = pair_score_map.get((u, v), pair_score_map.get((v, u), 0.0))
        ranks.append(tiebroken_rank(pair_scores_arr, target))

    # ── AUC ──────────────────────────────────────────────────────────────
    pos_scores = np.array([pair_score_map.get(p, pair_score_map.get((p[1], p[0]), 0.0))
                            for p in test_edges])
    neg_cands = [p for p in non_train_pairs
                 if p not in test_set and (p[1], p[0]) not in test_set]
    n_neg = min(len(pos_scores), len(neg_cands))
    neg_sample = [neg_cands[i] for i in rng.choice(len(neg_cands), n_neg, replace=False)]
    neg_scores = np.array([pair_score_map.get(p, pair_score_map.get((p[1], p[0]), 0.0))
                            for p in neg_sample])

    y_true  = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5

    return ranks, auc, len(non_train_pairs)


# ---------------------------------------------------------------------------
# Unified comparison on a shared train/test split
# ---------------------------------------------------------------------------

def compare_methods_shared_split(G, heuristics=('adamic_adar',),
                                  test_ratio=0.1, seed=42,
                                  seal_kwargs=None):
    """
    Run all link-prediction methods on the SAME train/test edge split.

    Parameters
    ----------
    G           : networkx.Graph
    heuristics  : tuple of predictor names ('jaccard', 'adamic_adar')
    test_ratio  : fraction of edges held out
    seed        : random seed for the split
    seal_kwargs : dict of keyword arguments passed to evaluate_link_prediction_seal

    Returns
    -------
    dict  {method_name: {'ranks': list, 'entropy': float, 'auc': float}}
    """
    N = G.number_of_nodes()
    node_list, train_edges, test_edges = split_edges(G, test_ratio=test_ratio, seed=seed)

    results = {}

    for predictor in heuristics:
        ranks, auc, n_cands = evaluate_link_prediction_heuristic(
            G, train_edges, test_edges, predictor=predictor, seed=seed
        )
        results[predictor] = {
            'ranks':   ranks,
            'entropy': calculate_entropy(ranks, N, n_candidates=n_cands),
            'auc':     auc,
        }

    kw = seal_kwargs or {}
    seal_ranks, seal_auc, seal_n_cands = evaluate_link_prediction_seal(
        G, train_edges, test_edges, seed=seed, **kw
    )
    results['seal'] = {
        'ranks':   seal_ranks,
        'entropy': calculate_entropy(seal_ranks, N, n_candidates=seal_n_cands),
        'auc':     seal_auc,
    }

    return results


def calculate_entropy(ranks, N, n_candidates=None):
    """
    Calculate entropy H based on rank distribution using N/2 bins.

    Args:
        ranks       : list of ranks from link prediction (one per test edge)
        N           : number of nodes in the network
        n_candidates: actual number of scored non-training pairs (max possible rank).
                      Defaults to N²/2 if not provided, but passing the real value
                      is more accurate for sparse graphs.
    Returns:
        H: entropy value
    """
    n_bins   = N // 2
    max_rank = n_candidates if n_candidates is not None else (N * N) // 2

    bin_width  = max_rank / n_bins
    bin_edges  = np.arange(0, max_rank + bin_width, bin_width)

    bin_assignments = np.digitize(ranks, bin_edges)
    bin_counts      = Counter(bin_assignments)

    total_ranks  = len(ranks)
    probabilities = []
    for bin_idx in range(1, n_bins + 1):
        count = bin_counts.get(bin_idx, 0)
        p = count / total_ranks
        if p > 0:
            probabilities.append(p)

    return -sum(p * np.log2(p) for p in probabilities)
