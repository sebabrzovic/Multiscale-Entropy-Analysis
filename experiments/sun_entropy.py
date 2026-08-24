"""
Adamic-Adar prediction entropy with the analytic normalisation of Sun et al. (2020).

    Sun, Zhao, Sun, Chen, Zhang. "Revealing the predictability of intrinsic structure
    in complex networks." Nature Communications 11:574 (2020).

Replaces the simulated Erdos-Renyi denominator, which is structurally broken for
Adamic-Adar: AA scores zero for any pair with no common neighbour, so on a sparse ER
graph nearly every candidate pair ties at zero, the held-out ranks pile into one or two
bins, and H(G_R) -> 0 by construction. Averaging draws does not help -- the denominator
is small and *stable* (median 0.30, relative spread 0.25 across ten draws), not noisy.

The analytic form instead divides by the entropy of a uniform distribution over the
N/2 rank bins:

    H*(G) = H(G) / (log2(N) - 1),      log2(N) - 1 = log2(N/2)

which is deterministic, bounded away from zero, and puts H* in [0, 1] by construction.

Tie-breaking
------------
AA produces massive ties at zero, and the tie convention silently determines the
result. Ranks are assigned RANDOMLY within a tied block:

    rank = n_greater + U{1, ..., n_ties}

Measured on one sparse ER graph (N=197, E=400): random gives H* = 0.970, taking the
best rank in the block gives 0.092, the worst gives 0.098. The latter two are artefacts
of the convention rather than properties of the graph. The RNG is seeded per network and
the seed recorded.

Leave-one-out and why the scores need computing only once
---------------------------------------------------------
AA(u,v) counts common neighbours of u and v, weighted by 1/log(deg). Deleting the edge
(u,v) itself does not change that set -- (u,v) is not incident to any common neighbour of
u and v -- so the held-out edge's own score is unchanged by its removal. Removal perturbs
only the degrees of u and v, a second-order effect on other pairs. The full pair-score
matrix can therefore be computed once as A * diag(1/log d) * A rather than once per
held-out edge, which is what makes leave-one-out tractable at this corpus size.
"""

import numpy as np
import scipy.sparse as sp


def aa_score_matrix(A, degrees):
    """All-pairs Adamic-Adar via a sparse triple product.

    AA(u,v) = sum_{w in N(u) & N(v)} 1 / log(deg w), which is exactly
    (A @ diag(1/log d) @ A)[u, v].
    """
    with np.errstate(divide='ignore'):
        w = 1.0 / np.log(degrees.astype(float))
    # deg 0 and 1 contribute nothing: log(1) = 0 gives inf, and a degree-0 node is in
    # no pair's neighbourhood. Zeroing them is the standard convention.
    w[~np.isfinite(w)] = 0.0
    D = sp.diags(w)
    return (A @ D @ A).tocsr()


def prediction_entropy(G, seed=0, return_detail=False):
    """Sun et al. prediction entropy for Adamic-Adar on an undirected graph.

    Returns H* in [0, 1] (or a detail dict). Rank bins have width N, giving
    n_bins = ceil(C(N,2)/N); ranks beyond the last bin are absorbed into it.
    """
    import networkx as nx

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(G.nodes())
    N = len(nodes)
    if N < 3 or G.number_of_edges() == 0:
        raise ValueError(f'graph too small: N={N}, E={G.number_of_edges()}')

    idx = {u: i for i, u in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=float)
    A.data[:] = 1.0                                   # unweighted
    deg = np.asarray(A.sum(axis=1)).ravel()

    S = aa_score_matrix(A, deg)

    iu, ju = np.triu_indices(N, k=1)
    scores = np.asarray(S[iu, ju]).ravel()
    is_edge = np.asarray(A[iu, ju]).ravel() > 0

    # Candidate set: every non-edge, plus the held-out edge itself. Under
    # leave-one-out each edge is scored against the non-edges, so the negatives are
    # shared and only the positive changes.
    neg = scores[~is_edge]
    pos = scores[is_edge]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('no positives or no negatives to rank against')

    order = np.sort(neg)
    # For each positive: how many negatives score strictly more, and how many tie.
    n_greater = len(order) - np.searchsorted(order, pos, side='right')
    n_equal = (np.searchsorted(order, pos, side='right')
               - np.searchsorted(order, pos, side='left'))

    rng = np.random.default_rng(seed)
    # +1 for the held-out edge, which is itself in the tied block.
    n_ties = n_equal + 1
    ranks = n_greater + rng.integers(1, n_ties + 1)

    n_bins = int(np.ceil((N * (N - 1) / 2) / N))       # ~ N/2
    bins = np.minimum((ranks - 1) // N, n_bins - 1)    # clip into the top bin
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())

    denom = np.log2(N) - 1.0                           # = log2(N/2)
    H_star = H / denom

    if not (-1e-9 <= H_star <= 1.0 + 1e-9):
        raise AssertionError(
            f'H* = {H_star:.4f} outside [0, 1] (N={N}, H={H:.4f}, denom={denom:.4f})')
    H_star = float(np.clip(H_star, 0.0, 1.0))

    if return_detail:
        mean_k = 2 * G.number_of_edges() / N
        return dict(H_star=H_star, H_raw=H, denom=denom, N=N,
                    E=G.number_of_edges(), mean_k=mean_k, seed=seed,
                    n_bins=n_bins,
                    # Sun et al. derive the ER result for ln N << <k> << N - ln N.
                    sparse_regime_ok=bool(np.log(N) < mean_k),
                    small_n=bool(N < 32))
    return H_star


def analytic_entropy_from_ranks(ranks, N):
    """Sun et al. entropy from a list of held-out-edge ranks.

    Shared by every predictor so Adamic-Adar and SEAL are normalised identically and
    their H* values are directly comparable.

        bin(r)  = floor((r-1)/N)          bins of fixed width N
        n_bins  = ceil(C(N,2)/N)          ~ N/2
        H       = -sum_j p_j log2 p_j
        H*      = H / (log2(N) - 1)

    Note the fixed bin width. The pipeline's original calculate_entropy sets
    bin_width = n_candidates/(N/2), which adapts to the number of candidate pairs. For
    a sparse graph the two agree (n_candidates ~ C(N,2) gives width ~ N), but for a
    dense graph the adaptive width is far smaller than N, so the two schemes disagree
    exactly where the graph is dense. Width N is the published choice.

    Returns (H_raw, denom, H_star).
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    if N < 3:
        raise ValueError(f'N={N} too small to normalise')
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
