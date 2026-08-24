import numpy as np
import math
import pygsp as gsp
from pygsp import graphs, filters, reduction
import scipy as sp
from scipy import sparse
import networkx as nx

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from sortedcontainers import SortedList
from . import graph_utils

import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import algorithm.calculo_entropia as ce

def coarsen(
    G,
    K=10,
    r=0.5,
    max_levels=10,
    method="variation_neighborhood",
    Uk=None,
    lk=None,
    max_level_r=0.99,
):
    """
    This function provides a common interface for coarsening algorithms that contract subgraphs

    Parameters
    ----------
    G : pygsp Graph
    K : int
        The size of the subspace we are interested in preserving.
    r : float between (0,1)
        The desired reduction defined as 1 - n/N.
    method : String
        ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'] 
    
    Returns
    -------
    C : np.array of size n x N
        The coarsening matrix.
    Gc : pygsp Graph
        The smaller graph.
    Call : list of np.arrays
        Coarsening matrices for each level
    Gall : list of (n_levels+1) pygsp Graphs
        All graphs involved in the multilevel coarsening

    Example
    -------
    C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
    """
    r = np.clip(r, 0, 0.999)
    G0 = G
    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    C = sp.sparse.eye(N, format="csc")
    Gc = G

    Call, Gall = [], []
    Gall.append(G)

    for level in range(1, max_levels + 1):

        G = Gc

        # how much more we need to reduce the current graph
        r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

        if "variation" in method:

            if level == 1:
                if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                    mask = lk < 1e-10
                    lk[mask] = 1
                    lsinv = lk ** (-0.5)
                    lsinv[mask] = 0
                    B = Uk[:, :K] @ np.diag(lsinv[:K])
                else:
                    offset = 2 * max(G.dw)
                    T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                    lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                    lk = (offset - lk)[::-1]
                    Uk = Uk[:, ::-1]
                    mask = lk < 1e-10
                    lk[mask] = 1
                    lsinv = lk ** (-0.5)
                    lsinv[mask] = 0
                    B = Uk @ np.diag(lsinv)
                A = B
            else:
                B = iC.dot(B)
                d, V = np.linalg.eig(B.T @ (G.L).dot(B))
                mask = d == 0
                d[mask] = 1
                dinvsqrt = d ** (-1 / 2)
                dinvsqrt[mask] = 0
                A = B @ np.diag(dinvsqrt) @ V

            if method == "variation_edges":
                coarsening_list = contract_variation_edges(
                    G, K=K, A=A, r=r_cur
                )
            else:
                coarsening_list = contract_variation_linear(
                    G, K=K, A=A, r=r_cur, mode=method
                )

        else:
            weights = get_proximity_measure(G, method, K=K)

            coarsening_list = matching_greedy(G, weights=weights, r=r_cur)

        iC = get_coarsening_matrix(G, coarsening_list)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes

        C = iC.dot(C)
        Call.append(iC)

        Wc = graph_utils.zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = gsp.graphs.Graph(Wc)
        else:
            Gc = gsp.graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
        Gall.append(Gc)

        n = Gc.N

        if n <= n_target:
            break

    return C, Gc, Call, Gall

################################################################################
# Plotting for paper
################################################################################

def coarsen_vector(x, C):
    return (C.power(2)).dot(x)


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))


def get_coarsening_matrix(G, partitioning):
    """
    This function should be called in order to build the coarsening matrix C.

    Parameters
    ----------
    G : the graph to be coarsened
    partitioning : a list of subgraphs to be contracted

    Returns
    -------
    C : the new coarsening matrix

    Example
    -------
    C = contract(gsp.graphs.sensor(20),[0,1]) ??
    """

    # C = np.eye(G.N)
    C = sp.sparse.eye(G.N, format="lil")

    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)

        # add v_j's to v_i's row
        C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)

        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C


def get_entropy_metadata_aritmethicEncoding(G):
    """Calculate entropy metrics for a graph and its random counterpart"""
    n = len(G)
    e = G.number_of_edges()
    
    # Create Erdos-Renyi random graph with same n,e
    p = (2.0 * e) / (n * (n-1)) if n > 1 else 0
    
    def create_filtered_random_graphs(target_edges, error_percentage):
        """Helper function to create random graphs with edge count filtering"""
        filtered_graphs = []
        min_edges = int(target_edges * (1 - error_percentage/100))
        max_edges = int(target_edges * (1 + error_percentage/100))
        
        attempts = 0
        max_attempts = 100
        
        while len(filtered_graphs) < 10 and attempts < max_attempts:
            G_random = nx.erdos_renyi_graph(n, p)
            edge_count = G_random.number_of_edges()
            if min_edges <= edge_count <= max_edges:
                filtered_graphs.append(G_random)
            attempts += 1
            
        return filtered_graphs
    
    # Start with 1% error and increase if needed
    error = 1
    random_graphs = []
    
    while not random_graphs and error <= 10:  # Set upper limit of 10%
        random_graphs = create_filtered_random_graphs(e, error)
        if not random_graphs:
            pass  # No valid graphs found at this error threshold, increasing
            error += 1
    
    if not random_graphs:
        raise ValueError("Could not generate valid random graphs even at 10% error margin")
    
    # Generated random graphs for entropy normalization
    
    compresion_G, compresion_G_random, compressionB1B2, avg_compressionB1B2_random = ce.entropiaArithmeticEncoding(G, random_graphs)
    
    return {
        "Entropy Normalizado": (compresion_G)/(compresion_G_random),
        "Grafo": (compresion_G),
        "Grafo_r": (compresion_G_random),   
        "Entropy B1 + B2 Normalizado": (compressionB1B2)/(avg_compressionB1B2_random),
        "B1 + B2": (compressionB1B2),
        "B1_r + B2_r": (avg_compressionB1B2_random)
        
    }
    
def contract_variation_edges(G, A=None, K=10, r=0.5):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    N, deg, M = G.N, G.dw, G.Ne
    ones = np.ones(2)
    Pibot = np.eye(2) - np.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(G, A, edge):
        edge, w = edge[:2].astype(int), edge[2]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    # cost function for the edge
    def subgraph_cost_old(G, A, edge):
        w = G.W[edge[0], edge[1]]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    # edges = np.array(G.get_edge_list()[0:2])
    edges = np.array(G.get_edge_list())
    weights = np.array([subgraph_cost(G, A, edges[:, e]) for e in range(M)])
    # weights = np.zeros(M)
    # for e in range(M):
    #    weights[e] = subgraph_cost_old(G, A, edges[:,e])

    # find a heavy weight matching
    coarsening_list = matching_greedy(G, weights=-weights, r=r)

    return coarsening_list


def contract_variation_linear(G, A=None, K=10, r=0.5, mode="neighborhood"):
    """
    Sequential contraction with local variation and general families.
    This is an implemmentation that improves running speed,
    at the expense of being more greedy (and thus having slightly larger error).

    See contract_variation() for documentation.
    """

    N, deg, W_lil = G.N, G.dw, G.W.tolil()

    # The following is correct only for a single level of coarsening.
    # Normally, A should be passed as an argument.
    if A is None:
        lk, Uk = sp.sparse.linalg.eigsh(
            G.L, k=K, which="SM", tol=1e-3
        )  # this is not optimized!
        lk[0] = 1
        lsinv = lk ** (-0.5)
        lsinv[0] = 0
        lk[0] = 0
        D_lsinv = np.diag(lsinv)
        A = Uk @ np.diag(lsinv)

    # cost function for the subgraph induced by nodes array
    def subgraph_cost(nodes):
        nc = len(nodes)
        ones = np.ones(nc)
        W = W_lil[nodes, :][:, nodes]  # .tocsc()
        L = np.diag(2 * deg[nodes] - W.dot(ones)) - W
        B = (np.eye(nc) - np.outer(ones, ones) / nc) @ A[nodes, :]
        return np.linalg.norm(B.T @ L @ B) / (nc - 1)

    class CandidateSet:
        def __init__(self, candidate_list):
            self.set = candidate_list
            self.cost = subgraph_cost(candidate_list)

        def __lt__(self, other):
            return self.cost < other.cost

    family = []
    W_bool = G.A + sp.sparse.eye(G.N, dtype=bool, format="csr")
    if "neighborhood" in mode:
        for i in range(N):
            # i_set = G.A[i,:].indices # graph_utils.get_neighbors(G, i)
            # i_set = np.append(i_set, i)
            i_set = W_bool[i, :].indices
            family.append(CandidateSet(i_set))

    if "cliques" in mode:
        import networkx as nx

        Gnx = nx.from_scipy_sparse_matrix(G.W)
        for clique in nx.find_cliques(Gnx):
            family.append(CandidateSet(np.array(clique)))

    else:
        if "edges" in mode:
            edges = np.array(G.get_edge_list()[0:2])
            for e in range(0, edges.shape[1]):
                family.append(CandidateSet(edges[:, e]))
        if "triangles" in mode:
            triangles = set([])
            edges = np.array(G.get_edge_list()[0:2])
            for e in range(0, edges.shape[1]):
                [u, v] = edges[:, e]
                for w in range(G.N):
                    if G.W[u, w] > 0 and G.W[v, w] > 0:
                        triangles.add(frozenset([u, v, w]))
            triangles = list(map(lambda x: np.array(list(x)), triangles))
            for triangle in triangles:
                family.append(CandidateSet(triangle))

    family = SortedList(family)
    marked = np.zeros(G.N, dtype=bool)

    # ----------------------------------------------------------------------------
    # Construct a (minimum weight) independent set.
    # ----------------------------------------------------------------------------
    coarsening_list = []
    # n, n_target = N, (1-r)*N
    n_reduce = np.floor(r * N)  # how many nodes do we need to reduce/eliminate?

    while len(family) > 0:

        i_cset = family.pop(index=0)
        i_set = i_cset.set

        # check if marked
        i_marked = marked[i_set]

        if not any(i_marked):

            n_gain = len(i_set) - 1
            if n_gain > n_reduce:
                continue  # this helps avoid over-reducing

            # all vertices are unmarked: add i_set to the coarsening list
            marked[i_set] = True
            coarsening_list.append(i_set)
            # n -= len(i_set) - 1
            n_reduce -= n_gain

            # if n <= n_target: break
            if n_reduce <= 0:
                break

        # may be worth to keep this set
        else:
            i_set = i_set[~i_marked]
            if len(i_set) > 1:
                # todo1: check whether to add to coarsening_list before adding to family
                # todo2: currently this will also select contraction sets that are disconnected
                # should we eliminate those?
                i_cset.set = i_set
                i_cset.cost = subgraph_cost(i_set)
                family.add(i_cset)

    return coarsening_list


################################################################################
# Edge-based contraction algorithms
################################################################################


def generate_test_vectors(
    G, num_vectors=10, method="Gauss-Seidel", iterations=5, lambda_cut=0.1
):

    L = G.L
    N = G.N
    X = np.random.randn(N, num_vectors) / np.sqrt(N)

    if method == "GS" or method == "Gauss-Seidel":

        L_upper = sp.sparse.triu(L, 1, format="csc")
        L_lower_diag = sp.sparse.triu(L, 0, format="csc").T

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                x = -sp.sparse.linalg.spsolve_triangular(L_lower_diag, L_upper @ x)
            X[:, j] = x
        return X

    if method == "JC" or method == "Jacobi":

        deg = G.dw.astype(float)
        D = sp.sparse.diags(deg, 0)
        deginv = deg ** (-1)
        deginv[deginv == np.inf] = 0
        Dinv = sp.sparse.diags(deginv, 0)
        M = Dinv.dot(D - L)

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                x = 0.5 * x + 0.5 * M.dot(x)
            X[:, j] = x
        return X

    elif method == "Chebychev":
        from pygsp import filters

        f = filters.Filter(G, lambda x: ((x <= lambda_cut) * 1).astype(np.float32))
        return f.filter(X, method="chebyshev", order=50)


def get_proximity_measure(G, name, K=10):

    N = G.N
    W = G.W  # np.array(G.W.toarray(), dtype=np.float32)
    deg = G.dw  # np.sum(W, axis=0)
    edges = np.array(G.get_edge_list()[0:2])
    weights = np.array(G.get_edge_list()[2])
    M = edges.shape[1]

    num_vectors = K  # int(1*K*np.log(K))
    if "lanczos" in name:
        l_lan, X_lan = sp.sparse.linalg.eigsh(G.L, k=K, which="SM", tol=1e-2)
    elif "cheby" in name:
        X_cheby = generate_test_vectors(
            G, num_vectors=num_vectors, method="Chebychev", lambda_cut=G.e[K + 1]
        )
    elif "JC" in name:
        X_jc = generate_test_vectors(
            G, num_vectors=num_vectors, method="JC", iterations=20
        )
    elif "GS" in name:
        X_gs = generate_test_vectors(
            G, num_vectors=num_vectors, method="GS", iterations=1
        )
    if "expected" in name:
        X = X_lan
        assert not np.isnan(X).any()
        assert X.shape[0] == N
        K = X.shape[1]

    proximity = np.zeros(M, dtype=np.float32)

    # heuristic for mutligrid
    if name == "heavy_edge":
        wmax = np.array(np.max(G.W, 0).todense())[0] + 1e-5
        for e in range(0, M):
            proximity[e] = weights[e] / max(
                wmax[edges[:, e]]
            )  # select edges with large proximity
        return proximity

    # heuristic for mutligrid
    elif name == "algebraic_JC":
        proximity += np.inf
        for e in range(0, M):
            i, j = edges[:, e]
            for kIdx in range(num_vectors):
                xk = X_jc[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(np.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

        return proximity

    # heuristic for mutligrid
    elif name == "affinity_GS":
        c = np.zeros((N, N))
        for e in range(0, M):
            i, j = edges[:, e]
            c[i, j] = (X_gs[i, :] @ X_gs[j, :].T) ** 2 / (
                (X_gs[i, :] @ X_gs[i, :].T) ** 2 * (X_gs[j, :] @ X_gs[j, :].T) ** 2
            )  # select

        c += c.T
        c -= np.diag(np.diag(c))
        for e in range(0, M):
            i, j = edges[:, e]
            proximity[e] = c[i, j] / (max(c[i, :]) * max(c[j, :]))

        return proximity

    for e in range(0, M):
        i, j = edges[:, e]

        if name == "heavy_edge_degree":
            proximity[e] = (
                deg[i] + deg[j] + 2 * G.W[i, j]
            )  # select edges with large proximity

        # loose as little information as possible (custom)
        elif "min_expected_loss" in name:
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [proximity[e], (xk[i] - xk[j]) ** 2]
                )  # select edges with small proximity
        # loose as little gradient information as possible (custom)
        elif name == "min_expected_gradient_loss":
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2 * (deg[i] + deg[j] + 2 * G.W[i, j]),
                    ]
                )  # select edges with small proximity

        # relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss":
            for kIdx in range(1, K):
                xk = G.U[:, kIdx]
                lk = G.e[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4)
                        / lk,
                    ]
                )  # select edges with small proximity

        # fast relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_lanczos":
            for kIdx in range(1, K):
                xk = X_lan[:, kIdx]
                lk = l_lan[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0.5 * (lk + lk))
                        / lk,
                    ]
                )  # select edges with small proximity

        # approximate relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_cheby":
            for kIdx in range(num_vectors):
                xk = X_cheby[:, kIdx]
                lk = xk.T @ G.L @ xk
                proximity[e] = sum(
                    [
                        proximity[e],
                        (
                            (xk[i] - xk[j]) ** 2
                            * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0 * lk)
                            / lk
                        ),
                    ]
                )  # select edges with small proximity

        # heuristic for mutligrid (algebraic multigrid)
        elif name == "algebraic_GS":
            proximity[e] = np.inf
            for kIdx in range(num_vectors):
                xk = X_gs[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(np.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

    if ("rss" in name) or ("expected" in name):
        proximity = -proximity

    return proximity


def matching_greedy(G, weights, r=0.4):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    r : float
        The desired dimensionality reduction (r = 1 - n/N)

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    N = G.N

    # the edge set
    edges = np.array(G.get_edge_list()[0:2])
    M = edges.shape[1]

    idx = np.argsort(-weights)
    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    marked = np.zeros(N, dtype=bool)

    n, n_target = N, (1 - r) * N
    while len(candidate_edges) > 0:

        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)

        # check if marked
        if any(marked[[i, j]]):
            continue

        marked[[i, j]] = True
        n -= 1

        # add it to the matching
        matching.append(np.array([i, j]))

        # termination condition
        if n <= n_target:
            break

    return np.array(matching)
