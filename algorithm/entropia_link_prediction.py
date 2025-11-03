import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict

def evaluate_link_prediction(G, predictor='jaccard'):
    """
    Evaluate link prediction using leave-one-out cross-validation.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph
    predictor : str
        The link prediction method to use ('jaccard', 'adamic_adar', or 'common_neighbor')
        
    Returns:
    --------
    ranks : list
        List of rankings for each removed edge
    """
    
    # Choose the predictor function
    if predictor == 'jaccard':
        pred_func = nx.jaccard_coefficient
    elif predictor == 'adamic_adar':
        pred_func = nx.adamic_adar_index
    else:
        raise ValueError("Invalid predictor. Choose 'jaccard', 'adamic_adar', or 'common_neighbor'")
    
    # Store the rankings for each edge
    ranks = []
    
    # Make a copy of the graph to work with
    G_copy = G.copy()
    
    # Get all edges from the original graph
    edges = list(G.edges())
    
    for edge in edges:
        # Remove the edge temporarily
        G_copy.remove_edge(*edge)
        
        # Get all unlinked pairs plus our removed edge
        unlinked_pairs = get_unlinked_pairs(G_copy, edge)
        
        # Calculate prediction scores only for unlinked pairs
        scores = []
        for pair in unlinked_pairs:
            # For jaccard and adamic_adar, calculate for the pair
            preds = list(pred_func(G_copy, [(pair[0], pair[1])]))[0]
            score = preds[2]  # The score is the third element in the tuple
            scores.append((pair, score))
        
        # Get the rank of our removed edge
        rank = get_rank(scores, edge)
        ranks.append(rank)
        
        # Add the edge back
        G_copy.add_edge(*edge)
    
    return ranks

def get_unlinked_pairs(G, removed_edge):
    """Get all unlinked node pairs plus the removed edge."""
    nodes = list(G.nodes())
    unlinked_pairs = []
    
    # Add all non-existing edges (unlinked pairs)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edge = (nodes[i], nodes[j])
            # Add if it's either an unlinked pair or our removed edge
            if not G.has_edge(*edge) or edge == removed_edge or edge[::-1] == removed_edge:
                unlinked_pairs.append(edge)
    
    return unlinked_pairs

def get_rank(scores, target_edge):
    """Get the rank of the target edge among all possibilities."""
    # Sort scores in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Find the rank of our target edge
    for i, (edge, score) in enumerate(sorted_scores):
        if edge == target_edge or edge[::-1] == target_edge:
            return i + 1  # Add 1 because ranks start at 1, not 0
            
    return len(sorted_scores)  # If not found (shouldn't happen)

def calculate_entropy(ranks, N):
    """
    Calculate entropy H based on rank distribution using N/2 bins
    Args:
        ranks: list of ranks from link prediction
        N: number of nodes in the network
    Returns:
        H: entropy value
        bin_probabilities: dictionary with bin counts for verification
    """
    # Create N/2 bins
    n_bins = N // 2
    
    # Calculate bin edges
    # The paper implies equal width bins up to N²/2
    max_rank = (N * N) // 2
    bin_width = max_rank / n_bins
    bin_edges = np.arange(0, max_rank + bin_width, bin_width)
    
    # Assign ranks to bins
    bin_assignments = np.digitize(ranks, bin_edges)
    
    # Count frequencies in each bin
    bin_counts = Counter(bin_assignments)
    
    # Calculate probabilities
    total_ranks = len(ranks)
    probabilities = []
    
    # For each bin, calculate its probability
    for bin_idx in range(1, n_bins + 1):  # numpy's digitize starts from 1
        count = bin_counts.get(bin_idx, 0)
        p = count / total_ranks
        if p > 0:  # Only include non-zero probabilities in entropy calculation
            probabilities.append(p)
    
    # Calculate entropy
    H = -sum(p * np.log2(p) for p in probabilities)
    
    return H


def create_EdosReyni(G, max_attempts=50, error_percentage=1):
    """
    Create an Erdős-Rényi random graph with same number of nodes and edges as input graph,
    within specified error percentage for edge count
    
    Args:
        G: Input graph
        max_attempts: Maximum number of attempts to create graph with correct edge count
        error_percentage: Allowed percentage error in edge count (default 1%)
    Returns:
        Random graph with approximately same n,e, or None if no valid graph found
    """
    n = len(G)
    e = G.number_of_edges()
    
    # Calculate probability p for desired edge count
    p = (2.0 * e) / (n * (n-1)) if n > 1 else 0
    
    # Calculate acceptable edge range
    min_edges = int(e * (1 - error_percentage/100))
    max_edges = int(e * (1 + error_percentage/100))
    
    # Try to create graph up to max_attempts times
    for _ in range(max_attempts):
        G_random = nx.erdos_renyi_graph(n, p)
        edge_count = G_random.number_of_edges()
        
        if min_edges <= edge_count <= max_edges:
            print(f"Created random graph with {edge_count} edges (original had {e})")
            return G_random
    
    print(f"Failed to create random graph with edge count in range [{min_edges}, {max_edges}] after {max_attempts} attempts")
    return None



def compare_real_vs_random(G, predictor='jaccard'):
    """
    Compare link prediction entropy of a real graph vs random graph
    
    Args:
        G: Input graph
    Returns:
        Dictionary with entropy values and basic statistics
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()
    
    print(f"Analyzing graph with {N} nodes and {E} edges")
    
    # Calculate entropy for real graph
    print("Calculating entropy for real graph...")
    ranks = evaluate_link_prediction(G, predictor)
    real_entropy = calculate_entropy(ranks, N)
    
    # Create random graph
    print("Creating random graph and calculating its entropy...")
    G_random = create_EdosReyni(G)
    if G_random is None:
        return None
        
    random_rank = evaluate_link_prediction(G_random, predictor)
    random_entropy = calculate_entropy(random_rank, N)
    
    results = {
        'graph_stats': {
            'nodes': N,
            'edges': E
        },
        'real_graph': {
            'entropy': real_entropy
        },
        'random_graph': {
            'entropy': random_entropy
        }
    }
    
    # Print results
    print("\nResults:")
    print(f"Real graph entropy: {real_entropy}")
    print(f"Random graph entropy: {random_entropy}")
    
    return results
