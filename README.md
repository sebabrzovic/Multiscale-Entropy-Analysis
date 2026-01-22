# Multiscale Entropy Analysis of Complex Networks

Official code for analysis of multiscale entropy in various real and not real graphs

This repository contains the complete code and experiments for the paper **"Networks Multiscale Entropy Analysis"**, which presents a theoretical framework for analyzing the structural complexity and predictability of complex networks through multiscale entropy.

## Overview

The project extends compression-based entropy analysis to the multiscale domain through spectral graph reduction techniques. This allows quantifying how structural complexity evolves as the network is gradually reduced, capturing hierarchical patterns at multiple scales.

**Main contributions:**
- Multiscale entropy framework for networks via spectral reduction
- Significant improvement in entropy-predictability relationship vs. single-scale models
- Identification of characteristic entropy profiles by network families
- Reduced computational costs by enabling analysis at reduced scales

## Repository Structure

```
.
├── algorithm/
│   └── Entropy_Experiments/
│       ├── Correlation_Analysis/
│       ├── LinkPrediction_Experiments/
│       └── Network_Families/
├── Real_World_Networks/
├── calculo_entropia.py
├── coarsening_utils.py
├── entropia_link_prediction.py
├── graph_lib.py
├── graph_utils.py
├── maxWeightMatching.py
├── Multiscale_Entropy.pdf
└── README.md
```

---

## Folder Descriptions

### `algorithm/Entropy_Experiments/`

Contains all multiscale entropy experiments organized by analysis type.

#### **`Network_Families/`**
Experiments with synthetic and real-world networks to reproduce the paper's main figures.

**Key files:**
- `*.pkl`: Generated/loaded graphs (synthetic: Ring, Barabási-Albert, Random Regular, Grid; real: ICON corpus)
- `synthetic_graphs_analysis.json`: Multiscale entropy results for synthetic graphs
- `graph_families_analysis.json`: Complete results for 439 undirected networks from the ICON dataset
- `graph_families_multi_entropy_analysis.json`: Results for 60 selected networks with multiple entropy metrics

**Implemented experiments:**
- **Figure 1**: Multiscale entropy in synthetic networks (4 families × 3 sizes × 10 instances)
- **Figures 2 & 3**: Entropy trajectories in real networks stratified by domain and size
- **Figure A3** (Appendix): Representative visualizations of 6 domains across scales

**Method:**
1. Graph generation/loading
2. Spectral reduction at [100%, 80%, 60%, 40%, 20%] of original size
3. Compression entropy calculation at each scale
4. Normalization using Erdős-Rényi random graphs with equal density

#### **`LinkPrediction_Experiments/`**
Analysis of the relationship between structural entropy and link predictability.

**Key files:**
- `graph_families_multi_entropy_analysis.json`: Data for 60 networks with link prediction entropies
- Result files for linear regression and residual analysis

**Implemented experiments:**
- **Table 3**: Linear regression models (1-5 scales) predicting link prediction entropy
- **Figure 6**: Predicted vs. actual values for Model 1 (single-scale) vs. Model 5 (multiscale)
- **Figure A2** (Appendix): Residual analysis by domain

**Method:**
1. Leave-one-out cross-validation on each edge
2. Similarity score calculation (Jaccard, Adamic-Adar) for all unconnected pairs
3. Empirical ranking distribution construction
4. Shannon entropy calculation over ranking bins
5. Normalization with random graphs
6. Linear regression using reduction entropies as predictors

#### **`Correlation_Analysis/`**
Clustering analysis for network classification based on multiscale entropy profiles.

**Key files:**
- `graph_families_analysis.json`: Input data with 5D entropy vectors
- K-means clustering result files

**Implemented experiments:**
- **Figure 4**: PCA visualization of K-means clusters (k=3)
- **Table 2**: Cluster composition by network family

**Method:**
1. Extract 5D entropy vectors [L*(100%), L*(80%), L*(60%), L*(40%), L*(20%)]
2. Standardization with StandardScaler
3. K-means clustering (k=3, random_state=42)
4. PCA reduction to 2D for visualization
5. Cluster composition analysis by domain

**Identified clusters:**
- **Cluster 1** (Hybrid): Dominated by social networks (111/126)
- **Cluster 2** (Increasing): Economic/technological networks (127/133)
- **Cluster 3** (Stable): Transportation/informational networks (29/36)

---

### `Real_World_Networks/`

Analysis and processing folder for real networks from the ICON dataset (Index of Complex Networks).

**Expected files:**
- `undirected_networks.pkl`: DataFrame with 439 filtered undirected networks
- Domain-specific analysis scripts
- Exploratory analysis notebooks

**Included domains:**
- **Biological**: Food webs, protein-protein interactions, metabolic networks, connectomes
- **Social**: Affiliation, offline, communication, collaboration
- **Economic**: Governance, trade, employment
- **Technological**: Digital circuits, software, communication, water distribution
- **Transportation**: Public transport, roads, airports
- **Informational**: Citations, web graphs, language

---

## Main Python Modules

These files contain the fundamental functions used in all experiments.

### `calculo_entropia.py`
**Compression-based entropy calculation functions.**

**Main functions:**
- `Encoder(M)`: Encodes graph into binary sequences B1, B2 using SZIP algorithm
- `M_adyacencia(G)`: Converts NetworkX Graph to adjacency matrix
- `get_optimized_compression_length(binary_string)`: Optimized arithmetic compression
- `entropiaArithmeticEncoding(G, ListaGrafosRandom)`: Calculates normalized entropy vs. random graphs
- `entropiaArithmeticTheoretic(G)`: Theoretical entropy without reference graphs

**SZIP method:**
1. Iterative vertex partitioning
2. Neighbor count encoding in B1 (multi-bit)
3. Binary encoding in B2 (single-bit for singletons)
4. Arithmetic compression of concatenated sequences

**Normalization:**
- Generates 10 Erdős-Rényi graphs with identical (n, e)
- Allows ±1% error in edge count
- Returns ratio: L(G) / E[L(G_random)]

### `coarsening_utils.py`
**Multiscale spectral graph reduction implementation.**

**Main functions:**
- `coarsen(G, K, r, method, algorithm)`: Main multiscale reduction interface
- `plot_coarsening(Gall, Call, entropies)`: Reduction hierarchy visualization
- `plot_reduction_levels_horizontal(G, reduction_levels)`: Horizontal visual comparison
- `get_entropy_metadata(G)`: Calculates entropy metrics for visualization
- `get_entropy_metadata_aritmethicEncoding(G)`: Variant with arithmetic encoding

**Supported reduction methods:**
- `variation_neighborhood`: Neighborhood-based contraction
- `variation_edges`: Edge-based contraction
- `heavy_edge`: Heavy edge matching
- `algebraic_JC`: Algebraic multigrid (Jacobi)
- `affinity_GS`: Affinity (Gauss-Seidel)

**Multiscale strategy:**
1. Construct sequence G₀ → G₁ → ... → Gₖ
2. Reduction matrix Cₗ at each level
3. Laplacian update: Lₗ = Cₗ^⊺ Lₗ₋₁ Cₗ^+
4. Stopping criteria: target size, max levels, minimum reduction

**Spectral guarantees:**
- Restricted Spectral Similarity (RSS): ‖x - C⁺Cx‖_L ≤ ε‖x‖_L
- Preservation of first K eigenspaces
- Principal eigenvalue alignment

### `entropia_link_prediction.py`
**Link prediction and prediction entropy calculation functions.**

**Main functions:**
- `evaluate_link_prediction(G, predictor)`: Leave-one-out cross-validation
- `calculate_entropy(ranks, N)`: Shannon entropy over ranking distribution
- `create_EdosReyni(G, error_percentage)`: Reference random graph generation
- `compare_real_vs_random(G, predictor)`: Real vs. random entropy comparison

**Supported predictors:**
- **Jaccard coefficient**: |Γ(u) ∩ Γ(v)| / |Γ(u) ∪ Γ(v)|
- **Adamic-Adar index**: Σ_{w∈Γ(u)∩Γ(v)} 1/log|Γ(w)|

**Evaluation protocol:**
1. For each edge (u,v) in E:
   - Temporarily remove (u,v)
   - Compute scores for all unconnected pairs + (u,v)
   - Rank by descending score
   - Record rank r_i of (u,v)
2. Build distribution D = {r₁, r₂, ..., r_E}
3. Divide into N/2 equally-spaced bins
4. Calculate H = -Σ p_j log₂(p_j)

### `graph_lib.py` & `graph_utils.py`
**Auxiliary utilities for graph manipulation.**

Expected functions:
- Format conversions (NetworkX ↔ PyGSP ↔ matrices)
- Basic metric calculation (degree, components, distances)
- Linear algebra operations on graphs
- Visualization helpers

### `maxWeightMatching.py`
**Maximum weight matching algorithm for optimal reduction.**

Used by `coarsening_utils.py` when `algorithm='optimal'`:
- Implements maximum weight matching in weighted graphs
- Complexity O(N³)
- Used for optimal selection of node pairs to contract

---

## JSON Data Structure

### Multiscale results format

```json
{
  "Domain": {
    "Network_Name": {
      "Name": "string",
      "Subdomain": "string",
      "Node_Type": "string",
      "Edge_Type": "string",
      "reductions": {
        "100": {
          "graph_portion": 100,
          "number_nodes": int,
          "number_edges": int,
          "ave_degree": float,
          "entropy_arithmetic": {
            "graph": float,
            "random": float,
            "normalized": float
          },
          "entropy_linkPrediction_Jaccard": {
            "graph": float,
            "random": float,
            "normalized": float
          },
          "entropy_linkPrediction_AdamicAdar": {
            "graph": float,
            "random": float,
            "normalized": float
          }
        },
        "80": { ... },
        "60": { ... },
        "40": { ... },
        "20": { ... }
      }
    }
  }
}
```

**Entropy fields:**
- `graph`: Raw value for the network
- `random`: Average of 10 equiprobable Erdős-Rényi graphs
- `normalized`: Graph/random ratio (main metric)

---
