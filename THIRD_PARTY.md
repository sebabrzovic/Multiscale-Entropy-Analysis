# Third-party code and data

Parts of this repository are adapted from other projects. They are listed here with
what was taken, what was changed, and where the original lives. Anyone redistributing
this code should check the upstream licenses, which govern those portions.

## Graph coarsening — `algorithm/coarsening_utils.py`, `algorithm/graph_utils.py`

Adapted from **`graph-coarsening`** by Andreas Loukas, the reference implementation for
*Graph reduction with spectral and cut guarantees*, JMLR 20(116):1–42, 2019.

- Upstream: <https://github.com/loukasa/graph-coarsening>
- Taken: the local variation coarsening algorithms (`coarsen`,
  `contract_variation_edges`, `contract_variation_linear`, `get_proximity_measure`,
  `matching_greedy`, `get_coarsening_matrix`, `coarsen_matrix`, `coarsen_vector`,
  `generate_test_vectors`) and the PyGSP helpers in `graph_utils.py`.
- Changed: `get_entropy_metadata_aritmethicEncoding` is ours, bridging coarsening to the
  compression entropy in `calculo_entropia.py`. NumPy 2.x fixes (`np.int` → `int`,
  `np.float` → `float`, `np.Inf` → `np.inf`) were applied throughout. The optimal
  (blossom) matching path and the Kron reduction, sparsification, plotting and
  quality-evaluation sections were removed — nothing in this project reached them.
- The Kron reduction section in the original notes that it is itself adapted from PyGSP,
  which took it from the MatlabBGL library.

## SEAL link prediction — `algorithm/entropia_link_prediction.py`

The functions `_drnl_node_labeling`, `_extract_enclosing_subgraph`, `_DGCNN` and
`evaluate_link_prediction_seal` are adapted from the official PyTorch Geometric example.

- Upstream: <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py>
- Method: Zhang & Chen, *Link Prediction Based on Graph Neural Networks*, NeurIPS 2018.
- Changed: wrapped to score a fixed candidate set and report prediction entropy
  alongside AUC, with the candidate count capped on large graphs.

## Network corpus — `data/CommunityFitNet_updated.pickle`

The **CommunityFitNet** corpus, 572 networks assembled from the Index of Complex
Networks (ICON).

- Upstream: <https://github.com/Aghasemian/CommunityFitNet>
- Reference: Ghasemian, Hosseinmardi & Clauset, *Evaluating overfit and underfit in
  models of network community structure*, IEEE TKDE, 2019.
- Used by `experiments/build_full_corpus.py`, which filters it to the 558 networks with
  at least 20 nodes and 30 edges and writes `data/all_networks.pkl`.

## Compression entropy

`algorithm/calculo_entropia.py` is our own implementation of the SZIP structural
encoding of Choi & Szpankowski, *Compression of graphical structures: fundamental
limits, algorithms, and experiments*, IEEE Trans. Inf. Theory 58(2), 2012. The
arithmetic coding step uses the `arithmetic-compressor` package.
