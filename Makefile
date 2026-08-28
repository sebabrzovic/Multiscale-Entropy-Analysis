# Reproduction driver for "Multiscale Entropy Analysis of Complex Networks".
#
# Three stages, cheapest last:
#
#   corpus   data/CommunityFitNet_updated.pickle -> data/all_networks.pkl   (seconds)
#   compute  the corpus -> results/*.csv                       (GPU, many hours)
#   figures  results/*.csv -> figures/*.pdf, tables/*.tex       (seconds, no torch)
#
# `figures` is the one you want if you only need to regenerate what the paper shows.
# `compute` re-derives the numbers themselves and is the expensive boundary.

PYTHON ?= python3
PAPER  ?= $(HOME)/Documents/Multiscale_Entropy

.PHONY: help corpus compute figures sync all clean-outputs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## ── stage 1 ──────────────────────────────────────────────────────────────────
corpus: data/all_networks.pkl  ## Build the 558-network corpus from CommunityFitNet

data/all_networks.pkl: data/CommunityFitNet_updated.pickle experiments/build_full_corpus.py
	$(PYTHON) experiments/build_full_corpus.py

## ── stage 2 (expensive) ──────────────────────────────────────────────────────
compute: compute-entropy compute-robustness compute-runtime compute-synthetic  ## All of stage 2 (hours, needs a GPU)

compute-entropy: data/all_networks.pkl  ## Multiscale entropy + SEAL/Adamic-Adar link prediction
	$(PYTHON) run_real_networks_experiment.py

compute-robustness: data/all_networks.pkl  ## Second coarsening algorithm, for the robustness appendix
	$(PYTHON) experiments/coarsening_robustness.py

compute-runtime: data/all_networks.pkl  ## Per-stage runtime and peak memory
	$(PYTHON) experiments/runtime_benchmark.py --n-networks 50 --repeats 3

compute-synthetic:  ## Figure 1: entropy across synthetic families (no corpus needed)
	$(PYTHON) experiments/synthetic_families.py

## ── stage 3 (cheap) ──────────────────────────────────────────────────────────
figures:  ## Regenerate every figure and table in the paper from results/
	$(PYTHON) experiments/rebuild_regression.py
	$(PYTHON) experiments/clustering_analysis.py
	$(PYTHON) experiments/trajectory_figures.py
	$(PYTHON) experiments/runtime_benchmark.py --analyze
	$(PYTHON) experiments/coarsening_robustness.py --analyze
	@if [ -f results/synthetic_families.csv ]; then \
	  $(PYTHON) experiments/synthetic_families.py --analyze; \
	else \
	  echo "skipping Fig. 1: results/synthetic_families.csv not present"; \
	  echo "  build it with 'make compute-synthetic' (~2 h, no GPU needed)"; \
	fi
	@echo
	@echo "figures/ and tables/ are up to date. 'make sync' copies them into the paper."

sync: figures  ## Copy the generated PDFs into the manuscript's image tree
	@test -d "$(PAPER)/imagenes/correlacion" || \
	  { echo "Paper tree not found at $(PAPER); set PAPER=/path/to/manuscript"; exit 1; }
	cp figures/predicted_vs_actual_*.pdf "$(PAPER)/imagenes/correlacion/"
	cp figures/kmeans_pca_clusters_single.pdf "$(PAPER)/imagenes/correlacion/"
	cp figures/runtime_scaling.pdf figures/runtime_by_level.pdf "$(PAPER)/imagenes/correlacion/"
	cp figures/entropy_trajectories_by_domain.pdf "$(PAPER)/imagenes/grafos_Reales/"
	cp figures/synthetic_families.pdf "$(PAPER)/imagenes/reduccion_grafos_conocidos/"
	@echo "Figures copied. Table bodies are pasted into body.tex/appendix.tex by hand —"
	@echo "see tables/*.tex and the README note on keeping them in step."

all: corpus compute figures  ## Everything, from the raw corpus up

clean-outputs:  ## Delete generated figures and tables (results/ is left alone)
	rm -f figures/*.pdf tables/*.tex
