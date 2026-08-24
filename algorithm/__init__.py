"""Multiscale entropy of graphs: spectral coarsening, compression entropy, link prediction.

Only `coarsen` and `get_entropy_metadata_aritmethicEncoding` are part of the public
surface; see README.md for the pipeline that uses them.
"""
from .coarsening_utils import coarsen, get_entropy_metadata_aritmethicEncoding

__all__ = ['coarsen', 'get_entropy_metadata_aritmethicEncoding']
