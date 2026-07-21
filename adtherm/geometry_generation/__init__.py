"""Geometry generation for AdTherm.

Builds ASE Atoms objects sampling an adsorbate's rigid-body degrees of
freedom on a surface, for evaluation with DFT and downstream surrogate
training.
"""

from .generator import Generator, AutoGenerator

__all__ = ["Generator", "AutoGenerator"]
