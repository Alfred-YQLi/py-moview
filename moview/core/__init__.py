"""Numerical core: data model, basis evaluation, grids, isosurfaces, checks."""

from __future__ import annotations

from .bonds import compute_bonds
from .evaluate import evaluate_basis, evaluate_orbital, evaluate_orbitals, evaluate_shell
from .grid import (
    GridSpec,
    OrbitalGrid,
    SeparableTerms,
    compute_orbital_grid,
    compute_orbital_grids,
    make_grid_spec,
)
from .overlap import OrthonormalityReport, orthonormality, overlap_matrix
from .surface import SurfaceMesh, extract_isosurfaces, surface_for_orbital
from .wavefunction import Shell, Wavefunction

__all__ = [
    "GridSpec",
    "OrbitalGrid",
    "OrthonormalityReport",
    "SeparableTerms",
    "Shell",
    "SurfaceMesh",
    "Wavefunction",
    "compute_bonds",
    "compute_orbital_grid",
    "compute_orbital_grids",
    "evaluate_basis",
    "evaluate_orbital",
    "evaluate_orbitals",
    "evaluate_shell",
    "extract_isosurfaces",
    "make_grid_spec",
    "orthonormality",
    "overlap_matrix",
    "surface_for_orbital",
]
