"""MOview - molecular orbital wavefunction reader and OpenGL isosurface viewer.

Package layout::

    moview.core      numerics: Wavefunction model, basis evaluation, grids,
                     isosurfaces, bonds, overlap/orthonormality checks
    moview.parsers   FCHK and Molden readers
    moview.gui       PyQt6/OpenGL application (imported lazily)
    moview.cli       command line entry point (GUI and batch mode)
"""

from __future__ import annotations

from .config import AppConfig, ColorPreset, ConfigError, load_config
from .constants import BOHR_TO_ANG, HARTREE_TO_EV
from .core import (
    GridSpec,
    OrbitalGrid,
    Shell,
    SurfaceMesh,
    Wavefunction,
    compute_bonds,
    compute_orbital_grid,
    compute_orbital_grids,
    extract_isosurfaces,
    make_grid_spec,
    orthonormality,
    overlap_matrix,
    surface_for_orbital,
)
from .elements import atom_symbol
from .parsers import detect_wavefunction_format, parse_wavefunction

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "BOHR_TO_ANG",
    "ColorPreset",
    "ConfigError",
    "GridSpec",
    "HARTREE_TO_EV",
    "OrbitalGrid",
    "Shell",
    "SurfaceMesh",
    "Wavefunction",
    "__version__",
    "atom_symbol",
    "compute_bonds",
    "compute_orbital_grid",
    "compute_orbital_grids",
    "detect_wavefunction_format",
    "extract_isosurfaces",
    "load_config",
    "make_grid_spec",
    "orthonormality",
    "overlap_matrix",
    "parse_wavefunction",
    "surface_for_orbital",
]
