"""Isosurface extraction (marching cubes) for both orbital phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import BOHR_TO_ANG, DEFAULT_GRID_CHUNK_POINTS
from .grid import OrbitalGrid, compute_orbital_grid
from .wavefunction import Wavefunction


@dataclass
class SurfaceMesh:
    """Triangle mesh in angstrom with outward winding and outward unit vertex normals."""

    vertices: np.ndarray  # (n, 3) float32
    faces: np.ndarray  # (m, 3) uint32
    normals: np.ndarray  # (n, 3) float32

    @classmethod
    def empty(cls) -> SurfaceMesh:
        return cls(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
            np.empty((0, 3), dtype=np.float32),
        )

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.vertices.nbytes + self.faces.nbytes + self.normals.nbytes)


def _lobe(grid: OrbitalGrid, level: float) -> SurfaceMesh:
    """Surface of the lobe enclosing values beyond ``level`` (above if > 0, below if < 0)."""
    field = grid.values
    vmin = float(np.nanmin(field))
    vmax = float(np.nanmax(field))
    if not (vmin < level < vmax):
        return SurfaceMesh.empty()
    try:
        from skimage import measure
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Isosurface extraction needs scikit-image: python -m pip install scikit-image"
        ) from exc

    spacing = float(grid.spec.spacing)
    vertices, faces, normals, _values = measure.marching_cubes(
        field, level=level, spacing=(spacing, spacing, spacing), allow_degenerate=False
    )
    # scikit-image returns normals along -grad(psi) and the opposite winding.
    # Positive lobe: -grad points outward -> keep normals, flip winding.
    # Negative lobe: -grad points inward  -> flip normals, keep winding.
    if level > 0.0:
        faces = faces[:, ::-1]
    else:
        normals = -normals
    vertices = (vertices + grid.spec.origin) * BOHR_TO_ANG
    return SurfaceMesh(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
        np.ascontiguousarray(normals, dtype=np.float32),
    )


def extract_isosurfaces(grid: OrbitalGrid, iso: float) -> tuple[SurfaceMesh, SurfaceMesh, float]:
    """Return ``(positive, negative, level)`` meshes for ``psi = +iso`` and ``psi = -iso``."""
    level = float(iso)
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError("Isovalue must be a finite positive value")
    return _lobe(grid, level), _lobe(grid, -level), level


def surface_for_orbital(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    grid_size: int,
    iso: float,
    margin_bohr: float,
    *,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> tuple[SurfaceMesh, SurfaceMesh, OrbitalGrid]:
    grid = compute_orbital_grid(
        wavefunction, spin, orbital_index0, grid_size, margin_bohr, chunk_points=chunk_points
    )
    positive, negative, _level = extract_isosurfaces(grid, iso)
    return positive, negative, grid
