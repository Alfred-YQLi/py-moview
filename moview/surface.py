from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .constants import BOHR_TO_ANG
from .grid import DEFAULT_GRID_CHUNK_POINTS, OrbitalGrid, compute_orbital_grid
from .wavefunction import Wavefunction


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _empty_mesh() -> SurfaceMesh:
    return SurfaceMesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32))


def smooth_surface_mesh(mesh: SurfaceMesh, iterations: int = 0, relaxation: float = 0.18) -> SurfaceMesh:
    if mesh.vertices.shape[0] < 4 or mesh.faces.shape[0] < 4 or iterations <= 0:
        return mesh
    vertices = mesh.vertices.astype(np.float32, copy=True)
    faces = mesh.faces.astype(np.int64, copy=False)
    edges = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    degree = np.bincount(src, minlength=vertices.shape[0]).astype(np.float32)
    movable = degree > 0
    for _ in range(iterations):
        neighbor_sum = np.zeros_like(vertices)
        np.add.at(neighbor_sum, src, vertices[dst])
        averaged = vertices.copy()
        averaged[movable] = neighbor_sum[movable] / degree[movable, None]
        vertices[movable] = (1.0 - relaxation) * vertices[movable] + relaxation * averaged[movable]
    return SurfaceMesh(vertices, mesh.faces)


def _marching_mesh(field: np.ndarray, level: float, origin: np.ndarray, spacing: np.ndarray) -> SurfaceMesh:
    if not np.isfinite(level):
        return _empty_mesh()
    vmin = float(np.nanmin(field))
    vmax = float(np.nanmax(field))
    if not (vmin < level < vmax):
        return _empty_mesh()
    try:
        from skimage import measure
    except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
        print(
            "Missing dependency: scikit-image\n"
            "Install with:\n"
            "pip install numpy scikit-image pyqtgraph PyQt6 PyOpenGL",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    vertices, faces, _normals, _values = measure.marching_cubes(
        field,
        level=level,
        spacing=tuple(float(v) for v in spacing),
        allow_degenerate=False,
    )
    vertices = (vertices + origin) * BOHR_TO_ANG
    return smooth_surface_mesh(
        SurfaceMesh(vertices.astype(np.float32, copy=False), faces.astype(np.uint32, copy=False))
    )


def extract_isosurfaces(grid: OrbitalGrid, iso: float) -> tuple[SurfaceMesh, SurfaceMesh, float]:
    level = float(iso)
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError("Isovalue must be a finite positive value")
    return (
        _marching_mesh(grid.values, level, grid.origin, grid.spacing),
        _marching_mesh(grid.values, -level, grid.origin, grid.spacing),
        level,
    )


def surface_for_orbital(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    grid_size: int,
    iso: float,
    margin_bohr: float,
    *,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> tuple[SurfaceMesh, SurfaceMesh, float, tuple[int, int, int]]:
    grid = compute_orbital_grid(
        wavefunction,
        spin,
        orbital_index0,
        grid_size,
        margin_bohr,
        chunk_points=chunk_points,
    )
    pos, neg, level = extract_isosurfaces(grid, iso)
    return pos, neg, level, grid.shape
