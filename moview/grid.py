from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .basis.evaluate import (
    evaluate_mo,
    evaluate_mo_float32,
    evaluate_mos,
    evaluate_mos_float32,
    evaluate_shell_float32,
)
from .basis.gaussian import n_shell_functions
from .wavefunction import Wavefunction, Shell


MIN_GRID_SIZE = 8
DEFAULT_GRID_CHUNK_POINTS = 262_144


@dataclass(frozen=True)
class GridSpec:
    grid_size: int
    margin_bohr: float
    shape: tuple[int, int, int]
    spacing: np.ndarray
    origin: np.ndarray

    @property
    def n_points(self) -> int:
        return math.prod(self.shape)

    def scalar_nbytes(self, dtype: np.dtype | type = np.float32) -> int:
        return self.n_points * np.dtype(dtype).itemsize


def make_grid_spec(
    wavefunction: Wavefunction,
    grid_size: int,
    margin_bohr: float,
) -> GridSpec:
    grid_size = int(grid_size)
    margin_bohr = float(margin_bohr)
    if grid_size < MIN_GRID_SIZE:
        raise ValueError(f"Grid size must be at least {MIN_GRID_SIZE}")
    if not math.isfinite(margin_bohr) or margin_bohr < 0.0:
        raise ValueError("Grid margin must be a finite non-negative value")

    coords = np.asarray(wavefunction.coordinates_bohr, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] != 3:
        raise ValueError("Wavefunction coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("Wavefunction coordinates contain non-finite values")

    low = coords.min(axis=0) - margin_bohr
    high = coords.max(axis=0) + margin_bohr
    lengths = np.maximum(high - low, 1.0)
    longest_axis = int(np.argmax(lengths))
    longest = float(lengths[longest_axis])
    spacing_value = longest / (grid_size - 1)
    counts = np.ceil(lengths / spacing_value).astype(np.int64) + 1
    counts = np.clip(counts, MIN_GRID_SIZE, grid_size)
    counts[longest_axis] = grid_size
    return GridSpec(
        grid_size=grid_size,
        margin_bohr=margin_bohr,
        shape=tuple(int(value) for value in counts),
        spacing=np.full(3, spacing_value, dtype=np.float64),
        origin=low,
    )


def iter_grid_point_chunks(
    spec: GridSpec,
    *,
    dtype: np.dtype | type = np.float64,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> Iterator[tuple[int, int, np.ndarray]]:
    chunk_points = max(1, int(chunk_points))
    ny, nz = spec.shape[1], spec.shape[2]
    yz_plane = ny * nz
    target_dtype = np.dtype(dtype)
    origin = spec.origin.astype(target_dtype, copy=False)
    spacing = spec.spacing.astype(target_dtype, copy=False)
    for start in range(0, spec.n_points, chunk_points):
        stop = min(spec.n_points, start + chunk_points)
        flat = np.arange(start, stop, dtype=np.int64)
        ix, remainder = np.divmod(flat, yz_plane)
        iy, iz = np.divmod(remainder, nz)
        points = np.empty((stop - start, 3), dtype=target_dtype)
        points[:, 0] = origin[0] + ix * spacing[0]
        points[:, 1] = origin[1] + iy * spacing[1]
        points[:, 2] = origin[2] + iz * spacing[2]
        yield start, stop, points


def make_grid(
    wavefunction: Wavefunction,
    grid_size: int,
    margin_bohr: float,
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray]:
    spec = make_grid_spec(wavefunction, grid_size, margin_bohr)
    points = np.empty((spec.n_points, 3), dtype=np.float64)
    for start, stop, chunk in iter_grid_point_chunks(spec):
        points[start:stop] = chunk
    return points, spec.shape, spec.spacing, spec.origin


@dataclass
class OrbitalGrid:
    spin: str
    orbital_index0: int
    grid_size: int
    margin_bohr: float
    values: np.ndarray
    shape: tuple[int, int, int]
    spacing: np.ndarray
    origin: np.ndarray


@dataclass
class BasisGrid:
    grid_size: int
    margin_bohr: float
    shape: tuple[int, int, int]
    spacing: np.ndarray
    origin: np.ndarray
    basis_values: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.basis_values.nbytes)


def estimate_basis_grid_bytes(
    wavefunction: Wavefunction,
    grid_size: int,
    margin_bohr: float,
) -> int:
    spec = make_grid_spec(wavefunction, grid_size, margin_bohr)
    return wavefunction.n_basis * spec.scalar_nbytes(np.float32)


def _compute_orbital_grids_chunked(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    grid_size: int,
    margin_bohr: float,
    *,
    use_float32: bool,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> list[OrbitalGrid]:
    indices = list(orbital_indices0)
    if not indices:
        return []
    coeff_matrix = wavefunction.coefficients(spin)
    index_array = np.asarray(indices, dtype=np.int64)
    if int(index_array.min()) < 0 or int(index_array.max()) >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index is outside 1..{coeff_matrix.shape[0]}")

    spec = make_grid_spec(wavefunction, grid_size, margin_bohr)
    value_dtype = np.float32 if use_float32 else np.float64
    values_by_orbital = [np.empty(spec.n_points, dtype=value_dtype) for _ in indices]
    multi_evaluator = evaluate_mos_float32 if use_float32 else evaluate_mos
    single_evaluator = evaluate_mo_float32 if use_float32 else evaluate_mo
    for start, stop, points in iter_grid_point_chunks(
        spec,
        dtype=value_dtype,
        chunk_points=chunk_points,
    ):
        if len(indices) == 1:
            values_by_orbital[0][start:stop] = single_evaluator(
                wavefunction,
                spin,
                indices[0],
                points,
            )
        else:
            chunk_values = multi_evaluator(wavefunction, spin, indices, points)
            for row, values in enumerate(values_by_orbital):
                values[start:stop] = chunk_values[row]

    grids: list[OrbitalGrid] = []
    for orbital_index0, flat_values in zip(indices, values_by_orbital):
        values = flat_values.reshape(spec.shape)
        grids.append(
            OrbitalGrid(
                spin=spin,
                orbital_index0=orbital_index0,
                grid_size=spec.grid_size,
                margin_bohr=spec.margin_bohr,
                values=values,
                shape=spec.shape,
                spacing=spec.spacing,
                origin=spec.origin,
            )
        )
    return grids


def compute_orbital_grid(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    grid_size: int,
    margin_bohr: float,
    *,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> OrbitalGrid:
    return _compute_orbital_grids_chunked(
        wavefunction,
        spin,
        [orbital_index0],
        grid_size,
        margin_bohr,
        use_float32=False,
        chunk_points=chunk_points,
    )[0]


def compute_orbital_grid_float32(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    grid_size: int,
    margin_bohr: float,
    *,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> OrbitalGrid:
    return _compute_orbital_grids_chunked(
        wavefunction,
        spin,
        [orbital_index0],
        grid_size,
        margin_bohr,
        use_float32=True,
        chunk_points=chunk_points,
    )[0]


def compute_basis_grid(
    wavefunction: Wavefunction,
    grid_size: int,
    margin_bohr: float,
    workers: int = 1,
) -> BasisGrid:
    points, shape, spacing, origin = make_grid(wavefunction, grid_size, margin_bohr)
    points32 = points.astype(np.float32, copy=False)
    basis_values = np.empty((wavefunction.n_basis, points32.shape[0]), dtype=np.float32)

    shell_jobs: list[tuple[int, Shell]] = []
    basis_index = 0
    for shell in wavefunction.shells:
        shell_jobs.append((basis_index, shell))
        basis_index += n_shell_functions(shell.shell_type)

    def fill_shell(job: tuple[int, Shell]) -> None:
        start, shell = job
        components = evaluate_shell_float32(shell, points32)
        for offset, component in enumerate(components):
            basis_values[start + offset] = component

    worker_count = max(1, int(workers))
    if worker_count > 1 and len(shell_jobs) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(fill_shell, shell_jobs))
    else:
        for job in shell_jobs:
            fill_shell(job)

    return BasisGrid(
        grid_size=grid_size,
        margin_bohr=margin_bohr,
        shape=shape,
        spacing=spacing,
        origin=origin,
        basis_values=basis_values,
    )


def compute_orbital_grid_from_basis(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    basis_grid: BasisGrid,
) -> OrbitalGrid:
    coeff_matrix = wavefunction.coefficients(spin)
    if orbital_index0 < 0 or orbital_index0 >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index {orbital_index0 + 1} is outside 1..{coeff_matrix.shape[0]}")
    coeffs = coeff_matrix[orbital_index0].astype(np.float32, copy=False)
    values = (coeffs @ basis_grid.basis_values).reshape(basis_grid.shape)
    return OrbitalGrid(
        spin=spin,
        orbital_index0=orbital_index0,
        grid_size=basis_grid.grid_size,
        margin_bohr=basis_grid.margin_bohr,
        values=values,
        shape=basis_grid.shape,
        spacing=basis_grid.spacing,
        origin=basis_grid.origin,
    )


def compute_orbital_grids_from_basis(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    basis_grid: BasisGrid,
) -> list[OrbitalGrid]:
    indices = list(orbital_indices0)
    if not indices:
        return []
    coeff_matrix = wavefunction.coefficients(spin)
    index_array = np.asarray(indices, dtype=np.int64)
    if int(index_array.min()) < 0 or int(index_array.max()) >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index is outside 1..{coeff_matrix.shape[0]}")
    selected_coeffs = coeff_matrix[index_array].astype(np.float32, copy=False)
    value_rows = selected_coeffs @ basis_grid.basis_values
    grids: list[OrbitalGrid] = []
    for row_idx, orbital_index0 in enumerate(indices):
        values = value_rows[row_idx].reshape(basis_grid.shape).copy()
        grids.append(
            OrbitalGrid(
                spin=spin,
                orbital_index0=orbital_index0,
                grid_size=basis_grid.grid_size,
                margin_bohr=basis_grid.margin_bohr,
                values=values,
                shape=basis_grid.shape,
                spacing=basis_grid.spacing,
                origin=basis_grid.origin,
            )
        )
    return grids


def compute_orbital_grids(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    grid_size: int,
    margin_bohr: float,
    *,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> list[OrbitalGrid]:
    return _compute_orbital_grids_chunked(
        wavefunction,
        spin,
        orbital_indices0,
        grid_size,
        margin_bohr,
        use_float32=False,
        chunk_points=chunk_points,
    )


def compute_orbital_grids_float32(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    grid_size: int,
    margin_bohr: float,
    *,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> list[OrbitalGrid]:
    return _compute_orbital_grids_chunked(
        wavefunction,
        spin,
        orbital_indices0,
        grid_size,
        margin_bohr,
        use_float32=True,
        chunk_points=chunk_points,
    )
