"""Regular 3D grids and orbital fields sampled on them.

Grid values are stored as ``values[ix, iy, iz]`` at
``origin + (ix, iy, iz) * spacing`` (bohr).

Fields are computed with the *separable* (tensor-product) form of Gaussian
basis functions.  Every Cartesian primitive factorizes,

    (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-a r^2) = X(x) Y(y) Z(z),

so an MO on a regular grid is a sum of rank-1 tensors

    psi[i, j, k] = sum_t c_t X_t[i] Y_t[j] Z_t[k].

``exp`` is only evaluated on the three 1D axes, and the 3D sum is one matrix
product ``(X * c) @ (Y (x) Z)^T`` per block of y-rows, done by BLAS.  The result
is exact (it equals point-wise evaluation to ~1e-13 in float64) and is 10-25x
faster than evaluating every Gaussian at every grid point, without caching any
basis-function values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..constants import BOHR_TO_ANG, DEFAULT_GRID_CHUNK_POINTS, MIN_GRID_SIZE
from .basis import CARTESIAN_POWERS, cartesian_factors
from .evaluate import _check_indices
from .primitives import cartesian_primitives
from .wavefunction import Wavefunction


@dataclass(frozen=True)
class GridSpec:
    grid_size: int  # requested points along the longest axis
    margin_bohr: float
    shape: tuple[int, int, int]
    spacing: float  # bohr, identical along every axis
    origin: np.ndarray  # bohr

    @property
    def n_points(self) -> int:
        return math.prod(self.shape)

    def scalar_nbytes(self, dtype: np.dtype | type = np.float32) -> int:
        return self.n_points * np.dtype(dtype).itemsize

    def axis(self, dim: int) -> np.ndarray:
        return self.origin[dim] + self.spacing * np.arange(self.shape[dim], dtype=np.float64)

    @property
    def high(self) -> np.ndarray:
        return self.origin + self.spacing * (np.asarray(self.shape, dtype=np.float64) - 1.0)

    def corners_angstrom(self) -> np.ndarray:
        low, high = self.origin, self.high
        corners = np.array(
            [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
            dtype=np.float64,
        )
        return corners * BOHR_TO_ANG


def make_grid_spec(wavefunction: Wavefunction, grid_size: int, margin_bohr: float) -> GridSpec:
    """Box around all atoms, expanded by ``margin_bohr``, with cubic voxels.

    ``grid_size`` points span the longest edge; the other edges use the same
    spacing (at least ``MIN_GRID_SIZE`` points each).
    """
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
    spacing = float(lengths[longest_axis]) / (grid_size - 1)
    counts = np.ceil(lengths / spacing - 1.0e-9).astype(np.int64) + 1
    counts = np.clip(counts, MIN_GRID_SIZE, grid_size)
    counts[longest_axis] = grid_size
    # Centre the (possibly slightly larger) box on the molecule along every axis.
    covered = spacing * (counts - 1)
    origin = 0.5 * (low + high) - 0.5 * covered
    return GridSpec(grid_size, margin_bohr, tuple(int(c) for c in counts), spacing, origin)


@dataclass
class OrbitalGrid:
    spin: str
    orbital_index0: int
    spec: GridSpec
    values: np.ndarray  # shape == spec.shape

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.spec.shape

    @property
    def grid_size(self) -> int:
        return self.spec.grid_size

    @property
    def margin_bohr(self) -> float:
        return self.spec.margin_bohr

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)


@dataclass(frozen=True)
class SeparableTerms:
    """One column per (primitive, Cartesian component) of the basis."""

    cartesian_index: np.ndarray  # Cartesian function each term belongs to
    alpha: np.ndarray
    center: np.ndarray  # (T, 3) bohr
    powers: np.ndarray  # (T, 3) lx, ly, lz
    weight: np.ndarray  # contraction coefficient x primitive normalization
    transform: np.ndarray  # (n_cartesian, n_basis): basis -> Cartesian coefficients

    @classmethod
    def from_wavefunction(cls, wavefunction: Wavefunction) -> SeparableTerms:
        prims, transform = cartesian_primitives(wavefunction)
        counts = np.array([len(CARTESIAN_POWERS[int(l)]) for l in prims.l])
        primitive = np.repeat(np.arange(prims.l.size), counts)
        component = np.concatenate([np.arange(n) for n in counts])
        powers = np.array([CARTESIAN_POWERS[int(prims.l[p])][c] for p, c in zip(primitive, component)], dtype=np.int64)
        factors = np.array([cartesian_factors(int(prims.l[p]))[c] for p, c in zip(primitive, component)])
        return cls(
            cartesian_index=prims.part_offset[prims.part[primitive]] + component,
            alpha=prims.alpha[primitive],
            center=prims.center[primitive],
            powers=powers,
            weight=prims.weight[primitive] * factors,
            transform=transform,
        )

    def coefficients(self, mo_coefficients: np.ndarray) -> np.ndarray:
        """Term coefficients ``(n_orbitals, T)`` for MO coefficient rows."""
        cartesian = mo_coefficients @ self.transform.T
        return cartesian[:, self.cartesian_index] * self.weight


def _axis_factors(axis: np.ndarray, center: np.ndarray, alpha: np.ndarray, power: np.ndarray, dtype) -> np.ndarray:
    """``(x - A)^l exp(-a (x - A)^2)`` for every grid coordinate and term: (n_axis, T)."""
    d = axis[:, None] - center[None, :]
    return (d ** power[None, :] * np.exp(-alpha[None, :] * d * d)).astype(dtype, copy=False)


def compute_orbital_grids(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    spec: GridSpec,
    *,
    dtype: np.dtype | type = np.float32,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
    terms: SeparableTerms | None = None,
) -> list[OrbitalGrid]:
    """Evaluate several orbitals on ``spec``.

    ``chunk_points`` bounds the grid points handled per matrix product (all x,
    a slab of y, all z); temporary memory is about ``chunk_points / nx``
    times the number of terms, in ``dtype``.
    """
    indices = [int(index) for index in orbital_indices0]
    if not indices:
        return []
    _check_indices(indices, wavefunction.n_orbitals(spin))
    dtype = np.dtype(dtype)
    terms = terms or SeparableTerms.from_wavefunction(wavefunction)
    coefficients = terms.coefficients(wavefunction.coefficients(spin)[np.asarray(indices)])
    keep = np.abs(coefficients).max(axis=0) > 0.0
    coefficients = coefficients[:, keep]
    alpha, center, powers = terms.alpha[keep], terms.center[keep], terms.powers[keep]

    nx, ny, nz = spec.shape
    n_orbitals = len(indices)
    x = _axis_factors(spec.axis(0), center[:, 0], alpha, powers[:, 0], dtype)
    y = _axis_factors(spec.axis(1), center[:, 1], alpha, powers[:, 1], dtype)
    z = _axis_factors(spec.axis(2), center[:, 2], alpha, powers[:, 2], dtype)
    # Fold the coefficients into the x factors: one (N*nx, T) left operand.
    left = (x[None, :, :] * coefficients[:, None, :].astype(dtype)).reshape(n_orbitals * nx, -1)
    fields = np.empty((n_orbitals, nx, ny, nz), dtype=dtype)
    rows = max(1, min(ny, int(chunk_points) // max(1, nx * nz)))
    for start in range(0, ny, rows):
        stop = min(ny, start + rows)
        right = (y[start:stop, None, :] * z[None, :, :]).reshape((stop - start) * nz, -1)  # Khatri-Rao product
        fields[:, :, start:stop, :] = (left @ right.T).reshape(n_orbitals, nx, stop - start, nz)
    if n_orbitals == 1:
        return [OrbitalGrid(spin, indices[0], spec, fields[0])]
    # One array per orbital so a cached field never pins the memory of its batch.
    return [OrbitalGrid(spin, index, spec, fields[row].copy()) for row, index in enumerate(indices)]


def compute_orbital_grid(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    grid_size: int,
    margin_bohr: float,
    *,
    dtype: np.dtype | type = np.float64,
    chunk_points: int = DEFAULT_GRID_CHUNK_POINTS,
) -> OrbitalGrid:
    spec = make_grid_spec(wavefunction, grid_size, margin_bohr)
    return compute_orbital_grids(wavefunction, spin, [orbital_index0], spec, dtype=dtype, chunk_points=chunk_points)[0]
