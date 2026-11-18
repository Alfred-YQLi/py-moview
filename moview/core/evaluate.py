"""Evaluate basis functions and molecular orbitals at arbitrary points (bohr)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .basis import CARTESIAN_POWERS, SPHERICAL_TO_CARTESIAN, cartesian_factors, radial_norm
from .wavefunction import Shell, Wavefunction

# Basis-function magnitude below which a shell is skipped for a block of points.
SCREENING_THRESHOLD = 1.0e-10
# Generous bound on |angular part| / r^l for both Cartesian and pure functions.
_ANGULAR_BOUND = 4.0


@dataclass(frozen=True)
class _ShellPart:
    l: int
    weights: np.ndarray  # contraction coefficient x radial normalization, per primitive
    exponents: np.ndarray
    transform: np.ndarray | None  # (n_cart, n_pure) or None for Cartesian parts
    offset: int  # first function of this part within the shell


def _shell_parts(shell: Shell) -> tuple[_ShellPart, ...]:
    exponents = np.asarray(shell.exponents, dtype=np.float64)
    if shell.shell_type == -1:
        p_coefficients = shell.sp_coefficients if shell.sp_coefficients is not None else shell.coefficients
        return (
            _ShellPart(0, shell.coefficients * radial_norm(0, exponents), exponents, None, 0),
            _ShellPart(1, p_coefficients * radial_norm(1, exponents), exponents, None, 1),
        )
    l = abs(shell.shell_type)
    if l not in CARTESIAN_POWERS:
        raise NotImplementedError(
            f"Unsupported shell type {shell.shell_type}; only S through H shells are supported"
        )
    transform = SPHERICAL_TO_CARTESIAN[l] if shell.shell_type < -1 else None
    return (_ShellPart(l, shell.coefficients * radial_norm(l, exponents), exponents, transform, 0),)


def evaluate_shell(shell: Shell, points: np.ndarray, dtype: np.dtype | type = np.float64) -> np.ndarray:
    """Values of every function of ``shell`` at ``points``; shape ``(n_functions, n_points)``."""
    dtype = np.dtype(dtype)
    rel = np.asarray(points, dtype=dtype) - np.asarray(shell.center, dtype=dtype)
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    r2 = x * x + y * y + z * z
    out = np.empty((shell.n_functions, rel.shape[0]), dtype=dtype)
    for part in _shell_parts(shell):
        radial = np.zeros(rel.shape[0], dtype=dtype)
        for weight, alpha in zip(part.weights, part.exponents):
            radial += dtype.type(weight) * np.exp(dtype.type(-alpha) * r2)
        powers = CARTESIAN_POWERS[part.l]
        if part.l == 0:
            cart = radial[None, :]
        else:
            xp = [None, x] + [x**k for k in range(2, part.l + 1)]
            yp = [None, y] + [y**k for k in range(2, part.l + 1)]
            zp = [None, z] + [z**k for k in range(2, part.l + 1)]
            factors = cartesian_factors(part.l).astype(dtype)
            cart = np.empty((len(powers), rel.shape[0]), dtype=dtype)
            for row, (lx, ly, lz) in enumerate(powers):
                value = radial * factors[row]
                if lx:
                    value = value * xp[lx]
                if ly:
                    value = value * yp[ly]
                if lz:
                    value = value * zp[lz]
                cart[row] = value
        if part.transform is None:
            out[part.offset : part.offset + cart.shape[0]] = cart
        else:
            n_pure = part.transform.shape[1]
            out[part.offset : part.offset + n_pure] = part.transform.T.astype(dtype) @ cart
    return out


@dataclass(frozen=True)
class BasisLayout:
    """Per-shell bookkeeping used to evaluate a whole basis efficiently."""

    shells: tuple[Shell, ...]
    offsets: np.ndarray  # first basis-function index of each shell
    centers: np.ndarray  # (n_shells, 3), bohr
    extents: np.ndarray  # radius (bohr) beyond which |chi| < SCREENING_THRESHOLD

    @classmethod
    def from_wavefunction(cls, wavefunction: Wavefunction, threshold: float = SCREENING_THRESHOLD) -> BasisLayout:
        shells = tuple(wavefunction.shells)
        counts = np.array([shell.n_functions for shell in shells], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1])) if shells else np.zeros(0, np.int64)
        centers = np.array([shell.center for shell in shells], dtype=np.float64).reshape(-1, 3)
        extents = np.array([_shell_extent(shell, threshold) for shell in shells], dtype=np.float64)
        return cls(shells, offsets, centers, extents)

    def active_shells(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Indices of shells that can be non-negligible inside the box ``[low, high]``."""
        nearest = np.clip(self.centers, low, high)
        distance = np.linalg.norm(self.centers - nearest, axis=1)
        return np.flatnonzero(distance <= self.extents)


def _shell_extent(shell: Shell, threshold: float) -> float:
    """Distance beyond which every function of the shell is below ``threshold``."""
    radius = 0.0
    for part in _shell_parts(shell):
        weights = np.abs(part.weights)
        alphas = part.exponents
        l = part.l

        def bound(r: float) -> float:
            # r^l exp(-a r^2) peaks at sqrt(l / 2a); beyond that it decreases, so
            # evaluating at max(r, peak) is an upper bound for all radii >= r.
            rr = np.maximum(r, np.sqrt(l / (2.0 * alphas)))
            return float(_ANGULAR_BOUND * np.sum(weights * rr**l * np.exp(-alphas * rr * rr)))

        low, high = 0.0, 4.0
        while bound(high) > threshold and high < 1.0e4:
            high *= 2.0
        for _ in range(48):
            mid = 0.5 * (low + high)
            if bound(mid) > threshold:
                low = mid
            else:
                high = mid
        radius = max(radius, high)
    return radius


def evaluate_basis(
    wavefunction: Wavefunction,
    points: np.ndarray,
    dtype: np.dtype | type = np.float32,
    *,
    layout: BasisLayout | None = None,
) -> np.ndarray:
    """All basis-function values at ``points``; shape ``(n_basis, n_points)``."""
    layout = layout or BasisLayout.from_wavefunction(wavefunction)
    points = np.asarray(points, dtype=np.float64)
    out = np.zeros((wavefunction.n_basis, points.shape[0]), dtype=dtype)
    if points.shape[0] == 0:
        return out
    for index in layout.active_shells(points.min(axis=0), points.max(axis=0)):
        shell = layout.shells[index]
        start = int(layout.offsets[index])
        out[start : start + shell.n_functions] = evaluate_shell(shell, points, dtype)
    return out


def evaluate_orbitals(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    points: np.ndarray,
    dtype: np.dtype | type = np.float64,
    *,
    coeff_cutoff: float = 1.0e-10,
    layout: BasisLayout | None = None,
) -> np.ndarray:
    """Values of several MOs at ``points``; shape ``(n_orbitals, n_points)``."""
    indices = np.asarray(list(orbital_indices0), dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    out = np.zeros((indices.size, points.shape[0]), dtype=dtype)
    if indices.size == 0 or points.shape[0] == 0:
        return out
    coefficients = wavefunction.coefficients(spin)
    _check_indices(indices, coefficients.shape[0])
    selected = coefficients[indices].astype(dtype, copy=False)
    layout = layout or BasisLayout.from_wavefunction(wavefunction)
    for index in layout.active_shells(points.min(axis=0), points.max(axis=0)):
        shell = layout.shells[index]
        start = int(layout.offsets[index])
        block = selected[:, start : start + shell.n_functions]
        if not np.any(np.abs(block) >= coeff_cutoff):
            continue
        out += block @ evaluate_shell(shell, points, dtype)
    return out


def evaluate_orbital(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    points: np.ndarray,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    return evaluate_orbitals(wavefunction, spin, [orbital_index0], points, dtype)[0]


def _check_indices(indices: Sequence[int] | np.ndarray, n_orbitals: int) -> None:
    indices = np.asarray(indices)
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_orbitals):
        bad = int(indices.min()) if int(indices.min()) < 0 else int(indices.max())
        raise IndexError(f"Orbital {bad + 1} is outside 1..{n_orbitals}")
