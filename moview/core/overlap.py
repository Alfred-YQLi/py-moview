"""Analytic overlap integrals and MO orthonormality diagnostics.

For a correctly interpreted file the MO coefficients satisfy ``C S C^T = I``.
This is used to detect writer-specific Molden conventions and is exposed in
batch mode as ``moview FILE --batch --check``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .basis import CARTESIAN_POWERS, cartesian_factors
from .primitives import CartesianPrimitives, cartesian_primitives
from .wavefunction import Wavefunction

# Primitive pairs with mu * R^2 above this contribute < exp(-60) and are skipped.
_SCREEN_EXPONENT = 60.0
_PAIR_CHUNK = 400_000


def _obara_saika_1d(la: int, lb: int, pa: np.ndarray, pb: np.ndarray, inv2p: np.ndarray, s00: np.ndarray) -> np.ndarray:
    """1D overlap table ``E[:, i, j]`` for i <= la, j <= lb."""
    table = np.zeros((pa.shape[0], la + 1, lb + 1))
    table[:, 0, 0] = s00
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                value = pa * table[:, i - 1, j]
                if i > 1:
                    value = value + (i - 1) * inv2p * table[:, i - 2, j]
                if j > 0:
                    value = value + j * inv2p * table[:, i - 1, j - 1]
            else:
                value = pb * table[:, 0, j - 1]
                if j > 1:
                    value = value + (j - 1) * inv2p * table[:, 0, j - 2]
            table[:, i, j] = value
    return table


def overlap_matrix(wavefunction: Wavefunction) -> np.ndarray:
    """AO overlap matrix ``S`` (n_basis x n_basis) in the file's basis-function order."""
    prims, transform = cartesian_primitives(wavefunction)
    n_cart = transform.shape[0]
    n_parts = prims.part_offset.size
    s_cart = np.zeros((n_cart, n_cart))
    for la in np.unique(prims.l):
        rows_a = np.flatnonzero(prims.l == la)
        for lb in np.unique(prims.l):
            if lb < la:
                continue
            rows_b = np.flatnonzero(prims.l == lb)
            step = max(1, _PAIR_CHUNK // max(1, rows_b.size))
            for start in range(0, rows_a.size, step):
                _accumulate_block(prims, int(la), int(lb), rows_a[start : start + step], rows_b, n_parts, s_cart)
    # Only la <= lb blocks were computed; mirror them.
    s_cart = np.where(s_cart != 0.0, s_cart, s_cart.T)
    return transform.T @ s_cart @ transform


def _accumulate_block(prims: CartesianPrimitives, la: int, lb: int, rows_a: np.ndarray, rows_b: np.ndarray, n_parts: int, out: np.ndarray) -> None:
    i, j = np.meshgrid(rows_a, rows_b, indexing="ij")
    i, j = i.ravel(), j.ravel()
    a, b = prims.alpha[i], prims.alpha[j]
    p = a + b
    mu = a * b / p
    ab = prims.center[i] - prims.center[j]
    keep = mu * np.einsum("ij,ij->i", ab, ab) < _SCREEN_EXPONENT
    if not np.any(keep):
        return
    i, j, a, b, p, mu, ab = i[keep], j[keep], a[keep], b[keep], p[keep], mu[keep], ab[keep]
    center_p = (a[:, None] * prims.center[i] + b[:, None] * prims.center[j]) / p[:, None]
    pa = center_p - prims.center[i]
    pb = center_p - prims.center[j]
    inv2p = 0.5 / p
    tables = [
        _obara_saika_1d(la, lb, pa[:, k], pb[:, k], inv2p, np.sqrt(math.pi / p) * np.exp(-mu * ab[:, k] ** 2))
        for k in range(3)
    ]
    weight = prims.weight[i] * prims.weight[j]
    key = prims.part[i] * n_parts + prims.part[j]
    unique_keys, inverse = np.unique(key, return_inverse=True)
    row_base = prims.part_offset[unique_keys // n_parts]
    col_base = prims.part_offset[unique_keys % n_parts]
    factors_a = cartesian_factors(la)
    factors_b = cartesian_factors(lb)
    for ca, (ax, ay, az) in enumerate(CARTESIAN_POWERS[la]):
        for cb, (bx, by, bz) in enumerate(CARTESIAN_POWERS[lb]):
            values = weight * tables[0][:, ax, bx] * tables[1][:, ay, by] * tables[2][:, az, bz]
            summed = np.bincount(inverse, weights=values, minlength=unique_keys.size)
            out[row_base + ca, col_base + cb] += summed * (factors_a[ca] * factors_b[cb])


@dataclass(frozen=True)
class OrthonormalityReport:
    spin: str
    n_checked: int
    max_norm_error: float  # max |<i|i> - 1|
    max_overlap: float  # max |<i|j>| for i != j

    @property
    def worst(self) -> float:
        return max(self.max_norm_error, self.max_overlap)


def orthonormality(
    wavefunction: Wavefunction,
    overlap: np.ndarray | None = None,
    *,
    max_orbitals: int | None = None,
) -> list[OrthonormalityReport]:
    """Check ``C S C^T = I`` per spin (optionally on the orbitals nearest the HOMO)."""
    overlap = overlap_matrix(wavefunction) if overlap is None else overlap
    reports: list[OrthonormalityReport] = []
    for spin in wavefunction.spins:
        coefficients = wavefunction.coefficients(spin)
        n_orb = coefficients.shape[0]
        if max_orbitals is not None and n_orb > max_orbitals:
            homo = wavefunction.homo_index(spin)
            start = max(0, min(homo - max_orbitals // 2, n_orb - max_orbitals))
            coefficients = coefficients[start : start + max_orbitals]
        gram = coefficients @ overlap @ coefficients.T
        diagonal = np.diag(gram).copy()
        np.fill_diagonal(gram, 0.0)
        reports.append(
            OrthonormalityReport(
                spin,
                int(coefficients.shape[0]),
                float(np.max(np.abs(diagonal - 1.0))) if diagonal.size else 0.0,
                float(np.max(np.abs(gram))) if gram.size else 0.0,
            )
        )
    return reports
