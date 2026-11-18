"""Gaussian basis-function conventions.

Conventions (shared by FCHK and Molden):

* A normalized Cartesian primitive is
  ``N(lx,ly,lz,a) x^lx y^ly z^lz exp(-a r^2)`` where every Cartesian component
  is normalized individually.  ``N`` factorizes into a radial part
  ``n_l(a) = (2a/pi)^(3/4) (4a)^(l/2)`` and an angular part
  ``f(lx,ly,lz) = 1 / sqrt((2lx-1)!! (2ly-1)!! (2lz-1)!!)``.
* Pure (spherical) functions are ordered ``m = 0, +1, -1, +2, -2, ...`` and are
  linear combinations of the normalized Cartesian functions given by
  :data:`SPHERICAL_TO_CARTESIAN` (columns = pure functions).  The tables were
  verified against real solid harmonics without Condon-Shortley phase.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_FUNCTIONS_PER_SHELL = {
    -5: 11, -4: 9, -3: 7, -2: 5, -1: 4,
    0: 1, 1: 3, 2: 6, 3: 10, 4: 15, 5: 21,
}

MAX_ANGULAR_MOMENTUM = 5

# Cartesian component order for each angular momentum (Gaussian/Molden order;
# the Molden parser reorders its different 15G convention to this one).
CARTESIAN_POWERS: dict[int, tuple[tuple[int, int, int], ...]] = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    2: ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    3: (
        (3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0),
        (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1),
    ),
    4: (
        (0, 0, 4), (0, 1, 3), (0, 2, 2), (0, 3, 1), (0, 4, 0),
        (1, 0, 3), (1, 1, 2), (1, 2, 1), (1, 3, 0), (2, 0, 2),
        (2, 1, 1), (2, 2, 0), (3, 0, 1), (3, 1, 0), (4, 0, 0),
    ),
    5: (
        (0, 0, 5), (0, 1, 4), (0, 2, 3), (0, 3, 2), (0, 4, 1), (0, 5, 0),
        (1, 0, 4), (1, 1, 3), (1, 2, 2), (1, 3, 1), (1, 4, 0),
        (2, 0, 3), (2, 1, 2), (2, 2, 1), (2, 3, 0),
        (3, 0, 2), (3, 1, 1), (3, 2, 0),
        (4, 0, 1), (4, 1, 0), (5, 0, 0),
    ),
}


def n_shell_functions(shell_type: int) -> int:
    """Number of basis functions of a shell type (0 for unknown types)."""
    return _FUNCTIONS_PER_SHELL.get(int(shell_type), 0)


def odd_double_factorial(n: int) -> int:
    """``n!!`` for odd ``n``; returns 1 for ``n <= 0``."""
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def radial_norm(l: int, alpha: np.ndarray | float) -> np.ndarray | float:
    """Angular-momentum dependent part of the primitive normalization."""
    alpha = np.asarray(alpha, dtype=np.float64)
    return (2.0 * alpha / math.pi) ** 0.75 * (4.0 * alpha) ** (0.5 * l)


def cartesian_factor(lx: int, ly: int, lz: int) -> float:
    return 1.0 / math.sqrt(
        odd_double_factorial(2 * lx - 1)
        * odd_double_factorial(2 * ly - 1)
        * odd_double_factorial(2 * lz - 1)
    )


def primitive_norm(lx: int, ly: int, lz: int, alpha: float) -> float:
    """Normalization constant of a single Cartesian primitive."""
    return float(radial_norm(lx + ly + lz, alpha)) * cartesian_factor(lx, ly, lz)


@lru_cache(maxsize=None)
def cartesian_factors(l: int) -> np.ndarray:
    return np.array([cartesian_factor(*p) for p in CARTESIAN_POWERS[l]], dtype=np.float64)


def contracted_norm(l: int, exponents: np.ndarray, coefficients: np.ndarray) -> float:
    """Norm of ``sum_p c_p g_p`` for normalized primitives of angular momentum ``l``.

    Uses ``<g_p|g_q> = (2 sqrt(a_p a_q) / (a_p + a_q))^(l + 3/2)``, which holds for
    every normalized Cartesian component and for pure functions alike.
    """
    a = np.asarray(exponents, dtype=np.float64)
    c = np.asarray(coefficients, dtype=np.float64)
    overlap = (2.0 * np.sqrt(np.outer(a, a)) / np.add.outer(a, a)) ** (l + 1.5)
    return float(math.sqrt(max(float(c @ overlap @ c), 0.0)))


def _build_spherical_to_cartesian() -> dict[int, np.ndarray]:
    s = math.sqrt
    d = np.zeros((6, 5), dtype=np.float64)
    d[0:3, 0] = [-0.5, -0.5, 1.0]
    d[4, 1] = 1.0
    d[5, 2] = 1.0
    d[0:2, 3] = [s(3.0) / 2.0, -s(3.0) / 2.0]
    d[3, 4] = 1.0

    f = np.zeros((10, 7), dtype=np.float64)
    f[2, 0] = 1.0
    f[5, 0] = -1.5 / s(5.0)
    f[8, 0] = -1.5 / s(5.0)
    f[0, 1] = -s(3.0 / 8.0)
    f[3, 1] = -s(3.0 / 40.0)
    f[6, 1] = s(6.0 / 5.0)
    f[1, 2] = -s(3.0 / 8.0)
    f[4, 2] = -s(3.0 / 40.0)
    f[7, 2] = s(6.0 / 5.0)
    f[5, 3] = s(3.0) / 2.0
    f[8, 3] = -s(3.0) / 2.0
    f[9, 4] = 1.0
    f[0, 5] = s(5.0 / 8.0)
    f[3, 5] = -3.0 / s(8.0)
    f[1, 6] = -s(5.0 / 8.0)
    f[4, 6] = 3.0 / s(8.0)

    g = np.zeros((15, 9), dtype=np.float64)
    g[0, 0] = 1.0
    g[2, 0] = -3.0 * s(3.0 / 35.0)
    g[4, 0] = 3.0 / 8.0
    g[9, 0] = -3.0 * s(3.0 / 35.0)
    g[11, 0] = 3.0 / 4.0 * s(3.0 / 35.0)
    g[14, 0] = 3.0 / 8.0
    g[5, 1] = 2.0 * s(5.0 / 14.0)
    g[7, 1] = -1.5 / s(14.0)
    g[12, 1] = -1.5 * s(5.0 / 14.0)
    g[1, 2] = 2.0 * s(5.0 / 14.0)
    g[3, 2] = -1.5 * s(5.0 / 14.0)
    g[10, 2] = -1.5 / s(14.0)
    g[2, 3] = -3.0 * s(3.0 / 28.0)
    g[4, 3] = s(5.0) / 4.0
    g[9, 3] = 3.0 * s(3.0 / 28.0)
    g[14, 3] = -s(5.0) / 4.0
    g[6, 4] = 3.0 / s(7.0)
    g[8, 4] = -s(5.0 / 28.0)
    g[13, 4] = -s(5.0 / 28.0)
    g[7, 5] = -3.0 / s(8.0)
    g[12, 5] = s(5.0 / 8.0)
    g[3, 6] = -s(5.0 / 8.0)
    g[10, 6] = 3.0 / s(8.0)
    g[4, 7] = s(35.0) / 8.0
    g[11, 7] = -3.0 / 4.0 * s(3.0)
    g[14, 7] = s(35.0) / 8.0
    g[8, 8] = -s(5.0) / 2.0
    g[13, 8] = s(5.0) / 2.0

    h = np.zeros((21, 11), dtype=np.float64)
    h[0, 0] = 1.0
    h[11, 0] = -5.0 / s(21.0)
    h[2, 0] = -5.0 / s(21.0)
    h[18, 0] = 5.0 / 8.0
    h[4, 0] = 5.0 / 8.0
    h[13, 0] = s(15.0 / 7.0) / 4.0
    h[6, 1] = s(5.0 / 3.0)
    h[15, 1] = -3.0 * s(5.0 / 28.0)
    h[8, 1] = -3.0 / s(28.0)
    h[20, 1] = s(15.0) / 8.0
    h[10, 1] = s(5.0 / 3.0) / 8.0
    h[17, 1] = s(5.0 / 7.0) / 4.0
    h[1, 2] = s(5.0 / 3.0)
    h[3, 2] = -3.0 * s(5.0 / 28.0)
    h[12, 2] = -3.0 / s(28.0)
    h[5, 2] = s(15.0) / 8.0
    h[19, 2] = s(5.0 / 3.0) / 8.0
    h[14, 2] = s(5.0 / 7.0) / 4.0
    h[11, 3] = s(5.0) / 2.0
    h[2, 3] = -s(5.0) / 2.0
    h[18, 3] = -s(35.0 / 3.0) / 4.0
    h[4, 3] = s(35.0 / 3.0) / 4.0
    h[7, 4] = s(5.0 / 3.0)
    h[16, 4] = -s(5.0 / 12.0)
    h[9, 4] = -s(5.0 / 12.0)
    h[15, 5] = s(5.0 / 6.0)
    h[8, 5] = -s(1.5)
    h[20, 5] = -s(17.5) / 8.0
    h[10, 5] = s(17.5) / 8.0
    h[17, 5] = s(5.0 / 6.0) / 4.0
    h[3, 6] = -s(5.0 / 6.0)
    h[12, 6] = s(1.5)
    h[19, 6] = -s(17.5) / 8.0
    h[5, 6] = s(17.5) / 8.0
    h[14, 6] = -s(5.0 / 6.0) / 4.0
    h[18, 7] = s(35.0) / 8.0
    h[4, 7] = s(35.0) / 8.0
    h[13, 7] = -0.75 * s(3.0)
    h[16, 8] = s(5.0) / 2.0
    h[9, 8] = -s(5.0) / 2.0
    h[20, 9] = 3.0 / 8.0 * s(3.5)
    h[10, 9] = 5.0 / 8.0 * s(3.5)
    h[17, 9] = -1.25 * s(1.5)
    h[5, 10] = 3.0 / 8.0 * s(3.5)
    h[19, 10] = 5.0 / 8.0 * s(3.5)
    h[14, 10] = -1.25 * s(1.5)
    return {2: d, 3: f, 4: g, 5: h}


SPHERICAL_TO_CARTESIAN: dict[int, np.ndarray] = _build_spherical_to_cartesian()
for _matrix in SPHERICAL_TO_CARTESIAN.values():
    _matrix.setflags(write=False)


def shell_transform(shell_type: int) -> np.ndarray:
    """Matrix (n_cartesian, n_functions) mapping Cartesian components to the shell's functions.

    SP shells (-1) are handled by the callers as an S part plus a P part.
    """
    l = abs(int(shell_type))
    if l > MAX_ANGULAR_MOMENTUM or shell_type == -1:
        raise NotImplementedError(f"Unsupported shell type {shell_type}")
    if shell_type >= 0 or l < 2:
        return np.eye(len(CARTESIAN_POWERS[l]), dtype=np.float64)
    return SPHERICAL_TO_CARTESIAN[l]
