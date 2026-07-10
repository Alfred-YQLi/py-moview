from __future__ import annotations

import math


def _odd_double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for value in range(n, 0, -2):
        out *= value
    return out


def primitive_norm(lx: int, ly: int, lz: int, alpha: float) -> float:
    """Cartesian primitive Gaussian normalization."""
    lsum = lx + ly + lz
    prefactor = (2.0 * alpha / math.pi) ** 0.75
    numerator = (4.0 * alpha) ** lsum
    denom = (
        _odd_double_factorial(2 * lx - 1)
        * _odd_double_factorial(2 * ly - 1)
        * _odd_double_factorial(2 * lz - 1)
    )
    return prefactor * math.sqrt(numerator / denom)


def n_shell_functions(shell_type: int) -> int:
    return {
        -5: 11,
        -4: 9,
        -3: 7,
        -2: 5,
        -1: 4,
        0: 1,
        1: 3,
        2: 6,
        3: 10,
        4: 15,
        5: 21,
    }.get(shell_type, 0)


CARTESIAN_POWERS: dict[int, list[tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (1, 2, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 0, 2),
        (0, 1, 2),
        (0, 2, 1),
        (1, 1, 1),
    ],
    4: [
        (0, 0, 4),
        (0, 1, 3),
        (0, 2, 2),
        (0, 3, 1),
        (0, 4, 0),
        (1, 0, 3),
        (1, 1, 2),
        (1, 2, 1),
        (1, 3, 0),
        (2, 0, 2),
        (2, 1, 1),
        (2, 2, 0),
        (3, 0, 1),
        (3, 1, 0),
        (4, 0, 0),
    ],
    5: [
        (0, 0, 5),
        (0, 1, 4),
        (0, 2, 3),
        (0, 3, 2),
        (0, 4, 1),
        (0, 5, 0),
        (1, 0, 4),
        (1, 1, 3),
        (1, 2, 2),
        (1, 3, 1),
        (1, 4, 0),
        (2, 0, 3),
        (2, 1, 2),
        (2, 2, 1),
        (2, 3, 0),
        (3, 0, 2),
        (3, 1, 1),
        (3, 2, 0),
        (4, 0, 1),
        (4, 1, 0),
        (5, 0, 0),
    ],
}
