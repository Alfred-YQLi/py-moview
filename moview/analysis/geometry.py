from __future__ import annotations

import numpy as np

from ..constants import COVALENT_RADII


def compute_bonds(atomic_numbers: np.ndarray, coords_ang: np.ndarray) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(atomic_numbers)):
        zi = int(atomic_numbers[i])
        if zi == 0:
            continue
        ri = COVALENT_RADII.get(zi, 0.75)
        for j in range(i + 1, len(atomic_numbers)):
            zj = int(atomic_numbers[j])
            if zj == 0:
                continue
            rj = COVALENT_RADII.get(zj, 0.75)
            cutoff = 1.22 * (ri + rj) + 0.16
            dist = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
            if 0.25 < dist < cutoff:
                bonds.append((i, j))
    return bonds
