"""Connectivity from covalent radii."""

from __future__ import annotations

import numpy as np

from ..elements import covalent_radius

# Bonded if d < r_i + r_j + BOND_TOLERANCE (angstrom), the Jmol/Open Babel rule.
BOND_TOLERANCE = 0.45
MIN_BOND_LENGTH = 0.40


def compute_bonds(atomic_numbers: np.ndarray, coords_ang: np.ndarray) -> np.ndarray:
    """Return bonded atom pairs ``(i, j)`` with ``i < j`` as an ``(n_bonds, 2)`` int array."""
    z = np.asarray(atomic_numbers, dtype=np.int64)
    coords = np.asarray(coords_ang, dtype=np.float64).reshape(-1, 3)
    real = np.flatnonzero(z > 0)  # ghost atoms / point charges never bond
    if real.size < 2:
        return np.empty((0, 2), dtype=np.int64)
    radii = np.array([covalent_radius(value) for value in z[real]])
    points = coords[real]
    max_cutoff = 2.0 * float(radii.max()) + BOND_TOLERANCE
    try:
        from scipy.spatial import cKDTree

        candidates = cKDTree(points).query_pairs(max_cutoff, output_type="ndarray")
    except ImportError:  # pragma: no cover - scipy ships with scikit-image
        i, j = np.triu_indices(real.size, k=1)
        candidates = np.column_stack((i, j))
    if candidates.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    i, j = candidates[:, 0], candidates[:, 1]
    distance = np.linalg.norm(points[i] - points[j], axis=1)
    bonded = (distance > MIN_BOND_LENGTH) & (distance < radii[i] + radii[j] + BOND_TOLERANCE)
    pairs = np.sort(np.column_stack((real[i[bonded]], real[j[bonded]])), axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]
