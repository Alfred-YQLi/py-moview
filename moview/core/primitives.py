"""Primitive-level view of a basis, shared by the overlap code and the grid evaluator.

``cartesian_primitives`` returns every primitive Gaussian (angular momentum,
exponent, centre, contraction x normalization weight, owning Cartesian shell
part) and ``T`` with ``chi_basis = T^T phi_cartesian``, i.e. MO coefficients
over Cartesian components are ``T @ c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import CARTESIAN_POWERS, radial_norm, shell_transform
from .wavefunction import Wavefunction


@dataclass(frozen=True)
class CartesianPrimitives:
    """Every contracted shell expanded into primitives of Cartesian shell parts."""

    l: np.ndarray
    alpha: np.ndarray
    center: np.ndarray
    weight: np.ndarray  # contraction coefficient x radial normalization
    part: np.ndarray  # index of the Cartesian shell part
    part_offset: np.ndarray  # first Cartesian index of each part


def cartesian_primitives(wavefunction: Wavefunction) -> tuple[CartesianPrimitives, np.ndarray]:
    """Expand shells into Cartesian parts and build ``T`` (n_cartesian x n_basis)."""
    ls, alphas, centers, weights, parts = [], [], [], [], []
    part_offsets: list[int] = []
    blocks: list[tuple[int, int, np.ndarray]] = []
    n_cart = 0
    n_basis = 0
    for shell in wavefunction.shells:
        if shell.shell_type == -1:
            p_coefficients = shell.sp_coefficients if shell.sp_coefficients is not None else shell.coefficients
            pieces = ((0, shell.coefficients, np.eye(1), 0), (1, p_coefficients, np.eye(3), 1))
        else:
            l = abs(shell.shell_type)
            pieces = ((l, shell.coefficients, shell_transform(shell.shell_type), 0),)
        for l, coefficients, transform, function_offset in pieces:
            part_index = len(part_offsets)
            part_offsets.append(n_cart)
            blocks.append((n_cart, n_basis + function_offset, transform))
            exps = np.asarray(shell.exponents, dtype=np.float64)
            ls.append(np.full(exps.size, l))
            alphas.append(exps)
            centers.append(np.repeat(np.asarray(shell.center, dtype=np.float64)[None, :], exps.size, axis=0))
            weights.append(np.asarray(coefficients, dtype=np.float64) * radial_norm(l, exps))
            parts.append(np.full(exps.size, part_index))
            n_cart += len(CARTESIAN_POWERS[l])
        n_basis += shell.n_functions
    transform_matrix = np.zeros((n_cart, n_basis))
    for row, col, block in blocks:
        transform_matrix[row : row + block.shape[0], col : col + block.shape[1]] = block
    primitives = CartesianPrimitives(
        np.concatenate(ls),
        np.concatenate(alphas),
        np.concatenate(centers),
        np.concatenate(weights),
        np.concatenate(parts),
        np.asarray(part_offsets, dtype=np.int64),
    )
    return primitives, transform_matrix
