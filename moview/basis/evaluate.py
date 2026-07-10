from __future__ import annotations

from typing import Iterable

import numpy as np

from ..wavefunction import Wavefunction, Shell
from .gaussian import CARTESIAN_POWERS, n_shell_functions, primitive_norm
from .spherical import SPHERICAL_TO_CARTESIAN


def _primitive_cartesian_value(
    lx: int,
    ly: int,
    lz: int,
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    exp_term: np.ndarray,
) -> np.ndarray:
    out = primitive_norm(lx, ly, lz, alpha) * exp_term
    if lx:
        out = out * (x**lx)
    if ly:
        out = out * (y**ly)
    if lz:
        out = out * (z**lz)
    return out


def _primitive_cartesian_value32(
    lx: int,
    ly: int,
    lz: int,
    alpha: np.float32,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    exp_term: np.ndarray,
) -> np.ndarray:
    out = np.float32(primitive_norm(lx, ly, lz, float(alpha))) * exp_term
    if lx:
        out = out * (x**lx)
    if ly:
        out = out * (y**ly)
    if lz:
        out = out * (z**lz)
    return out


def evaluate_shell(shell: Shell, points: np.ndarray) -> list[np.ndarray]:
    rel = points - shell.center
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    r2 = x * x + y * y + z * z
    st = shell.shell_type

    if st == -1:
        comps = [np.zeros(points.shape[0], dtype=np.float64) for _ in range(4)]
        powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for pidx, alpha in enumerate(shell.exponents):
            exp_term = np.exp(-float(alpha) * r2)
            coeffs = [float(shell.coefficients[pidx])]
            p_coeff = (
                float(shell.sp_coefficients[pidx])
                if shell.sp_coefficients is not None
                else float(shell.coefficients[pidx])
            )
            coeffs.extend([p_coeff, p_coeff, p_coeff])
            for comp_idx, (lx, ly, lz) in enumerate(powers):
                comps[comp_idx] += coeffs[comp_idx] * _primitive_cartesian_value(
                    lx, ly, lz, float(alpha), x, y, z, exp_term
                )
        return comps

    angular_momentum = abs(st)
    if angular_momentum not in CARTESIAN_POWERS:
        raise NotImplementedError(f"Unsupported shell type {st}; only S through H shells are supported")
    powers = CARTESIAN_POWERS[angular_momentum]
    if st >= 0:
        matrix = np.eye(len(powers), dtype=np.float64)
    else:
        matrix = SPHERICAL_TO_CARTESIAN[angular_momentum]

    comps = [np.zeros(points.shape[0], dtype=np.float64) for _ in range(matrix.shape[1])]
    nonzero_by_cart = [np.flatnonzero(np.abs(matrix[row]) > 1.0e-14) for row in range(matrix.shape[0])]
    for pidx, alpha in enumerate(shell.exponents):
        alpha_f = float(alpha)
        coeff = float(shell.coefficients[pidx])
        exp_term = np.exp(-alpha_f * r2)
        for cart_idx, (lx, ly, lz) in enumerate(powers):
            comp_indices = nonzero_by_cart[cart_idx]
            if comp_indices.size == 0:
                continue
            term = coeff * _primitive_cartesian_value(lx, ly, lz, alpha_f, x, y, z, exp_term)
            for comp_idx in comp_indices:
                comps[int(comp_idx)] += matrix[cart_idx, int(comp_idx)] * term
    return comps


def evaluate_shell_float32(shell: Shell, points: np.ndarray) -> list[np.ndarray]:
    points32 = points.astype(np.float32, copy=False)
    center = shell.center.astype(np.float32, copy=False)
    rel = points32 - center
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    r2 = x * x + y * y + z * z
    st = shell.shell_type

    if st == -1:
        comps = [np.zeros(points32.shape[0], dtype=np.float32) for _ in range(4)]
        powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        exponents = shell.exponents.astype(np.float32, copy=False)
        coefficients = shell.coefficients.astype(np.float32, copy=False)
        sp_coefficients = (
            shell.sp_coefficients.astype(np.float32, copy=False)
            if shell.sp_coefficients is not None
            else coefficients
        )
        for pidx, alpha in enumerate(exponents):
            exp_term = np.exp(-alpha * r2)
            coeffs = [coefficients[pidx], sp_coefficients[pidx], sp_coefficients[pidx], sp_coefficients[pidx]]
            for comp_idx, (lx, ly, lz) in enumerate(powers):
                comps[comp_idx] += coeffs[comp_idx] * _primitive_cartesian_value32(
                    lx, ly, lz, alpha, x, y, z, exp_term
                )
        return comps

    angular_momentum = abs(st)
    if angular_momentum not in CARTESIAN_POWERS:
        raise NotImplementedError(f"Unsupported shell type {st}; only S through H shells are supported")
    powers = CARTESIAN_POWERS[angular_momentum]
    if st >= 0:
        matrix = np.eye(len(powers), dtype=np.float32)
    else:
        matrix = SPHERICAL_TO_CARTESIAN[angular_momentum].astype(np.float32, copy=False)

    comps = [np.zeros(points32.shape[0], dtype=np.float32) for _ in range(matrix.shape[1])]
    nonzero_by_cart = [np.flatnonzero(np.abs(matrix[row]) > 1.0e-14) for row in range(matrix.shape[0])]
    exponents = shell.exponents.astype(np.float32, copy=False)
    coefficients = shell.coefficients.astype(np.float32, copy=False)
    for pidx, alpha in enumerate(exponents):
        coeff = coefficients[pidx]
        exp_term = np.exp(-alpha * r2)
        for cart_idx, (lx, ly, lz) in enumerate(powers):
            comp_indices = nonzero_by_cart[cart_idx]
            if comp_indices.size == 0:
                continue
            term = coeff * _primitive_cartesian_value32(lx, ly, lz, alpha, x, y, z, exp_term)
            for comp_idx in comp_indices:
                comps[int(comp_idx)] += matrix[cart_idx, int(comp_idx)] * term
    return comps


def evaluate_mo(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    points: np.ndarray,
    coeff_cutoff: float = 1.0e-10,
) -> np.ndarray:
    coeff_matrix = wavefunction.coefficients(spin)
    if orbital_index0 < 0 or orbital_index0 >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index {orbital_index0 + 1} is outside 1..{coeff_matrix.shape[0]}")
    coeffs = coeff_matrix[orbital_index0]
    values = np.zeros(points.shape[0], dtype=np.float64)
    basis_index = 0
    for shell in wavefunction.shells:
        nfunc = n_shell_functions(shell.shell_type)
        shell_coeffs = coeffs[basis_index : basis_index + nfunc]
        basis_index += nfunc
        if np.max(np.abs(shell_coeffs)) < coeff_cutoff:
            continue
        components = evaluate_shell(shell, points)
        for component, coeff in zip(components, shell_coeffs):
            if abs(coeff) >= coeff_cutoff:
                values += coeff * component
    return values


def evaluate_mos(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    points: np.ndarray,
    coeff_cutoff: float = 1.0e-10,
) -> np.ndarray:
    indices = np.array(list(orbital_indices0), dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, points.shape[0]), dtype=np.float64)

    coeff_matrix = wavefunction.coefficients(spin)
    if int(indices.min()) < 0 or int(indices.max()) >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index is outside 1..{coeff_matrix.shape[0]}")

    selected_coeffs = coeff_matrix[indices]
    values = np.zeros((indices.size, points.shape[0]), dtype=np.float64)
    basis_index = 0
    for shell in wavefunction.shells:
        nfunc = n_shell_functions(shell.shell_type)
        shell_coeffs = selected_coeffs[:, basis_index : basis_index + nfunc]
        basis_index += nfunc
        active_components = np.max(np.abs(shell_coeffs), axis=0) >= coeff_cutoff
        if not np.any(active_components):
            continue
        components = evaluate_shell(shell, points)
        component_matrix = np.vstack(
            [components[int(comp_idx)] for comp_idx in np.flatnonzero(active_components)]
        )
        values += shell_coeffs[:, active_components] @ component_matrix
    return values


def evaluate_mos_float32(
    wavefunction: Wavefunction,
    spin: str,
    orbital_indices0: Iterable[int],
    points: np.ndarray,
    coeff_cutoff: float = 1.0e-10,
) -> np.ndarray:
    """Evaluate several orbitals with bounded float32 working memory."""
    indices = np.asarray(list(orbital_indices0), dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, points.shape[0]), dtype=np.float32)

    coeff_matrix = wavefunction.coefficients(spin)
    if int(indices.min()) < 0 or int(indices.max()) >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index is outside 1..{coeff_matrix.shape[0]}")

    selected_coeffs = coeff_matrix[indices].astype(np.float32, copy=False)
    values = np.zeros((indices.size, points.shape[0]), dtype=np.float32)
    basis_index = 0
    for shell in wavefunction.shells:
        nfunc = n_shell_functions(shell.shell_type)
        shell_coeffs = selected_coeffs[:, basis_index : basis_index + nfunc]
        basis_index += nfunc
        active_components = np.max(np.abs(shell_coeffs), axis=0) >= coeff_cutoff
        if not np.any(active_components):
            continue
        components = evaluate_shell_float32(shell, points)
        component_matrix = np.vstack(
            [components[int(comp_idx)] for comp_idx in np.flatnonzero(active_components)]
        )
        values += shell_coeffs[:, active_components] @ component_matrix
    return values


def evaluate_mo_float32(
    wavefunction: Wavefunction,
    spin: str,
    orbital_index0: int,
    points: np.ndarray,
    coeff_cutoff: float = 1.0e-10,
) -> np.ndarray:
    coeff_matrix = wavefunction.coefficients(spin)
    if orbital_index0 < 0 or orbital_index0 >= coeff_matrix.shape[0]:
        raise IndexError(f"Orbital index {orbital_index0 + 1} is outside 1..{coeff_matrix.shape[0]}")
    coeffs = coeff_matrix[orbital_index0].astype(np.float32, copy=False)
    values = np.zeros(points.shape[0], dtype=np.float32)
    basis_index = 0
    for shell in wavefunction.shells:
        nfunc = n_shell_functions(shell.shell_type)
        shell_coeffs = coeffs[basis_index : basis_index + nfunc]
        basis_index += nfunc
        if np.max(np.abs(shell_coeffs)) < coeff_cutoff:
            continue
        components = evaluate_shell_float32(shell, points)
        for component, coeff in zip(components, shell_coeffs):
            if abs(coeff) >= coeff_cutoff:
                values += coeff * component
    return values
