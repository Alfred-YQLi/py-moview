"""Synthetic wavefunctions with exactly orthonormal MOs, and file writers for them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moview.core.basis import CARTESIAN_POWERS, cartesian_factors, contracted_norm, radial_norm
from moview.core.overlap import overlap_matrix
from moview.core.wavefunction import Shell, Wavefunction
from moview.parsers.molden import MOLDEN_CARTESIAN_G_ORDER

FIXTURE_DIR_ENV = "MOVIEW_FIXTURES"

# Atoms in bohr: a bent "water" so every shell overlaps with something.
ATOMS = ((8, (0.0, 0.0, 0.0)), (1, (0.0, 1.43, -1.10)), (1, (0.0, -1.43, -1.10)))

# (atom, l, exponents, contraction coefficients for normalized primitives)
SHELLS = (
    (0, 0, (130.7, 23.81, 6.444), (0.154, 0.535, 0.445)),
    (0, 0, (0.38,), (1.0,)),
    (0, 1, (5.03, 1.17), (0.156, 0.607)),
    (0, 2, (1.2,), (1.0,)),
    (0, 3, (0.9,), (1.0,)),
    (0, 4, (0.8,), (1.0,)),
    (1, 0, (3.43, 0.62), (0.154, 0.535)),
    (1, 1, (0.75,), (1.0,)),
    (2, 0, (3.43, 0.62), (0.154, 0.535)),
    (2, 1, (0.75,), (1.0,)),
)


def make_wavefunction(*, cartesian: bool = False, unrestricted: bool = False, seed: int = 0) -> Wavefunction:
    """Restricted closed shell (5 doubly occupied) or unrestricted (5 alpha, 4 beta)."""
    numbers = np.array([z for z, _ in ATOMS], dtype=np.int32)
    coords = np.array([xyz for _, xyz in ATOMS], dtype=np.float64)
    shells = []
    for atom, l, exponents, coefficients in SHELLS:
        exps = np.array(exponents, dtype=np.float64)
        coefs = np.array(coefficients, dtype=np.float64)
        coefs = coefs / contracted_norm(l, exps, coefs)
        shell_type = l if (cartesian or l < 2) else -l
        shells.append(Shell(shell_type, coords[atom].copy(), exps, coefs))
    n_basis = sum(shell.n_functions for shell in shells)
    template = Wavefunction(
        path=Path("synthetic"), title="synthetic", method_line="", atomic_numbers=numbers,
        coordinates_bohr=coords, n_alpha=5, n_beta=5 if not unrestricted else 4, n_basis=n_basis,
        shell_types=np.array([s.shell_type for s in shells], dtype=np.int32),
        shell_to_atom=np.array([atom for atom, *_ in SHELLS], dtype=np.int32), shells=shells,
        alpha_energies=np.linspace(-1.0, 1.0, n_basis), beta_energies=None,
        alpha_coefficients=np.eye(n_basis), beta_coefficients=None,
    )
    overlap = overlap_matrix(template)
    values, vectors = np.linalg.eigh(overlap)
    inverse_sqrt = vectors @ np.diag(values**-0.5) @ vectors.T
    rng = np.random.default_rng(seed)

    def orbitals() -> np.ndarray:
        q, _ = np.linalg.qr(rng.normal(size=(n_basis, n_basis)))
        return q @ inverse_sqrt

    template.alpha_coefficients = orbitals()
    if unrestricted:
        template.beta_coefficients = orbitals()
        template.beta_energies = np.linspace(-0.9, 1.1, n_basis)
    return template


def _molden_g_reorder(row: np.ndarray, shells: list[Shell]) -> np.ndarray:
    out = row.copy()
    index = {power: k for k, power in enumerate(MOLDEN_CARTESIAN_G_ORDER)}
    order = np.array([index[p] for p in CARTESIAN_POWERS[4]])
    start = 0
    for shell in shells:
        if shell.shell_type == 4:
            out[start + order] = row[start : start + 15]
        start += shell.n_functions
    return out


def write_molden(
    path: Path,
    wf: Wavefunction,
    *,
    orca: bool = False,
    axis_cartesian: bool = False,
    with_sym: bool = True,
    angstrom: bool = False,
) -> Path:
    from moview.constants import BOHR_TO_ANG
    from moview.elements import atom_symbol

    lines = ["[Molden Format]", "[Title]", " Molden file created by orca_2mkl" if orca else " synthetic test"]
    lines.append("[Atoms] Angs" if angstrom else "[Atoms] AU")
    for index, (z, xyz) in enumerate(zip(wf.atomic_numbers, wf.coordinates_bohr), start=1):
        xyz = xyz * BOHR_TO_ANG if angstrom else xyz
        lines.append(f"{atom_symbol(z)} {index} {z} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    lines.append("[GTO]")
    for atom in range(wf.n_atoms):
        lines.append(f"  {atom + 1} 0")
        for shell, shell_atom in zip(wf.shells, wf.shell_to_atom):
            if shell_atom != atom:
                continue
            l = abs(shell.shell_type)
            coefficients = shell.coefficients * (radial_norm(l, shell.exponents) if orca else 1.0)
            lines.append(f"{'spdfgh'[l]} {len(shell.exponents)} 1.00")
            lines.extend(f"  {a:.10E} {c:.10E}" for a, c in zip(shell.exponents, coefficients))
        lines.append("")
    if any(shell.shell_type < -1 for shell in wf.shells):
        lines += ["[5D]", "[7F]", "[9G]"]

    # Convert MOview coefficients into what this writer would have put in the file.
    scale = np.ones(wf.n_basis)
    start = 0
    for shell in wf.shells:
        l = abs(shell.shell_type)
        if orca and shell.shell_type in (-3, -4):
            scale[start + 5 : start + (7 if l == 3 else 9)] = -1.0
        if axis_cartesian and shell.shell_type >= 2:
            factors = cartesian_factors(l)
            scale[start : start + factors.size] = factors / factors[0]
        start += shell.n_functions

    lines.append("[MO]")
    blocks = [("Alpha", wf.alpha_energies, wf.alpha_coefficients, wf.occupations("alpha"))]
    if wf.beta_coefficients is not None:
        blocks.append(("Beta", wf.beta_energies, wf.beta_coefficients, wf.occupations("beta")))
    for spin, energies, coefficients, occupations in blocks:
        for energy, row, occupation in zip(energies, coefficients, occupations):
            if with_sym:
                lines.append(" Sym= A")
            lines += [f" Ene= {energy:.8f}", f" Spin= {spin}", f" Occup= {occupation:.6f}"]
            row = _molden_g_reorder(row * scale, wf.shells)
            lines.extend(f"{k + 1:5d} {value: .12E}" for k, value in enumerate(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fchk_ints(label: str, values) -> list[str]:
    values = list(values)
    out = [f"{label:<40}   I   N={len(values):12d}"]
    for i in range(0, len(values), 6):
        out.append("".join(f"{int(v):12d}" for v in values[i : i + 6]))
    return out


def _fchk_reals(label: str, values, *, fortran_tiny: bool = False) -> list[str]:
    values = list(np.ravel(values))
    out = [f"{label:<40}   R   N={len(values):12d}"]
    for i in range(0, len(values), 5):
        chunk = [f"{float(v):16.8E}" for v in values[i : i + 5]]
        out.append("".join(chunk))
    if fortran_tiny:
        # Fortran drops the 'E' for 3-digit exponents: "1.00000000-100".
        out[1] = "  1.00000000-100" + out[1][16:]
    return out


def write_fchk(path: Path, wf: Wavefunction) -> Path:
    shells = wf.shells
    lines = ["synthetic fchk", "SP        RHF                                                         test"]
    lines.append(f"{'Number of atoms':<40}   I     {wf.n_atoms:12d}")
    # A character array whose text would look like a header if not skipped.
    lines += [f"{'Job labels':<40}   C   N=           2", "R          I   N=           9"]
    lines.append(f"{'Number of alpha electrons':<40}   I     {wf.n_alpha:12d}")
    lines.append(f"{'Number of beta electrons':<40}   I     {wf.n_beta:12d}")
    lines.append(f"{'Number of basis functions':<40}   I     {wf.n_basis:12d}")
    lines += _fchk_ints("Atomic numbers", wf.atomic_numbers)
    lines += _fchk_reals("Current cartesian coordinates", wf.coordinates_bohr)
    lines += _fchk_ints("Shell types", [s.shell_type for s in shells])
    lines += _fchk_ints("Number of primitives per shell", [len(s.exponents) for s in shells])
    lines += _fchk_ints("Shell to atom map", wf.shell_to_atom + 1)
    lines += _fchk_reals("Primitive exponents", np.concatenate([s.exponents for s in shells]))
    lines += _fchk_reals("Contraction coefficients", np.concatenate([s.coefficients for s in shells]))
    lines += _fchk_reals("Alpha Orbital Energies", wf.alpha_energies)
    coefficients = wf.alpha_coefficients.copy()
    lines += _fchk_reals("Alpha MO coefficients", coefficients, fortran_tiny=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
