"""Molden (.molden / .molden.input) reader.

Supports ``[Atoms]`` (AU or Angs), ``[GTO]`` with S..H and SP shells,
``[5D]/[5D7F]/[5D10F]/[7F]/[9G]/[11H]`` angular flags, and alpha/beta ``[MO]``
blocks with sparse coefficient lists.

Different programs disagree on details the format leaves open.  After reading,
the coefficients are interpreted under candidate conventions and the one that
makes the MOs orthonormal (``C S C^T = I``) is kept:

* ``standard`` - contraction coefficients refer to normalized primitives;
  contracted functions are renormalized.
* ``orca`` - ``orca_2mkl`` multiplies every contraction coefficient by the
  primitive's radial normalization and flips the sign of the pure
  f(+3), f(-3), g(+3), g(-3), g(+4), g(-4) functions.
* ``+axis-cartesian`` - Cartesian d/f/g/h components share the normalization of
  their ``x^l`` component instead of being normalized individually.
"""

from __future__ import annotations

import re
from array import array
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..constants import BOHR_TO_ANG
from ..core.basis import CARTESIAN_POWERS, cartesian_factors, contracted_norm, n_shell_functions, radial_norm
from ..core.wavefunction import Shell, Wavefunction
from ..elements import SYMBOL_TO_ATOMIC_NUMBER
from .numbers import fortran_float

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")

# Molden's 15G Cartesian order; reordered to CARTESIAN_POWERS[4] on input.
MOLDEN_CARTESIAN_G_ORDER = (
    (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1),
    (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0),
    (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2),
)
_SHELL_LABELS = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "sp": -1}
# Pure-function positions (m = +3, -3, +4, -4) whose sign ORCA flips.
_ORCA_FLIPPED_COMPONENTS = {3: (5, 6), 4: (5, 6, 7, 8)}

# Accept a convention immediately below this error; warn above WARN level.
_CONVENTION_EXACT = 1.0e-5
_CONVENTION_WARN = 1.0e-2
# Larger bases are checked on a window of orbitals around the HOMO.
_FULL_CHECK_MAX_BASIS = 3000
_CHECK_WINDOW = 400
_CHECK_MAX_BASIS = 12_000


@dataclass
class _RawShell:
    atom: int
    label: str
    exponents: np.ndarray
    coefficients: np.ndarray
    sp_coefficients: np.ndarray | None


@dataclass
class _MOBlock:
    spin: str = "alpha"
    energy: float = float("nan")
    occupation: float = 0.0
    # Compact C arrays: a large [MO] section has millions of coefficients.
    indices: array | None = None
    values: array | None = None


class _MoldenReader:
    """Single-pass reader collecting raw sections and MO blocks."""

    def __init__(self, path: Path):
        self.path = path
        self.title = ""
        self.atom_lines: list[str] = []
        self.atom_unit = ""
        self.gto_lines: list[str] = []
        self.pure = {2: False, 3: False, 4: False, 5: False}
        self.blocks: list[_MOBlock] = []
        self.has_mo_section = False

    def read(self) -> None:
        section = None
        title_lines: list[str] = []
        block: _MOBlock | None = None
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                match = _SECTION.match(raw)
                if match is not None:
                    name = match.group(1).strip().lower()
                    section = name
                    self._apply_flag(name)
                    if name == "atoms":
                        self.atom_unit = match.group(2).strip().upper()
                    elif name == "mo":
                        self.has_mo_section = True
                    continue
                if section == "mo":
                    block = self._mo_line(raw, block)
                elif section == "gto":
                    self.gto_lines.append(raw)
                elif section == "atoms":
                    self.atom_lines.append(raw)
                elif section == "title":
                    title_lines.append(raw.strip())
        self.title = next((line for line in title_lines if line), "")

    def _apply_flag(self, name: str) -> None:
        if name in ("5d", "5d7f"):
            self.pure[2] = self.pure[3] = True
        elif name == "5d10f":
            self.pure[2], self.pure[3] = True, False
        elif name == "6d":
            self.pure[2] = False
        elif name == "7f":
            self.pure[3] = True
        elif name == "10f":
            self.pure[3] = False
        elif name == "9g":
            self.pure[4] = True
        elif name == "15g":
            self.pure[4] = False
        elif name == "11h":
            self.pure[5] = True
        elif name == "21h":
            self.pure[5] = False

    def _mo_line(self, raw: str, block: _MOBlock | None) -> _MOBlock | None:
        text = raw.strip()
        if not text:
            return block
        first = text[0]
        if first.isdigit():
            parts = text.split()
            if len(parts) >= 2:
                if block is None:
                    block = self._new_block()
                block.indices.append(int(parts[0]))
                try:
                    block.values.append(float(parts[1]))
                except ValueError:
                    block.values.append(fortran_float(parts[1]))
            return block
        if "=" not in text:
            return block
        key, _, value = text.partition("=")
        key = key.strip().lower()
        # A keyword after coefficient lines starts the next orbital, whether or
        # not the writer emits "Sym=".
        if block is None or block.indices:
            block = self._new_block()
        if key == "ene":
            block.energy = fortran_float(value)
        elif key == "spin":
            spin = value.strip().lower()
            if spin not in ("alpha", "beta"):
                raise ValueError(f"Invalid Molden MO spin: {value.strip()!r}")
            block.spin = spin
        elif key == "occup":
            block.occupation = fortran_float(value)
        return block

    def _new_block(self) -> _MOBlock:
        block = _MOBlock(indices=array("q"), values=array("d"))
        self.blocks.append(block)
        return block


def _parse_atoms(reader: _MoldenReader) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    numbers: list[int] = []
    coords: list[tuple[float, float, float]] = []
    sequence: dict[int, int] = {}
    for line in reader.atom_lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            z = int(parts[2])
        except ValueError:
            symbol = re.sub(r"[^A-Za-z]", "", parts[0]).upper()
            z = SYMBOL_TO_ATOMIC_NUMBER.get(symbol, -1)
            if z < 0:
                raise ValueError(f"Unknown Molden atom symbol: {parts[0]!r}") from None
        try:
            sequence[int(parts[1])] = len(numbers)
        except ValueError:
            pass
        numbers.append(z)
        coords.append((fortran_float(parts[3]), fortran_float(parts[4]), fortran_float(parts[5])))
    if not numbers:
        raise ValueError("Molden [Atoms] section is missing or contains no atoms")
    coords_arr = np.asarray(coords, dtype=np.float64)
    if "AU" not in reader.atom_unit:
        coords_arr = coords_arr / BOHR_TO_ANG
    return np.asarray(numbers, dtype=np.int32), coords_arr, sequence


def _parse_gto(reader: _MoldenReader, n_atoms: int, sequence: dict[int, int]) -> list[_RawShell]:
    shells: list[_RawShell] = []
    atom: int | None = None
    lines = reader.gto_lines
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0].lstrip("+-").isdigit():
            number = int(parts[0])
            atom = sequence.get(number, number - 1)
            if not 0 <= atom < n_atoms:
                raise ValueError(f"Molden [GTO] atom index out of range: {parts[0]}")
            continue
        label = parts[0].lower()
        if atom is None or len(parts) < 2:
            continue
        if label not in _SHELL_LABELS:
            if parts[1].isdigit():
                raise ValueError(f"Unsupported Molden shell label: {parts[0]!r}")
            continue
        n_primitives = int(parts[1])
        if n_primitives < 1:
            raise ValueError(f"Molden shell {label!r} must contain at least one primitive")
        exponents, coefficients, sp_coefficients = [], [], []
        for _ in range(n_primitives):
            if i >= len(lines):
                raise ValueError("Molden [GTO] ended inside a shell")
            values = lines[i].split()
            i += 1
            if len(values) < (3 if label == "sp" else 2):
                raise ValueError(f"Invalid Molden primitive line: {lines[i - 1].strip()!r}")
            exponents.append(fortran_float(values[0]))
            coefficients.append(fortran_float(values[1]))
            if label == "sp":
                sp_coefficients.append(fortran_float(values[2]))
        shells.append(
            _RawShell(
                atom,
                label,
                np.asarray(exponents),
                np.asarray(coefficients),
                np.asarray(sp_coefficients) if label == "sp" else None,
            )
        )
    if not shells:
        raise ValueError("Molden [GTO] section is missing or contains no shells")
    return shells


def _shell_type(label: str, pure: dict[int, bool]) -> int:
    base = _SHELL_LABELS[label]
    return -base if base >= 2 and pure[base] else base


def _assemble_coefficients(blocks: list[_MOBlock], spin: str, n_basis: int, g_reorders: list[tuple[int, np.ndarray]]):
    selected = [block for block in blocks if block.spin == spin and block.indices]
    if not selected:
        return None, None, None
    matrix = np.zeros((len(selected), n_basis))
    for row, block in enumerate(selected):
        index = np.frombuffer(block.indices, dtype=np.int64)
        if index.min() < 1 or index.max() > n_basis:
            bad = int(index.min()) if index.min() < 1 else int(index.max())
            raise ValueError(f"Molden MO coefficient index {bad} is outside 1..{n_basis}")
        matrix[row, index - 1] = np.frombuffer(block.values, dtype=np.float64)
    for start, order in g_reorders:
        matrix[:, start : start + order.size] = matrix[:, start + order]
    energies = np.array([block.energy for block in selected])
    occupations = np.array([block.occupation for block in selected])
    return energies, occupations, matrix


def _read_raw(path: Path) -> tuple[Wavefunction, list[_RawShell], _MoldenReader]:
    reader = _MoldenReader(path)
    reader.read()
    if not reader.has_mo_section:
        raise ValueError("Molden file is missing [MO]")
    atomic_numbers, coords, sequence = _parse_atoms(reader)
    raw_shells = _parse_gto(reader, len(atomic_numbers), sequence)

    shells: list[Shell] = []
    g_reorders: list[tuple[int, np.ndarray]] = []
    molden_g_index = {power: k for k, power in enumerate(MOLDEN_CARTESIAN_G_ORDER)}
    g_order = np.array([molden_g_index[power] for power in CARTESIAN_POWERS[4]])
    n_basis = 0
    for raw in raw_shells:
        shell_type = _shell_type(raw.label, reader.pure)
        if shell_type == 4:
            g_reorders.append((n_basis, g_order))
        shells.append(Shell(shell_type, coords[raw.atom].copy(), raw.exponents, raw.coefficients, raw.sp_coefficients))
        n_basis += n_shell_functions(shell_type)

    alpha = _assemble_coefficients(reader.blocks, "alpha", n_basis, g_reorders)
    beta = _assemble_coefficients(reader.blocks, "beta", n_basis, g_reorders)
    if alpha[2] is None:
        raise ValueError("Molden [MO] section contains no alpha orbitals")
    alpha_energies, alpha_occ, alpha_coefficients = alpha
    beta_energies, beta_occ, beta_coefficients = beta

    if beta_coefficients is None:
        n_alpha = int(np.count_nonzero(alpha_occ > 1.0e-8))
        n_beta = int(np.count_nonzero(alpha_occ > 1.5))
    else:
        n_alpha = int(np.count_nonzero(alpha_occ > 1.0e-8))
        n_beta = int(np.count_nonzero(beta_occ > 1.0e-8))

    wavefunction = Wavefunction(
        path=path,
        title=reader.title or path.name,
        method_line="Molden",
        atomic_numbers=atomic_numbers,
        coordinates_bohr=coords,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_basis=n_basis,
        shell_types=np.array([shell.shell_type for shell in shells], dtype=np.int32),
        shell_to_atom=np.array([raw.atom for raw in raw_shells], dtype=np.int32),
        shells=shells,
        alpha_energies=alpha_energies,
        beta_energies=beta_energies,
        alpha_coefficients=alpha_coefficients,
        beta_coefficients=beta_coefficients,
        alpha_occupations=alpha_occ,
        beta_occupations=beta_occ,
        source_format="molden",
    )
    return wavefunction, raw_shells, reader


# ---------------------------------------------------------------- conventions
def _normalized_shells(shells: list[Shell], *, orca: bool) -> list[Shell]:
    fixed: list[Shell] = []
    for shell in shells:
        if shell.shell_type == -1:
            parts = [(0, shell.coefficients), (1, shell.sp_coefficients)]
        else:
            parts = [(abs(shell.shell_type), shell.coefficients)]
        new = []
        for l, coefficients in parts:
            c = np.asarray(coefficients, dtype=np.float64)
            if orca:
                c = c / radial_norm(l, shell.exponents)
            norm = contracted_norm(l, shell.exponents, c)
            new.append(c / norm if norm > 0.0 else c)
        fixed.append(replace(shell, coefficients=new[0], sp_coefficients=new[1] if len(new) > 1 else None))
    return fixed


def _coefficient_scale(shells: list[Shell], *, orca_signs: bool, axis_cartesian: bool) -> np.ndarray | None:
    """Per-basis-function factors converting file MO coefficients to MOview's convention."""
    scale = np.ones(sum(shell.n_functions for shell in shells))
    changed = False
    start = 0
    for shell in shells:
        l = abs(shell.shell_type)
        if orca_signs and shell.shell_type < -1 and l in _ORCA_FLIPPED_COMPONENTS:
            for k in _ORCA_FLIPPED_COMPONENTS[l]:
                scale[start + k] = -1.0
                changed = True
        if axis_cartesian and shell.shell_type >= 2:
            factors = cartesian_factors(l)
            scale[start : start + factors.size] = factors[0] / factors
            changed = True
        start += shell.n_functions
    return scale if changed else None


@dataclass(frozen=True)
class _Convention:
    name: str
    orca: bool
    axis_cartesian: bool


def _apply(base: Wavefunction, shells: list[Shell], convention: _Convention) -> Wavefunction:
    scale = _coefficient_scale(shells, orca_signs=convention.orca, axis_cartesian=convention.axis_cartesian)
    alpha = base.alpha_coefficients if scale is None else base.alpha_coefficients * scale
    beta = base.beta_coefficients
    if beta is not None and scale is not None:
        beta = beta * scale
    return replace(base, shells=shells, alpha_coefficients=alpha, beta_coefficients=beta, notes=list(base.notes))


def _orthonormality_error(wavefunction: Wavefunction, overlap: np.ndarray) -> float:
    from ..core.overlap import orthonormality

    window = None if wavefunction.n_basis <= _FULL_CHECK_MAX_BASIS else _CHECK_WINDOW
    return max(report.worst for report in orthonormality(wavefunction, overlap, max_orbitals=window))


def _resolve_conventions(base: Wavefunction, title: str) -> Wavefunction:
    from ..core.overlap import overlap_matrix

    looks_like_orca = "orca" in title.lower()
    has_cartesian = any(shell.shell_type >= 2 for shell in base.shells)
    candidates = [_Convention("ORCA", True, False), _Convention("standard", False, False)]
    if not looks_like_orca:
        candidates.reverse()
    if has_cartesian:
        candidates += [replace(c, name=c.name + " + axis-normalized Cartesian", axis_cartesian=True) for c in candidates]

    shells_for = {
        False: _normalized_shells(base.shells, orca=False),
        True: _normalized_shells(base.shells, orca=True),
    }
    if base.n_basis > _CHECK_MAX_BASIS:
        chosen = _apply(base, shells_for[candidates[0].orca], candidates[0])
        chosen.notes.append(f"Molden: {candidates[0].name} conventions assumed (basis too large to verify).")
        return chosen

    overlaps: dict[bool, np.ndarray] = {}
    best: tuple[float, Wavefunction, _Convention] | None = None
    for convention in candidates:
        if convention.orca not in overlaps:
            overlaps[convention.orca] = overlap_matrix(replace(base, shells=shells_for[convention.orca]))
        candidate = _apply(base, shells_for[convention.orca], convention)
        error = _orthonormality_error(candidate, overlaps[convention.orca])
        if best is None or error < best[0]:
            best = (error, candidate, convention)
        if error < _CONVENTION_EXACT:
            break
    assert best is not None
    error, chosen, convention = best
    chosen.notes.append(f"Molden: {convention.name} conventions (MO orthonormality error {error:.1e}).")
    if error > _CONVENTION_WARN:
        chosen.notes.append(
            "Warning: no known Molden convention makes these MOs orthonormal; orbital shapes may be wrong."
        )
    return chosen


def parse_molden(path: Path, *, resolve_conventions: bool = True) -> Wavefunction:
    base, _raw, reader = _read_raw(path)
    if not resolve_conventions:
        return replace(base, shells=_normalized_shells(base.shells, orca=False))
    return _resolve_conventions(base, reader.title)
