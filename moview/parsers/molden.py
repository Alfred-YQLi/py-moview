from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..basis.gaussian import CARTESIAN_POWERS, n_shell_functions
from ..constants import BOHR_TO_ANG, SYMBOL_TO_ATOMIC_NUMBER
from ..wavefunction import Shell, Wavefunction
from .common import _as_float


def _section_header(line: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", line)
    if match is None:
        return None
    return match.group(1).strip().lower(), match.group(2).strip()


# Molden's required 15G order differs from the evaluator's Cartesian order.
_MOLDEN_CARTESIAN_G_POWERS = (
    (4, 0, 0),
    (0, 4, 0),
    (0, 0, 4),
    (3, 1, 0),
    (3, 0, 1),
    (1, 3, 0),
    (0, 3, 1),
    (1, 0, 3),
    (0, 1, 3),
    (2, 2, 0),
    (2, 0, 2),
    (0, 2, 2),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
)


class MoldenParser:
    SHELL_LABELS = {
        "s": (0, 1),
        "p": (1, 3),
        "d": (2, 6),
        "f": (3, 10),
        "g": (4, 15),
        "h": (5, 21),
        "sp": (-1, 4),
    }

    def __init__(self, path: Path):
        self.path = path
        self.title = path.name
        self.sections: dict[str, tuple[list[str], str]] = {}
        self.spherical_d = False
        self.spherical_f = False
        self.spherical_g = False
        self.spherical_h = False
        self._coefficient_reorders: list[tuple[int, np.ndarray]] = []

    def parse(self) -> Wavefunction:
        self.sections = self._read_structure_sections()
        self.title = self._parse_title()
        atomic_numbers, coords_bohr = self._parse_atoms()
        shells, shell_to_atom = self._parse_gto(coords_bohr)
        self._coefficient_reorders = self._build_coefficient_reorders(shells)
        n_basis = sum(n_shell_functions(shell.shell_type) for shell in shells)
        (
            alpha_energies,
            alpha_occ,
            alpha_coeffs,
            beta_energies,
            beta_occ,
            beta_coeffs,
        ) = self._parse_mo(n_basis)
        n_alpha = int(np.count_nonzero(alpha_occ > 1.0e-8))
        n_beta = int(np.count_nonzero(beta_occ > 1.0e-8)) if beta_occ is not None else n_alpha
        return Wavefunction(
            path=self.path,
            title=self.title,
            method_line="Molden",
            atomic_numbers=atomic_numbers,
            coordinates_bohr=coords_bohr,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_basis=n_basis,
            shell_types=np.asarray([shell.shell_type for shell in shells], dtype=np.int32),
            shell_to_atom=shell_to_atom,
            shells=shells,
            alpha_energies=alpha_energies,
            beta_energies=beta_energies,
            alpha_coefficients=alpha_coeffs,
            beta_coefficients=beta_coeffs,
            alpha_occupations=alpha_occ,
            beta_occupations=beta_occ,
            source_format="molden",
        )

    def _apply_angular_convention(self, name: str) -> None:
        if name in {"5d", "5d7f"}:
            self.spherical_d = True
            self.spherical_f = True
        elif name == "5d10f":
            self.spherical_d = True
            self.spherical_f = False
        elif name == "6d":
            self.spherical_d = False
        elif name == "7f":
            self.spherical_f = True
        elif name == "10f":
            self.spherical_f = False
        elif name == "9g":
            self.spherical_g = True
        elif name == "15g":
            self.spherical_g = False
        elif name == "11h":
            self.spherical_h = True
        elif name == "21h":
            self.spherical_h = False

    def _read_structure_sections(self) -> dict[str, tuple[list[str], str]]:
        self.spherical_d = False
        self.spherical_f = False
        self.spherical_g = False
        self.spherical_h = False
        sections: dict[str, tuple[list[str], str]] = {}
        current_lines: list[str] | None = None
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                header = _section_header(line)
                if header is None:
                    if current_lines is not None:
                        current_lines.append(line)
                    continue
                name, suffix = header
                self._apply_angular_convention(name)
                if name == "mo":
                    break
                if name in {"title", "atoms", "gto"} and name not in sections:
                    current_lines = []
                    sections[name] = (current_lines, suffix)
                else:
                    current_lines = None
        return sections

    def _parse_title(self) -> str:
        title_section = self.sections.get("title")
        if title_section is None:
            return self.path.name
        lines, _suffix = title_section
        for line in lines:
            text = line.strip()
            if text:
                return text
        return self.path.name

    def _parse_atoms(self) -> tuple[np.ndarray, np.ndarray]:
        section = self.sections.get("atoms")
        if section is None:
            raise ValueError("Molden file is missing [Atoms]")
        lines, suffix = section
        unit = suffix.upper()
        atomic_numbers: list[int] = []
        coords: list[tuple[float, float, float]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                continue
            symbol = parts[0].upper()
            try:
                atomic_number = int(parts[2])
            except ValueError:
                atomic_number = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
                if atomic_number is None:
                    raise ValueError(f"Unknown Molden atom symbol: {parts[0]!r}") from None
            atomic_numbers.append(atomic_number)
            coords.append((_as_float(parts[3]), _as_float(parts[4]), _as_float(parts[5])))
        if not atomic_numbers:
            raise ValueError("Molden [Atoms] section contains no atoms")
        coords_arr = np.asarray(coords, dtype=np.float64)
        if "AU" not in unit:
            coords_arr = coords_arr / BOHR_TO_ANG
        return np.asarray(atomic_numbers, dtype=np.int32), coords_arr

    def _molden_shell_type(self, label: str) -> int:
        base_type, _count = self.SHELL_LABELS[label]
        if label == "d" and self.spherical_d:
            return -2
        if label == "f" and self.spherical_f:
            return -3
        if label == "g" and self.spherical_g:
            return -4
        if label == "h" and self.spherical_h:
            return -5
        return base_type

    @staticmethod
    def _build_coefficient_reorders(shells: list[Shell]) -> list[tuple[int, np.ndarray]]:
        source_index = {
            power: index for index, power in enumerate(_MOLDEN_CARTESIAN_G_POWERS)
        }
        g_to_internal = np.asarray(
            [source_index[power] for power in CARTESIAN_POWERS[4]],
            dtype=np.intp,
        )
        reorders: list[tuple[int, np.ndarray]] = []
        basis_start = 0
        for shell in shells:
            if shell.shell_type == 4:
                reorders.append((basis_start, g_to_internal))
            basis_start += n_shell_functions(shell.shell_type)
        return reorders

    def _parse_gto(self, coords_bohr: np.ndarray) -> tuple[list[Shell], np.ndarray]:
        section = self.sections.get("gto")
        if section is None:
            raise ValueError("Molden file is missing [GTO]")
        lines, _suffix = section
        shells: list[Shell] = []
        shell_to_atom: list[int] = []
        atom_index0: int | None = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            parts = line.split()
            if parts[0].lstrip("+-").isdigit():
                atom_index0 = int(parts[0]) - 1
                if atom_index0 < 0 or atom_index0 >= coords_bohr.shape[0]:
                    raise ValueError(f"Molden [GTO] atom index out of range: {parts[0]}")
                continue
            label = parts[0].lower()
            if atom_index0 is None or len(parts) < 2:
                continue
            if label not in self.SHELL_LABELS:
                if parts[1].lstrip("+-").isdigit():
                    raise ValueError(f"Unsupported Molden shell label: {parts[0]!r}")
                continue
            n_prim = int(parts[1])
            if n_prim < 1:
                raise ValueError(f"Molden shell {label!r} must contain at least one primitive")
            if len(parts) >= 3:
                _as_float(parts[2])
            exponents: list[float] = []
            coefficients: list[float] = []
            sp_coefficients: list[float] | None = [] if label == "sp" else None
            for _ in range(n_prim):
                if i >= len(lines):
                    raise ValueError("Molden [GTO] ended inside a shell")
                prim_parts = lines[i].split()
                i += 1
                if len(prim_parts) < 2:
                    raise ValueError("Invalid Molden primitive line")
                exponents.append(_as_float(prim_parts[0]))
                coefficients.append(_as_float(prim_parts[1]))
                if sp_coefficients is not None:
                    if len(prim_parts) < 3:
                        raise ValueError("Molden sp shell primitive is missing p coefficient")
                    sp_coefficients.append(_as_float(prim_parts[2]))
            shells.append(
                Shell(
                    shell_type=self._molden_shell_type(label),
                    center=coords_bohr[atom_index0].copy(),
                    exponents=np.asarray(exponents, dtype=np.float64),
                    coefficients=np.asarray(coefficients, dtype=np.float64),
                    sp_coefficients=(
                        np.asarray(sp_coefficients, dtype=np.float64)
                        if sp_coefficients is not None
                        else None
                    ),
                )
            )
            shell_to_atom.append(atom_index0)
        if not shells:
            raise ValueError("Molden [GTO] section contains no basis shells")
        return shells, np.asarray(shell_to_atom, dtype=np.int32)

    def _finish_mo_block(
        self,
        block: dict[str, object],
        n_basis: int,
        alpha_rows: list[np.ndarray],
        alpha_energies: list[float],
        alpha_occ: list[float],
        beta_rows: list[np.ndarray],
        beta_energies: list[float],
        beta_occ: list[float],
    ) -> None:
        coeff_pairs = block.get("coefficients")
        if not coeff_pairs:
            return
        row = np.zeros(n_basis, dtype=np.float64)
        for basis_idx, coeff in coeff_pairs:  # type: ignore[union-attr]
            if not 1 <= basis_idx <= n_basis:
                raise ValueError(
                    f"Molden MO coefficient index {basis_idx} is outside 1..{n_basis}"
                )
            row[basis_idx - 1] = coeff
        for basis_start, source_offsets in self._coefficient_reorders:
            basis_end = basis_start + source_offsets.size
            row[basis_start:basis_end] = row[basis_start:basis_end][source_offsets]
        spin = str(block.get("spin", "alpha")).strip().lower()
        if spin not in {"alpha", "beta"}:
            raise ValueError(f"Invalid Molden MO spin: {block.get('spin')!r}")
        energy = float(block.get("energy", np.nan))
        occupation = float(block.get("occupation", 0.0))
        if spin == "beta":
            beta_rows.append(row)
            beta_energies.append(energy)
            beta_occ.append(occupation)
        else:
            alpha_rows.append(row)
            alpha_energies.append(energy)
            alpha_occ.append(occupation)

    def _parse_mo(
        self,
        n_basis: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        alpha_rows: list[np.ndarray] = []
        alpha_energies: list[float] = []
        alpha_occ: list[float] = []
        beta_rows: list[np.ndarray] = []
        beta_energies: list[float] = []
        beta_occ: list[float] = []
        block: dict[str, object] = {}
        in_mo = False
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                header = _section_header(line)
                if header is not None:
                    name, _suffix = header
                    if name == "mo":
                        in_mo = True
                        continue
                    if in_mo:
                        break
                    continue
                if not in_mo:
                    continue
                text = line.strip()
                if not text:
                    continue
                lower = text.lower()
                if lower.startswith("sym="):
                    self._finish_mo_block(
                        block,
                        n_basis,
                        alpha_rows,
                        alpha_energies,
                        alpha_occ,
                        beta_rows,
                        beta_energies,
                        beta_occ,
                    )
                    block = {"coefficients": []}
                elif lower.startswith("ene="):
                    block["energy"] = _as_float(text.split("=", 1)[1])
                elif lower.startswith("spin="):
                    block["spin"] = text.split("=", 1)[1].strip()
                elif lower.startswith("occup="):
                    block["occupation"] = _as_float(text.split("=", 1)[1])
                else:
                    parts = text.split()
                    if len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                        coefficients = block.setdefault("coefficients", [])
                        if not isinstance(coefficients, list):
                            raise TypeError("Molden MO coefficient container is not a list")
                        coefficients.append((int(parts[0]), _as_float(parts[1])))
        if not in_mo:
            raise ValueError("Molden file is missing [MO]")
        self._finish_mo_block(
            block,
            n_basis,
            alpha_rows,
            alpha_energies,
            alpha_occ,
            beta_rows,
            beta_energies,
            beta_occ,
        )
        if not alpha_rows:
            raise ValueError("Molden [MO] section contains no alpha orbitals")
        alpha_coeffs = np.vstack(alpha_rows)
        beta_coeffs = np.vstack(beta_rows) if beta_rows else None
        return (
            np.asarray(alpha_energies, dtype=np.float64),
            np.asarray(alpha_occ, dtype=np.float64),
            alpha_coeffs,
            np.asarray(beta_energies, dtype=np.float64) if beta_rows else None,
            np.asarray(beta_occ, dtype=np.float64) if beta_rows else None,
            beta_coeffs,
        )
