from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import BOHR_TO_ANG


@dataclass
class Shell:
    shell_type: int
    center: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray
    sp_coefficients: np.ndarray | None = None


@dataclass
class Wavefunction:
    path: Path
    title: str
    method_line: str
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    n_alpha: int
    n_beta: int
    n_basis: int
    shell_types: np.ndarray
    shell_to_atom: np.ndarray
    shells: list[Shell]
    alpha_energies: np.ndarray
    beta_energies: np.ndarray | None
    alpha_coefficients: np.ndarray
    beta_coefficients: np.ndarray | None
    alpha_occupations: np.ndarray | None = None
    beta_occupations: np.ndarray | None = None
    source_format: str = "fchk"

    @property
    def coordinates_angstrom(self) -> np.ndarray:
        return self.coordinates_bohr * BOHR_TO_ANG

    @property
    def is_unrestricted(self) -> bool:
        return self.beta_coefficients is not None

    @staticmethod
    def _validate_spin(spin: str) -> None:
        if spin not in {"alpha", "beta"}:
            raise ValueError(f"Spin must be 'alpha' or 'beta', got {spin!r}")

    def energies(self, spin: str) -> np.ndarray:
        self._validate_spin(spin)
        if spin == "beta" and self.beta_energies is not None:
            return self.beta_energies
        return self.alpha_energies

    def coefficients(self, spin: str) -> np.ndarray:
        self._validate_spin(spin)
        if spin == "beta" and self.beta_coefficients is not None:
            return self.beta_coefficients
        return self.alpha_coefficients

    def occupation(self, spin: str, orbital_index0: int) -> float:
        energies = self.energies(spin)
        if orbital_index0 < 0 or orbital_index0 >= len(energies):
            raise IndexError(
                f"Orbital index {orbital_index0 + 1} is outside 1..{len(energies)}"
            )
        if spin == "beta" and self.beta_occupations is not None:
            return float(self.beta_occupations[orbital_index0])
        if spin != "beta" and self.alpha_occupations is not None:
            return float(self.alpha_occupations[orbital_index0])
        if self.beta_coefficients is None:
            if orbital_index0 < self.n_beta:
                return 2.0
            if orbital_index0 < self.n_alpha:
                return 1.0
            return 0.0
        if spin == "alpha":
            return 1.0 if orbital_index0 < self.n_alpha else 0.0
        return 1.0 if orbital_index0 < self.n_beta else 0.0

    def default_orbital(self, spin: str) -> int:
        energies = self.energies(spin)
        if len(energies) == 0:
            raise ValueError(f"No {spin} orbitals are available")
        count = self.n_beta if spin == "beta" else self.n_alpha
        return min(max(0, count - 1), len(energies) - 1)

    def lumo_orbital(self, spin: str) -> int:
        energies = self.energies(spin)
        for idx in range(len(energies)):
            if self.occupation(spin, idx) <= 0:
                return idx
        return len(energies) - 1
