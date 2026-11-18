"""Data model shared by all parsers, evaluators and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from ..constants import BOHR_TO_ANG
from .basis import n_shell_functions

SPINS = ("alpha", "beta")
OCCUPIED_THRESHOLD = 1.0e-8


@dataclass
class Shell:
    """One contracted Gaussian shell.

    ``shell_type`` follows the Gaussian FCHK convention: 0=S, 1=P, -1=SP,
    2=Cartesian D (6), -2=pure D (5), 3/-3 F, 4/-4 G, 5/-5 H.
    ``coefficients`` multiply *normalized* primitive Gaussians.
    """

    shell_type: int
    center: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray
    sp_coefficients: np.ndarray | None = None

    @property
    def angular_momentum(self) -> int:
        return 1 if self.shell_type == -1 else abs(self.shell_type)

    @property
    def is_pure(self) -> bool:
        return self.shell_type < -1

    @property
    def n_functions(self) -> int:
        return n_shell_functions(self.shell_type)


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
    # Human-readable remarks from the parser (applied conventions, checks).
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def coordinates_angstrom(self) -> np.ndarray:
        return self.coordinates_bohr * BOHR_TO_ANG

    @property
    def n_atoms(self) -> int:
        return int(len(self.atomic_numbers))

    @property
    def is_unrestricted(self) -> bool:
        return self.beta_coefficients is not None

    @property
    def spins(self) -> tuple[str, ...]:
        return SPINS if self.is_unrestricted else SPINS[:1]

    @staticmethod
    def _validate_spin(spin: str) -> None:
        if spin not in SPINS:
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

    def n_orbitals(self, spin: str) -> int:
        return int(self.coefficients(spin).shape[0])

    # -------------------------------------------------------------- occupation
    @cached_property
    def _occupation_table(self) -> dict[str, np.ndarray]:
        return {spin: self._compute_occupations(spin) for spin in SPINS}

    def _compute_occupations(self, spin: str) -> np.ndarray:
        n_orb = self.n_orbitals(spin)
        explicit = self.beta_occupations if spin == "beta" else self.alpha_occupations
        if spin == "beta" and self.beta_occupations is None and not self.is_unrestricted:
            explicit = self.alpha_occupations
        if explicit is not None:
            return np.asarray(explicit, dtype=np.float64)[:n_orb]
        index = np.arange(n_orb)
        if not self.is_unrestricted:
            # Restricted (closed or open shell): doubly occupied up to n_beta,
            # singly occupied up to n_alpha.
            return np.where(index < self.n_beta, 2.0, np.where(index < self.n_alpha, 1.0, 0.0))
        count = self.n_alpha if spin == "alpha" else self.n_beta
        return np.where(index < count, 1.0, 0.0)

    def occupations(self, spin: str) -> np.ndarray:
        self._validate_spin(spin)
        return self._occupation_table[spin]

    def occupation(self, spin: str, orbital_index0: int) -> float:
        occupations = self.occupations(spin)
        if orbital_index0 < 0 or orbital_index0 >= occupations.size:
            raise IndexError(f"Orbital index {orbital_index0 + 1} is outside 1..{occupations.size}")
        return float(occupations[orbital_index0])

    def homo_index(self, spin: str) -> int:
        """Highest occupied orbital (0-based); 0 when nothing is occupied."""
        occupied = np.flatnonzero(self.occupations(spin) > OCCUPIED_THRESHOLD)
        if self.n_orbitals(spin) == 0:
            raise ValueError(f"No {spin} orbitals are available")
        return int(occupied[-1]) if occupied.size else 0

    def lumo_index(self, spin: str) -> int:
        """Lowest unoccupied orbital (0-based); the last orbital if all are occupied."""
        empty = np.flatnonzero(self.occupations(spin) <= OCCUPIED_THRESHOLD)
        return int(empty[0]) if empty.size else self.n_orbitals(spin) - 1

    def frontier_label(self, spin: str, orbital_index0: int) -> str:
        """Return ``HOMO``, ``HOMO-2``, ``LUMO+1`` ... or an empty string far from the gap."""
        homo = self.homo_index(spin)
        lumo = self.lumo_index(spin)
        if orbital_index0 <= homo and self.occupation(spin, orbital_index0) > OCCUPIED_THRESHOLD:
            delta = homo - orbital_index0
            return "HOMO" if delta == 0 else (f"HOMO-{delta}" if delta <= 9 else "")
        if orbital_index0 >= lumo:
            delta = orbital_index0 - lumo
            return "LUMO" if delta == 0 else (f"LUMO+{delta}" if delta <= 9 else "")
        return ""
