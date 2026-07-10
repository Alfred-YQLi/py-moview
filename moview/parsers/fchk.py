from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..basis.gaussian import n_shell_functions
from ..wavefunction import Wavefunction, Shell
from .common import _as_float


class FCHKParser:
    HEADER_RE = re.compile(r"^(?P<label>.*?)\s+(?P<kind>[IRC])\s+(?:N=\s*(?P<n>\d+)|(?P<value>.*?))\s*$")

    def __init__(self, path: Path):
        self.path = path
        self.title = path.name
        self.method_line = ""
        self.fields: dict[str, object] = {}

    def parse(self) -> Wavefunction:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()

        if lines:
            self.title = lines[0].strip()
        if len(lines) > 1:
            self.method_line = lines[1].strip()

        i = 2 if len(lines) > 1 else 0
        while i < len(lines):
            line = lines[i].rstrip("\n")
            match = self.HEADER_RE.match(line)
            if not match:
                i += 1
                continue

            label = match.group("label").strip()
            kind = match.group("kind")
            n_text = match.group("n")
            value = match.group("value") or ""
            i += 1

            if n_text is None:
                parts = value.split()
                if kind == "I":
                    self.fields[label] = int(parts[-1])
                elif kind == "R":
                    self.fields[label] = _as_float(parts[-1])
                else:
                    self.fields[label] = value.strip()
                continue

            n_items = int(n_text)
            if kind == "C":
                self.fields[label] = []
                continue

            block_lines: list[str] = []
            token_count = 0
            while i < len(lines) and token_count < n_items:
                block_lines.append(lines[i])
                token_count += len(lines[i].split())
                i += 1
            if token_count < n_items:
                raise ValueError(f"Field {label!r} ended early: expected {n_items}, got {token_count}")
            block_text = "".join(block_lines)
            if kind == "I":
                array = np.fromstring(block_text, dtype=np.int32, sep=" ", count=n_items)
            else:
                array = np.fromstring(
                    block_text.replace("D", "E").replace("d", "E"),
                    dtype=np.float64,
                    sep=" ",
                    count=n_items,
                )
            if array.size != n_items:
                raise ValueError(f"Field {label!r} ended early: expected {n_items}, got {array.size}")
            self.fields[label] = array

        return self._build_wavefunction()

    def _field(self, *names: str):
        for name in names:
            if name in self.fields:
                return self.fields[name]
        joined = ", ".join(names)
        raise KeyError(f"Missing required fchk field: {joined}")

    def _optional_array(self, *names: str) -> np.ndarray | None:
        for name in names:
            value = self.fields.get(name)
            if isinstance(value, np.ndarray):
                return value
        return None

    @staticmethod
    def _reshape_coefficients(raw: np.ndarray, n_basis: int, label: str) -> np.ndarray:
        if raw.size % n_basis != 0:
            raise ValueError(f"{label} length {raw.size} is not divisible by basis count {n_basis}")
        return raw.reshape((raw.size // n_basis, n_basis))

    @staticmethod
    def _fit_energies(energies: np.ndarray, count: int) -> np.ndarray:
        if energies.size == count:
            return energies
        if energies.size > count:
            return energies[:count].copy()
        out = np.full(count, np.nan, dtype=np.float64)
        out[: energies.size] = energies
        return out

    def _build_wavefunction(self) -> Wavefunction:
        atomic_numbers = np.asarray(self._field("Atomic numbers"), dtype=np.int32)
        coords_raw = np.asarray(
            self._field("Current cartesian coordinates"),
            dtype=np.float64,
        )
        if coords_raw.size % 3:
            raise ValueError("Current cartesian coordinates length is not divisible by 3")
        coords = coords_raw.reshape(-1, 3)
        if atomic_numbers.size != coords.shape[0]:
            raise ValueError(
                f"Atom count mismatch: {atomic_numbers.size} atomic numbers, "
                f"{coords.shape[0]} coordinate rows"
            )
        n_alpha = int(self._field("Number of alpha electrons"))
        n_beta = int(self._field("Number of beta electrons"))
        n_basis = int(self._field("Number of basis functions"))
        if n_basis <= 0:
            raise ValueError(f"Number of basis functions must be positive, got {n_basis}")
        shell_types = np.asarray(self._field("Shell types"), dtype=np.int32)
        nprim = np.asarray(self._field("Number of primitives per shell"), dtype=np.int32)
        shell_to_atom = np.asarray(self._field("Shell to atom map"), dtype=np.int32) - 1
        n_shells = shell_types.size
        if nprim.size != n_shells or shell_to_atom.size != n_shells:
            raise ValueError(
                "Shell types, primitive counts, and shell-to-atom map must have equal lengths"
            )
        if np.any(nprim < 0):
            raise ValueError("Number of primitives per shell cannot be negative")
        if np.any(shell_to_atom < 0) or np.any(shell_to_atom >= atomic_numbers.size):
            raise ValueError("Shell to atom map contains an out-of-range atom index")
        exponents = np.asarray(self._field("Primitive exponents"), dtype=np.float64)
        coefficients = np.asarray(self._field("Contraction coefficients"), dtype=np.float64)
        sp_coefficients = self._optional_array("P(S=P) Contraction coefficients")
        primitive_count = int(nprim.sum())
        if exponents.size != primitive_count or coefficients.size != primitive_count:
            raise ValueError(
                f"Primitive count mismatch: shell layout gives {primitive_count}, "
                f"exponents={exponents.size}, coefficients={coefficients.size}"
            )
        if sp_coefficients is not None and sp_coefficients.size != primitive_count:
            raise ValueError(
                "P(S=P) contraction coefficient count does not match primitive count"
            )
        shell_coords_arr = self._optional_array("Coordinates of each shell")
        if shell_coords_arr is None:
            shell_coords = coords[shell_to_atom]
        else:
            shell_coords_raw = np.asarray(shell_coords_arr, dtype=np.float64)
            if shell_coords_raw.size != n_shells * 3:
                raise ValueError("Coordinates of each shell do not match the shell count")
            shell_coords = shell_coords_raw.reshape(-1, 3)

        shells: list[Shell] = []
        prim_start = 0
        for idx, shell_type in enumerate(shell_types):
            count = int(nprim[idx])
            prim_slice = slice(prim_start, prim_start + count)
            p_coeffs = None
            if shell_type == -1 and sp_coefficients is not None:
                p_coeffs = sp_coefficients[prim_slice].copy()
            shells.append(
                Shell(
                    shell_type=int(shell_type),
                    center=shell_coords[idx].copy(),
                    exponents=exponents[prim_slice].copy(),
                    coefficients=coefficients[prim_slice].copy(),
                    sp_coefficients=p_coeffs,
                )
            )
            prim_start += count

        expected_basis = sum(n_shell_functions(int(st)) for st in shell_types)
        if expected_basis != n_basis:
            raise ValueError(f"Basis count mismatch: fchk says {n_basis}, shell layout gives {expected_basis}")

        alpha_energies = np.asarray(
            self._field("Alpha Orbital Energies", "alpha orbital energies", "orbital energies"),
            dtype=np.float64,
        )
        beta_energies = self._optional_array("Beta Orbital Energies", "beta orbital energies")
        alpha_raw = np.asarray(
            self._field("Alpha MO coefficients", "alpha MO coefficients", "MO coefficients"),
            dtype=np.float64,
        )
        beta_raw = self._optional_array("Beta MO coefficients", "beta MO coefficients")

        alpha_coefficients = self._reshape_coefficients(alpha_raw, n_basis, "Alpha MO coefficients")
        beta_coefficients = (
            self._reshape_coefficients(np.asarray(beta_raw, dtype=np.float64), n_basis, "Beta MO coefficients")
            if beta_raw is not None
            else None
        )
        alpha_energies = self._fit_energies(alpha_energies, alpha_coefficients.shape[0])
        if beta_coefficients is not None:
            beta_energies = (
                self._fit_energies(
                    np.asarray(beta_energies, dtype=np.float64),
                    beta_coefficients.shape[0],
                )
                if beta_energies is not None
                else np.full(beta_coefficients.shape[0], np.nan, dtype=np.float64)
            )

        return Wavefunction(
            path=self.path,
            title=self.title,
            method_line=self.method_line,
            atomic_numbers=atomic_numbers,
            coordinates_bohr=coords,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_basis=n_basis,
            shell_types=shell_types,
            shell_to_atom=shell_to_atom,
            shells=shells,
            alpha_energies=alpha_energies,
            beta_energies=beta_energies,
            alpha_coefficients=alpha_coefficients,
            beta_coefficients=beta_coefficients,
        )
