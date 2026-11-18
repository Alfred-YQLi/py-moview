"""Gaussian formatted checkpoint (.fchk / .fch) reader."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.basis import n_shell_functions
from ..core.wavefunction import Shell, Wavefunction
from .numbers import fortran_float, parse_floats, parse_ints

# "Label (up to 40 chars)   T   value"  or  "Label   T   N=  count"
_HEADER = re.compile(r"^(?P<label>\S.*?)\s+(?P<kind>[IRCLH])\s+(?:N=\s*(?P<count>\d+)|(?P<value>\S.*?))\s*$")
# Values per line for array types that are not split on whitespace.
_TEXT_ITEMS_PER_LINE = {"C": 5, "H": 9, "L": 72}
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd]?[-+]\d+|[EeDd]\d+)?")
_GLUED_TEXT = re.compile(r"\d[A-Za-z]{2,}")


@dataclass
class _Array:
    """An array field kept as text until it is needed (most fields never are)."""

    label: str
    kind: str
    count: int
    text: str

    def values(self) -> np.ndarray:
        try:
            return parse_ints(self.text, self.count) if self.kind == "I" else parse_floats(self.text, self.count)
        except ValueError as exc:
            raise ValueError(f"FCHK field {self.label!r}: {exc}") from None


def _take_values(line: str, need: int) -> tuple[str, str, int]:
    """Split ``line`` after ``need`` numbers: ``(numeric part, remainder, n_found)``."""
    cut, found = 0, 0
    for match in _NUMBER.finditer(line):
        cut = match.end()
        found += 1
        if found == need:
            break
    return line[:cut], line[cut:], found


def read_fchk_fields(path: Path) -> tuple[str, str, dict[str, object]]:
    """Return ``(title, method_line, fields)``; arrays are returned as lazy ``_Array``s."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    title = lines[0].strip() if lines else path.name
    method_line = " ".join(lines[1].split()) if len(lines) > 1 else ""
    fields: dict[str, object] = {}
    i = 2
    while i < len(lines):
        match = _HEADER.match(lines[i].rstrip("\n"))
        i += 1
        if match is None:
            continue
        label = match.group("label").strip()
        kind = match.group("kind")
        count_text = match.group("count")
        if count_text is None:
            value = match.group("value").strip()
            if kind == "I":
                fields[label] = int(value.split()[-1])
            elif kind == "R":
                fields[label] = fortran_float(value.split()[-1])
            else:
                fields[label] = value
            continue

        count = int(count_text)
        if kind in _TEXT_ITEMS_PER_LINE:
            # Character/logical arrays: skip their fixed-width lines so text
            # inside them is never mistaken for a field header.
            n_lines = math.ceil(count / _TEXT_ITEMS_PER_LINE[kind]) if count else 0
            fields[label] = "".join(lines[i : i + n_lines])
            i += n_lines
            continue

        parts: list[str] = []
        tokens = 0
        while i < len(lines) and tokens < count:
            line = lines[i]
            n_tokens = len(line.split())
            if tokens + n_tokens > count or (tokens + n_tokens == count and _GLUED_TEXT.search(line)):
                # Last line of the block.  Some writers (or concatenated files)
                # glue the next header onto it: "2.35E-04Beta Orbital Energies ...".
                numeric, rest, found = _take_values(line, count - tokens)
                parts.append(numeric)
                tokens += found
                if rest.strip():
                    lines[i] = rest.lstrip()  # re-read the remainder as a header
                else:
                    i += 1
                break
            parts.append(line)
            tokens += n_tokens
            i += 1
        fields[label] = _Array(label, kind, count, "".join(parts))
    return title, method_line, fields


def _require(fields: dict[str, object], *names: str):
    for name in names:
        if name in fields:
            value = fields[name]
            return value.values() if isinstance(value, _Array) else value
    raise ValueError(f"Missing required FCHK field: {' / '.join(names)}")


def _optional_array(fields: dict[str, object], *names: str) -> np.ndarray | None:
    for name in names:
        value = fields.get(name)
        if isinstance(value, _Array):
            return value.values()
    return None


def _fit_length(values: np.ndarray | None, count: int) -> np.ndarray:
    """Pad with NaN / truncate so there is exactly one entry per orbital."""
    out = np.full(count, np.nan, dtype=np.float64)
    if values is not None:
        n = min(count, values.size)
        out[:n] = values[:n]
    return out


def parse_fchk(path: Path) -> Wavefunction:
    title, method_line, fields = read_fchk_fields(path)

    atomic_numbers = np.asarray(_require(fields, "Atomic numbers"), dtype=np.int32)
    coords_raw = np.asarray(_require(fields, "Current cartesian coordinates"), dtype=np.float64)
    if coords_raw.size % 3:
        raise ValueError("Current cartesian coordinates length is not divisible by 3")
    coords = coords_raw.reshape(-1, 3)
    if atomic_numbers.size != coords.shape[0]:
        raise ValueError(
            f"Atom count mismatch: {atomic_numbers.size} atomic numbers, {coords.shape[0]} coordinate rows"
        )

    n_alpha = int(_require(fields, "Number of alpha electrons"))
    n_beta = int(_require(fields, "Number of beta electrons"))
    n_basis = int(_require(fields, "Number of basis functions"))
    if n_basis <= 0:
        raise ValueError(f"Number of basis functions must be positive, got {n_basis}")

    shell_types = np.asarray(_require(fields, "Shell types"), dtype=np.int32)
    n_primitives = np.asarray(_require(fields, "Number of primitives per shell"), dtype=np.int32)
    shell_to_atom = np.asarray(_require(fields, "Shell to atom map"), dtype=np.int32) - 1
    n_shells = shell_types.size
    if n_primitives.size != n_shells or shell_to_atom.size != n_shells:
        raise ValueError("Shell types, primitive counts and shell-to-atom map must have equal lengths")
    if np.any(n_primitives < 1):
        raise ValueError("Every shell needs at least one primitive")
    if np.any(shell_to_atom < 0) or np.any(shell_to_atom >= atomic_numbers.size):
        raise ValueError("Shell to atom map contains an out-of-range atom index")
    unknown = sorted({int(t) for t in shell_types if n_shell_functions(int(t)) == 0})
    if unknown:
        raise ValueError(f"Unsupported FCHK shell types: {unknown}")

    exponents = np.asarray(_require(fields, "Primitive exponents"), dtype=np.float64)
    coefficients = np.asarray(_require(fields, "Contraction coefficients"), dtype=np.float64)
    sp_coefficients = _optional_array(fields, "P(S=P) Contraction coefficients")
    n_total = int(n_primitives.sum())
    if exponents.size != n_total or coefficients.size != n_total:
        raise ValueError(
            f"Primitive count mismatch: shells need {n_total}, "
            f"found {exponents.size} exponents and {coefficients.size} coefficients"
        )
    if sp_coefficients is not None and sp_coefficients.size != n_total:
        raise ValueError("P(S=P) contraction coefficient count does not match the primitive count")

    shell_centers = _optional_array(fields, "Coordinates of each shell")
    if shell_centers is None:
        shell_centers = coords[shell_to_atom]
    else:
        if shell_centers.size != n_shells * 3:
            raise ValueError("Coordinates of each shell do not match the shell count")
        shell_centers = shell_centers.reshape(-1, 3)

    shells: list[Shell] = []
    offsets = np.concatenate(([0], np.cumsum(n_primitives)))
    for index, shell_type in enumerate(shell_types):
        primitives = slice(int(offsets[index]), int(offsets[index + 1]))
        shells.append(
            Shell(
                shell_type=int(shell_type),
                center=shell_centers[index].copy(),
                exponents=exponents[primitives].copy(),
                coefficients=coefficients[primitives].copy(),
                sp_coefficients=(
                    sp_coefficients[primitives].copy()
                    if shell_type == -1 and sp_coefficients is not None
                    else None
                ),
            )
        )

    expected = sum(n_shell_functions(int(t)) for t in shell_types)
    if expected != n_basis:
        raise ValueError(f"Basis count mismatch: file says {n_basis}, shells give {expected}")

    def coefficient_matrix(raw: np.ndarray, label: str) -> np.ndarray:
        if raw.size % n_basis:
            raise ValueError(f"{label} length {raw.size} is not divisible by the basis size {n_basis}")
        return raw.reshape(-1, n_basis)

    alpha_coefficients = coefficient_matrix(
        np.asarray(_require(fields, "Alpha MO coefficients"), dtype=np.float64), "Alpha MO coefficients"
    )
    beta_raw = _optional_array(fields, "Beta MO coefficients")
    beta_coefficients = None if beta_raw is None else coefficient_matrix(beta_raw, "Beta MO coefficients")

    alpha_energies = _fit_length(_optional_array(fields, "Alpha Orbital Energies"), alpha_coefficients.shape[0])
    beta_energies = None
    if beta_coefficients is not None:
        beta_energies = _fit_length(_optional_array(fields, "Beta Orbital Energies"), beta_coefficients.shape[0])

    return Wavefunction(
        path=path,
        title=title,
        method_line=method_line,
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
        source_format="fchk",
    )
