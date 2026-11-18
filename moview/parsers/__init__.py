"""Wavefunction file readers: format detection and dispatch."""

from __future__ import annotations

from pathlib import Path

from ..core.wavefunction import Wavefunction
from .fchk import parse_fchk
from .molden import parse_molden

SUPPORTED_FORMATS = ("fchk", "molden")
FILE_DIALOG_FILTER = (
    "Wavefunction files (*.fchk *.fch *.molden *.molden.input *.input *.mold);;"
    "Gaussian formatted checkpoint (*.fchk *.fch);;"
    "Molden (*.molden *.molden.input *.input *.mold);;"
    "All files (*)"
)


def detect_wavefunction_format(path: Path) -> str:
    """Guess the format from the file contents, then from its name."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(262_144).lower()
    if "[molden format]" in sample or ("[atoms]" in sample and "[gto]" in sample):
        return "molden"
    if "number of basis functions" in sample or "alpha mo coefficients" in sample:
        return "fchk"
    name = path.name.lower()
    if name.endswith((".fchk", ".fch")):
        return "fchk"
    if "molden" in name or name.endswith(".mold"):
        return "molden"
    raise ValueError(f"Could not determine the wavefunction format of {path}")


def parse_wavefunction(path: Path | str, file_format: str | None = None) -> Wavefunction:
    path = Path(path)
    resolved = (file_format or detect_wavefunction_format(path)).lower()
    if resolved == "fchk":
        return parse_fchk(path)
    if resolved == "molden":
        return parse_molden(path)
    raise ValueError(f"Unsupported wavefunction format: {resolved}")


__all__ = [
    "FILE_DIALOG_FILTER",
    "SUPPORTED_FORMATS",
    "detect_wavefunction_format",
    "parse_fchk",
    "parse_molden",
    "parse_wavefunction",
]
