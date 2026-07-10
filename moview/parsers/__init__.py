from __future__ import annotations

from pathlib import Path

from ..wavefunction import Wavefunction
from .fchk import FCHKParser
from .molden import MoldenParser
from .orca import ORCAParser


def detect_wavefunction_format(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(262_144)
    lower = sample.lower()
    if "[molden format]" in lower or ("[atoms]" in lower and "[gto]" in lower):
        return "molden"
    if "number of basis functions" in lower or "alpha mo coefficients" in lower:
        return "fchk"
    name = path.name.lower()
    if name.endswith((".fchk", ".fch")):
        return "fchk"
    if "molden" in name or name.endswith((".molden", ".mol")):
        return "molden"
    raise ValueError(f"Could not determine wavefunction file type for {path}")


def parse_wavefunction(path: Path, file_format: str | None = None) -> Wavefunction:
    resolved_format = (file_format or detect_wavefunction_format(path)).lower()
    if resolved_format == "fchk":
        return FCHKParser(path).parse()
    if resolved_format == "molden":
        return MoldenParser(path).parse()
    raise ValueError(f"Unsupported wavefunction file type: {resolved_format}")


__all__ = [
    "FCHKParser",
    "MoldenParser",
    "ORCAParser",
    "detect_wavefunction_format",
    "parse_wavefunction",
]
