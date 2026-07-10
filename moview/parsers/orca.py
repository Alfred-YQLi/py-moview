from __future__ import annotations

from pathlib import Path

from ..wavefunction import Wavefunction


class ORCAParser:
    """Placeholder for ORCA wavefunction parsing.

    The original single-file viewer did not implement ORCA parsing; this module
    exists so future ORCA support has a stable home without touching the fchk or
    Molden parsers.
    """

    def __init__(self, path: Path):
        self.path = path

    def parse(self) -> Wavefunction:
        raise NotImplementedError("ORCA wavefunction parsing is not implemented yet")
