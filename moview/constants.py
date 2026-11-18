"""Physical constants and application-wide numeric defaults."""

from __future__ import annotations

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_EV = 27.211386245988

DEFAULT_ISOVALUE = 0.05
DEFAULT_GRID_SIZE = 81
DEFAULT_MARGIN_BOHR = 4.0

# Smallest grid edge accepted anywhere (CLI, config and GUI).
MIN_GRID_SIZE = 8
# Grid sizes above this value ask for confirmation in the GUI and warn in batch mode.
HIGH_GRID_WARNING_THRESHOLD = 256
# Grid points per matrix product when evaluating fields (bounds temporary memory).
DEFAULT_GRID_CHUNK_POINTS = 262_144
