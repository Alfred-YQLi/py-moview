from __future__ import annotations

from collections.abc import Iterable

from ..config import DEFAULT_ATOM_LABEL_MODE, DEFAULT_ATOM_STYLE, DEFAULT_SURFACE_STYLE
from ..constants import atom_symbol
from ..grid import GridSpec


ATOM_STYLE_OPTIONS = (
    ("Ball & stick", "ball_stick"),
    ("Space filling", "space_filling"),
    ("Licorice", "licorice"),
)
SURFACE_STYLE_OPTIONS = (
    ("Glass", "glass"),
    ("Solid", "solid"),
    ("Wireframe", "wireframe"),
    ("Solid + edges", "solid_edges"),
)
ATOM_LABEL_OPTIONS = (
    ("Off", "off"),
    ("Number", "number"),
    ("Element", "element"),
    ("Number + element", "number_element"),
)
ATOM_LABEL_PLACEMENT_OPTIONS = (
    ("Attached", "attached"),
    ("Floating", "floating"),
)

HIGH_GRID_WARNING_THRESHOLD = 256
QT_GRID_MAXIMUM = 2_147_483_647
GRID_PREFETCH_DEBOUNCE_MS = 650


def atom_label_texts(atomic_numbers: Iterable[int], mode: str) -> list[str]:
    if mode not in {value for _label, value in ATOM_LABEL_OPTIONS}:
        raise ValueError(f"Unknown atom label mode: {mode}")
    labels: list[str] = []
    for index, atomic_number in enumerate(atomic_numbers, start=1):
        z = int(atomic_number)
        if z == 0 or mode == "off":
            labels.append("")
        elif mode == "number":
            labels.append(str(index))
        elif mode == "element":
            labels.append(atom_symbol(z))
        else:
            labels.append(f"{index}{atom_symbol(z)}")
    return labels


def format_byte_size(nbytes: int) -> str:
    value = float(max(0, nbytes))
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit in {"B", "KiB"} else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def high_grid_warning_text(spec: GridSpec) -> str:
    scalar_bytes = spec.scalar_nbytes()
    return (
        f"Grid {spec.grid_size} creates {spec.n_points:,} points "
        f"({spec.shape[0]} x {spec.shape[1]} x {spec.shape[2]}).\n\n"
        f"One float32 orbital field needs about {format_byte_size(scalar_bytes)} before "
        "surface extraction. The calculation can take a long time and may use substantial "
        "additional working memory.\n\nContinue with this resolution?"
    )
