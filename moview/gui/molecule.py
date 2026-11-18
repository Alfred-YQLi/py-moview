"""Triangle meshes for atoms and bonds (NumPy only; no OpenGL)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..elements import covalent_radius, element_color

ATOM_STYLES = ("ball_stick", "space_filling", "licorice")
LICORICE_RADIUS = 0.15
BALL_STICK_BOND_RADIUS = 0.075


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray  # (n, 3) float32, angstrom
    normals: np.ndarray  # (n, 3) float32
    colors: np.ndarray  # (n, 4) float32
    faces: np.ndarray  # (m, 3) uint32

    @classmethod
    def empty(cls) -> TriangleMesh:
        return cls(
            np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty((0, 4), np.float32), np.empty((0, 3), np.uint32)
        )

    @property
    def nbytes(self) -> int:
        return int(self.vertices.nbytes + self.normals.nbytes + self.colors.nbytes + self.faces.nbytes)


def atom_radius(atomic_number: int, style: str = "ball_stick", scale: float = 1.0) -> float:
    covalent = covalent_radius(atomic_number)
    if style == "ball_stick":
        radius = min(0.42, max(0.16, 0.30 * covalent + 0.06))
    elif style == "space_filling":
        radius = min(1.30, max(0.30, 0.85 * covalent + 0.25))
    elif style == "licorice":
        radius = LICORICE_RADIUS
    else:
        raise ValueError(f"Unknown atom style: {style}")
    return radius * max(0.05, float(scale))


@lru_cache(maxsize=4)
def unit_sphere(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere of radius 1: ``(vertices, faces)``; vertices double as normals."""
    theta = np.linspace(0.0, math.pi, rows + 1)
    phi = np.linspace(0.0, 2.0 * math.pi, cols, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    vertices = np.column_stack((np.sin(t).ravel() * np.cos(p).ravel(), np.sin(t).ravel() * np.sin(p).ravel(), np.cos(t).ravel()))
    faces = []
    for r in range(rows):
        for c in range(cols):
            a = r * cols + c
            b = r * cols + (c + 1) % cols
            d = (r + 1) * cols + c
            e = (r + 1) * cols + (c + 1) % cols
            if r > 0:
                faces.append((a, d, b))
            if r < rows - 1:
                faces.append((b, d, e))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.uint32)


@lru_cache(maxsize=4)
def unit_tube(segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open cylinder of radius 1 from z=0 to z=1: ``(vertices, normals, faces)``."""
    angle = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(segments)))
    vertices = np.vstack((ring, ring + (0.0, 0.0, 1.0)))
    normals = np.vstack((ring, ring))
    faces = []
    for k in range(segments):
        n = (k + 1) % segments
        faces.append((k, n, segments + k))
        faces.append((n, segments + n, segments + k))
    return vertices.astype(np.float32), normals.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def _frame(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix whose third column is ``direction`` (unit)."""
    z = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    helper = np.array((1.0, 0.0, 0.0)) if abs(z[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def build_molecule_mesh(
    atomic_numbers: np.ndarray,
    coords_ang: np.ndarray,
    bonds: np.ndarray,
    style: str = "ball_stick",
    atom_scale: float = 1.0,
) -> TriangleMesh:
    if style not in ATOM_STYLES:
        raise ValueError(f"Unknown atom style: {style}")
    n_atoms = len(atomic_numbers)
    detail = (14, 24) if style == "space_filling" or n_atoms < 200 else (10, 16)
    sphere_v, sphere_f = unit_sphere(*detail)
    vertices, normals, colors, faces = [], [], [], []
    offset = 0

    def add(v: np.ndarray, n: np.ndarray, rgb, f: np.ndarray) -> None:
        nonlocal offset
        vertices.append(v.astype(np.float32))
        normals.append(n.astype(np.float32))
        colors.append(np.tile(np.array((*rgb, 1.0), dtype=np.float32), (v.shape[0], 1)))
        faces.append(f + offset)
        offset += v.shape[0]

    coords = np.asarray(coords_ang, dtype=np.float64)
    radii = np.array([atom_radius(z, style, atom_scale) for z in atomic_numbers])
    for index, z in enumerate(atomic_numbers):
        if int(z) <= 0:
            continue
        add(sphere_v * radii[index] + coords[index], sphere_v, element_color(int(z)), sphere_f)

    if style != "space_filling":
        tube_v, tube_n, tube_f = unit_tube(16 if n_atoms < 200 else 10)
        bond_radius = LICORICE_RADIUS if style == "licorice" else BALL_STICK_BOND_RADIUS * max(0.6, min(1.4, atom_scale))
        for i, j in np.asarray(bonds, dtype=np.int64).reshape(-1, 2):
            start, end = coords[i], coords[j]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length < 1.0e-6:
                continue
            rotation = _frame(vector)
            middle = start + 0.5 * vector
            # Two half-bonds, each in the color of its atom.
            for origin, atom in ((start, i), (middle, j)):
                scaled = tube_v * np.array((bond_radius, bond_radius, 0.5 * length))
                add(scaled @ rotation.T + origin, tube_n @ rotation.T, element_color(int(atomic_numbers[atom])), tube_f)

    if not vertices:
        return TriangleMesh.empty()
    return TriangleMesh(np.vstack(vertices), np.vstack(normals), np.vstack(colors), np.vstack(faces).astype(np.uint32))


def label_texts(atomic_numbers: np.ndarray, mode: str) -> list[str]:
    from ..elements import atom_symbol

    if mode not in ("off", "number", "element", "number_element"):
        raise ValueError(f"Unknown atom label mode: {mode}")
    labels = []
    for index, z in enumerate(atomic_numbers, start=1):
        z = int(z)
        if mode == "off" or z == 0:
            labels.append("")
        elif mode == "number":
            labels.append(str(index))
        elif mode == "element":
            labels.append(atom_symbol(z))
        else:
            labels.append(f"{atom_symbol(z)}{index}")
    return labels
