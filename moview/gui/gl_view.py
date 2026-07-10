from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..constants import BOHR_TO_ANG, COVALENT_RADII, ELEMENT_COLORS
from ..grid import OrbitalGrid
from ..surface import SurfaceMesh
from ..wavefunction import Wavefunction
from .presentation import atom_label_texts


FALLBACK_ELEMENT_COLORS = dict(ELEMENT_COLORS)


_GUI_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from PyQt6 import QtCore, QtGui, QtOpenGL, QtWidgets

    from OpenGL import GL
    from OpenGL.GL import shaders as gl_shaders

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
    from pyqtgraph.opengl import shaders as pg_shaders
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
    _GUI_IMPORT_ERROR = exc

    class _MissingQtWidgets:
        QWidget = object
        QMainWindow = object

    class _MissingGL:
        GLViewWidget = object

    QtCore = QtGui = QtOpenGL = pg = pg_shaders = GL = gl_shaders = None
    QtWidgets = _MissingQtWidgets()
    gl = _MissingGL()
    GLGraphicsItem = object


def _require_gui_dependencies() -> None:
    if _GUI_IMPORT_ERROR is None:
        return
    missing = _GUI_IMPORT_ERROR.name or "PyQtGraph/OpenGL dependency"
    print(
        f"Missing dependency: {missing}\n"
        "Install with:\n"
        "pip install numpy scikit-image pyqtgraph PyQt6 PyOpenGL",
        file=sys.stderr,
    )
    raise SystemExit(1) from _GUI_IMPORT_ERROR

VMD_ROTATE_DEG_PER_PIXEL = 1.0 / 3.0
OPENGL_SURFACE_FACE_LIMIT = 160_000
PREFETCH_OCCUPIED_BACK = 12
PREFETCH_VIRTUAL_FORWARD = 12
PREFETCH_BATCH_SIZE = 24
CORE_PREFETCH_OCCUPIED_BACK = 30
CORE_PREFETCH_VIRTUAL_FORWARD = 15
LOW_PREFETCH_GRID = 56
MAX_COMPARE_ORBITALS = 9
MOLECULE_SPHERE_ROWS = 10
MOLECULE_SPHERE_COLS = 16
MOLECULE_BOND_COLS = 14
MOLECULE_BOND_RADIUS = 0.055
SCENE_BACKGROUND_HEX = "#fafafa"
FOG_FLAT_SHADER = "orbitalFogFlat"
FOG_SHADED_SHADER = "orbitalFogShaded"
FOG_STRENGTH = 0.74
LABEL_ATLAS_FONT_PIXELS = 64
LABEL_WORLD_HEIGHT_PER_POINT = 0.032

PERIODIC_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
DISPLAY_SYMBOL_TO_ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(PERIODIC_SYMBOLS) if symbol}


def axis_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float64)
    if axis == "y":
        return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)
    if axis == "z":
        return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    raise ValueError(f"Unknown rotation axis: {axis}")


def default_scene_rotation() -> np.ndarray:
    return axis_rotation_matrix("x", -24.0) @ axis_rotation_matrix("y", 34.0)


def rgb_from_hex(color: str) -> np.ndarray:
    text = color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #rrggbb color, got {color!r}")
    return np.array([int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


SCENE_BACKGROUND_RGB = tuple(float(value) for value in rgb_from_hex(SCENE_BACKGROUND_HEX))


def load_gview_element_colors(path: Path | None = None) -> dict[int, tuple[float, float, float]]:
    color_paths = [path] if path is not None else [
        Path(__file__).resolve().parents[1] / "gview_color.tcl",
        Path(__file__).resolve().parents[2] / "gview_color.tcl",
    ]
    color_ids: dict[int, tuple[float, float, float]] = {}
    element_colors: dict[int, tuple[float, float, float]] = {}
    lines: list[str] | None = None
    for color_path in color_paths:
        if color_path is None:
            continue
        try:
            lines = color_path.read_text(encoding="utf-8").splitlines()
            break
        except OSError:
            continue
    if lines is None:
        return dict(FALLBACK_ELEMENT_COLORS)
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and parts[:3] == ["color", "change", "rgb"]:
            try:
                rgb = (float(parts[4]), float(parts[5]), float(parts[6]))
                max_channel = max(rgb)
                if max_channel > 100.0:
                    rgb = tuple(value / 255.0 for value in rgb)
                elif max_channel > 1.0:
                    rgb = tuple(value / 100.0 for value in rgb)
                color_ids[int(parts[3])] = tuple(max(0.0, min(1.0, value)) for value in rgb)
            except ValueError:
                continue
        elif len(parts) >= 4 and parts[:2] == ["color", "Element"]:
            z = DISPLAY_SYMBOL_TO_ATOMIC_NUMBER.get(parts[2])
            if z is None:
                continue
            try:
                color_id = int(parts[3])
            except ValueError:
                continue
            rgb = color_ids.get(color_id)
            if rgb is not None:
                element_colors[z] = rgb
    if not element_colors:
        return dict(FALLBACK_ELEMENT_COLORS)
    merged = dict(FALLBACK_ELEMENT_COLORS)
    merged.update(element_colors)
    return merged


ELEMENT_COLORS = load_gview_element_colors()


def _register_fog_shaders() -> None:
    if FOG_FLAT_SHADER in pg_shaders.ShaderProgram.names:
        return
    fog_uniform = {
        "u_fog": [
            SCENE_BACKGROUND_RGB[0],
            SCENE_BACKGROUND_RGB[1],
            SCENE_BACKGROUND_RGB[2],
            0.0,
            1.0,
            FOG_STRENGTH,
        ]
    }
    pg_shaders.ShaderProgram(
        FOG_FLAT_SHADER,
        [
            pg_shaders.VertexShader(
                """
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying float v_eye_depth;
                void main() {
                    gl_Position = u_mvp * a_position;
                    v_color = a_color;
                    v_eye_depth = max(gl_Position.w, 0.0);
                }
                """
            ),
            pg_shaders.FragmentShader(
                """
                #ifdef GL_ES
                precision mediump float;
                #endif
                uniform float u_fog[6];
                varying vec4 v_color;
                varying float v_eye_depth;
                void main() {
                    float fog = smoothstep(u_fog[3], u_fog[4], v_eye_depth) * u_fog[5];
                    vec3 bg = vec3(u_fog[0], u_fog[1], u_fog[2]);
                    gl_FragColor = vec4(mix(v_color.rgb, bg, fog), v_color.a);
                }
                """
            ),
        ],
        uniforms=fog_uniform,
    )
    pg_shaders.ShaderProgram(
        FOG_SHADED_SHADER,
        [
            pg_shaders.VertexShader(
                """
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                varying float v_eye_depth;
                void main() {
                    gl_Position = u_mvp * a_position;
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    v_eye_depth = max(gl_Position.w, 0.0);
                }
                """
            ),
            pg_shaders.FragmentShader(
                """
                #ifdef GL_ES
                precision mediump float;
                #endif
                uniform float u_fog[6];
                varying vec4 v_color;
                varying vec3 v_normal;
                varying float v_eye_depth;
                void main() {
                    vec3 light_dir = normalize(vec3(0.65, -0.85, -1.0));
                    float diffuse = max(dot(v_normal, light_dir), 0.0);
                    float highlight = pow(diffuse, 12.0);
                    vec3 lit = v_color.rgb * (0.46 + 0.54 * diffuse) + vec3(1.0) * (0.10 * highlight);
                    float fog = smoothstep(u_fog[3], u_fog[4], v_eye_depth) * u_fog[5];
                    vec3 bg = vec3(u_fog[0], u_fog[1], u_fog[2]);
                    gl_FragColor = vec4(mix(clamp(lit, 0.0, 1.0), bg, fog), v_color.a);
                }
                """
            ),
        ],
        uniforms=fog_uniform,
    )


def update_fog_shader_params(distance: float, radius: float) -> None:
    _register_fog_shaders()
    radius = max(float(radius), 0.1)
    start = max(0.01, float(distance) - 0.20 * radius)
    end = max(start + 0.10 * radius, float(distance) + 1.05 * radius)
    fog = [
        SCENE_BACKGROUND_RGB[0],
        SCENE_BACKGROUND_RGB[1],
        SCENE_BACKGROUND_RGB[2],
        start,
        end,
        FOG_STRENGTH,
    ]
    pg_shaders.ShaderProgram.names[FOG_FLAT_SHADER]["u_fog"] = fog
    pg_shaders.ShaderProgram.names[FOG_SHADED_SHADER]["u_fog"] = fog


def glass_facecolors(triangles: np.ndarray, color: str) -> np.ndarray:
    if triangles.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    base = rgb_from_hex(color)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    light_dir = np.array((-0.34, -0.46, 0.82), dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    view_dir = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    lambert = np.abs(normals @ light_dir)
    rim = np.power(1.0 - np.clip(np.abs(normals @ view_dir), 0.0, 1.0), 0.55)
    highlight = np.power(np.clip(normals @ light_dir, 0.0, 1.0), 10.0)
    shade = 0.50 + 0.36 * lambert
    rgb = base[None, :] * shade[:, None]
    rgb += (1.0 - base[None, :]) * (0.22 * rim[:, None] + 0.34 * highlight[:, None])
    return np.column_stack((np.clip(rgb, 0.0, 1.0), np.full(triangles.shape[0], 0.62))).astype(np.float32)


def glass_edgecolor(color: str) -> tuple[float, float, float, float]:
    base = rgb_from_hex(color)
    edge_rgb = np.clip(base * 0.20, 0.0, 1.0)
    return (float(edge_rgb[0]), float(edge_rgb[1]), float(edge_rgb[2]), 0.46)


def grid_box_points(grid: OrbitalGrid) -> np.ndarray:
    high = grid.origin + grid.spacing * (np.array(grid.shape, dtype=np.float64) - 1.0)
    corners = np.array(
        [
            [grid.origin[0], grid.origin[1], grid.origin[2]],
            [high[0], grid.origin[1], grid.origin[2]],
            [grid.origin[0], high[1], grid.origin[2]],
            [grid.origin[0], grid.origin[1], high[2]],
            [high[0], high[1], grid.origin[2]],
            [high[0], grid.origin[1], high[2]],
            [grid.origin[0], high[1], high[2]],
            [high[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )
    return corners * BOHR_TO_ANG


def mesh_faces(mesh: SurfaceMesh, face_limit: int = OPENGL_SURFACE_FACE_LIMIT) -> np.ndarray:
    if mesh.n_faces == 0:
        return np.empty((0, 3), dtype=np.uint32)
    stride = max(1, int(math.ceil(mesh.faces.shape[0] / face_limit)))
    return mesh.faces[::stride].astype(np.uint32, copy=False)


def mesh_item(
    mesh: SurfaceMesh,
    color: str,
    style: str = "glass",
    *,
    face_limit: int = OPENGL_SURFACE_FACE_LIMIT,
) -> gl.GLMeshItem:
    if style not in {"glass", "solid", "wireframe", "solid_edges"}:
        raise ValueError(f"Unknown surface style: {style}")
    faces = mesh_faces(mesh, face_limit)
    vertices = mesh.vertices.astype(np.float32, copy=False)
    if style != "glass":
        base = rgb_from_hex(color)
        edge_rgb = np.clip(base * 0.32, 0.0, 1.0)
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        if style == "wireframe":
            return gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=False,
                drawFaces=False,
                drawEdges=True,
                edgeColor=(float(base[0]), float(base[1]), float(base[2]), 0.94),
                computeNormals=False,
                glOptions="translucent",
            )
        return gl.GLMeshItem(
            meshdata=mesh_data,
            color=(float(base[0]), float(base[1]), float(base[2]), 0.96),
            smooth=True,
            drawFaces=True,
            drawEdges=style == "solid_edges",
            edgeColor=(float(edge_rgb[0]), float(edge_rgb[1]), float(edge_rgb[2]), 0.72),
            computeNormals=True,
            shader=FOG_SHADED_SHADER,
            glOptions="opaque",
        )

    triangles = vertices[faces] if faces.size else np.empty((0, 3, 3), dtype=np.float32)
    mesh_data = gl.MeshData(
        vertexes=vertices,
        faces=faces,
        faceColors=glass_facecolors(triangles, color),
    )
    return gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=False,
        drawFaces=True,
        drawEdges=False,
        edgeColor=glass_edgecolor(color),
        computeNormals=False,
        shader=FOG_FLAT_SHADER,
        glOptions="translucent",
    )


@dataclass(frozen=True)
class MoleculeGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.vertices.nbytes + self.faces.nbytes + self.vertex_colors.nbytes)


def atom_display_radius(
    atomic_number: int,
    style: str = "ball_stick",
    scale: float = 1.0,
) -> float:
    covalent = COVALENT_RADII.get(int(atomic_number), 0.75)
    if style == "ball_stick":
        radius = max(0.13, min(0.33, 0.34 * covalent))
    elif style == "space_filling":
        radius = max(0.22, min(0.90, 0.72 * covalent))
    elif style == "licorice":
        radius = 0.13
    else:
        raise ValueError(f"Unknown atom style: {style}")
    return radius * max(0.05, float(scale))


def _rgba_rows(rgb: tuple[float, float, float], count: int, alpha: float = 1.0) -> np.ndarray:
    rgba = np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)
    return np.tile(rgba, (count, 1))


def _rotation_from_z_axis(direction: np.ndarray) -> np.ndarray:
    z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    dot = float(np.clip(z_axis @ unit, -1.0, 1.0))
    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.999999:
        return np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64)
    cross = np.cross(z_axis, unit)
    skew = np.array(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    sin2 = float(cross @ cross)
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / sin2)


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-8:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    base = gl.MeshData.cylinder(rows=1, cols=MOLECULE_BOND_COLS, radius=[radius, radius], length=length)
    rotation = _rotation_from_z_axis(vector)
    vertices = base.vertexes().astype(np.float64, copy=False) @ rotation.T + start
    return vertices.astype(np.float32, copy=False), base.faces().astype(np.uint32, copy=False)


def build_molecule_geometry(
    wavefunction: Wavefunction,
    bonds: list[tuple[int, int]],
    style: str = "ball_stick",
    atom_scale: float = 1.0,
) -> MoleculeGeometry:
    if style not in {"ball_stick", "space_filling", "licorice"}:
        raise ValueError(f"Unknown atom style: {style}")
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_color_parts: list[np.ndarray] = []
    offset = 0

    coords = wavefunction.coordinates_angstrom.astype(np.float64, copy=False)
    sphere_rows = 14 if style == "space_filling" else MOLECULE_SPHERE_ROWS
    sphere_cols = 20 if style == "space_filling" else MOLECULE_SPHERE_COLS
    for atom_idx, atomic_number in enumerate(wavefunction.atomic_numbers):
        z = int(atomic_number)
        if z == 0:
            continue
        color = ELEMENT_COLORS.get(z, (0.55, 0.58, 0.64))
        sphere = gl.MeshData.sphere(
            rows=sphere_rows,
            cols=sphere_cols,
            radius=atom_display_radius(z, style, atom_scale),
        )
        vertices = sphere.vertexes().astype(np.float32, copy=True)
        vertices += coords[atom_idx].astype(np.float32, copy=False)
        faces = sphere.faces().astype(np.uint32, copy=False)
        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        vertex_color_parts.append(_rgba_rows(color, vertices.shape[0]))
        offset += vertices.shape[0]

    if style != "space_filling":
        bond_color = (0.42, 0.45, 0.50)
        bond_radius = 0.10 if style == "licorice" else MOLECULE_BOND_RADIUS
        for i, j in bonds:
            vertices, faces = cylinder_between(coords[i], coords[j], bond_radius)
            if vertices.size == 0:
                continue
            vertices_parts.append(vertices)
            faces_parts.append(faces + offset)
            vertex_color_parts.append(_rgba_rows(bond_color, vertices.shape[0], alpha=0.96))
            offset += vertices.shape[0]

    if not vertices_parts:
        return MoleculeGeometry(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
            np.empty((0, 4), dtype=np.float32),
        )

    return MoleculeGeometry(
        np.vstack(vertices_parts).astype(np.float32, copy=False),
        np.vstack(faces_parts).astype(np.uint32, copy=False),
        np.vstack(vertex_color_parts).astype(np.float32, copy=False),
    )


def molecule_mesh_item(
    wavefunction: Wavefunction,
    bonds: list[tuple[int, int]],
    style: str = "ball_stick",
    atom_scale: float = 1.0,
    geometry: MoleculeGeometry | None = None,
) -> gl.GLMeshItem:
    geometry = geometry or build_molecule_geometry(wavefunction, bonds, style, atom_scale)
    if geometry.vertices.size == 0:
        return gl.GLMeshItem()

    mesh_data = gl.MeshData(
        vertexes=geometry.vertices,
        faces=geometry.faces,
        vertexColors=geometry.vertex_colors,
    )
    return gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=True,
        drawFaces=True,
        drawEdges=False,
        shader=FOG_SHADED_SHADER,
        glOptions="opaque",
    )


@dataclass(frozen=True)
class LabelAtlasEntry:
    u_left: float
    v_top: float
    u_right: float
    v_bottom: float
    aspect_ratio: float


@lru_cache(maxsize=8)
def build_label_atlas(
    labels: tuple[str, ...],
) -> tuple[np.ndarray, tuple[LabelAtlasEntry, ...]]:
    if not labels:
        return np.empty((0, 0, 4), dtype=np.uint8), ()

    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(LABEL_ATLAS_FONT_PIXELS)
    font.setWeight(QtGui.QFont.Weight.DemiBold)
    padding = 7.0
    halo_width = 5.0
    paths: list[QtGui.QPainterPath] = []
    bounds: list[QtCore.QRectF] = []
    sizes: list[tuple[int, int]] = []
    for label in labels:
        path = QtGui.QPainterPath()
        path.addText(QtCore.QPointF(0.0, 0.0), font, str(label))
        bound = path.boundingRect()
        width = max(1, int(math.ceil(bound.width() + 2.0 * padding + halo_width)))
        height = max(1, int(math.ceil(bound.height() + 2.0 * padding + halo_width)))
        paths.append(path)
        bounds.append(bound)
        sizes.append((width, height))

    total_area = sum(width * height for width, height in sizes)
    target_width = max(
        max(width for width, _height in sizes),
        min(2048, max(256, int(math.ceil(math.sqrt(total_area) * 1.35)))),
    )
    placements: list[tuple[int, int, int, int]] = []
    x = 0
    y = 0
    row_height = 0
    used_width = 0
    for width, height in sizes:
        if x and x + width > target_width:
            x = 0
            y += row_height
            row_height = 0
        placements.append((x, y, width, height))
        x += width
        used_width = max(used_width, x)
        row_height = max(row_height, height)
    atlas_width = max(1, used_width)
    atlas_height = max(1, y + row_height)

    image = QtGui.QImage(
        atlas_width,
        atlas_height,
        QtGui.QImage.Format.Format_RGBA8888,
    )
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHints(
        QtGui.QPainter.RenderHint.Antialiasing
        | QtGui.QPainter.RenderHint.TextAntialiasing
    )
    halo_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 238), halo_width)
    halo_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    text_brush = QtGui.QBrush(QtGui.QColor("#111114"))
    entries: list[LabelAtlasEntry] = []
    for path, bound, (x, y, width, height) in zip(paths, bounds, placements):
        translated = QtGui.QPainterPath(path)
        translated.translate(
            x + 0.5 * (width - bound.width()) - bound.left(),
            y + 0.5 * (height - bound.height()) - bound.top(),
        )
        painter.strokePath(translated, halo_pen)
        painter.fillPath(translated, text_brush)
        entries.append(
            LabelAtlasEntry(
                u_left=x / atlas_width,
                v_top=y / atlas_height,
                u_right=(x + width) / atlas_width,
                v_bottom=(y + height) / atlas_height,
                aspect_ratio=width / height,
            )
        )
    painter.end()

    pointer = image.constBits()
    pointer.setsize(image.sizeInBytes())
    rows = np.frombuffer(pointer, dtype=np.uint8).reshape(atlas_height, image.bytesPerLine())
    atlas = rows[:, : atlas_width * 4].reshape(atlas_height, atlas_width, 4).copy()
    atlas = np.ascontiguousarray(atlas)
    atlas.setflags(write=False)
    return atlas, tuple(entries)


def attached_label_vertices(
    positions: np.ndarray,
    entries: Sequence[LabelAtlasEntry],
    depth_offsets: np.ndarray,
    font_size: int,
) -> np.ndarray:
    world_height = max(0.18, int(font_size) * LABEL_WORLD_HEIGHT_PER_POINT)
    rows: list[tuple[float, ...]] = []
    for center, entry, depth_offset in zip(positions, entries, depth_offsets):
        half_height = 0.5 * world_height
        half_width = half_height * entry.aspect_ratio
        left, right = -half_width, half_width
        bottom, top = -half_height, half_height
        u0, u1 = entry.u_left, entry.u_right
        v0, v1 = entry.v_top, entry.v_bottom
        corners = (
            (left, bottom, u0, v1),
            (right, bottom, u1, v1),
            (right, top, u1, v0),
            (left, bottom, u0, v1),
            (right, top, u1, v0),
            (left, top, u0, v0),
        )
        for corner_x, corner_y, u, v in corners:
            rows.append(
                (
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    corner_x,
                    corner_y,
                    u,
                    v,
                    float(depth_offset),
                )
            )
    return np.ascontiguousarray(rows, dtype=np.float32).reshape((-1, 8))


ATTACHED_LABEL_SHADER_LEGACY = {
    "vertex": """
        uniform mat4 u_modelview;
        uniform mat4 u_projection;
        attribute vec3 a_center;
        attribute vec2 a_corner;
        attribute vec2 a_uv;
        attribute float a_depth_offset;
        varying vec2 v_uv;

        void main() {
            vec4 center = u_modelview * vec4(a_center, 1.0);
            center.z += a_depth_offset;
            gl_Position = u_projection * (center + vec4(a_corner, 0.0, 0.0));
            v_uv = a_uv;
        }
    """,
    "fragment": """
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform sampler2D u_atlas;
        varying vec2 v_uv;

        void main() {
            vec4 color = texture2D(u_atlas, v_uv);
            if (color.a < 0.025) discard;
            gl_FragColor = color;
        }
    """,
}

ATTACHED_LABEL_SHADER_CORE = {
    "vertex": """
        uniform mat4 u_modelview;
        uniform mat4 u_projection;
        in vec3 a_center;
        in vec2 a_corner;
        in vec2 a_uv;
        in float a_depth_offset;
        out vec2 v_uv;

        void main() {
            vec4 center = u_modelview * vec4(a_center, 1.0);
            center.z += a_depth_offset;
            gl_Position = u_projection * (center + vec4(a_corner, 0.0, 0.0));
            v_uv = a_uv;
        }
    """,
    "fragment": """
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform sampler2D u_atlas;
        in vec2 v_uv;
        out vec4 fragColor;

        void main() {
            vec4 color = texture(u_atlas, v_uv);
            if (color.a < 0.025) discard;
            fragColor = color;
        }
    """,
}


class AttachedAtomLabelItem(GLGraphicsItem):
    """Render atom-attached labels as one depth-tested billboard batch."""

    _shader_programs: dict[tuple[bool, int, int], object] = {}

    def __init__(
        self,
        positions: np.ndarray,
        labels: Sequence[str],
        font_size: int,
        atom_radii: np.ndarray,
    ):
        super().__init__()
        positions = np.asarray(positions, dtype=np.float32)
        atom_radii = np.asarray(atom_radii, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Atom label positions must have shape (n, 3)")
        if positions.shape[0] != len(labels) or atom_radii.shape != (len(labels),):
            raise ValueError("Attached atom label data counts differ")
        self.positions = positions
        self.labels = tuple(str(label) for label in labels)
        self.font_size = max(6, int(font_size))
        self.atom_radii = atom_radii
        self.depth_offsets = atom_radii * 1.035 + 0.01
        self.atlas_rgba, entries = build_label_atlas(self.labels)
        self.vertex_data = attached_label_vertices(
            self.positions,
            entries,
            self.depth_offsets,
            self.font_size,
        )
        self.label_count = len(self.labels)
        self._vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self._vbo_dirty = True
        self._texture: int | None = None
        self.setGLOptions("translucent")
        self.setDepthValue(100)

    @classmethod
    def shader_program(cls):
        context = QtGui.QOpenGLContext.currentContext()
        if context is None:
            raise RuntimeError("Attached labels require a current OpenGL context")
        major, minor = context.format().version()
        key = (bool(context.isOpenGLES()), int(major), int(minor))
        cached = cls._shader_programs.get(key)
        if cached is not None:
            return cached
        if context.isOpenGLES():
            if (major, minor) >= (3, 0):
                version = "#version 300 es\n"
                sources = ATTACHED_LABEL_SHADER_CORE
            else:
                version = "#version 100\n"
                sources = ATTACHED_LABEL_SHADER_LEGACY
        elif (major, minor) >= (3, 1):
            version = "#version 140\n"
            sources = ATTACHED_LABEL_SHADER_CORE
        else:
            version = "#version 120\n"
            sources = ATTACHED_LABEL_SHADER_LEGACY
        program = gl_shaders.compileProgram(
            gl_shaders.compileShader([version, sources["vertex"]], GL.GL_VERTEX_SHADER),
            gl_shaders.compileShader([version, sources["fragment"]], GL.GL_FRAGMENT_SHADER),
        )
        for location, name in enumerate(
            ("a_center", "a_corner", "a_uv", "a_depth_offset")
        ):
            GL.glBindAttribLocation(program, location, name)
        GL.glLinkProgram(program)
        cls._shader_programs[key] = program
        return program

    def _upload_vbo(self) -> None:
        if not self._vbo.isCreated():
            self._vbo.create()
        self._vbo.bind()
        self._vbo.allocate(self.vertex_data, self.vertex_data.nbytes)
        self._vbo.release()
        self._vbo_dirty = False

    def _upload_texture(self) -> None:
        if self._texture is None:
            self._texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        height, width = self.atlas_rgba.shape[:2]
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            self.atlas_rgba,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def release_gl_resources(self) -> None:
        if self._texture is not None:
            GL.glDeleteTextures([self._texture])
            self._texture = None
        if self._vbo.isCreated():
            self._vbo.destroy()
        self._vbo_dirty = True

    def paint(self) -> None:
        if not self.vertex_data.size:
            return
        self.setupGLState()
        if self._vbo_dirty:
            self._upload_vbo()
        if self._texture is None:
            self._upload_texture()

        modelview = np.asarray(self.modelViewMatrix().data(), dtype=np.float32)
        projection = np.asarray(self.projectionMatrix().data(), dtype=np.float32)
        stride = 8 * np.dtype(np.float32).itemsize
        attributes = ((0, 3, 0), (1, 2, 3), (2, 2, 5), (3, 1, 7))
        self._vbo.bind()
        for location, size, offset_floats in attributes:
            GL.glVertexAttribPointer(
                location,
                size,
                GL.GL_FLOAT,
                False,
                stride,
                GL.GLvoidp(offset_floats * np.dtype(np.float32).itemsize),
            )
            GL.glEnableVertexAttribArray(location)
        self._vbo.release()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        program = self.shader_program()
        with program:
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(program, "u_modelview"),
                1,
                False,
                modelview,
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(program, "u_projection"),
                1,
                False,
                projection,
            )
            GL.glUniform1i(GL.glGetUniformLocation(program, "u_atlas"), 0)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_data.shape[0])

        for location, _size, _offset in attributes:
            GL.glDisableVertexAttribArray(location)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


class AtomLabelItem(GLGraphicsItem):
    """Draw all atom labels in one QPainter pass."""

    def __init__(
        self,
        positions: np.ndarray,
        labels: Sequence[str],
        font_size: int,
    ):
        super().__init__()
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Atom label positions must have shape (n, 3)")
        if positions.shape[0] != len(labels):
            raise ValueError("Atom label position and text counts differ")
        self.positions = positions
        self.labels = tuple(str(label) for label in labels)
        self.font = QtGui.QFont()
        self.font.setPointSize(max(6, int(font_size)))
        self.font.setWeight(QtGui.QFont.Weight.DemiBold)
        self.text_color = QtGui.QColor("#111114")
        self.halo_color = QtGui.QColor(255, 255, 255, 235)
        self.setGLOptions("additive")
        self.setDepthValue(100)

    def paint(self) -> None:
        view = self.view()
        if view is None or not self.labels:
            return
        self.setupGLState()
        rect = QtCore.QRectF(view.rect())
        ndc_to_viewport = QtGui.QMatrix4x4()
        ndc_to_viewport.viewport(rect.left(), rect.bottom(), rect.width(), -rect.height())
        projection = ndc_to_viewport * self.mvpMatrix()

        painter = QtGui.QPainter(view)
        painter.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.TextAntialiasing
        )
        painter.setFont(self.font)
        metrics = QtGui.QFontMetricsF(self.font)
        halo_pen = QtGui.QPen(self.halo_color, 3.0)
        halo_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        view_bounds = QtCore.QRectF(5.0, 5.0, max(1.0, view.width() - 10.0), max(1.0, view.height() - 10.0))
        directions = (
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (0.0, 1.0),
        )
        placed: list[QtCore.QRectF] = []
        layouts: list[tuple[QtCore.QPointF, QtCore.QRectF, str, float]] = []
        for position, label in zip(self.positions, self.labels):
            if not label:
                continue
            mapped = projection.map(
                QtGui.QVector3D(float(position[0]), float(position[1]), float(position[2]))
            )
            x, y = float(mapped.x()), float(mapped.y())
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if x < -80.0 or y < -80.0 or x > view.width() + 80.0 or y > view.height() + 80.0:
                continue
            width = metrics.horizontalAdvance(label)
            height = metrics.height()
            base_point = QtCore.QPointF(x, y)
            best_rect: QtCore.QRectF | None = None
            best_score = math.inf
            for radius in (height * 0.72, height * 1.35, height * 2.05, height * 2.80):
                for dx, dy in directions:
                    center = QtCore.QPointF(x + dx * radius, y + dy * radius)
                    candidate = QtCore.QRectF(
                        center.x() - width * 0.5 - 3.0,
                        center.y() - height * 0.5 - 2.0,
                        width + 6.0,
                        height + 4.0,
                    )
                    overlap = 0.0
                    for occupied in placed:
                        intersection = candidate.intersected(occupied)
                        if not intersection.isEmpty():
                            overlap += intersection.width() * intersection.height()
                    visible_rect = candidate.intersected(view_bounds)
                    outside = candidate.width() * candidate.height()
                    if not visible_rect.isEmpty():
                        outside -= visible_rect.width() * visible_rect.height()
                    score = overlap + 8.0 * max(0.0, outside)
                    if score < best_score:
                        best_rect = candidate
                        best_score = score
                    if score == 0.0:
                        break
                if best_score == 0.0:
                    break
            if best_rect is None:
                continue
            placed.append(best_rect.adjusted(-1.5, -1.5, 1.5, 1.5))
            layouts.append((base_point, best_rect, label, width))

        leader_pen = QtGui.QPen(QtGui.QColor(71, 84, 98, 120), 1.0)
        leader_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(leader_pen)
        for base_point, label_rect, _label, _width in layouts:
            center = label_rect.center()
            if math.hypot(center.x() - base_point.x(), center.y() - base_point.y()) > metrics.height():
                painter.drawLine(base_point, center)

        for _base_point, label_rect, label, width in layouts:
            center = label_rect.center()
            path = QtGui.QPainterPath()
            path.addText(
                QtCore.QPointF(
                    center.x() - width * 0.5,
                    center.y() + (metrics.ascent() - metrics.descent()) * 0.5,
                ),
                self.font,
                label,
            )
            painter.strokePath(path, halo_pen)
            painter.fillPath(path, self.text_color)
        painter.end()


def atom_label_item(
    wavefunction: Wavefunction,
    mode: str,
    font_size: int,
    *,
    placement: str = "attached",
    atom_style: str = "ball_stick",
    atom_scale: float = 1.0,
) -> AtomLabelItem | AttachedAtomLabelItem | None:
    if placement not in {"attached", "floating"}:
        raise ValueError(f"Unknown atom label placement: {placement}")
    labels = atom_label_texts(wavefunction.atomic_numbers, mode)
    visible = [index for index, label in enumerate(labels) if label]
    if not visible:
        return None
    positions = wavefunction.coordinates_angstrom[np.asarray(visible, dtype=np.int64)]
    visible_labels = [labels[index] for index in visible]
    if placement == "floating":
        return AtomLabelItem(positions, visible_labels, font_size)
    radii = np.asarray(
        [
            atom_display_radius(int(wavefunction.atomic_numbers[index]), atom_style, atom_scale)
            for index in visible
        ],
        dtype=np.float32,
    )
    return AttachedAtomLabelItem(positions, visible_labels, font_size, radii)


def remove_gl_item(view, item) -> None:
    release = getattr(item, "release_gl_resources", None)
    if callable(release):
        try:
            context = view.context()
            if context is not None and context.isValid():
                view.makeCurrent()
                try:
                    release()
                finally:
                    view.doneCurrent()
        except Exception:
            pass
    view.removeItem(item)


class OrbitalGLView(gl.GLViewWidget):
    def __init__(self, owner: "OpenGLViewer", slot: "SceneSlot | None" = None):
        super().__init__()
        self.owner = owner
        self.slot = slot
        self.setBackgroundColor(SCENE_BACKGROUND_HEX)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCameraParams(distance=12.0, elevation=90.0, azimuth=-90.0, fov=18.0)
        self._dragging = False
        self._drag_mode = "rotate"
        self._last_pos: QtCore.QPointF | None = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.owner.set_active_slot(self.slot)
        if self.owner.center_pick_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.owner.pick_rotation_center(self.slot, event.position())
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_mode = "rotate"
        elif event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            self._drag_mode = "roll"
        else:
            event.ignore()
            return
        self._dragging = True
        self._last_pos = event.position()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._dragging or self._last_pos is None:
            event.ignore()
            return
        pos = event.position()
        dx = float(pos.x() - self._last_pos.x())
        dy = float(pos.y() - self._last_pos.y())
        self._last_pos = pos
        self.owner.apply_mouse_rotation(self.slot, dx, dy, self._drag_mode)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._dragging = False
        self._last_pos = None
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.owner.zoom_view(1.12 if delta > 0 else 1.0 / 1.12, slot=self.slot)
        event.accept()
