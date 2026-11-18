"""OpenGL scene items: shaded meshes (surfaces, molecule) and atom labels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .qt import GL, GLGraphicsItem, QtCore, QtGui, QtOpenGL, gl_shaders

# --------------------------------------------------------------------- shaders
_MESH_VERTEX = """
uniform mat4 u_mvp;
uniform mat4 u_modelview;
uniform mat3 u_normal;
ATTR vec3 a_position;
ATTR vec3 a_normal;
ATTR vec4 a_color;
VARY_OUT vec3 v_normal;
VARY_OUT vec3 v_eye;
VARY_OUT vec4 v_color;
void main() {
    vec4 eye = u_modelview * vec4(a_position, 1.0);
    v_eye = eye.xyz;
    v_normal = u_normal * a_normal;
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""

_MESH_FRAGMENT = """
uniform vec4 u_material;   // ambient, diffuse, specular, shininess
uniform vec4 u_fog;        // rgb, strength
uniform vec2 u_fog_range;  // start, end (eye distance)
uniform float u_glass;     // 0 = opaque, otherwise base opacity
uniform float u_lit;       // 0 = flat color (lines)
VARY_IN vec3 v_normal;
VARY_IN vec3 v_eye;
VARY_IN vec4 v_color;
FRAG_DECL
void main() {
    vec3 base = v_color.rgb;
    vec3 color = base;
    float alpha = v_color.a;
    vec3 view = normalize(-v_eye);
    if (u_lit > 0.5) {
        vec3 n = normalize(v_normal);
        if (dot(n, view) < 0.0) n = -n;  // two-sided lighting
        vec3 key = normalize(vec3(-0.40, 0.55, 0.73));
        vec3 fill = normalize(vec3(0.65, -0.30, 0.45));
        float diffuse = max(dot(n, key), 0.0) + 0.30 * max(dot(n, fill), 0.0);
        float spec = pow(max(dot(n, normalize(key + view)), 0.0), u_material.w);
        color = base * (u_material.x + u_material.y * diffuse) + vec3(u_material.z * spec);
        if (u_glass > 0.0) {
            float rim = pow(1.0 - max(dot(n, view), 0.0), 2.2);
            alpha = clamp(u_glass + (1.0 - u_glass) * 0.85 * rim, 0.0, 1.0);
            color = mix(color, vec3(1.0), 0.18 * rim);
        }
    }
    float fog = clamp((-v_eye.z - u_fog_range.x) / max(u_fog_range.y - u_fog_range.x, 1.0e-3), 0.0, 1.0) * u_fog.a;
    FRAG_COLOR = vec4(mix(clamp(color, 0.0, 1.0), u_fog.rgb, fog), alpha);
}
"""

_LABEL_VERTEX = """
uniform mat4 u_modelview;
uniform mat4 u_projection;
ATTR vec3 a_center;
ATTR vec2 a_corner;
ATTR vec2 a_uv;
ATTR float a_depth_offset;
VARY_OUT vec2 v_uv;
void main() {
    vec4 center = u_modelview * vec4(a_center, 1.0);
    center.z += a_depth_offset;
    gl_Position = u_projection * (center + vec4(a_corner, 0.0, 0.0));
    v_uv = a_uv;
}
"""

_LABEL_FRAGMENT = """
uniform sampler2D u_atlas;
VARY_IN vec2 v_uv;
FRAG_DECL
void main() {
    vec4 color = TEXTURE(u_atlas, v_uv);
    if (color.a < 0.025) discard;
    FRAG_COLOR = color;
}
"""


def _glsl(source: str, stage: str, modern: bool, es: bool) -> str:
    if modern:
        header = "#version 300 es\nprecision mediump float;\n" if es else "#version 140\n"
        replacements = {
            "ATTR": "in",
            "VARY_OUT": "out",
            "VARY_IN": "in",
            "FRAG_DECL": "out vec4 fragColor;",
            "FRAG_COLOR": "fragColor",
            "TEXTURE": "texture",
        }
    else:
        header = "#version 100\nprecision mediump float;\n" if es else "#version 120\n"
        replacements = {
            "ATTR": "attribute",
            "VARY_OUT": "varying",
            "VARY_IN": "varying",
            "FRAG_DECL": "",
            "FRAG_COLOR": "gl_FragColor",
            "TEXTURE": "texture2D",
        }
    for token, value in replacements.items():
        source = source.replace(token, value)
    return header + source


_PROGRAMS: dict[tuple, int] = {}


def _program(name: str, vertex: str, fragment: str, attributes: Sequence[str]) -> int:
    context = QtGui.QOpenGLContext.currentContext()
    if context is None:
        raise RuntimeError("No current OpenGL context")
    major, minor = context.format().version()
    es = bool(context.isOpenGLES())
    modern = (major, minor) >= ((3, 0) if es else (3, 1))
    # All MOview GL widgets share one context group (AA_ShareOpenGLContexts).
    key = (name, es, modern)
    program = _PROGRAMS.get(key)
    if program is not None:
        return program
    program = gl_shaders.compileProgram(
        gl_shaders.compileShader(_glsl(vertex, "vertex", modern, es), GL.GL_VERTEX_SHADER),
        gl_shaders.compileShader(_glsl(fragment, "fragment", modern, es), GL.GL_FRAGMENT_SHADER),
        validate=False,
    )
    for location, attribute in enumerate(attributes):
        GL.glBindAttribLocation(program, location, attribute)
    GL.glLinkProgram(program)
    _PROGRAMS[key] = program
    return program


def _blend_keep_alpha() -> None:
    """Alpha blending that leaves the framebuffer opaque.

    Qt composites a QOpenGLWidget as premultiplied alpha; writing alpha < 1
    would make colors overflow (red glass turns cyan on light backgrounds).
    """
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE)


def _uniform_location(program: int, name: str) -> int:
    return GL.glGetUniformLocation(program, name)


def _matrix(m) -> np.ndarray:
    return np.asarray(m.data(), dtype=np.float32)


# ------------------------------------------------------------------ materials
@dataclass(frozen=True)
class Material:
    ambient: float = 0.34
    diffuse: float = 0.66
    specular: float = 0.30
    shininess: float = 36.0
    glass: float = 0.0  # base opacity for glass, 0 = opaque


SOLID = Material()
GLASS = Material(ambient=0.40, diffuse=0.62, specular=0.38, shininess=48.0, glass=0.42)
ATOMS = Material(ambient=0.30, diffuse=0.70, specular=0.42, shininess=56.0)


class ShadedMeshItem(GLGraphicsItem):
    """Indexed triangle mesh drawn with MOview's lighting shader.

    ``mode`` is ``solid``, ``glass``, ``wireframe`` or ``solid_edges``.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        faces: np.ndarray,
        *,
        color: tuple[float, float, float, float] | None = None,
        colors: np.ndarray | None = None,
        mode: str = "solid",
        material: Material = SOLID,
        edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5),
    ):
        super().__init__()
        if mode not in ("solid", "glass", "wireframe", "solid_edges"):
            raise ValueError(f"Unknown surface style: {mode}")
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self.colors = None if colors is None else np.ascontiguousarray(colors, dtype=np.float32)
        self.color = color or (0.7, 0.7, 0.7, 1.0)
        self.mode = mode
        self.material = material
        self.edge_color = edge_color
        self.edges = self._unique_edges(self.faces) if mode in ("wireframe", "solid_edges") else None
        self._buffers: dict[str, QtOpenGL.QOpenGLBuffer] = {}
        self.setDepthValue(10 if mode == "glass" else 0)

    @staticmethod
    def _unique_edges(faces: np.ndarray) -> np.ndarray:
        if faces.size == 0:
            return np.empty((0, 2), dtype=np.uint32)
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        return np.ascontiguousarray(np.unique(np.sort(edges, axis=1), axis=0), dtype=np.uint32)

    # -- GL resources ----------------------------------------------------
    def _buffer(self, name: str, data: np.ndarray, kind) -> QtOpenGL.QOpenGLBuffer:
        buffer = self._buffers.get(name)
        if buffer is None:
            buffer = QtOpenGL.QOpenGLBuffer(kind)
            buffer.create()
            buffer.bind()
            buffer.allocate(data, data.nbytes)
            buffer.release()
            self._buffers[name] = buffer
        return buffer

    def release_gl_resources(self) -> None:
        for buffer in self._buffers.values():
            if buffer.isCreated():
                buffer.destroy()
        self._buffers.clear()

    # -- painting --------------------------------------------------------
    def _fog_uniforms(self) -> tuple[list[float], list[float]]:
        view = self.view()
        fog = getattr(view, "fog", None)
        if fog is None:
            return [0.0, 0.0, 0.0, 0.0], [1.0, 2.0]
        return list(fog[0]), list(fog[1])

    def paint(self) -> None:
        if self.vertices.shape[0] == 0 or self.faces.size == 0:
            return
        vbuf = QtOpenGL.QOpenGLBuffer.Type.VertexBuffer
        ibuf = QtOpenGL.QOpenGLBuffer.Type.IndexBuffer
        positions = self._buffer("position", self.vertices, vbuf)
        normals = self._buffer("normal", self.normals, vbuf)
        colors = self._buffer("color", self.colors, vbuf) if self.colors is not None else None
        program = _program("mesh", _MESH_VERTEX, _MESH_FRAGMENT, ("a_position", "a_normal", "a_color"))

        GL.glUseProgram(program)
        GL.glUniformMatrix4fv(_uniform_location(program, "u_mvp"), 1, False, _matrix(self.mvpMatrix()))
        GL.glUniformMatrix4fv(_uniform_location(program, "u_modelview"), 1, False, _matrix(self.modelViewMatrix()))
        GL.glUniformMatrix3fv(_uniform_location(program, "u_normal"), 1, False, _matrix(self.modelViewMatrix().normalMatrix()))
        m = self.material
        GL.glUniform4f(_uniform_location(program, "u_material"), m.ambient, m.diffuse, m.specular, m.shininess)
        fog, fog_range = self._fog_uniforms()
        GL.glUniform4f(_uniform_location(program, "u_fog"), *fog)
        GL.glUniform2f(_uniform_location(program, "u_fog_range"), *fog_range)

        for location, buffer in ((0, positions), (1, normals)):
            buffer.bind()
            GL.glVertexAttribPointer(location, 3, GL.GL_FLOAT, False, 0, None)
            buffer.release()
            GL.glEnableVertexAttribArray(location)
        if colors is not None:
            colors.bind()
            GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, False, 0, None)
            colors.release()
            GL.glEnableVertexAttribArray(2)
        else:
            GL.glDisableVertexAttribArray(2)
            GL.glVertexAttrib4f(2, *self.color)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        try:
            if self.mode in ("solid", "solid_edges"):
                self._draw_solid(program)
            elif self.mode == "glass":
                self._draw_glass(program)
            if self.mode in ("wireframe", "solid_edges"):
                self._draw_edges(program, colors is not None)
        finally:
            GL.glDisableVertexAttribArray(0)
            GL.glDisableVertexAttribArray(1)
            GL.glDisableVertexAttribArray(2)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glUseProgram(0)

    def _draw_triangles(self) -> None:
        index = self._buffer("faces", self.faces, QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)
        index.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, self.faces.size, GL.GL_UNSIGNED_INT, None)
        index.release()

    def _draw_solid(self, program: int) -> None:
        GL.glUniform1f(_uniform_location(program, "u_glass"), 0.0)
        GL.glUniform1f(_uniform_location(program, "u_lit"), 1.0)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        if self.mode == "solid_edges":
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0, 1.0)
        self._draw_triangles()

    def _draw_glass(self, program: int) -> None:
        GL.glUniform1f(_uniform_location(program, "u_glass"), float(self.material.glass))
        GL.glUniform1f(_uniform_location(program, "u_lit"), 1.0)
        _blend_keep_alpha()
        GL.glDepthMask(GL.GL_FALSE)  # glass never hides what is behind it
        GL.glEnable(GL.GL_CULL_FACE)
        for culled in (GL.GL_FRONT, GL.GL_BACK):  # far side first, then near side
            GL.glCullFace(culled)
            self._draw_triangles()

    def _draw_edges(self, program: int, has_color_buffer: bool) -> None:
        if self.edges is None or self.edges.size == 0:
            return
        GL.glUniform1f(_uniform_location(program, "u_glass"), 0.0)
        GL.glUniform1f(_uniform_location(program, "u_lit"), 0.0)
        _blend_keep_alpha()
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisableVertexAttribArray(2)
        GL.glVertexAttrib4f(2, *self.edge_color)
        index = self._buffer("edges", self.edges, QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)
        index.bind()
        GL.glDrawElements(GL.GL_LINES, self.edges.size, GL.GL_UNSIGNED_INT, None)
        index.release()


# ---------------------------------------------------------------------- labels
@dataclass(frozen=True)
class _AtlasEntry:
    u_left: float
    v_top: float
    u_right: float
    v_bottom: float
    aspect: float


ATLAS_FONT_PIXELS = 64
LABEL_HEIGHT_PER_POINT = 0.030  # angstrom per font point for attached labels


@lru_cache(maxsize=8)
def build_label_atlas(labels: tuple[str, ...], text_color: str, halo_color: str) -> tuple[np.ndarray, tuple[_AtlasEntry, ...]]:
    """Rasterize all labels into one RGBA texture (one draw call for every label)."""
    if not labels:
        return np.zeros((1, 1, 4), dtype=np.uint8), ()
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(ATLAS_FONT_PIXELS)
    font.setWeight(QtGui.QFont.Weight.DemiBold)
    padding, halo = 7.0, 9.0
    paths, bounds, sizes = [], [], []
    for label in labels:
        path = QtGui.QPainterPath()
        path.addText(QtCore.QPointF(0.0, 0.0), font, label)
        rect = path.boundingRect()
        paths.append(path)
        bounds.append(rect)
        sizes.append((max(1, math.ceil(rect.width() + 2 * padding + halo)), max(1, math.ceil(rect.height() + 2 * padding + halo))))
    target_width = max(max(w for w, _ in sizes), min(4096, max(256, math.ceil(math.sqrt(sum(w * h for w, h in sizes)) * 1.3))))
    placements, x, y, row_height, used_width = [], 0, 0, 0, 0
    for width, height in sizes:
        if x and x + width > target_width:
            x, y, row_height = 0, y + row_height, 0
        placements.append((x, y, width, height))
        x += width
        used_width = max(used_width, x)
        row_height = max(row_height, height)
    atlas_w, atlas_h = max(1, used_width), max(1, y + row_height)
    image = QtGui.QImage(atlas_w, atlas_h, QtGui.QImage.Format.Format_RGBA8888)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
    halo_qcolor = QtGui.QColor(halo_color)
    halo_qcolor.setAlpha(235)
    halo_pen = QtGui.QPen(halo_qcolor, halo)
    halo_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    entries = []
    for path, rect, (x, y, width, height) in zip(paths, bounds, placements):
        moved = QtGui.QPainterPath(path)
        moved.translate(x + 0.5 * (width - rect.width()) - rect.left(), y + 0.5 * (height - rect.height()) - rect.top())
        painter.strokePath(moved, halo_pen)
        painter.fillPath(moved, QtGui.QBrush(QtGui.QColor(text_color)))
        entries.append(_AtlasEntry(x / atlas_w, y / atlas_h, (x + width) / atlas_w, (y + height) / atlas_h, width / height))
    painter.end()
    pointer = image.constBits()
    pointer.setsize(image.sizeInBytes())
    rows = np.frombuffer(pointer, dtype=np.uint8).reshape(atlas_h, image.bytesPerLine())
    atlas = np.ascontiguousarray(rows[:, : atlas_w * 4].reshape(atlas_h, atlas_w, 4))
    atlas.setflags(write=False)
    return atlas, tuple(entries)


class AttachedLabelItem(GLGraphicsItem):
    """Depth-tested billboards placed just in front of each atom (single draw call)."""

    def __init__(self, positions: np.ndarray, labels: Sequence[str], font_size: int, radii: np.ndarray, text_color: str, halo_color: str):
        super().__init__()
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self.labels = tuple(labels)
        self.atlas, entries = build_label_atlas(self.labels, text_color, halo_color)
        height = max(0.18, int(font_size) * LABEL_HEIGHT_PER_POINT)
        depth = np.asarray(radii, dtype=np.float32) * 1.04 + 0.02
        rows = []
        for center, entry, offset in zip(positions, entries, depth):
            hh = 0.5 * height
            hw = hh * entry.aspect
            corners = (
                (-hw, -hh, entry.u_left, entry.v_bottom),
                (hw, -hh, entry.u_right, entry.v_bottom),
                (hw, hh, entry.u_right, entry.v_top),
                (-hw, -hh, entry.u_left, entry.v_bottom),
                (hw, hh, entry.u_right, entry.v_top),
                (-hw, hh, entry.u_left, entry.v_top),
            )
            for cx, cy, u, v in corners:
                rows.append((*center, cx, cy, u, v, offset))
        self.vertex_data = np.ascontiguousarray(rows, dtype=np.float32).reshape(-1, 8)
        self._vbo: QtOpenGL.QOpenGLBuffer | None = None
        self._texture: int | None = None
        self.setDepthValue(100)

    def release_gl_resources(self) -> None:
        if self._texture is not None:
            GL.glDeleteTextures([self._texture])
            self._texture = None
        if self._vbo is not None and self._vbo.isCreated():
            self._vbo.destroy()
        self._vbo = None

    def paint(self) -> None:
        if not self.vertex_data.size:
            return
        if self._vbo is None:
            self._vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
            self._vbo.create()
            self._vbo.bind()
            self._vbo.allocate(self.vertex_data, self.vertex_data.nbytes)
            self._vbo.release()
        if self._texture is None:
            self._texture = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
            for parameter, value in (
                (GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR),
                (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
            ):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, parameter, value)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            height, width = self.atlas.shape[:2]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, self.atlas)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        program = _program("label", _LABEL_VERTEX, _LABEL_FRAGMENT, ("a_center", "a_corner", "a_uv", "a_depth_offset"))
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        _blend_keep_alpha()
        stride = 8 * 4
        self._vbo.bind()
        for location, size, offset in ((0, 3, 0), (1, 2, 3), (2, 2, 5), (3, 1, 7)):
            GL.glVertexAttribPointer(location, size, GL.GL_FLOAT, False, stride, GL.GLvoidp(offset * 4))
            GL.glEnableVertexAttribArray(location)
        self._vbo.release()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glUseProgram(program)
        GL.glUniformMatrix4fv(_uniform_location(program, "u_modelview"), 1, False, _matrix(self.modelViewMatrix()))
        GL.glUniformMatrix4fv(_uniform_location(program, "u_projection"), 1, False, _matrix(self.projectionMatrix()))
        GL.glUniform1i(_uniform_location(program, "u_atlas"), 0)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_data.shape[0])
        GL.glUseProgram(0)
        for location in range(4):
            GL.glDisableVertexAttribArray(location)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_BLEND)


class FloatingLabelItem(GLGraphicsItem):
    """Screen-space labels with simple collision avoidance and leader lines."""

    _DIRECTIONS = ((0.0, -1.0), (1.0, -1.0), (-1.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))

    def __init__(self, positions: np.ndarray, labels: Sequence[str], font_size: int, text_color: str, halo_color: str):
        super().__init__()
        self.positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self.labels = tuple(labels)
        self.font = QtGui.QFont()
        self.font.setPointSize(max(6, int(font_size)))
        self.font.setWeight(QtGui.QFont.Weight.DemiBold)
        self.text_color = QtGui.QColor(text_color)
        self.halo_color = QtGui.QColor(halo_color)
        self.halo_color.setAlpha(235)
        self.leader_color = QtGui.QColor(text_color)
        self.leader_color.setAlpha(110)
        self.setDepthValue(100)

    def paint(self) -> None:
        view = self.view()
        if view is None or not self.labels:
            return
        width, height = float(view.width()), float(view.height())
        to_screen = QtGui.QMatrix4x4()
        to_screen.viewport(0.0, height, width, -height)
        projection = to_screen * self.mvpMatrix()
        painter = QtGui.QPainter(view)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        metrics = QtGui.QFontMetricsF(self.font)
        bounds = QtCore.QRectF(4.0, 4.0, max(1.0, width - 8.0), max(1.0, height - 8.0))
        placed: list[QtCore.QRectF] = []
        layouts = []
        for position, label in zip(self.positions, self.labels):
            if not label:
                continue
            mapped = projection.map(QtGui.QVector3D(*map(float, position)))
            x, y = float(mapped.x()), float(mapped.y())
            if not (math.isfinite(x) and math.isfinite(y)) or not (-80 < x < width + 80 and -80 < y < height + 80):
                continue
            text_w, text_h = metrics.horizontalAdvance(label), metrics.height()
            best, best_score = None, math.inf
            for radius in (0.72 * text_h, 1.35 * text_h, 2.05 * text_h):
                for dx, dy in self._DIRECTIONS:
                    rect = QtCore.QRectF(x + dx * radius - 0.5 * text_w - 3, y + dy * radius - 0.5 * text_h - 2, text_w + 6, text_h + 4)
                    score = sum(
                        r.width() * r.height() for r in (rect.intersected(other) for other in placed) if not r.isEmpty()
                    )
                    visible = rect.intersected(bounds)
                    outside = rect.width() * rect.height() - (0.0 if visible.isEmpty() else visible.width() * visible.height())
                    score += 8.0 * outside
                    if score < best_score:
                        best, best_score = rect, score
                    if score == 0.0:
                        break
                if best_score == 0.0:
                    break
            if best is not None:
                placed.append(best.adjusted(-1.5, -1.5, 1.5, 1.5))
                layouts.append((QtCore.QPointF(x, y), best, label, text_w))
        painter.setPen(QtGui.QPen(self.leader_color, 1.0))
        for anchor, rect, _label, _w in layouts:
            if math.hypot(rect.center().x() - anchor.x(), rect.center().y() - anchor.y()) > metrics.height():
                painter.drawLine(anchor, rect.center())
        halo_pen = QtGui.QPen(self.halo_color, 3.0)
        halo_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        for _anchor, rect, label, text_w in layouts:
            path = QtGui.QPainterPath()
            path.addText(
                QtCore.QPointF(rect.center().x() - 0.5 * text_w, rect.center().y() + 0.5 * (metrics.ascent() - metrics.descent())),
                self.font,
                label,
            )
            painter.strokePath(path, halo_pen)
            painter.fillPath(path, self.text_color)
        painter.end()
