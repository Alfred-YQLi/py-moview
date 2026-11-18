"""3D viewport widgets: the OpenGL view and the framed cell around it."""

from __future__ import annotations

import math

import numpy as np

from .qt import GL, QtCore, QtGui, QtWidgets, Signal, create_opengl_surface_format, gl, pg
from .theme import Theme

FIELD_OF_VIEW = 20.0
ROTATE_DEG_PER_PIXEL = 0.4
MIN_ZOOM, MAX_ZOOM = 0.25, 6.0


def axis_rotation(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "x":
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)
    if axis == "y":
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)
    if axis == "z":
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)
    raise ValueError(f"Unknown rotation axis: {axis}")


def default_rotation() -> np.ndarray:
    return axis_rotation("x", -24.0) @ axis_rotation("y", 34.0)


class AxesWidget(QtWidgets.QWidget):
    """Orientation triad in the lower-left corner of a view.

    A transparent child widget painted with an ordinary QPainter, so it is
    composited correctly on screen, in window grabs and in saved images.
    """

    SIZE = 96

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.rotation: np.ndarray | None = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.resize(self.SIZE, self.SIZE)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if self.rotation is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        length = 28.0
        origin = QtCore.QPointF(self.width() * 0.5, self.height() * 0.5)
        font = QtGui.QFont()
        font.setPointSizeF(9.0)
        font.setBold(True)
        painter.setFont(font)
        rotation = self.rotation
        for i in sorted(range(3), key=lambda k: rotation[2, k]):  # far axes first
            vector = rotation[:, i]
            end = origin + QtCore.QPointF(vector[0] * length, -vector[1] * length)
            color = QtGui.QColor(("#e5484d", "#30a46c", "#3e63dd")[i])
            if vector[2] < -0.3:
                color.setAlphaF(0.55)  # pointing away from the viewer
            painter.setPen(QtGui.QPen(color, 2.2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
            painter.drawLine(origin, end)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(end, 2.6, 2.6)
            label = origin + QtCore.QPointF(vector[0] * (length + 11), -vector[1] * (length + 11))
            painter.setPen(color)
            painter.drawText(QtCore.QRectF(label.x() - 8, label.y() - 8, 16, 16), QtCore.Qt.AlignmentFlag.AlignCenter, "XYZ"[i])
        painter.end()


class OrbitalView(gl.GLViewWidget):
    """GL widget with VMD-like mouse controls; emits intents, owns no scene state.

    The camera never rotates (x right, y up, z towards the viewer); the scene
    rotation is applied to the items, which keeps lighting fixed to the viewer.
    """

    activated = Signal()
    rotate_requested = Signal(float, float, str)  # dx, dy (pixels), "rotate" | "roll"
    pan_requested = Signal(float, float)  # dx, dy (pixels)
    zoom_requested = Signal(float)  # multiplicative factor
    pick_requested = Signal(QtCore.QPointF)

    def __init__(self, theme: Theme):
        super().__init__()
        self.setFormat(create_opengl_surface_format())
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(160, 120)
        self.setCameraParams(distance=12.0, elevation=90.0, azimuth=-90.0, fov=FIELD_OF_VIEW)
        self.fog: tuple[tuple[float, float, float, float], tuple[float, float]] | None = None
        self.axes = AxesWidget(self)
        self.pick_mode = False
        self._drag: str | None = None
        self._last: QtCore.QPointF | None = None
        self.theme = theme
        self.set_theme(theme)

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.setBackgroundColor(theme.viewport)
        self.update()

    @property
    def show_axes(self) -> bool:
        return not self.axes.isHidden()

    @show_axes.setter
    def show_axes(self, visible: bool) -> None:
        self.axes.setVisible(bool(visible))

    def set_axes_rotation(self, rotation: np.ndarray) -> None:
        self.axes.rotation = rotation
        self.axes.update()

    # ------------------------------------------------------------ painting
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.axes.move(6, self.height() - self.axes.height() - 6)

    def paintGL(self) -> None:  # noqa: N802 - Qt override
        GL.glDepthMask(GL.GL_TRUE)  # glClear only clears depth when writes are enabled
        super().paintGL()

    # ---------------------------------------------------------- projection
    def project(self, points: np.ndarray, model: np.ndarray) -> np.ndarray:
        """Screen coordinates (logical pixels) of model-space points."""
        width, height = max(1, self.width()), max(1, self.height())
        region = (0, 0, width, height)
        matrix = self.projectionMatrix(region, region) * self.viewMatrix()
        mvp = np.array(matrix.data(), dtype=np.float64).reshape(4, 4).T @ model
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ mvp.T
        ndc = homogeneous[:, :2] / homogeneous[:, 3:4]
        return np.column_stack(((ndc[:, 0] + 1.0) * 0.5 * width, (1.0 - ndc[:, 1]) * 0.5 * height))

    # -------------------------------------------------------------- input
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.activated.emit()
        self.setFocus()
        button = event.button()
        if self.pick_mode and button == QtCore.Qt.MouseButton.LeftButton:
            self.pick_requested.emit(event.position())
            event.accept()
            return
        shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if button == QtCore.Qt.MouseButton.LeftButton:
            self._drag = "pan" if shift else "rotate"
        elif button == QtCore.Qt.MouseButton.MiddleButton:
            self._drag = "pan"
        elif button == QtCore.Qt.MouseButton.RightButton:
            self._drag = "roll"
        else:
            event.ignore()
            return
        self._last = event.position()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._drag is None or self._last is None:
            event.ignore()
            return
        position = event.position()
        dx, dy = position.x() - self._last.x(), position.y() - self._last.y()
        self._last = position
        if self._drag == "pan":
            self.pan_requested.emit(dx, dy)
        else:
            self.rotate_requested.emit(dx, dy, self._drag)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drag = None
        self._last = None
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.zoom_requested.emit(math.pow(1.0015, delta))
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        step = 20.0
        if key == QtCore.Qt.Key.Key_Left:
            self.rotate_requested.emit(-step, 0.0, "rotate")
        elif key == QtCore.Qt.Key.Key_Right:
            self.rotate_requested.emit(step, 0.0, "rotate")
        elif key == QtCore.Qt.Key.Key_Up:
            self.rotate_requested.emit(0.0, -step, "rotate")
        elif key == QtCore.Qt.Key.Key_Down:
            self.rotate_requested.emit(0.0, step, "rotate")
        elif key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self.zoom_requested.emit(1.15)
        elif key == QtCore.Qt.Key.Key_Minus:
            self.zoom_requested.emit(1.0 / 1.15)
        else:
            event.ignore()
            return
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        event.ignore()

    def remove_item(self, item) -> None:
        """Free GPU buffers of ``item`` (needs a current context) and drop it."""
        release = getattr(item, "release_gl_resources", None)
        if callable(release) and self.context() is not None and self.context().isValid():
            self.makeCurrent()
            try:
                release()
            finally:
                self.doneCurrent()
        if item in self.items:
            self.removeItem(item)


class Viewport(QtWidgets.QFrame):
    """One comparison cell: header line + GL view, with its own camera state."""

    def __init__(self, index: int, theme: Theme):
        super().__init__()
        self.index = index
        self.setObjectName("viewportFrame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        header = QtWidgets.QWidget()
        header.setObjectName("viewportHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 7, 12, 5)
        header_layout.setSpacing(10)
        self.title = QtWidgets.QLabel("")
        self.title.setObjectName("viewTitle")
        self.title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.meta = QtWidgets.QLabel("")
        self.meta.setObjectName("viewMeta")
        self.meta.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.title, 1)
        header_layout.addWidget(self.meta)
        self.view = OrbitalView(theme)
        layout.addWidget(header)
        layout.addWidget(self.view, 1)
        self.reset_camera_state()
        self.orbital: int | None = None
        self.result = None  # RenderResult shown here
        self.surface_items: list = []
        self.molecule_items: list = []

    def reset_camera_state(self) -> None:
        self.rotation = default_rotation()
        self.zoom = 1.0
        self.pivot: np.ndarray | None = None  # rotation centre (angstrom)
        self.pan = np.zeros(3)
        self.home_pivot: np.ndarray | None = None
        self.radius = 5.0

    def set_active(self, active: bool) -> None:
        self.setProperty("active", bool(active))
        self.style().unpolish(self)
        self.style().polish(self)

    # -------------------------------------------------------------- camera
    def model_matrix(self) -> np.ndarray:
        pivot = self.pivot if self.pivot is not None else np.zeros(3)
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = pivot - self.rotation @ pivot + self.pan
        return matrix

    def apply(self, fog_color: tuple[float, float, float], fog_strength: float) -> None:
        matrix = self.model_matrix()
        transform = pg.Transform3D(*matrix.ravel().tolist())
        for item in self.molecule_items + self.surface_items:
            item.setTransform(transform)
        center = self.pivot if self.pivot is not None else np.zeros(3)
        radius = max(0.5, self.radius / max(self.zoom, 1.0e-3))
        distance = max(2.0, radius / math.tan(math.radians(FIELD_OF_VIEW) * 0.5))
        self.view.setCameraParams(
            center=QtGui.QVector3D(*map(float, center)), distance=distance, elevation=90.0, azimuth=-90.0, fov=FIELD_OF_VIEW
        )
        scene = self.radius
        self.view.fog = ((*fog_color, fog_strength), (distance - 0.25 * scene, distance + 1.25 * scene))
        self.view.set_axes_rotation(self.rotation)
        self.view.update()

    def rotate(self, dx: float, dy: float, mode: str) -> None:
        if mode == "roll":
            self.rotation = axis_rotation("z", -dx * ROTATE_DEG_PER_PIXEL) @ self.rotation
        else:
            self.rotation = axis_rotation("y", dx * ROTATE_DEG_PER_PIXEL) @ axis_rotation("x", dy * ROTATE_DEG_PER_PIXEL) @ self.rotation

    def pan_by(self, dx: float, dy: float) -> None:
        height = max(1, self.view.height())
        distance = float(self.view.opts["distance"])
        world_per_pixel = 2.0 * distance * math.tan(math.radians(FIELD_OF_VIEW) * 0.5) / height
        self.pan = self.pan + np.array((dx * world_per_pixel, -dy * world_per_pixel, 0.0))

    def zoom_by(self, factor: float) -> None:
        self.zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))

    def copy_camera_from(self, other: Viewport) -> None:
        self.rotation = other.rotation.copy()
        self.zoom = other.zoom
        self.pan = other.pan.copy()
        if other.pivot is not None:
            self.pivot = other.pivot.copy()

    # ----------------------------------------------------------------- items
    def clear_items(self, which: str = "all") -> None:
        groups = ("surface_items", "molecule_items") if which == "all" else (f"{which}_items",)
        for group in groups:
            for item in getattr(self, group):
                self.view.remove_item(item)
            setattr(self, group, [])

    def add_items(self, which: str, items) -> None:
        bucket = getattr(self, f"{which}_items")
        for item in items:
            self.view.addItem(item)
            bucket.append(item)
