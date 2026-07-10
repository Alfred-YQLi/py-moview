from __future__ import annotations

import numpy as np

from ..grid import OrbitalGrid
from ..surface import SurfaceMesh
from .gl_view import QtCore, QtGui, QtWidgets, OrbitalGLView, SCENE_BACKGROUND_HEX, default_scene_rotation


class ValueSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int) if QtCore is not None else None

    def __init__(
        self,
        minimum: int,
        maximum: int,
        value: int,
        *,
        divisor: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.divisor = float(divisor)
        self.decimals = int(decimals)
        self.suffix = suffix
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.value_label = QtWidgets.QLabel()
        self.value_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.value_label.setMinimumWidth(52)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.setValue(value)
        self._update_label(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def setValueQuietly(self, value: int) -> None:
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(blocked)
        self._update_label(self.slider.value())

    def _on_value_changed(self, value: int) -> None:
        self._update_label(value)
        self.valueChanged.emit(value)

    def _update_label(self, value: int) -> None:
        number = value / self.divisor
        self.value_label.setText(f"{number:.{self.decimals}f}{self.suffix}")


class CornerAxesWidget(QtWidgets.QWidget):
    def __init__(self, owner: "OpenGLViewer", slot: "SceneSlot | None" = None):
        super().__init__()
        self.owner = owner
        self.slot = slot
        self.setFixedSize(150, 150)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if getattr(self.owner, "corner_check", None) is not None and not self.owner.corner_check.isChecked():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(SCENE_BACKGROUND_HEX))
        painter.setPen(QtGui.QPen(QtGui.QColor("#d4d4d8"), 1.0))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        rotation = self.slot.scene_rotation if self.slot is not None else np.eye(3, dtype=np.float64)
        axes = np.eye(3, dtype=np.float64) @ rotation.T
        colors = (QtGui.QColor("#ef4444"), QtGui.QColor("#22c55e"), QtGui.QColor("#3b82f6"))
        labels = ("X", "Y", "Z")
        center = QtCore.QPointF(self.width() * 0.5, self.height() * 0.51)
        scale = min(self.width(), self.height()) * 0.27

        projected: list[tuple[float, int, np.ndarray]] = []
        for idx, vec in enumerate(axes):
            screen = np.array((vec[0] - 0.38 * vec[1], -vec[2] + 0.38 * vec[1]), dtype=np.float64)
            projected.append((float(vec[1]), idx, screen))
        painter.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold))
        for _depth, idx, screen in sorted(projected, key=lambda item: item[0]):
            length = float(np.linalg.norm(screen))
            if length < 1.0e-6:
                continue
            unit = screen / length
            end = center + QtCore.QPointF(float(screen[0] * scale), float(screen[1] * scale))
            painter.setPen(QtGui.QPen(colors[idx], 2.4, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
            painter.drawLine(center, end)

            arrow_base = end - QtCore.QPointF(float(unit[0] * 10.0), float(unit[1] * 10.0))
            normal = np.array((-unit[1], unit[0]), dtype=np.float64)
            points = [
                end,
                arrow_base + QtCore.QPointF(float(normal[0] * 4.5), float(normal[1] * 4.5)),
                arrow_base - QtCore.QPointF(float(normal[0] * 4.5), float(normal[1] * 4.5)),
            ]
            painter.setBrush(QtGui.QBrush(colors[idx]))
            painter.drawPolygon(QtGui.QPolygonF(points))

            label_pos = end + QtCore.QPointF(float(unit[0] * 7.0), float(unit[1] * 7.0))
            painter.setPen(QtGui.QPen(colors[idx], 1.0))
            painter.drawText(
                QtCore.QRectF(label_pos.x() - 10, label_pos.y() - 10, 20, 20),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                labels[idx],
            )


class ViewHost(QtWidgets.QWidget):
    def __init__(self, owner: "OpenGLViewer", slot: "SceneSlot | None" = None):
        super().__init__()
        self.slot = slot
        self.main_view = OrbitalGLView(owner, slot)
        self.corner_view = CornerAxesWidget(owner, slot)
        self.main_view.setParent(self)
        self.corner_view.setParent(self)
        self.corner_view.raise_()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.main_view.setGeometry(self.rect())
        size = min(150, max(110, self.width() // 6))
        margin = 18
        self.corner_view.setGeometry(self.width() - size - margin, self.height() - size - margin, size, size)


class SceneSlot:
    def __init__(self, owner: "OpenGLViewer", index: int):
        self.owner = owner
        self.index = index
        self.frame = QtWidgets.QWidget()
        self.frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.layout = QtWidgets.QVBoxLayout(self.frame)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.title_label = QtWidgets.QLabel("")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 10pt; color: #18181b;")
        self.view_host = ViewHost(owner, self)
        self.view_host.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.view = self.view_host.main_view
        self.corner_view = self.view_host.corner_view
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.view_host, stretch=1)
        self.reset_state()

    def reset_state(self) -> None:
        self.scene_rotation = default_scene_rotation()
        self.base_center: np.ndarray | None = None
        self.base_radius = 1.0
        self.view_center: np.ndarray | None = None
        self.view_zoom = 1.0
        self.center_atom_idx: int | None = None
        self.surface_items: list[object] = []
        self.molecule_items: list[object] = []
        self.corner_items: list[object] = []
        self.scene_limit_arrays: list[np.ndarray] = []
        self.orbital_index0: int | None = None
        self.grid: OrbitalGrid | None = None
        self.level: float | None = None
        self.positive_mesh: SurfaceMesh | None = None
        self.negative_mesh: SurfaceMesh | None = None
