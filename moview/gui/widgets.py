"""Reusable form widgets with a consistent look."""

from __future__ import annotations

import math

from .qt import QtCore, QtGui, QtWidgets, Signal


class SectionHeader(QtWidgets.QLabel):
    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(text.upper(), parent)
        self.setObjectName("sectionTitle")


class FormGrid(QtWidgets.QGridLayout):
    """Two-column grid: muted labels on the left, editors on the right."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setHorizontalSpacing(12)
        self.setVerticalSpacing(8)
        self.setColumnStretch(1, 1)
        self._row = 0

    def add_row(self, label: str, widget: QtWidgets.QWidget | QtWidgets.QLayout, tooltip: str = "") -> QtWidgets.QLabel:
        caption = QtWidgets.QLabel(label)
        caption.setObjectName("muted")
        caption.setMinimumWidth(74)
        if tooltip:
            caption.setToolTip(tooltip)
        self.addWidget(caption, self._row, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        if isinstance(widget, QtWidgets.QLayout):
            self.addLayout(widget, self._row, 1)
        else:
            self.addWidget(widget, self._row, 1)
        self._row += 1
        return caption

    def add_full_row(self, widget: QtWidgets.QWidget | QtWidgets.QLayout) -> None:
        if isinstance(widget, QtWidgets.QLayout):
            self.addLayout(widget, self._row, 0, 1, 2)
        else:
            self.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1


class ValueSlider(QtWidgets.QWidget):
    """Slider with a right-aligned value readout (integer model, scaled display)."""

    valueChanged = Signal(int)

    def __init__(self, minimum: int, maximum: int, value: int, *, divisor: float = 1.0, decimals: int = 0, suffix: str = ""):
        super().__init__()
        self.divisor = float(divisor)
        self.decimals = int(decimals)
        self.suffix = suffix
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.readout = QtWidgets.QLabel()
        self.readout.setObjectName("muted")
        self.readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.readout.setMinimumWidth(44)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.readout)
        self.slider.setValue(int(value))
        self._update_readout(self.slider.value())
        self.slider.valueChanged.connect(self._changed)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(int(value))

    def _changed(self, value: int) -> None:
        self._update_readout(value)
        self.valueChanged.emit(value)

    def _update_readout(self, value: int) -> None:
        self.readout.setText(f"{value / self.divisor:.{self.decimals}f}{self.suffix}")


class CompactDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Double spin box that shows the shortest faithful representation (``0.05``)."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        return f"{value:.6g}"

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt override
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        stripped = text.strip()
        if stripped in ("", ".", "-", "e", "E") or stripped.endswith(("e", "E", "e-", "E-")):
            return QtGui.QValidator.State.Intermediate, text, pos
        try:
            float(stripped)
        except ValueError:
            return QtGui.QValidator.State.Invalid, text, pos
        return QtGui.QValidator.State.Acceptable, text, pos


class Stepper(QtWidgets.QWidget):
    """Numeric field with compact - / + buttons (instead of tiny native arrows)."""

    def __init__(self, box: QtWidgets.QAbstractSpinBox):
        super().__init__()
        self.box = box
        box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        box.setKeyboardTracking(False)
        box.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        box.setMinimumWidth(56)  # the size hint of a huge range would be very wide
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(box, 1)
        for text, step in (("−", -1), ("+", 1)):
            button = QtWidgets.QPushButton(text)
            button.setObjectName("stepper")
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            button.setAutoRepeat(True)
            button.clicked.connect(lambda _checked=False, s=step: box.stepBy(s))
            layout.addWidget(button)


class SegmentedControl(QtWidgets.QWidget):
    """Row of mutually exclusive buttons; emits the selected key."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], current: str | None = None):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        for position, (key, text) in enumerate(options):
            button = QtWidgets.QPushButton(text)
            button.setObjectName("segment")
            button.setCheckable(True)
            button.setProperty("first", position == 0)
            button.setProperty("last", position == len(options) - 1)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.group.addButton(button)
            self.buttons[key] = button
            layout.addWidget(button)
            button.toggled.connect(lambda checked, k=key: checked and self.changed.emit(k))
        if current is None and options:
            current = options[0][0]
        if current in self.buttons:
            self.buttons[current].setChecked(True)

    def current(self) -> str:
        for key, button in self.buttons.items():
            if button.isChecked():
                return key
        return ""

    def set_current(self, key: str, *, quiet: bool = False) -> None:
        if key not in self.buttons:
            return
        blocker = QtCore.QSignalBlocker(self) if quiet else None
        self.buttons[key].setChecked(True)
        del blocker

    def set_enabled_keys(self, keys: set[str]) -> None:
        for key, button in self.buttons.items():
            button.setEnabled(key in keys)


class LogSlider(QtWidgets.QSlider):
    """Horizontal slider over a logarithmic value range."""

    valueChangedFloat = Signal(float)
    STEPS = 1000

    def __init__(self, minimum: float, maximum: float, value: float):
        super().__init__(QtCore.Qt.Orientation.Horizontal)
        self.log_min = math.log10(minimum)
        self.log_max = math.log10(maximum)
        self.setRange(0, self.STEPS)
        self.set_float(value, quiet=True)
        self.valueChanged.connect(lambda step: self.valueChangedFloat.emit(self.float_value()))

    def float_value(self) -> float:
        fraction = self.value() / self.STEPS
        return 10.0 ** (self.log_min + fraction * (self.log_max - self.log_min))

    def set_float(self, value: float, *, quiet: bool = True) -> None:
        value = min(max(value, 10.0**self.log_min), 10.0**self.log_max)
        step = round((math.log10(value) - self.log_min) / (self.log_max - self.log_min) * self.STEPS)
        blocked = self.blockSignals(quiet)
        self.setValue(int(step))
        self.blockSignals(blocked)


def combo(options, current=None) -> QtWidgets.QComboBox:
    widget = QtWidgets.QComboBox()
    widget.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    widget.setMinimumContentsLength(4)
    for item in options:
        if len(item) == 3:
            text, data, item_icon = item
            widget.addItem(item_icon, text, data)
        else:
            text, data = item
            widget.addItem(text, data)
    if current is not None:
        index = widget.findData(current)
        if index >= 0:
            widget.setCurrentIndex(index)
    return widget
