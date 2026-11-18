"""Side panels: file summary, orbital table and the settings inspector."""

from __future__ import annotations

import html
import math
import re

import numpy as np

from ..config import AppConfig
from ..constants import HARTREE_TO_EV, MIN_GRID_SIZE
from ..core.wavefunction import OCCUPIED_THRESHOLD, Wavefunction
from .icons import dot_icon, swatch_icon
from .qt import QtCore, QtGui, QtWidgets, Signal
from .theme import Theme
from .widgets import (
    CompactDoubleSpinBox,
    FormGrid,
    LogSlider,
    SectionHeader,
    SegmentedControl,
    Stepper,
    ValueSlider,
    combo,
)

MAX_COMPARE = 9
GRID_SPIN_MAXIMUM = 2_147_483_647

SURFACE_STYLES = (("Glass", "glass"), ("Solid", "solid"), ("Wireframe", "wireframe"), ("Solid + edges", "solid_edges"))
ATOM_STYLES = (("Ball and stick", "ball_stick"), ("Space filling", "space_filling"), ("Licorice", "licorice"))
LABEL_MODES = (("None", "off"), ("Number", "number"), ("Element", "element"), ("Element + number", "number_element"))
LABEL_PLACEMENTS = (("Attached", "attached"), ("Floating", "floating"))
THEME_CHOICES = (("System", "system"), ("Light", "light"), ("Dark", "dark"))


def parse_orbital_list(text: str, n_orbitals: int) -> list[int]:
    """Parse ``"45, 46 48-50"`` (1-based, ranges either direction) into 0-based indices."""
    indices: list[int] = []
    normalized = re.sub(r"\s*-\s*", "-", text.strip())
    for token in re.split(r"[\s,;]+", normalized):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            step = 1 if end >= start else -1
            indices.extend(range(start - 1, end - 1 + step, step))
        elif token.isdigit():
            indices.append(int(token) - 1)
        else:
            raise ValueError(f"Invalid orbital token: {token!r}")
    result: list[int] = []
    for index in indices:
        if not 0 <= index < n_orbitals:
            raise ValueError(f"Orbital {index + 1} is outside 1..{n_orbitals}")
        if index not in result:
            result.append(index)
    if len(result) > MAX_COMPARE:
        raise ValueError(f"Compare at most {MAX_COMPARE} orbitals at once")
    return result


# ----------------------------------------------------------------- file card
class FileCard(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.name = QtWidgets.QLabel("No file loaded")
        self.name.setObjectName("fileName")
        self.name.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.format = QtWidgets.QLabel("")
        self.format.setObjectName("pill")
        self.format.hide()
        top.addWidget(self.name, 1)
        top.addWidget(self.format)
        layout.addLayout(top)
        self.details = QtWidgets.QLabel("Open a .fchk or .molden file to begin.")
        self.details.setObjectName("muted")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.notes = QtWidgets.QLabel("")
        self.notes.setObjectName("note")
        self.notes.setWordWrap(True)
        self.notes.hide()
        layout.addWidget(self.notes)

    def show_wavefunction(self, wf: Wavefunction) -> None:
        metrics = QtGui.QFontMetrics(self.name.font())
        self.name.setText(metrics.elidedText(wf.path.name, QtCore.Qt.TextElideMode.ElideMiddle, 210))
        self.name.setToolTip(str(wf.path))
        self.format.setText(wf.source_format.upper())
        self.format.show()
        electrons = f"{wf.n_alpha}α + {wf.n_beta}β electrons"
        kind = "unrestricted" if wf.is_unrestricted else "restricted"
        lines = [f"{wf.n_atoms} atoms · {wf.n_basis} basis functions", f"{electrons} · {kind}"]
        if wf.method_line and wf.method_line.lower() != "molden":
            lines.append(html.escape(wf.method_line))
        self.details.setText("<br>".join(lines))
        notes = [note for note in wf.notes]
        warning = any(note.lower().startswith("warning") for note in notes)
        self.notes.setObjectName("warning" if warning else "note")
        self.notes.style().unpolish(self.notes)  # re-apply the style for the new object name
        self.notes.style().polish(self.notes)
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))


# -------------------------------------------------------------- orbital table
class OrbitalTableModel(QtCore.QAbstractTableModel):
    HEADERS = ("#", "", "Occ", "E / Eh", "E / eV")

    def __init__(self, theme: Theme):
        super().__init__()
        self.wf: Wavefunction | None = None
        self.spin = "alpha"
        self.theme = theme
        self._rows: list[tuple] = []
        self._make_icons()

    def _make_icons(self) -> None:
        self._occupied_icon = dot_icon(self.theme.occupied, True)
        self._partial_icon = dot_icon(self.theme.occupied, False)
        self._virtual_icon = dot_icon(self.theme.text_faint, False)

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._make_icons()
        if self._rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._rows) - 1, len(self.HEADERS) - 1))

    def set_orbitals(self, wf: Wavefunction | None, spin: str) -> None:
        self.beginResetModel()
        self.wf, self.spin = wf, spin
        self._rows = []
        if wf is not None:
            energies = wf.energies(spin)
            occupations = wf.occupations(spin)
            full = 2.0 if not wf.is_unrestricted else 1.0
            for index in range(wf.n_orbitals(spin)):
                energy = float(energies[index]) if index < energies.size else math.nan
                occupation = float(occupations[index])
                self._rows.append((index, wf.frontier_label(spin, index), occupation, energy, occupation >= full - 1.0e-6))
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == QtCore.Qt.Orientation.Horizontal and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and section >= 2:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return None

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        orbital, label, occupation, energy, full = self._rows[index.row()]
        column = index.column()
        Role = QtCore.Qt.ItemDataRole
        if role == Role.DisplayRole:
            if column == 0:
                return str(orbital + 1)
            if column == 1:
                return label
            if column == 2:
                return f"{occupation:.4g}"
            if column == 3:
                return "—" if not math.isfinite(energy) else f"{energy:.5f}"
            if column == 4:
                return "—" if not math.isfinite(energy) else f"{energy * HARTREE_TO_EV:.3f}"
        if role == Role.DecorationRole and column == 0:
            if occupation > OCCUPIED_THRESHOLD:
                return self._occupied_icon if full else self._partial_icon
            return self._virtual_icon
        if role == Role.TextAlignmentRole:
            if column >= 2:
                return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if role == Role.ForegroundRole:
            if column == 1 and label in ("HOMO", "LUMO"):
                return QtGui.QBrush(QtGui.QColor(self.theme.accent))
            if column == 1 or occupation <= OCCUPIED_THRESHOLD:
                return QtGui.QBrush(QtGui.QColor(self.theme.text_muted))
        if role == Role.FontRole and column == 1 and label:
            font = QtGui.QFont()
            font.setBold(label in ("HOMO", "LUMO"))
            font.setPointSizeF(font.pointSizeF() * 0.92)
            return font
        if role == Role.UserRole:
            return orbital
        return None


class OrbitalPanel(QtWidgets.QWidget):
    """Spin selector, jump field, orbital table and comparison entry."""

    selection_changed = Signal(list)  # 0-based orbital indices (selection order)
    spin_changed = Signal(str)
    compare_requested = Signal(list)

    def __init__(self, theme: Theme):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(SectionHeader("Orbitals"))
        header.addStretch(1)
        self.spin = SegmentedControl([("alpha", "α"), ("beta", "β")])
        self.spin.setToolTip("Spin (unrestricted wavefunctions)")
        self.spin.changed.connect(self.spin_changed)
        header.addWidget(self.spin)
        layout.addLayout(header)

        self.jump = QtWidgets.QLineEdit()
        self.jump.setPlaceholderText("Go to orbital — e.g. 212, HOMO-1, LUMO")
        self.jump.setClearButtonEnabled(True)
        self.jump.returnPressed.connect(self._jump)
        self.jump.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)  # never receives focus implicitly
        layout.addWidget(self.jump)

        self.model = OrbitalTableModel(theme)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.verticalHeader().setMinimumSectionSize(22)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setToolTip("Click to render · ⌘/Ctrl- or Shift-click to compare up to 9 orbitals")
        horizontal = self.table.horizontalHeader()
        horizontal.setHighlightSections(False)
        horizontal.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        horizontal.setStretchLastSection(False)
        for column, width in ((0, 62), (2, 40), (3, 82), (4, 64)):
            horizontal.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        horizontal.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        compare_row = QtWidgets.QHBoxLayout()
        compare_row.setSpacing(6)
        self.compare = QtWidgets.QLineEdit()
        self.compare.setPlaceholderText("Compare: 210-214 or 205, 212")
        self.compare.returnPressed.connect(self._compare)
        self.compare.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        compare_button = QtWidgets.QPushButton("Compare")
        compare_button.clicked.connect(self._compare)
        compare_row.addWidget(self.compare, 1)
        compare_row.addWidget(compare_button)
        layout.addLayout(compare_row)
        self._selection_order: list[int] = []

    # ------------------------------------------------------------------ data
    def set_wavefunction(self, wf: Wavefunction | None, spin: str = "alpha") -> None:
        self.spin.set_enabled_keys(set(wf.spins) if wf is not None else {"alpha"})
        self.spin.set_current(spin, quiet=True)
        self._selection_order = []
        blocker = QtCore.QSignalBlocker(self.table.selectionModel())
        self.model.set_orbitals(wf, spin)
        del blocker

    def select(self, indices: list[int], *, emit: bool = True) -> None:
        model = self.table.selectionModel()
        blocker = QtCore.QSignalBlocker(model)
        selection = QtCore.QItemSelection()
        for index in indices:
            if 0 <= index < self.model.rowCount():
                selection.select(self.model.index(index, 0), self.model.index(index, self.model.columnCount() - 1))
        model.select(selection, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        if indices and 0 <= indices[0] < self.model.rowCount():
            model.setCurrentIndex(self.model.index(indices[0], 0), QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            self.table.scrollTo(self.model.index(indices[0], 0), QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        del blocker
        self._selection_order = [i for i in indices if 0 <= i < self.model.rowCount()]
        if emit:
            self.selection_changed.emit(list(self._selection_order))

    def selected(self) -> list[int]:
        return list(self._selection_order)

    def _selection_changed(self, *_args) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        order = [row for row in self._selection_order if row in rows] + [row for row in rows if row not in self._selection_order]
        if len(order) > MAX_COMPARE:
            order = order[-MAX_COMPARE:]
        self._selection_order = order
        if order:
            self.selection_changed.emit(list(order))

    def _jump(self) -> None:
        wf = self.model.wf
        text = self.jump.text().strip()
        if wf is None or not text:
            return
        from ..cli import resolve_orbital

        try:
            spec = text.lower().replace(" ", "")
            index = resolve_orbital(wf, self.model.spin, spec)
        except (ValueError, AssertionError):
            QtWidgets.QToolTip.showText(self.jump.mapToGlobal(QtCore.QPoint(0, self.jump.height())), f"Unknown orbital: {text}")
            return
        self.select([index])

    def _compare(self) -> None:
        wf = self.model.wf
        if wf is None:
            return
        try:
            indices = parse_orbital_list(self.compare.text(), wf.n_orbitals(self.model.spin))
        except ValueError as exc:
            QtWidgets.QToolTip.showText(self.compare.mapToGlobal(QtCore.QPoint(0, self.compare.height())), str(exc))
            return
        if indices:
            self.select(indices, emit=False)
            self.compare_requested.emit(indices)


# ----------------------------------------------------------------- inspector
class Inspector(QtWidgets.QScrollArea):
    """Isosurface, appearance and view settings."""

    compute_changed = Signal()  # grid or margin
    iso_changed = Signal(float)
    surface_changed = Signal()  # style or colors (no recomputation)
    molecule_changed = Signal()  # atom style/size, labels
    view_changed = Signal()  # sync, axes

    def __init__(self, config: AppConfig, grid: int, iso: float, margin: float):
        super().__init__()
        self.config = config
        self.setObjectName("inspector")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        body.setObjectName("inspectorBody")
        self.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        render, view = config.render, config.view

        # Isosurface ------------------------------------------------------
        layout.addWidget(SectionHeader("Isosurface"))
        form = FormGrid()
        iso_row = QtWidgets.QHBoxLayout()
        iso_row.setSpacing(8)
        self.iso_slider = LogSlider(1.0e-4, 1.0, iso)
        self.iso_box = CompactDoubleSpinBox()
        self.iso_box.setDecimals(8)
        self.iso_box.setRange(1.0e-8, 1.0e8)
        self.iso_box.setSingleStep(0.005)
        self.iso_box.setValue(iso)
        self.iso_box.setFixedWidth(76)
        self.iso_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.iso_box.setKeyboardTracking(False)
        self.iso_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        iso_row.addWidget(self.iso_slider, 1)
        iso_row.addWidget(self.iso_box)
        form.add_row("Isovalue", iso_row, "Surfaces are drawn at +value and −value")
        self.grid_box = QtWidgets.QSpinBox()
        self.grid_box.setRange(MIN_GRID_SIZE, GRID_SPIN_MAXIMUM)
        self.grid_box.setSingleStep(8)
        self.grid_box.setValue(max(MIN_GRID_SIZE, int(grid)))
        form.add_row("Grid", Stepper(self.grid_box), "Points along the longest box edge")
        self.margin_box = CompactDoubleSpinBox()
        self.margin_box.setRange(0.0, 30.0)
        self.margin_box.setDecimals(2)
        self.margin_box.setSingleStep(0.5)
        self.margin_box.setValue(max(0.0, margin))
        self.margin_box.setSuffix("")
        form.add_row("Margin", Stepper(self.margin_box), "Space around the molecule, in bohr")
        self.grid_info = QtWidgets.QLabel("")
        self.grid_info.setObjectName("faint")
        form.add_full_row(self.grid_info)
        layout.addLayout(form)

        # Appearance ------------------------------------------------------
        layout.addSpacing(6)
        layout.addWidget(SectionHeader("Appearance"))
        form = FormGrid()
        self.surface_style = combo(SURFACE_STYLES, render.surface_style)
        form.add_row("Surface", self.surface_style)
        colors = [(preset.name, preset.name, swatch_icon(preset.rgb)) for preset in config.colors]
        self.positive_color = combo(colors, render.positive_color)
        self.negative_color = combo(colors, render.negative_color)
        form.add_row("+ψ color", self.positive_color, "Color of the positive phase")
        form.add_row("−ψ color", self.negative_color, "Color of the negative phase")
        self.atom_style = combo(ATOM_STYLES, render.atom_style)
        form.add_row("Atoms", self.atom_style)
        self.atom_scale = ValueSlider(35, 220, round(render.atom_scale * 100), divisor=100.0, decimals=2, suffix="×")
        form.add_row("Atom size", self.atom_scale)
        self.label_mode = combo(LABEL_MODES, render.label_mode)
        self.label_placement = combo(LABEL_PLACEMENTS, render.label_placement)
        self.label_placement.setToolTip("Attached: drawn in 3D at each atom · Floating: avoids overlaps")
        form.add_row("Labels", self.label_mode)
        form.add_row("Placement", self.label_placement)
        self.label_size = ValueSlider(8, 28, render.label_size, suffix=" pt")
        form.add_row("Label size", self.label_size)
        layout.addLayout(form)

        # View ------------------------------------------------------------
        layout.addSpacing(6)
        layout.addWidget(SectionHeader("View"))
        form = FormGrid()
        self.sync_views = QtWidgets.QCheckBox("Rotate comparison views together")
        self.sync_views.setChecked(view.sync_views)
        form.add_full_row(self.sync_views)
        self.show_axes = QtWidgets.QCheckBox("Show axes")
        self.show_axes.setChecked(view.show_axes)
        form.add_full_row(self.show_axes)
        layout.addLayout(form)
        layout.addStretch(1)

        hint = QtWidgets.QLabel(
            "<b>Mouse</b> drag rotate · right-drag roll · shift/middle-drag pan · wheel zoom<br>"
            "<b>Keys</b> A / D previous / next orbital · arrows rotate · +/− zoom · "
            "C then click an atom to set the rotation centre"
        )
        hint.setObjectName("faint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Signals ----------------------------------------------------------
        self.iso_slider.valueChangedFloat.connect(self._slider_iso)
        self.iso_box.valueChanged.connect(self._box_iso)
        self.grid_box.valueChanged.connect(lambda _v: self.compute_changed.emit())
        self.margin_box.valueChanged.connect(lambda _v: self.compute_changed.emit())
        for widget in (self.surface_style, self.positive_color, self.negative_color):
            widget.currentIndexChanged.connect(lambda _i: self.surface_changed.emit())
        for widget in (self.atom_style, self.label_mode, self.label_placement):
            widget.currentIndexChanged.connect(lambda _i: self.molecule_changed.emit())
        self.atom_scale.valueChanged.connect(lambda _v: self.molecule_changed.emit())
        self.label_size.valueChanged.connect(lambda _v: self.molecule_changed.emit())
        self.sync_views.toggled.connect(lambda _v: self.view_changed.emit())
        self.show_axes.toggled.connect(lambda _v: self.view_changed.emit())

    # ------------------------------------------------------------- values
    def _slider_iso(self, value: float) -> None:
        value = float(f"{value:.3g}")
        blocker = QtCore.QSignalBlocker(self.iso_box)
        self.iso_box.setValue(value)
        del blocker
        self.iso_changed.emit(value)

    def _box_iso(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            return
        self.iso_slider.set_float(value, quiet=True)
        self.iso_changed.emit(float(value))

    def isovalue(self) -> float:
        return float(self.iso_box.value())

    def grid(self) -> int:
        return int(self.grid_box.value())

    def margin(self) -> float:
        return float(self.margin_box.value())

    def surface_style_value(self) -> str:
        return str(self.surface_style.currentData())

    def positive_preset(self):
        return self.config.color(str(self.positive_color.currentData()))

    def negative_preset(self):
        return self.config.color(str(self.negative_color.currentData()))

    def atom_style_value(self) -> str:
        return str(self.atom_style.currentData())

    def atom_scale_value(self) -> float:
        return self.atom_scale.value() / 100.0

    def label_mode_value(self) -> str:
        return str(self.label_mode.currentData())

    def label_placement_value(self) -> str:
        return str(self.label_placement.currentData())

    def label_size_value(self) -> int:
        return int(self.label_size.value())

    def set_grid_info(self, text: str) -> None:
        self.grid_info.setText(text)


def energy_text(energy: float) -> str:
    if not np.isfinite(energy):
        return "energy n/a"
    return f"{energy:.5f} Eh · {energy * HARTREE_TO_EV:.3f} eV"
