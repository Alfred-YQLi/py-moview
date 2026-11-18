"""Main window: wires panels, viewports, the render service and background jobs."""

from __future__ import annotations

import html
import math
import time
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..constants import BOHR_TO_ANG, HARTREE_TO_EV, HIGH_GRID_WARNING_THRESHOLD
from ..core.bonds import compute_bonds
from ..core.wavefunction import Wavefunction
from ..elements import atom_symbol
from ..parsers import FILE_DIALOG_FILTER, parse_wavefunction
from .gl_items import ATOMS, GLASS, SOLID, AttachedLabelItem, FloatingLabelItem, ShadedMeshItem
from .icons import icon
from .jobs import JobRunner
from .molecule import TriangleMesh, atom_radius, build_molecule_mesh, label_texts
from .panels import THEME_CHOICES, FileCard, Inspector, OrbitalPanel, energy_text
from .qt import QtCore, QtGui, QtWidgets
from .render_service import Cancelled, RenderResult, RenderService, estimate_points_text, high_grid_warning_text
from .theme import Theme, apply_theme, resolve_theme
from .viewport import OrbitalView, Viewport, default_rotation

MAX_VIEWPORTS = 9
SELECTION_DEBOUNCE_MS = 90
ISO_DEBOUNCE_MS = 120
COMPUTE_DEBOUNCE_MS = 650
APPEARANCE_DEBOUNCE_MS = 60
PREFETCH_DELAY_MS = 150  # neighbors are pre-rendered shortly after each render


def _grid_positions(count: int) -> list[tuple[int, int, int, int]]:
    if count <= 1:
        return [(0, 0, 1, 1)]
    if count == 2:
        return [(0, 0, 1, 1), (0, 1, 1, 1)]
    if count == 3:
        return [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)]
    columns = 2 if count == 4 else 3
    return [(i // columns, i % columns, 1, 1) for i in range(count)]


def _timer(parent: QtCore.QObject, interval: int, slot) -> QtCore.QTimer:
    timer = QtCore.QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(interval)
    timer.timeout.connect(slot)
    return timer


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, *, grid: int, isovalue: float, margin: float):
        super().__init__()
        self.config = config
        self.theme_setting = config.view.theme
        self.theme: Theme = resolve_theme(self.theme_setting)
        self.service = RenderService(config.resources)
        self.foreground = JobRunner(1, "moview-render", self)
        self.background = JobRunner(1, "moview-prefetch", self)

        self.wf: Wavefunction | None = None
        self.bonds = np.empty((0, 2), dtype=np.int64)
        self.generation = 0
        self.spin = "alpha"
        self.loading = False
        self.render_token = 0
        self.prefetch_token = 0
        self.acknowledged_grids: set[int] = set()
        self.molecule_meshes: dict[tuple[str, float], TriangleMesh] = {}
        self.displayed: list[int] = []
        self._pending_selection: list[int] = []

        self.setWindowTitle("MOview")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 640)
        self.setAcceptDrops(True)
        apply_theme(QtWidgets.QApplication.instance(), self.theme)

        self._build_toolbar()
        self._build_body(grid, isovalue, margin)
        self._build_statusbar()
        self._install_shortcuts()

        self.selection_timer = _timer(self, SELECTION_DEBOUNCE_MS, self._render_selection)
        self.iso_timer = _timer(self, ISO_DEBOUNCE_MS, self.render_displayed)
        self.compute_timer = _timer(self, COMPUTE_DEBOUNCE_MS, self._compute_settings_settled)
        self.molecule_timer = _timer(self, APPEARANCE_DEBOUNCE_MS, self.rebuild_molecules)
        self.prefetch_timer = _timer(self, PREFETCH_DELAY_MS, self.start_prefetch)
        self._update_actions()

    # ================================================================ layout
    def _build_toolbar(self) -> None:
        bar = QtWidgets.QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QtCore.QSize(18, 18))
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)
        self.toolbar = bar
        self.actions: dict[str, QtGui.QAction] = {}

        def add(key, text, slot, shortcut=None, tip="", icon_name=None):
            action = QtGui.QAction(text, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(QtGui.QKeySequence(shortcut))
            action.setToolTip(f"{tip or text}" + (f"  ({QtGui.QKeySequence(shortcut).toString(QtGui.QKeySequence.SequenceFormat.NativeText)})" if shortcut else ""))
            action.setData(icon_name)
            bar.addAction(action)
            self.actions[key] = action
            return action

        title = QtWidgets.QLabel("MOview")
        title.setObjectName("appTitle")
        title.setContentsMargins(6, 0, 14, 0)
        bar.addWidget(title)
        add("open", "Open", self.open_dialog, QtGui.QKeySequence.StandardKey.Open, "Open a wavefunction file", "open")
        bar.addSeparator()
        add("render", "Render", self.render_selection_now, "Ctrl+R", "Render the selected orbitals", "render")
        add("prev", "", lambda: self.step_orbital(-1), None, "Previous orbital (A)", "prev")
        add("next", "", lambda: self.step_orbital(1), None, "Next orbital (D)", "next")
        add("homo", "HOMO", lambda: self.jump_frontier("homo"), None, "Highest occupied orbital", "homo")
        add("lumo", "LUMO", lambda: self.jump_frontier("lumo"), None, "Lowest unoccupied orbital", "lumo")
        bar.addSeparator()
        add("reset", "Reset view", self.reset_view, "R", "Reset rotation, zoom and centre", "reset")
        add("snapshot", "Save image", self.save_image, QtGui.QKeySequence.StandardKey.Save, "Save the view as PNG", "camera")
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        add("left", "", self._toggle_left, None, "Show or hide the orbital list", "panel-left").setCheckable(True)
        add("right", "", self._toggle_right, None, "Show or hide the settings", "panel-right").setCheckable(True)
        self.actions["left"].setChecked(True)
        self.actions["right"].setChecked(True)
        add("theme", "", lambda: None, None, "Theme: system, light or dark", "moon")
        self._build_theme_menu()
        self.toolbar.widgetForAction(self.actions["render"]).setObjectName("primary")
        for key in ("prev", "next", "left", "right", "theme"):
            self.toolbar.widgetForAction(self.actions[key]).setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._refresh_icons()

    def _refresh_icons(self) -> None:
        for key, action in self.actions.items():
            name = action.data()
            if key == "theme":
                name = "sun" if self.theme.dark else "moon"
            if name:
                color = self.theme.accent_text if key == "render" else self.theme.text_muted
                action.setIcon(icon(name, color))

    def _build_body(self, grid: int, isovalue: float, margin: float) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)
        self.splitter = splitter

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(320)
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 14, 14, 12)
        side_layout.setSpacing(12)
        self.file_card = FileCard()
        side_layout.addWidget(self.file_card)
        self.orbitals = OrbitalPanel(self.theme)
        side_layout.addWidget(self.orbitals, 1)
        self.sidebar = sidebar

        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(12, 12, 12, 12)
        self.stack = QtWidgets.QStackedWidget()
        center_layout.addWidget(self.stack)
        self.stack.addWidget(self._empty_page())
        grid_host = QtWidgets.QWidget()
        self.viewport_grid = QtWidgets.QGridLayout(grid_host)
        self.viewport_grid.setContentsMargins(0, 0, 0, 0)
        self.viewport_grid.setSpacing(10)
        self.stack.addWidget(grid_host)
        self.grid_host = grid_host
        self.viewports: list[Viewport] = []
        self.visible_count = 0
        self.active: Viewport | None = None

        self.inspector = Inspector(self.config, grid, isovalue, margin)
        self.inspector.setMinimumWidth(300)

        splitter.addWidget(sidebar)
        splitter.addWidget(center)
        splitter.addWidget(self.inspector)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self._initial_sizes: list[int] | None = [370, 740, 330]

        self.orbitals.selection_changed.connect(self._on_selection)
        self.orbitals.compare_requested.connect(self.render_orbitals)
        self.orbitals.spin_changed.connect(self._on_spin)
        self.inspector.iso_changed.connect(lambda _v: self._on_iso())
        self.inspector.compute_changed.connect(self._on_compute_settings)
        self.inspector.surface_changed.connect(self.rebuild_surfaces)
        self.inspector.molecule_changed.connect(lambda: self.molecule_timer.start())
        self.inspector.view_changed.connect(self._on_view_settings)
        self._update_grid_info()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._initial_sizes is not None:
            # Applied once the window has its real width; earlier the splitter
            # squeezes the side panels to their minimum.
            left, _center, right = self._initial_sizes
            self._initial_sizes = None
            total = max(self.splitter.width(), left + right + 300)
            self.splitter.setSizes([left, total - left - right, right])

    def _empty_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addStretch(1)
        title = QtWidgets.QLabel("Open a wavefunction")
        title.setObjectName("emptyTitle")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint = QtWidgets.QLabel(
            "Gaussian formatted checkpoint (.fchk, .fch) or Molden (.molden, .molden.input)<br>"
            "Drop a file here, or use Open."
        )
        hint.setObjectName("emptyHint")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        button = QtWidgets.QPushButton("Open file…")
        button.setObjectName("primary")
        button.clicked.connect(self.open_dialog)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        row.addStretch(1)
        layout.addWidget(title)
        layout.addSpacing(6)
        layout.addWidget(hint)
        layout.addSpacing(14)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_statusbar(self) -> None:
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        self.status_label = QtWidgets.QLabel("Ready")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setTextVisible(False)
        self.progress.hide()
        status.addWidget(self.status_label, 1)
        status.addPermanentWidget(self.progress)

    def _install_shortcuts(self) -> None:
        for key, slot in (("A", lambda: self.step_orbital(-1)), ("D", lambda: self.step_orbital(1)), ("C", self.start_center_pick)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(lambda s=slot: None if self._typing() else s())

    def _typing(self) -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        return isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox))

    def _toggle_left(self) -> None:
        self.sidebar.setVisible(self.actions["left"].isChecked())

    def _toggle_right(self) -> None:
        self.inspector.setVisible(self.actions["right"].isChecked())

    def _build_theme_menu(self) -> None:
        """The single theme control: a toolbar menu (System / Light / Dark)."""
        menu = QtWidgets.QMenu(self)
        group = QtGui.QActionGroup(self)
        self.theme_actions: dict[str, QtGui.QAction] = {}
        for text, key in THEME_CHOICES:
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(key == self.theme_setting)
            action.triggered.connect(lambda _checked=False, k=key: self.set_theme(k))
            group.addAction(action)
            self.theme_actions[key] = action
        self.actions["theme"].setMenu(menu)
        button = self.toolbar.widgetForAction(self.actions["theme"])
        button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        hints = QtWidgets.QApplication.instance().styleHints()
        if hasattr(hints, "colorSchemeChanged"):  # follow macOS/Windows appearance live
            hints.colorSchemeChanged.connect(lambda _scheme: self.theme_setting == "system" and self.set_theme("system"))

    # ============================================================== status
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _busy(self, busy: bool) -> None:
        if busy:
            self.progress.setRange(0, 0)
            self.progress.show()
        else:
            self.progress.hide()

    def _update_actions(self) -> None:
        loaded = self.wf is not None and not self.loading
        for key in ("render", "prev", "next", "homo", "lumo", "reset", "snapshot"):
            self.actions[key].setEnabled(loaded)
        self.actions["open"].setEnabled(not self.loading)
        self.orbitals.setEnabled(loaded)

    # ============================================================== loading
    def open_dialog(self) -> None:
        start = str(self.wf.path.parent) if self.wf is not None else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open wavefunction", start, FILE_DIALOG_FILTER)
        if path:
            self.open_path(path, auto_render=True)

    def open_path(self, path: str | Path, *, file_format: str | None = None, auto_render: bool = True) -> None:
        path = Path(path)
        self.loading = True
        self._cancel_prefetch()
        self._update_actions()
        self.set_status(f"Reading {path.name}…")
        self._busy(True)

        def work():
            wavefunction = parse_wavefunction(path, file_format)
            return wavefunction, compute_bonds(wavefunction.atomic_numbers, wavefunction.coordinates_angstrom)

        self.foreground.submit(work, lambda result: self._loaded(result, auto_render), self._load_failed)

    def _load_failed(self, error: BaseException) -> None:
        self.loading = False
        self._busy(False)
        self._update_actions()
        self.set_status("Could not open the file.")
        QtWidgets.QMessageBox.critical(self, "Could not open file", str(error))

    def _loaded(self, result, auto_render: bool) -> None:
        wf, bonds = result
        self.loading = False
        self._busy(False)
        self.wf, self.bonds = wf, bonds
        self.generation = self.service.load(wf)
        self.render_token += 1
        self.acknowledged_grids.clear()
        self.molecule_meshes.clear()
        self.displayed = []
        self.spin = "alpha"
        self.setWindowTitle(f"{wf.path.name} — MOview")
        self.file_card.show_wavefunction(wf)
        self.orbitals.set_wavefunction(wf, self.spin)
        for viewport in self.viewports:
            viewport.clear_items()
            viewport.reset_camera_state()
            viewport.result = None
            viewport.orbital = None
        self.stack.setCurrentIndex(1)
        self._configure_viewports(1)
        self._update_grid_info()
        self._update_actions()
        homo = wf.homo_index(self.spin)
        self.orbitals.select([homo], emit=False)
        self.set_status(f"Loaded {wf.n_atoms} atoms, {wf.n_basis} basis functions.")
        if auto_render:
            self.render_orbitals([homo])
        else:
            self._show_molecule_only()
            self.prefetch_timer.start()

    def _show_molecule_only(self) -> None:
        viewport = self.viewports[0]
        self._ensure_molecule(viewport)
        viewport.title.setText(f"<b>{html.escape(self.wf.path.name)}</b>")
        viewport.meta.setText("Select an orbital to render")
        self._apply_camera(viewport)

    # ============================================================ viewports
    def _new_viewport(self) -> Viewport:
        viewport = Viewport(len(self.viewports), self.theme)
        view = viewport.view
        view.activated.connect(lambda v=viewport: self._activate(v))
        view.rotate_requested.connect(lambda dx, dy, mode, v=viewport: self._rotate(v, dx, dy, mode))
        view.pan_requested.connect(lambda dx, dy, v=viewport: self._pan(v, dx, dy))
        view.zoom_requested.connect(lambda factor, v=viewport: self._zoom(v, factor))
        view.pick_requested.connect(lambda pos, v=viewport: self._pick_center(v, pos))
        view.show_axes = self.inspector.show_axes.isChecked()
        self.viewports.append(viewport)
        return viewport

    def _configure_viewports(self, count: int) -> None:
        count = max(1, min(MAX_VIEWPORTS, count))
        while len(self.viewports) < count:
            self._new_viewport()
        if count != self.visible_count:
            self._layout_viewports(count)
        for viewport in self.viewports[count:]:
            viewport.clear_items()
            viewport.result = None
            viewport.orbital = None
        self.visible_count = count
        if self.active not in self.visible():
            self.active = self.viewports[0]
        for viewport in self.visible():
            viewport.set_active(count > 1 and viewport is self.active)
            if viewport.home_pivot is None and self.wf is not None:
                self._home_camera(viewport)
            if self.inspector.sync_views.isChecked() and viewport is not self.viewports[0]:
                viewport.copy_camera_from(self.viewports[0])

    def _layout_viewports(self, count: int) -> None:
        """Arrange ``count`` cells. Cells that stay visible are never hidden:
        hiding the focused view would hand keyboard focus to the next widget
        (the orbital text box), which then swallowed the A/D shortcuts."""
        focused = QtWidgets.QApplication.focusWidget()
        for viewport in self.viewports:
            self.viewport_grid.removeWidget(viewport)
        for row in range(3):
            self.viewport_grid.setRowStretch(row, 0)
            self.viewport_grid.setColumnStretch(row, 0)
        for viewport, (row, col, rows, cols) in zip(self.viewports, _grid_positions(count)):
            self.viewport_grid.addWidget(viewport, row, col, rows, cols)
            viewport.show()
            for r in range(row, row + rows):
                self.viewport_grid.setRowStretch(r, 1)
            for c in range(col, col + cols):
                self.viewport_grid.setColumnStretch(c, 1)
        for viewport in self.viewports[count:]:
            viewport.hide()
        if focused is not None and not focused.isVisible() and isinstance(focused, OrbitalView):
            self.viewports[0].view.setFocus()

    def visible(self) -> list[Viewport]:
        return self.viewports[: self.visible_count]

    def _home_camera(self, viewport: Viewport) -> None:
        coords = self.wf.coordinates_angstrom
        low, high = coords.min(axis=0), coords.max(axis=0)
        viewport.home_pivot = 0.5 * (low + high)
        viewport.pivot = viewport.home_pivot.copy()
        viewport.radius = max(2.0, 0.5 * float(np.linalg.norm(high - low)) + 1.0)
        viewport.zoom = self.config.view.zoom

    def _targets(self, viewport: Viewport) -> list[Viewport]:
        return self.visible() if self.inspector.sync_views.isChecked() else [viewport]

    def _apply_camera(self, viewport: Viewport) -> None:
        strength = 0.62 if self.theme.dark else 0.5
        viewport.apply(self.theme.rgb("viewport"), strength)

    def _activate(self, viewport: Viewport) -> None:
        self.active = viewport
        for v in self.visible():
            v.set_active(self.visible_count > 1 and v is viewport)

    def _rotate(self, viewport: Viewport, dx: float, dy: float, mode: str) -> None:
        for target in self._targets(viewport):
            target.rotate(dx, dy, mode)
            self._apply_camera(target)

    def _pan(self, viewport: Viewport, dx: float, dy: float) -> None:
        for target in self._targets(viewport):
            target.pan_by(dx, dy)
            self._apply_camera(target)

    def _zoom(self, viewport: Viewport, factor: float) -> None:
        for target in self._targets(viewport):
            target.zoom_by(factor)
            self._apply_camera(target)

    def reset_view(self) -> None:
        for viewport in self.visible():
            viewport.rotation = default_rotation()
            viewport.pan = np.zeros(3)
            if viewport.home_pivot is not None:
                viewport.pivot = viewport.home_pivot.copy()
            viewport.zoom = self.config.view.zoom
            self._apply_camera(viewport)

    def start_center_pick(self) -> None:
        if self.wf is None:
            return
        for viewport in self.visible():
            viewport.view.pick_mode = True
            viewport.view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.set_status("Click an atom to make it the rotation centre (Esc to cancel).")

    def _end_center_pick(self) -> None:
        for viewport in self.viewports:
            viewport.view.pick_mode = False
            viewport.view.unsetCursor()

    def _pick_center(self, viewport: Viewport, position: QtCore.QPointF) -> None:
        self._end_center_pick()
        coords = self.wf.coordinates_angstrom
        screen = viewport.view.project(coords, viewport.model_matrix())
        distance = np.hypot(screen[:, 0] - position.x(), screen[:, 1] - position.y())
        atom = int(np.argmin(distance))
        if distance[atom] > 30.0:
            self.set_status("No atom under the cursor; press C and try again.")
            return
        for target in self._targets(viewport):
            # The picked atom becomes the pivot and moves to the centre of the view.
            target.pivot = coords[atom].astype(np.float64)
            target.pan = np.zeros(3)
            self._apply_camera(target)
        self.set_status(f"Rotation centre: {atom_symbol(int(self.wf.atomic_numbers[atom]))}{atom + 1}")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._end_center_pick()
            self.set_status("Ready")
            return
        super().keyPressEvent(event)

    # ============================================================ selection
    def _on_selection(self, indices: list[int]) -> None:
        if self.wf is None or not indices:
            return
        self._pending_selection = list(indices)
        self.selection_timer.start()

    def _render_selection(self) -> None:
        self.render_orbitals(self._pending_selection or self.orbitals.selected())

    def render_selection_now(self) -> None:
        selected = self.orbitals.selected()
        if selected:
            self.render_orbitals(selected)

    def step_orbital(self, delta: int) -> None:
        if self.wf is None or self.loading:
            return
        current = self.displayed[0] if self.displayed else self.wf.homo_index(self.spin)
        index = min(max(current + delta, 0), self.wf.n_orbitals(self.spin) - 1)
        self.orbitals.select([index])

    def jump_frontier(self, which: str) -> None:
        if self.wf is None:
            return
        index = self.wf.homo_index(self.spin) if which == "homo" else self.wf.lumo_index(self.spin)
        self.orbitals.select([index])

    def _on_spin(self, spin: str) -> None:
        if self.wf is None or spin not in self.wf.spins:
            return
        self.spin = spin
        self._cancel_prefetch()
        self.orbitals.set_wavefunction(self.wf, spin)
        self.orbitals.select([self.wf.homo_index(spin)])

    # ============================================================ rendering
    def _confirm_grid(self, grid: int) -> bool:
        if grid <= HIGH_GRID_WARNING_THRESHOLD or grid in self.acknowledged_grids or self.wf is None:
            return True
        spec = self.service.grid_spec(grid, self.inspector.margin())
        answer = QtWidgets.QMessageBox.warning(
            self,
            "High-resolution grid",
            high_grid_warning_text(spec),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            self.set_status("High-resolution render cancelled.")
            return False
        self.acknowledged_grids.add(grid)
        return True

    def render_displayed(self) -> None:
        if self.displayed:
            self.render_orbitals(list(self.displayed))

    def render_orbitals(self, indices: list[int]) -> None:
        if self.wf is None or self.loading or not indices:
            return
        grid, margin, iso = self.inspector.grid(), self.inspector.margin(), self.inspector.isovalue()
        if not self._confirm_grid(grid):
            return
        indices = list(indices)[:MAX_VIEWPORTS]
        self.displayed = indices
        self._configure_viewports(len(indices))
        self.render_token += 1
        token, generation, spin = self.render_token, self.generation, self.spin

        missing = []
        for viewport, index in zip(self.visible(), indices):
            viewport.orbital = index
            cached = self.service.cached_result(spin, index, grid, margin, iso)
            if cached is not None:
                self._show_result(viewport, cached)
            else:
                missing.append(index)
                self._ensure_molecule(viewport)
                self._header(viewport, index, pending=True)
                self._apply_camera(viewport)
        if not missing:
            self._after_render(from_cache=True)
            return

        self._cancel_prefetch()
        self._busy(True)
        what = f"orbital {missing[0] + 1}" if len(missing) == 1 else f"{len(missing)} orbitals"
        self.set_status(f"Computing {what} on a {estimate_points_text(self.service.grid_spec(grid, margin))} grid…")

        def work():
            started = time.perf_counter()
            results = self.service.render(generation, spin, missing, grid, margin, iso, cancelled=lambda: token != self.render_token)
            return results, time.perf_counter() - started

        def done(payload) -> None:
            results, elapsed = payload
            if token != self.render_token or generation != self.generation:
                return
            by_index = {result.index: result for result in results}
            for viewport in self.visible():
                if viewport.orbital in by_index:
                    self._show_result(viewport, by_index[viewport.orbital])
            self._busy(False)
            self._after_render(from_cache=False, elapsed=elapsed)

        def failed(error: BaseException) -> None:
            if isinstance(error, Cancelled) or token != self.render_token:
                return
            self._busy(False)
            self.set_status("Render failed.")
            QtWidgets.QMessageBox.critical(self, "Render failed", str(error))

        self.foreground.submit(work, done, failed)

    def _after_render(self, from_cache: bool, elapsed: float | None = None) -> None:
        faces = sum(v.result.n_faces for v in self.visible() if v.result is not None)
        verb = "Shown from cache" if from_cache else f"Rendered in {1000 * (elapsed or 0):.0f} ms"
        if self.visible_count == 1 and self.visible()[0].result is not None:
            result = self.visible()[0].result
            text = f"{verb}: {self.spin} orbital {result.index + 1} · {result.positive.n_faces:,} + {result.negative.n_faces:,} triangles"
        else:
            text = f"{verb}: {self.visible_count} orbitals · {faces:,} triangles"
        limit = self.config.resources.surface_face_limit
        if any(v.result is not None and v.result.n_faces > 2 * limit for v in self.visible()):
            text += " · very dense surface, a coarser grid would be faster"
        self.set_status(text)
        self.prefetch_timer.start()

    def _header(self, viewport: Viewport, index: int, pending: bool = False) -> None:
        wf = self.wf
        spin_symbol = "α" if self.spin == "alpha" else "β"
        label = wf.frontier_label(self.spin, index)
        energy = float(wf.energies(self.spin)[index])
        label_html = f" <span style='color:{self.theme.accent}'><b>{label}</b></span>" if label else ""
        prefix = f"{spin_symbol} " if wf.is_unrestricted else ""
        muted = self.theme.text_muted
        compact = self.visible_count > 1  # comparison cells are narrow
        if compact:
            energy_html = "" if not math.isfinite(energy) else f"&nbsp;&nbsp;<span style='color:{muted}'>{energy * HARTREE_TO_EV:.3f} eV</span>"
        else:
            energy_html = f"&nbsp;&nbsp;<span style='color:{muted}'>{energy_text(energy)}</span>"
        viewport.title.setText(f"<b>{prefix}MO {index + 1}</b>{label_html}{energy_html}")
        occupation = wf.occupation(self.spin, index)
        if pending:
            viewport.meta.setText("computing…")
        elif compact:
            viewport.meta.setText(f"occ {occupation:g}")
        elif viewport.result is not None:
            shape = "×".join(str(n) for n in viewport.result.spec.shape)
            viewport.meta.setText(f"occ {occupation:g} · iso ±{viewport.result.level:g} · grid {shape}")

    def _show_result(self, viewport: Viewport, result: RenderResult) -> None:
        viewport.result = result
        viewport.orbital = result.index
        self._ensure_molecule(viewport)
        self._build_surfaces(viewport)
        self._header(viewport, result.index)
        self._apply_camera(viewport)

    # ============================================================ GL items
    def _build_surfaces(self, viewport: Viewport) -> None:
        viewport.clear_items("surface")
        result = viewport.result
        if result is None:
            return
        style = self.inspector.surface_style_value()
        items = []
        for mesh, preset in ((result.positive, self.inspector.positive_preset()), (result.negative, self.inspector.negative_preset())):
            if mesh.n_faces == 0:
                continue
            rgb = tuple(channel / 255.0 for channel in preset.rgb)
            if style == "wireframe":
                edge = (*rgb, 0.85)
            else:
                edge = (*(0.45 * c for c in rgb), 0.55)
            items.append(
                ShadedMeshItem(
                    mesh.vertices,
                    mesh.normals,
                    mesh.faces,
                    color=(*rgb, 1.0),
                    mode=style,
                    material=GLASS if style == "glass" else SOLID,
                    edge_color=edge,
                )
            )
        viewport.add_items("surface", items)

    def rebuild_surfaces(self) -> None:
        for viewport in self.visible():
            self._build_surfaces(viewport)
            self._apply_camera(viewport)

    def _molecule_mesh(self) -> TriangleMesh:
        key = (self.inspector.atom_style_value(), round(self.inspector.atom_scale_value(), 3))
        mesh = self.molecule_meshes.get(key)
        if mesh is None:
            mesh = build_molecule_mesh(self.wf.atomic_numbers, self.wf.coordinates_angstrom, self.bonds, *key)
            self.molecule_meshes[key] = mesh
        return mesh

    def _ensure_molecule(self, viewport: Viewport) -> None:
        if self.wf is None or viewport.molecule_items:
            return
        mesh = self._molecule_mesh()
        items = [ShadedMeshItem(mesh.vertices, mesh.normals, mesh.faces, colors=mesh.colors, mode="solid", material=ATOMS)]
        mode = self.inspector.label_mode_value()
        if mode != "off":
            labels = label_texts(self.wf.atomic_numbers, mode)
            keep = [i for i, text in enumerate(labels) if text]
            positions = self.wf.coordinates_angstrom[keep]
            texts = [labels[i] for i in keep]
            size = self.inspector.label_size_value()
            if self.inspector.label_placement_value() == "floating":
                items.append(FloatingLabelItem(positions, texts, size, self.theme.label_text, self.theme.label_halo))
            else:
                style, scale = self.inspector.atom_style_value(), self.inspector.atom_scale_value()
                radii = np.array([atom_radius(int(self.wf.atomic_numbers[i]), style, scale) for i in keep])
                items.append(AttachedLabelItem(positions, texts, size, radii, self.theme.label_text, self.theme.label_halo))
        viewport.add_items("molecule", items)

    def rebuild_molecules(self) -> None:
        if self.wf is None:
            return
        for viewport in self.visible():
            viewport.clear_items("molecule")
            self._ensure_molecule(viewport)
            self._apply_camera(viewport)

    # ============================================================ settings
    def _on_iso(self) -> None:
        self._cancel_prefetch()
        self.iso_timer.start()

    def _on_compute_settings(self) -> None:
        self._cancel_prefetch()
        self._update_grid_info()
        self.compute_timer.start()

    def _compute_settings_settled(self) -> None:
        self.render_displayed()

    def _update_grid_info(self) -> None:
        if self.wf is None:
            self.inspector.set_grid_info("")
            return
        spec = self.service.grid_spec(self.inspector.grid(), self.inspector.margin())
        spacing = spec.spacing * BOHR_TO_ANG
        self.inspector.set_grid_info(f"{estimate_points_text(spec)} · spacing {spacing:.3f} Å")

    def _on_view_settings(self) -> None:
        show_axes = self.inspector.show_axes.isChecked()
        for viewport in self.viewports:
            viewport.view.show_axes = show_axes
            viewport.view.update()
        if self.inspector.sync_views.isChecked() and self.visible_count > 1:
            reference = self.active or self.viewports[0]
            for viewport in self.visible():
                if viewport is not reference:
                    viewport.copy_camera_from(reference)
                    self._apply_camera(viewport)

    def set_theme(self, setting: str) -> None:
        self.theme_setting = setting
        if setting in getattr(self, "theme_actions", {}):
            self.theme_actions[setting].setChecked(True)
        self.theme = resolve_theme(setting)
        apply_theme(QtWidgets.QApplication.instance(), self.theme)
        self.orbitals.model.set_theme(self.theme)
        self._refresh_icons()
        for viewport in self.viewports:
            viewport.view.set_theme(self.theme)
        if self.wf is not None:
            self.rebuild_molecules()  # label colors follow the theme
            for viewport in self.visible():
                if viewport.orbital is not None:
                    self._header(viewport, viewport.orbital)

    # ============================================================ prefetch
    def _cancel_prefetch(self) -> None:
        self.prefetch_timer.stop()
        self.prefetch_token += 1
        self.background.cancel_pending()

    def start_prefetch(self) -> None:
        """Pre-render the orbitals next to the displayed one so A / D is instant."""
        if self.wf is None or self.loading:
            return
        if self.foreground.busy:
            self.prefetch_timer.start()
            return
        grid, margin, iso = self.inspector.grid(), self.inspector.margin(), self.inspector.isovalue()
        if grid > HIGH_GRID_WARNING_THRESHOLD and grid not in self.acknowledged_grids:
            return
        neighbors = self.service.neighbors_to_prefetch(self.spin, grid, margin, iso, self.displayed)
        if not neighbors:
            return
        self.prefetch_token += 1
        token, generation, spin = self.prefetch_token, self.generation, self.spin

        def work():
            if token != self.prefetch_token:
                raise Cancelled()
            return self.service.render(
                generation, spin, neighbors, grid, margin, iso, cancelled=lambda: token != self.prefetch_token, keep_fields=False
            )

        self.background.submit(work, lambda _results: None, lambda _error: None)

    # ============================================================== export
    def save_image(self) -> None:
        if self.wf is None:
            return
        orbital = self.displayed[0] + 1 if self.displayed else 0
        suggestion = str(self.wf.path.with_name(f"{self.wf.path.stem}_{self.spin}_MO{orbital}.png"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", suggestion, "PNG image (*.png)")
        if not path:
            return
        widget = self.grid_host if self.visible_count > 1 else self.visible()[0].view
        image = widget.grab().toImage()  # includes the axes overlay; device-pixel resolution
        if image.save(path):
            self.set_status(f"Saved {Path(path).name} ({image.width()}×{image.height()})")
        else:
            QtWidgets.QMessageBox.critical(self, "Save image", f"Could not write {path}")

    # ======================================================== drag & drop
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        urls = [url for url in event.mimeData().urls() if url.isLocalFile()]
        if urls:
            self.open_path(urls[0].toLocalFile(), auto_render=True)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.render_token += 1
        self.prefetch_token += 1
        for timer in (self.selection_timer, self.iso_timer, self.compute_timer, self.molecule_timer, self.prefetch_timer):
            timer.stop()
        self.foreground.shutdown()
        self.background.shutdown()
        super().closeEvent(event)
