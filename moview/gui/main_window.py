from __future__ import annotations

import html
import math
import re
import sys
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..analysis import compute_bonds
from ..config import AppConfig, ColorPreset, DEFAULT_CONFIG
from ..constants import DEFAULT_ISOVALUE, HARTREE_TO_EV, atom_symbol
from ..grid import (
    BasisGrid,
    OrbitalGrid,
    compute_basis_grid,
    compute_orbital_grids_float32,
    compute_orbital_grids_from_basis,
    estimate_basis_grid_bytes,
    make_grid_spec,
)
from ..parsers import parse_wavefunction
from ..surface import SurfaceMesh, extract_isosurfaces
from ..wavefunction import Wavefunction
from .gl_view import (
    CORE_PREFETCH_OCCUPIED_BACK,
    CORE_PREFETCH_VIRTUAL_FORWARD,
    LOW_PREFETCH_GRID,
    MAX_COMPARE_ORBITALS,
    PREFETCH_BATCH_SIZE,
    QtCore,
    QtGui,
    QtWidgets,
    VMD_ROTATE_DEG_PER_PIXEL,
    _require_gui_dependencies,
    atom_label_item,
    axis_rotation_matrix,
    build_molecule_geometry,
    default_scene_rotation,
    grid_box_points,
    mesh_item,
    MoleculeGeometry,
    molecule_mesh_item,
    remove_gl_item,
    pg,
    update_fog_shader_params,
)
from .layout import MainWindowLayoutMixin
from .presentation import (
    GRID_PREFETCH_DEBOUNCE_MS,
    HIGH_GRID_WARNING_THRESHOLD,
    high_grid_warning_text,
)
from .widgets import SceneSlot


class OpenGLViewer(MainWindowLayoutMixin, QtWidgets.QMainWindow):
    def __init__(
        self,
        input_path: str | None,
        default_grid: int,
        default_iso: float,
        default_margin: float,
        prefetch_workers: int,
        auto_render: bool,
        file_format: str | None = None,
        app_config: AppConfig | None = None,
    ):
        super().__init__()
        self.setWindowTitle("MOview - Molecular Orbital Viewer")
        self.resize(1320, 860)
        self.setMinimumSize(1040, 680)

        self.app_config = DEFAULT_CONFIG if app_config is None else app_config
        resources = self.app_config.resources
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.background_jobs = resources.background_jobs
        self.prefetch_executor = ThreadPoolExecutor(max_workers=self.background_jobs)
        self.wf: Wavefunction | None = None
        self.wavefunction_generation = 0
        self.bonds: list[tuple[int, int]] = []
        self.basis_cache: OrderedDict[tuple[int, int, float], BasisGrid] = OrderedDict()
        self.basis_cache_limit_bytes = resources.basis_cache_bytes
        self.max_basis_cache_entry_bytes = resources.max_basis_cache_entry_bytes
        self.basis_cache_lock = threading.RLock()
        self.grid_cache: OrbitalGrid | None = None
        self.render_cache: OrderedDict[
            tuple[str, int, int, float], tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]
        ] = OrderedDict()
        self.cache_limit = resources.render_cache_entries
        self.render_cache_limit_bytes = resources.render_cache_bytes
        self.prefetch_field_budget_bytes = resources.prefetch_field_budget_bytes
        self.max_prefetch_orbitals = resources.max_prefetch_orbitals
        self.grid_chunk_points = resources.grid_chunk_points
        self.surface_face_limit = resources.surface_face_limit
        self.default_zoom = self.app_config.view.zoom
        self.molecule_geometry_cache: dict[tuple[str, float], MoleculeGeometry] = {}
        self.acknowledged_high_grids: set[int] = set()
        self.loading = False
        self.job_token = 0
        self.iso_timer: QtCore.QTimer | None = None
        self.orbital_timer: QtCore.QTimer | None = None
        self.appearance_timer: QtCore.QTimer | None = None
        self.prefetch_restart_timer = QtCore.QTimer(self)
        self.prefetch_restart_timer.setSingleShot(True)
        self.prefetch_restart_timer.timeout.connect(
            lambda: self.restart_prefetch_after_settings_change()
        )
        self.future: Future | None = None
        self.prefetch_futures: dict[Future, tuple[tuple[int, int], ...]] = {}
        self.prefetch_queue: list[tuple[int, int]] = []
        self.prefetch_token = 0
        self.prefetch_total = 0
        self.prefetch_done = 0
        self.prefetch_spin: str | None = None
        self.prefetch_grid_size: int | None = None
        self.prefetch_margin: float | None = None
        self.prefetch_iso: float | None = None
        self.slots: list[SceneSlot] = []
        self.visible_slot_count = 0
        self.primary_slot: SceneSlot | None = None
        self.active_slot: SceneSlot | None = None
        self.center_pick_mode = False
        self.iso_upper = max(0.2, DEFAULT_ISOVALUE * 4.0)
        self.current_iso = DEFAULT_ISOVALUE
        self.shortcuts: list[QtGui.QShortcut] = []

        self._build_ui(default_grid, default_iso, default_margin)
        self._install_shortcuts()
        if input_path:
            QtCore.QTimer.singleShot(
                80,
                lambda: self.load_wavefunction(input_path, auto_render=auto_render, file_format=file_format),
            )

    def comparison_positions(self, count: int) -> list[tuple[int, int, int, int]]:
        if count <= 1:
            return [(0, 0, 1, 1)]
        if count == 2:
            return [(0, 0, 1, 1), (0, 1, 1, 1)]
        if count == 3:
            return [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)]
        cols = 2 if count <= 4 else 3
        return [(idx // cols, idx % cols, 1, 1) for idx in range(count)]

    def configure_scene_slots(self, count: int) -> None:
        count = max(1, min(MAX_COMPARE_ORBITALS, int(count)))
        self.clear_all_gl_items()
        self.scene_grid_widget.setUpdatesEnabled(False)
        while len(self.slots) < count:
            slot = SceneSlot(self, len(self.slots))
            slot.view_zoom = self.default_zoom
            self.slots.append(slot)
        try:
            while self.scene_grid.count():
                item = self.scene_grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            for idx, slot in enumerate(self.slots):
                slot.frame.setVisible(idx < count)
                slot.title_label.setVisible(idx < count and count > 1)
            positions = self.comparison_positions(count)
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            for slot, (row, col, row_span, col_span) in zip(self.slots[:count], positions):
                self.scene_grid.addWidget(slot.frame, row, col, row_span, col_span)
                used_rows.update(range(row, row + row_span))
                used_cols.update(range(col, col + col_span))
            for row in range(3):
                self.scene_grid.setRowStretch(row, 1 if row in used_rows else 0)
                self.scene_grid.setRowMinimumHeight(row, 0)
            for col in range(3):
                self.scene_grid.setColumnStretch(col, 1 if col in used_cols else 0)
                self.scene_grid.setColumnMinimumWidth(col, 0)
        finally:
            self.scene_grid_widget.setUpdatesEnabled(True)
        self.visible_slot_count = count
        self.primary_slot = self.slots[0]
        if self.active_slot not in self.slots[:count]:
            self.active_slot = self.primary_slot
        self.view_host = self.primary_slot.view_host
        self.view = self.primary_slot.view
        self.corner_view = self.primary_slot.corner_view

    def clear_all_gl_items(self) -> None:
        for slot in self.slots:
            self.clear_scene(slot)
            self.clear_corner(slot)

    def visible_slots(self) -> list[SceneSlot]:
        return self.slots[: self.visible_slot_count]

    def slot_or_primary(self, slot: SceneSlot | None = None) -> SceneSlot:
        if slot is not None:
            return slot
        if self.active_slot is not None:
            return self.active_slot
        assert self.primary_slot is not None
        return self.primary_slot

    def set_active_slot(self, slot: SceneSlot | None) -> None:
        if slot is not None:
            self.active_slot = slot

    def sync_rotation_enabled(self) -> bool:
        return getattr(self, "sync_views_check", None) is not None and self.sync_views_check.isChecked()

    def target_slots(self, slot: SceneSlot | None = None) -> list[SceneSlot]:
        if self.sync_rotation_enabled():
            return self.visible_slots()
        return [self.slot_or_primary(slot)]

    def on_sync_views_changed(self, _state: int) -> None:
        if not self.sync_rotation_enabled() or not self.visible_slots():
            return
        reference = self.visible_slots()[0]
        for slot in self.visible_slots()[1:]:
            slot.scene_rotation = reference.scene_rotation.copy()
            slot.view_zoom = reference.view_zoom
            slot.view_center = None if reference.view_center is None else reference.view_center.copy()
        self.update_scene_transform()

    def current_surface_style(self) -> str:
        return str(self.surface_style_combo.currentData() or self.app_config.render.surface_style)

    def current_positive_color(self) -> ColorPreset:
        name = str(
            self.positive_color_combo.currentData() or self.app_config.render.positive_color
        )
        return self.app_config.color(name)

    def current_negative_color(self) -> ColorPreset:
        name = str(
            self.negative_color_combo.currentData() or self.app_config.render.negative_color
        )
        return self.app_config.color(name)

    def current_atom_style(self) -> str:
        return str(self.atom_style_combo.currentData() or self.app_config.render.atom_style)

    def current_atom_scale(self) -> float:
        return self.atom_scale_control.value() / 100.0

    def current_atom_label_mode(self) -> str:
        return str(self.atom_label_combo.currentData() or self.app_config.render.label_mode)

    def current_atom_label_placement(self) -> str:
        return str(
            self.atom_label_placement_combo.currentData()
            or self.app_config.render.label_placement
        )

    def current_label_size(self) -> int:
        return self.label_size_control.value()

    def on_surface_style_changed(self, _index: int) -> None:
        for slot in self.visible_slots():
            self.rebuild_surface_items(slot)
            self.update_scene_transform(slot)

    def on_surface_color_changed(self, _index: int) -> None:
        for slot in self.visible_slots():
            self.rebuild_surface_items(slot)
            self.update_scene_transform(slot)
        self.refresh_scene_header()

    def schedule_molecule_refresh(self, _value: int = 0) -> None:
        if self.wf is None:
            return
        if self.appearance_timer is not None:
            self.appearance_timer.stop()
        self.appearance_timer = QtCore.QTimer(self)
        self.appearance_timer.setSingleShot(True)
        self.appearance_timer.timeout.connect(self.refresh_molecule_models)
        self.appearance_timer.start(60)

    def refresh_molecule_models(self) -> None:
        if self.wf is None:
            return
        self.molecule_geometry_cache.clear()
        for slot in self.visible_slots():
            for item in slot.molecule_items:
                try:
                    remove_gl_item(slot.view, item)
                except Exception:
                    continue
            slot.molecule_items = []
            self.ensure_molecule_model(slot)
            self.update_scene_transform(slot)

    def _install_shortcuts(self) -> None:
        bindings = [
            ("A", lambda: self.select_relative_orbital(-1)),
            ("D", lambda: self.select_relative_orbital(1)),
            ("C", self.enable_center_pick),
            ("Left", lambda: self.rotate_shortcut("y", -7.0)),
            ("Right", lambda: self.rotate_shortcut("y", 7.0)),
            ("Up", lambda: self.rotate_shortcut("x", 6.0)),
            ("Down", lambda: self.rotate_shortcut("x", -6.0)),
            ("+", lambda: self.zoom_view(1.15)),
            ("=", lambda: self.zoom_view(1.15)),
            ("-", lambda: self.zoom_view(1.0 / 1.15)),
        ]
        for sequence, callback in bindings:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(lambda cb=callback: self.trigger_shortcut(cb))
            self.shortcuts.append(shortcut)

    def focus_accepts_text(self, focus_widget=None) -> bool:
        widget = focus_widget if focus_widget is not None else QtWidgets.QApplication.focusWidget()
        protected_types = (
            QtWidgets.QAbstractItemView,
            QtWidgets.QAbstractSlider,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QComboBox,
            QtWidgets.QLineEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
        )
        while widget is not None:
            if isinstance(widget, protected_types):
                return True
            widget = widget.parentWidget()
        return False

    def trigger_shortcut(self, callback) -> None:
        if self.loading or self.focus_accepts_text():
            return
        callback()

    def rotate_shortcut(self, axis: str, angle_deg: float) -> None:
        self.apply_scene_rotation(angle_deg, axis)
        self.update_scene_transform()

    def open_file(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open wavefunction file",
            str(Path.cwd()),
            "Wavefunction files (*.fchk *.fch *.molden *.molden.input *.input);;"
            "Gaussian fchk (*.fchk *.fch);;Molden (*.molden *.molden.input *.input);;All files (*)",
        )
        if path:
            self.load_wavefunction(path, auto_render=True)

    def load_wavefunction(self, path: str, auto_render: bool, file_format: str | None = None) -> None:
        input_path = Path(path)
        self.loading = True
        self.wavefunction_generation += 1
        self.cancel_prefetch_work()
        self.prefetch_restart_timer.stop()
        if self.iso_timer is not None:
            self.iso_timer.stop()
        if self.orbital_timer is not None:
            self.orbital_timer.stop()
        for widget in (
            self.open_button,
            self.render_button,
            self.settings_tabs,
            self.homo_button,
            self.lumo_button,
            self.compare_entry,
            self.tree,
        ):
            widget.setEnabled(False)

        def parse_job() -> tuple[Wavefunction, list[tuple[int, int]]]:
            wavefunction = parse_wavefunction(input_path, file_format)
            bonds = compute_bonds(
                wavefunction.atomic_numbers,
                wavefunction.coordinates_angstrom,
            )
            return wavefunction, bonds

        self.submit_job(
            "Parsing wavefunction file...",
            parse_job,
            lambda result: self._finish_wavefunction_load(result, auto_render),
            error_title="Failed to load wavefunction file",
            error_status="Load failed",
            on_error=self._restore_load_controls,
        )

    def _restore_load_controls(self) -> None:
        self.loading = False
        has_wavefunction = self.wf is not None
        self.open_button.setEnabled(True)
        self.render_button.setEnabled(has_wavefunction)
        self.settings_tabs.setEnabled(True)
        for widget in (
            self.homo_button,
            self.lumo_button,
            self.compare_entry,
            self.tree,
        ):
            widget.setEnabled(has_wavefunction)

    def _finish_wavefunction_load(
        self,
        result: tuple[Wavefunction, list[tuple[int, int]]],
        auto_render: bool,
    ) -> None:
        self.wf, self.bonds = result
        self.grid_cache = None
        self.render_cache.clear()
        self.molecule_geometry_cache.clear()
        self.acknowledged_high_grids.clear()
        self.cancel_prefetch_work()
        self.clear_scene()
        for slot in self.slots:
            slot.reset_state()
            slot.view_zoom = self.default_zoom
        self.configure_scene_slots(1)
        self.zoom_control.setValueQuietly(int(round(self.default_zoom * 100.0)))
        self.update_base_view(self.wf.coordinates_angstrom, self.primary_slot)
        self.file_label.setText(
            f"{self.wf.path.name} ({self.wf.source_format})\n{self.wf.title}"
        )
        self.atom_label.setText(f"Atoms: {len(self.wf.atomic_numbers)}")
        self.basis_label.setText(f"Basis: {self.wf.n_basis}")
        self.spin_combo.blockSignals(True)
        self.spin_combo.clear()
        self.spin_combo.addItems(["alpha", "beta"] if self.wf.is_unrestricted else ["alpha"])
        self.spin_combo.setCurrentText("alpha")
        self.spin_combo.blockSignals(False)
        self.populate_orbitals()
        self._restore_load_controls()
        self.set_ready("Loaded. Double-click an orbital or press Render.")
        if auto_render:
            QtCore.QTimer.singleShot(120, self.render_selected)

    def populate_orbitals(self) -> None:
        if self.wf is None:
            return
        spin = self.spin_combo.currentText()
        self.tree.clear()
        for idx, energy in enumerate(self.wf.energies(spin)):
            occ = self.wf.occupation(spin, idx)
            eh = "nan" if not np.isfinite(energy) else f"{energy:.6f}"
            ev = "nan" if not np.isfinite(energy) else f"{energy * HARTREE_TO_EV:.3f}"
            item = QtWidgets.QTreeWidgetItem([str(idx + 1), f"{occ:g}", eh, ev])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, idx)
            self.tree.addTopLevelItem(item)
        self.select_orbital(self.wf.default_orbital(spin))
        self.grid_cache = None
        self.cancel_prefetch_work()
        self.draw_scene()
        QtCore.QTimer.singleShot(500, self.start_prefetch_common_orbitals)

    def selected_orbital(self) -> int | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        value = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        return int(value)

    def select_orbital(self, idx: int) -> None:
        if self.wf is None:
            return
        idx = max(0, min(idx, len(self.wf.energies(self.spin_combo.currentText())) - 1))
        item = self.tree.topLevelItem(idx)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        self.update_orbital_info()

    def select_frontier(self, which: str) -> None:
        if self.wf is None:
            return
        spin = self.spin_combo.currentText()
        idx = self.wf.default_orbital(spin) if which == "homo" else self.wf.lumo_orbital(spin)
        self.select_orbital(idx)
        self.render_selected()

    def update_orbital_info(self) -> None:
        if self.wf is None:
            self.orbital_info_label.setText("Select an orbital")
            return
        idx = self.selected_orbital()
        if idx is None:
            return
        spin = self.spin_combo.currentText()
        energy = self.wf.energies(spin)[idx]
        occ = self.wf.occupation(spin, idx)
        energy_text = "nan" if not np.isfinite(energy) else f"{energy:.6f} Eh / {energy * HARTREE_TO_EV:.3f} eV"
        self.orbital_info_label.setText(f"{spin} MO {idx + 1}: occ {occ:g}, E {energy_text}")

    def parse_compare_indices(self) -> list[int] | None:
        if self.wf is None:
            return None
        spin = self.spin_combo.currentText()
        n_orb = len(self.wf.energies(spin))
        text = self.compare_entry.text().strip()
        if not text:
            idx = self.selected_orbital()
            return [] if idx is None else [idx]
        indices: list[int] = []
        for token in re.split(r"[\s,;]+", text):
            if not token:
                continue
            if re.fullmatch(r"\d+\s*-\s*\d+", token):
                start_text, end_text = re.split(r"\s*-\s*", token)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                indices.extend(range(start - 1, end - 1 + step, step))
            else:
                try:
                    indices.append(int(token) - 1)
                except ValueError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Invalid comparison list",
                        f"Invalid orbital token: {token!r}",
                    )
                    return None
        deduped: list[int] = []
        for idx in indices:
            if idx < 0 or idx >= n_orb:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid comparison list",
                    f"Orbital {idx + 1} is outside 1..{n_orb}.",
                )
                return None
            if idx not in deduped:
                deduped.append(idx)
        if len(deduped) > MAX_COMPARE_ORBITALS:
            QtWidgets.QMessageBox.critical(
                self,
                "Too many orbitals",
                f"Compare at most {MAX_COMPARE_ORBITALS} orbitals at once.",
            )
            return None
        return deduped

    def core_prefetch_bounds(self, spin: str) -> tuple[int, int]:
        assert self.wf is not None
        homo = self.wf.default_orbital(spin)
        lumo = self.wf.lumo_orbital(spin)
        n_orb = len(self.wf.energies(spin))
        return max(0, homo - CORE_PREFETCH_OCCUPIED_BACK), min(n_orb - 1, lumo + CORE_PREFETCH_VIRTUAL_FORWARD)

    def display_grid_for_orbital(self, spin: str, idx: int) -> int:
        return int(self.grid_spin.value())

    def schedule_prefetch_for_compute_change(self, _value: object = None) -> None:
        self.cancel_prefetch_work()
        if self.wf is None or self.loading:
            return
        grid_size = int(self.grid_spin.value())
        if (
            grid_size > HIGH_GRID_WARNING_THRESHOLD
            and grid_size not in self.acknowledged_high_grids
        ):
            self.set_ready(
                f"Grid {grid_size} selected. Performance confirmation is required before pre-rendering."
            )
        else:
            self.set_ready("Grid settings changed. Pre-render will restart shortly.")
        self.arm_prefetch_restart()

    def arm_prefetch_restart(self) -> None:
        self.prefetch_restart_timer.stop()
        if self.wf is None or self.loading:
            return
        self.prefetch_restart_timer.start(GRID_PREFETCH_DEBOUNCE_MS)

    def restart_prefetch_after_settings_change(self) -> None:
        if self.wf is None or self.loading:
            return
        grid_size = int(self.grid_spin.value())
        if not self.confirm_grid_performance(grid_size):
            return
        self.start_prefetch_common_orbitals()

    def confirm_grid_performance(self, grid_size: int) -> bool:
        if self.wf is None or grid_size <= HIGH_GRID_WARNING_THRESHOLD:
            return True
        if grid_size in self.acknowledged_high_grids:
            return True
        spec = make_grid_spec(self.wf, grid_size, float(self.margin_spin.value()))
        choice = QtWidgets.QMessageBox.warning(
            self,
            "High resolution grid",
            high_grid_warning_text(spec),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if choice != QtWidgets.QMessageBox.StandardButton.Yes:
            self.set_ready("High resolution render cancelled.")
            return False
        self.acknowledged_high_grids.add(grid_size)
        return True

    def render_selected(self) -> None:
        if self.loading or self.wf is None:
            return
        if not self.confirm_grid_performance(int(self.grid_spin.value())):
            return
        indices = self.parse_compare_indices()
        if not indices:
            return
        self.render_orbitals(indices)

    def render_orbitals(self, indices: list[int]) -> None:
        assert self.wf is not None
        wavefunction = self.wf
        wavefunction_generation = self.wavefunction_generation
        margin = float(self.margin_spin.value())
        spin = self.spin_combo.currentText()
        iso = self.iso_value()
        self.configure_scene_slots(len(indices))
        for slot, idx in zip(self.visible_slots(), indices):
            slot.orbital_index0 = idx
            slot.title_label.setText(f"{spin} MO {idx + 1} | loading")
            self.clear_surfaces(slot)
            self.ensure_molecule_model(slot)
            self.update_scene_transform(slot)
        self.grid_cache = None
        ready: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        preview_ready: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        extract_from_cache: list[tuple[int, OrbitalGrid]] = []
        compute_specs: list[tuple[int, int, int]] = []
        for slot, idx in zip(self.visible_slots(), indices):
            grid_size = self.display_grid_for_orbital(spin, idx)
            key = self.cache_key(spin, idx, grid_size, margin)
            cached = self.render_cache.get(key)
            if cached is not None:
                self.render_cache.move_to_end(key)
                cached_grid, _cached_pos, _cached_neg, cached_level = cached
                if math.isclose(
                    iso,
                    cached_level,
                    rel_tol=1.0e-8,
                    abs_tol=1.0e-10,
                ):
                    ready.append((slot.index, cached))
                else:
                    extract_from_cache.append((slot.index, cached_grid))
                continue
            low_key = self.cache_key(spin, idx, LOW_PREFETCH_GRID, margin)
            low_cached = self.render_cache.get(low_key)
            if low_cached is not None and LOW_PREFETCH_GRID < grid_size:
                self.render_cache.move_to_end(low_key)
                low_grid, _low_pos, _low_neg, low_level = low_cached
                if math.isclose(
                    iso,
                    low_level,
                    rel_tol=1.0e-8,
                    abs_tol=1.0e-10,
                ):
                    preview_ready.append((slot.index, low_cached))
                else:
                    preview_ready.append((slot.index, self._extract_from_cached_grid(low_grid, iso)))
            compute_specs.append((slot.index, idx, grid_size))
        if not extract_from_cache and not compute_specs:
            self._finish_multi_render(ready, from_cache=True)
            return
        if preview_ready:
            self._finish_multi_render(ready + preview_ready, from_cache=True, restart_prefetch=False)
        self.submit_job(
            f"Rendering {len(indices)} orbital view(s)...",
            lambda: self._compute_multi_job(
                wavefunction,
                wavefunction_generation,
                spin,
                extract_from_cache,
                compute_specs,
                margin,
                iso,
            ),
            lambda computed: self._finish_multi_render(ready + computed, from_cache=not compute_specs),
        )

    def _extract_from_cached_grid(
        self,
        grid: OrbitalGrid,
        iso: float,
    ) -> tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]:
        pos, neg, level = extract_isosurfaces(grid, iso)
        return grid, pos, neg, level

    def _compute_multi_job(
        self,
        wavefunction: Wavefunction,
        wavefunction_generation: int,
        spin: str,
        extract_from_cache: list[tuple[int, OrbitalGrid]],
        compute_specs: list[tuple[int, int, int]],
        margin: float,
        iso: float,
    ) -> list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]]:
        out: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        for slot_index, grid in extract_from_cache:
            pos, neg, level = extract_isosurfaces(grid, iso)
            out.append((slot_index, (grid, pos, neg, level)))
        by_grid: dict[int, list[tuple[int, int]]] = {}
        for slot_index, idx, grid_size in compute_specs:
            by_grid.setdefault(grid_size, []).append((slot_index, idx))
        for grid_size, specs in by_grid.items():
            indices = [idx for _slot_index, idx in specs]
            slot_by_idx = {idx: slot_index for slot_index, idx in specs}
            if self.should_use_basis_grid(grid_size, margin, wavefunction):
                basis_grid = self.get_basis_grid(
                    grid_size,
                    margin,
                    wavefunction,
                    wavefunction_generation,
                )
                grids = compute_orbital_grids_from_basis(
                    wavefunction,
                    spin,
                    indices,
                    basis_grid,
                )
            else:
                grids = compute_orbital_grids_float32(
                    wavefunction,
                    spin,
                    indices,
                    grid_size,
                    margin,
                    chunk_points=self.grid_chunk_points,
                )
            for grid in grids:
                pos, neg, level = extract_isosurfaces(grid, iso)
                out.append((slot_by_idx[grid.orbital_index0], (grid, pos, neg, level)))
        return out

    def _finish_multi_render(
        self,
        results: list[tuple[int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]],
        from_cache: bool = False,
        restart_prefetch: bool = True,
    ) -> None:
        if not results:
            self.set_ready("No orbital rendered.")
            return
        results = sorted(results, key=lambda item: item[0])
        first_grid: OrbitalGrid | None = None
        total_pos = 0
        total_neg = 0
        for slot_index, result in results:
            if slot_index >= len(self.slots):
                continue
            grid, pos, neg, level = result
            slot = self.slots[slot_index]
            self.touch_cache(self.cache_key(grid.spin, grid.orbital_index0, grid.grid_size, grid.margin_bohr), result)
            self.draw_scene(pos, neg, grid, level, slot=slot)
            total_pos += pos.n_faces
            total_neg += neg.n_faces
            if first_grid is None:
                first_grid = grid
                self.grid_cache = grid
                self.update_iso_scale(grid, level)
        n_rendered = len(results)
        if n_rendered == 1 and first_grid is not None:
            prefix = "Cached" if from_cache else "Rendered"
            self.set_ready(f"{prefix} {total_pos} positive and {total_neg} negative triangles.")
        else:
            prefix = "Cached" if from_cache else "Rendered"
            self.set_ready(f"{prefix} {n_rendered} orbital views.")
        self.refresh_scene_header()
        self.slot_or_primary().view.setFocus()
        if restart_prefetch:
            QtCore.QTimer.singleShot(1200, self.start_prefetch_common_orbitals)

    def update_iso_scale(self, grid: OrbitalGrid, level: float | None = None) -> None:
        vmax = float(np.nanmax(np.abs(grid.values)))
        target = self.iso_value() if level is None else float(level)
        self.iso_upper = max(target * 4.0, vmax * 0.75, DEFAULT_ISOVALUE * 4.0)
        self.set_iso_slider_value(target, update_text=True)

    def iso_value(self) -> float:
        return self.current_iso

    def set_iso_slider_value(self, value: float, update_text: bool) -> None:
        value = max(1.0e-8, min(float(value), self.iso_upper))
        self.current_iso = value
        slider_value = max(1, min(1000, int(round(value / self.iso_upper * 1000.0))))
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(slider_value)
        self.iso_slider.blockSignals(False)
        if update_text:
            self.iso_entry.setText(f"{value:.5f}")

    def on_iso_drag(self, value: int) -> None:
        self.current_iso = max(1.0e-8, float(value) / 1000.0 * self.iso_upper)
        self.iso_entry.setText(f"{self.iso_value():.5f}")
        self.cancel_prefetch_work()
        self.arm_prefetch_restart()
        if self.grid_cache is None:
            return
        if self.iso_timer is not None:
            self.iso_timer.stop()
        self.iso_timer = QtCore.QTimer(self)
        self.iso_timer.setSingleShot(True)
        self.iso_timer.timeout.connect(self.render_cached_iso)
        self.iso_timer.start(260)

    def apply_iso_entry(self) -> None:
        try:
            value = float(self.iso_entry.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Invalid isovalue", "Isovalue must be numeric.")
            return
        if not math.isfinite(value) or value <= 0.0:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid isovalue",
                "Isovalue must be a finite positive number.",
            )
            return
        if value > self.iso_upper:
            self.iso_upper = value * 4.0
        self.cancel_prefetch_work()
        self.set_iso_slider_value(value, update_text=True)
        self.arm_prefetch_restart()
        self.render_cached_iso()

    def render_cached_iso(self) -> None:
        if self.wf is None:
            return
        wavefunction = self.wf
        wavefunction_generation = self.wavefunction_generation
        cached_slots = [(slot.index, slot.grid) for slot in self.visible_slots() if slot.grid is not None]
        if len(cached_slots) > 1:
            level = self.iso_value()
            self.submit_job(
                "Updating comparison isosurfaces...",
                lambda: self._compute_multi_job(
                    wavefunction,
                    wavefunction_generation,
                    self.spin_combo.currentText(),
                    cached_slots,
                    [],
                    float(self.margin_spin.value()),
                    level,
                ),
                lambda results: self._finish_multi_render(results, from_cache=True),
                show_progress=False,
            )
            return
        if self.grid_cache is None:
            return
        grid = self.grid_cache
        level = self.iso_value()
        self.submit_job(
            "Updating isosurface...",
            lambda: (*extract_isosurfaces(grid, level), grid),
            self._finish_cached_render,
            show_progress=False,
        )

    def _finish_cached_render(self, result: tuple[SurfaceMesh, SurfaceMesh, float, OrbitalGrid]) -> None:
        pos, neg, level, grid = result
        self.set_iso_slider_value(level, update_text=True)
        self.touch_cache(
            self.cache_key(grid.spin, grid.orbital_index0, grid.grid_size, grid.margin_bohr),
            (grid, pos, neg, level),
        )
        self.draw_scene(pos, neg, grid, level)
        self.set_ready(f"Rendered {pos.n_faces} positive and {neg.n_faces} negative triangles.")
        QtCore.QTimer.singleShot(1200, self.start_prefetch_common_orbitals)

    def submit_job(
        self,
        status: str,
        work,
        finish,
        show_progress: bool = True,
        *,
        error_title: str = "Render failed",
        error_status: str = "Render failed",
        on_error=None,
    ) -> None:
        self.job_token += 1
        token = self.job_token
        self.status_label.setText(status)
        if show_progress:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Rendering...")
        if self.future is not None and not self.future.done():
            self.future.cancel()
        future = self.executor.submit(work)
        self.future = future
        QtCore.QTimer.singleShot(
            80,
            lambda: self.poll_future(
                token,
                future,
                finish,
                error_title,
                error_status,
                on_error,
            ),
        )

    def poll_future(
        self,
        token: int,
        future: Future,
        finish,
        error_title: str = "Render failed",
        error_status: str = "Render failed",
        on_error=None,
    ) -> None:
        if not future.done():
            QtCore.QTimer.singleShot(
                80,
                lambda: self.poll_future(
                    token,
                    future,
                    finish,
                    error_title,
                    error_status,
                    on_error,
                ),
            )
            return
        if token != self.job_token:
            return
        self.progress.setRange(0, 1)
        self.progress.setFormat("")
        self.progress.setValue(0)
        try:
            finish(future.result())
        except Exception as exc:
            self.set_ready(error_status)
            if on_error is not None:
                on_error()
            QtWidgets.QMessageBox.critical(self, error_title, str(exc))

    def basis_key(
        self,
        wavefunction_generation: int,
        grid_size: int,
        margin: float,
    ) -> tuple[int, int, float]:
        return (int(wavefunction_generation), int(grid_size), round(float(margin), 6))

    def basis_cache_bytes_locked(self) -> int:
        return sum(grid.nbytes for grid in self.basis_cache.values())

    def prune_basis_cache_locked(self) -> None:
        while self.basis_cache and self.basis_cache_bytes_locked() > self.basis_cache_limit_bytes:
            self.basis_cache.popitem(last=False)

    def should_use_basis_grid(
        self,
        grid_size: int,
        margin: float,
        wavefunction: Wavefunction | None = None,
    ) -> bool:
        wavefunction = self.wf if wavefunction is None else wavefunction
        assert wavefunction is not None
        return (
            estimate_basis_grid_bytes(wavefunction, grid_size, margin)
            <= self.max_basis_cache_entry_bytes
        )

    def get_basis_grid(
        self,
        grid_size: int,
        margin: float,
        wavefunction: Wavefunction | None = None,
        wavefunction_generation: int | None = None,
    ) -> BasisGrid:
        wavefunction = self.wf if wavefunction is None else wavefunction
        if wavefunction_generation is None:
            wavefunction_generation = self.wavefunction_generation
        assert wavefunction is not None
        if not self.should_use_basis_grid(grid_size, margin, wavefunction):
            raise MemoryError(
                "Requested basis grid exceeds the in-memory cache budget; use chunked evaluation"
            )
        key = self.basis_key(wavefunction_generation, grid_size, margin)
        with self.basis_cache_lock:
            if wavefunction_generation == self.wavefunction_generation:
                stale_keys = [entry for entry in self.basis_cache if entry[0] != wavefunction_generation]
                for stale_key in stale_keys:
                    self.basis_cache.pop(stale_key, None)
            cached = self.basis_cache.get(key)
            if cached is not None:
                self.basis_cache.move_to_end(key)
                return cached
            basis_grid = compute_basis_grid(
                wavefunction,
                int(grid_size),
                float(margin),
                workers=max(1, self.prefetch_workers),
            )
            if wavefunction_generation == self.wavefunction_generation:
                self.basis_cache[key] = basis_grid
                self.basis_cache.move_to_end(key)
                self.prune_basis_cache_locked()
            return basis_grid

    def cache_key(self, spin: str, idx: int, grid_size: int, margin: float) -> tuple[str, int, int, float]:
        return (spin, idx, int(grid_size), round(float(margin), 6))

    def touch_cache(
        self,
        key: tuple[str, int, int, float],
        value: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float],
    ) -> None:
        if self.cache_entry_nbytes(value) > self.render_cache_limit_bytes:
            self.render_cache.pop(key, None)
            return
        self.render_cache[key] = value
        self.render_cache.move_to_end(key)
        while self.render_cache and (
            len(self.render_cache) > self.cache_limit
            or self.render_cache_nbytes() > self.render_cache_limit_bytes
        ):
            self.render_cache.popitem(last=False)

    @staticmethod
    def cache_entry_nbytes(
        value: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float],
    ) -> int:
        grid, pos, neg, _level = value
        return int(
            grid.values.nbytes
            + pos.vertices.nbytes
            + pos.faces.nbytes
            + neg.vertices.nbytes
            + neg.faces.nbytes
        )

    def render_cache_nbytes(self) -> int:
        return sum(self.cache_entry_nbytes(value) for value in self.render_cache.values())

    def cached_surface_matches(
        self,
        key: tuple[str, int, int, float],
        level: float,
    ) -> bool:
        cached = self.render_cache.get(key)
        return cached is not None and math.isclose(
            cached[3],
            level,
            rel_tol=1.0e-8,
            abs_tol=1.0e-10,
        )

    def cancel_prefetch_work(self) -> None:
        self.prefetch_token += 1
        self.prefetch_queue.clear()
        self.prefetch_total = 0
        self.prefetch_done = 0
        self.prefetch_spin = None
        self.prefetch_grid_size = None
        self.prefetch_margin = None
        self.prefetch_iso = None
        for future in list(self.prefetch_futures):
            future.cancel()
        self.prefetch_futures.clear()

    def homo_centered_prefetch_order(self, homo: int, n_orb: int) -> list[int]:
        indices: list[int] = [homo]
        max_delta = max(homo, n_orb - 1 - homo)
        for delta in range(1, max_delta + 1):
            occupied_idx = homo - delta
            virtual_idx = homo + delta
            if occupied_idx >= 0:
                indices.append(occupied_idx)
            if virtual_idx < n_orb:
                indices.append(virtual_idx)
        return indices

    def prefetch_tasks(
        self,
        spin: str,
        homo: int,
        n_orb: int,
        core_grid_size: int,
        max_tasks: int | None = None,
    ) -> list[tuple[int, int]]:
        low_idx, high_idx = self.core_prefetch_bounds(spin)
        order = self.homo_centered_prefetch_order(homo, n_orb)
        selected = self.selected_orbital()
        if selected is not None and low_idx <= selected <= high_idx:
            order = [selected] + [idx for idx in order if idx != selected]
        tasks = [(idx, int(core_grid_size)) for idx in order if low_idx <= idx <= high_idx]
        limit = self.max_prefetch_orbitals if max_tasks is None else max(0, int(max_tasks))
        return tasks[:limit]

    def prefetch_task_limit(self, grid_size: int, margin: float) -> int:
        assert self.wf is not None
        field_bytes = make_grid_spec(self.wf, grid_size, margin).scalar_nbytes(np.float32)
        if field_bytes > self.prefetch_field_budget_bytes:
            return 0
        return min(
            self.max_prefetch_orbitals,
            max(1, self.prefetch_field_budget_bytes // max(1, field_bytes)),
        )

    def update_prefetch_progress(self, text: str | None = None) -> None:
        if self.prefetch_total <= 0:
            return
        self.progress.setRange(0, self.prefetch_total)
        self.progress.setFormat("Pre-render %v/%m")
        self.progress.setValue(min(self.prefetch_done, self.prefetch_total))
        if text is not None:
            self.status_label.setText(text)

    def maybe_apply_prefetch_result(
        self,
        spin: str,
        idx: int,
        result_grid_size: int,
        result: tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float],
    ) -> None:
        if self.wf is None or spin != self.spin_combo.currentText():
            return
        desired_grid_size = self.display_grid_for_orbital(spin, idx)
        if result_grid_size > desired_grid_size:
            return
        if result_grid_size != desired_grid_size and result_grid_size != LOW_PREFETCH_GRID:
            return
        grid, pos, neg, level = result
        if not math.isclose(self.iso_value(), level, rel_tol=1.0e-8, abs_tol=1.0e-10):
            return
        updated_primary = False
        for slot in self.visible_slots():
            if slot.orbital_index0 != idx:
                continue
            current_grid_size = slot.grid.grid_size if slot.grid is not None else 0
            if current_grid_size >= result_grid_size:
                continue
            self.draw_scene(pos, neg, grid, level, slot=slot)
            if slot is self.primary_slot:
                updated_primary = True
        if updated_primary:
            self.grid_cache = grid
            self.update_iso_scale(grid, level)

    def start_prefetch_common_orbitals(self) -> None:
        if self.wf is None:
            return
        wavefunction = self.wf
        wavefunction_generation = self.wavefunction_generation
        if self.future is not None and not self.future.done():
            QtCore.QTimer.singleShot(1000, self.start_prefetch_common_orbitals)
            return
        spin = self.spin_combo.currentText()
        grid_size = int(self.grid_spin.value())
        margin = float(self.margin_spin.value())
        iso = self.iso_value()
        if (
            grid_size > HIGH_GRID_WARNING_THRESHOLD
            and grid_size not in self.acknowledged_high_grids
        ):
            return
        if self.prefetch_queue or self.prefetch_futures:
            same_prefetch = (
                self.prefetch_spin == spin
                and self.prefetch_grid_size == grid_size
                and self.prefetch_margin is not None
                and math.isclose(self.prefetch_margin, margin, rel_tol=0.0, abs_tol=1.0e-9)
                and self.prefetch_iso is not None
                and math.isclose(self.prefetch_iso, iso, rel_tol=1.0e-8, abs_tol=1.0e-10)
            )
            if same_prefetch:
                return
            self.cancel_prefetch_work()
        self.prefetch_token += 1
        token = self.prefetch_token
        homo = wavefunction.default_orbital(spin)
        n_orb = len(wavefunction.energies(spin))
        task_limit = self.prefetch_task_limit(grid_size, margin)
        if task_limit <= 0:
            self.set_ready(
                f"Grid {grid_size} exceeds the background pre-render memory budget; "
                "foreground Render remains available."
            )
            return
        tasks = self.prefetch_tasks(spin, homo, n_orb, grid_size, task_limit)
        self.prefetch_queue = [
            task
            for task in tasks
            if not self.cached_surface_matches(
                self.cache_key(spin, task[0], task[1], margin),
                iso,
            )
        ]
        self.prefetch_total = len(self.prefetch_queue)
        self.prefetch_done = 0
        for future in list(self.prefetch_futures):
            future.cancel()
        self.prefetch_futures.clear()
        self.prefetch_spin = spin
        self.prefetch_grid_size = grid_size
        self.prefetch_margin = margin
        self.prefetch_iso = iso
        if self.prefetch_total:
            self.update_prefetch_progress(
                f"Pre-render cache: 0/{self.prefetch_total} queued at grid {grid_size}."
            )
            self.pump_prefetch(
                token,
                wavefunction,
                wavefunction_generation,
                spin,
                grid_size,
                margin,
                iso,
            )
        else:
            self.progress.setRange(0, 1)
            self.progress.setFormat("")
            self.progress.setValue(0)

    def pump_prefetch(
        self,
        token: int,
        wavefunction: Wavefunction,
        wavefunction_generation: int,
        spin: str,
        grid_size: int,
        margin: float,
        iso: float,
    ) -> None:
        if (
            token != self.prefetch_token
            or self.wf is not wavefunction
            or self.wavefunction_generation != wavefunction_generation
        ):
            return
        done_futures = [future for future in self.prefetch_futures if future.done()]
        for future in done_futures:
            try:
                for idx, result_grid_size, result in future.result():
                    key = self.cache_key(spin, idx, result_grid_size, margin)
                    self.touch_cache(key, result)
                    self.prefetch_done += 1
                    self.maybe_apply_prefetch_result(spin, idx, result_grid_size, result)
            except Exception as exc:
                self.prefetch_queue.clear()
                self.set_ready(f"Background pre-render stopped: {exc}")
            self.prefetch_futures.pop(future, None)
            self.update_prefetch_progress(
                f"Pre-render cache: {self.prefetch_done}/{self.prefetch_total} orbitals ready."
            )
        if self.future is not None and not self.future.done():
            QtCore.QTimer.singleShot(
                250,
                lambda: self.pump_prefetch(
                    token,
                    wavefunction,
                    wavefunction_generation,
                    spin,
                    grid_size,
                    margin,
                    iso,
                ),
            )
            return
        while self.prefetch_queue and len(self.prefetch_futures) < self.background_jobs:
            batch_indices: list[tuple[int, int]] = []
            cached_grids: dict[tuple[int, int], OrbitalGrid] = {}
            skipped_cached = False
            while self.prefetch_queue and len(batch_indices) < PREFETCH_BATCH_SIZE:
                task = self.prefetch_queue.pop(0)
                key = self.cache_key(spin, task[0], task[1], margin)
                if self.cached_surface_matches(key, iso):
                    self.prefetch_done += 1
                    skipped_cached = True
                    continue
                cached = self.render_cache.get(key)
                if cached is not None:
                    cached_grids[task] = cached[0]
                batch_indices.append(task)
            if skipped_cached:
                self.update_prefetch_progress(
                    f"Pre-render cache: {self.prefetch_done}/{self.prefetch_total} orbitals ready."
                )
            if not batch_indices:
                continue
            future = self.prefetch_executor.submit(
                self.prefetch_batch,
                spin,
                batch_indices,
                margin,
                iso,
                cached_grids,
                wavefunction,
                wavefunction_generation,
            )
            self.prefetch_futures[future] = tuple(batch_indices)
        if not self.prefetch_queue and not self.prefetch_futures:
            if self.prefetch_total:
                self.update_prefetch_progress(
                    f"Pre-render cache complete: {self.prefetch_done}/{self.prefetch_total} orbitals."
                )
            return
        QtCore.QTimer.singleShot(
            180,
            lambda: self.pump_prefetch(
                token,
                wavefunction,
                wavefunction_generation,
                spin,
                grid_size,
                margin,
                iso,
            ),
        )

    def prefetch_batch(
        self,
        spin: str,
        tasks: list[tuple[int, int]],
        margin: float,
        iso: float,
        cached_grids: dict[tuple[int, int], OrbitalGrid] | None = None,
        wavefunction: Wavefunction | None = None,
        wavefunction_generation: int | None = None,
    ) -> list[tuple[int, int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]]:
        wavefunction = self.wf if wavefunction is None else wavefunction
        if wavefunction_generation is None:
            wavefunction_generation = self.wavefunction_generation
        assert wavefunction is not None
        out: list[tuple[int, int, tuple[OrbitalGrid, SurfaceMesh, SurfaceMesh, float]]] = []
        cached_grids = cached_grids or {}
        by_grid: dict[int, list[int]] = {}
        for idx, grid_size in tasks:
            cached_grid = cached_grids.get((idx, grid_size))
            if cached_grid is not None:
                pos, neg, level = extract_isosurfaces(cached_grid, iso)
                out.append((idx, grid_size, (cached_grid, pos, neg, level)))
            else:
                by_grid.setdefault(grid_size, []).append(idx)
        for grid_size, indices in by_grid.items():
            if self.should_use_basis_grid(grid_size, margin, wavefunction):
                basis_grid = self.get_basis_grid(
                    grid_size,
                    margin,
                    wavefunction,
                    wavefunction_generation,
                )
                grids = compute_orbital_grids_from_basis(
                    wavefunction,
                    spin,
                    indices,
                    basis_grid,
                )
            else:
                grids = compute_orbital_grids_float32(
                    wavefunction,
                    spin,
                    indices,
                    grid_size,
                    margin,
                    chunk_points=self.grid_chunk_points,
                )
            for grid in grids:
                pos, neg, level = extract_isosurfaces(grid, iso)
                out.append((grid.orbital_index0, grid.grid_size, (grid, pos, neg, level)))
        return out

    def clear_scene(self, slot: SceneSlot | None = None) -> None:
        slots = self.slots if slot is None else [slot]
        for scene_slot in slots:
            self.clear_surfaces(scene_slot)
            for item in scene_slot.molecule_items:
                try:
                    remove_gl_item(scene_slot.view, item)
                except Exception:
                    pass
            scene_slot.molecule_items = []

    def clear_surfaces(
        self,
        slot: SceneSlot | None = None,
        *,
        forget_geometry: bool = True,
    ) -> None:
        scene_slot = self.slot_or_primary(slot)
        for item in scene_slot.surface_items:
            try:
                scene_slot.view.removeItem(item)
            except Exception:
                pass
        scene_slot.surface_items = []
        if forget_geometry:
            scene_slot.positive_mesh = None
            scene_slot.negative_mesh = None

    def clear_corner(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        if hasattr(scene_slot.corner_view, "removeItem"):
            for item in scene_slot.corner_items:
                try:
                    scene_slot.corner_view.removeItem(item)
                except Exception:
                    pass
        scene_slot.corner_items = []
        scene_slot.corner_view.update()

    def add_surface_item(self, slot: SceneSlot, item) -> None:
        slot.view.addItem(item)
        slot.surface_items.append(item)

    def add_molecule_item(self, slot: SceneSlot, item) -> None:
        slot.view.addItem(item)
        slot.molecule_items.append(item)

    def rebuild_surface_items(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        self.clear_surfaces(scene_slot, forget_geometry=False)
        style = self.current_surface_style()
        if scene_slot.positive_mesh is not None and scene_slot.positive_mesh.n_faces:
            positive_color = self.current_positive_color()
            self.add_surface_item(
                scene_slot,
                mesh_item(
                    scene_slot.positive_mesh,
                    positive_color.hex_color,
                    style,
                    face_limit=self.surface_face_limit,
                ),
            )
        if scene_slot.negative_mesh is not None and scene_slot.negative_mesh.n_faces:
            negative_color = self.current_negative_color()
            self.add_surface_item(
                scene_slot,
                mesh_item(
                    scene_slot.negative_mesh,
                    negative_color.hex_color,
                    style,
                    face_limit=self.surface_face_limit,
                ),
            )

    def ensure_molecule_model(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        if self.wf is None or scene_slot.molecule_items:
            return
        atom_style = self.current_atom_style()
        atom_scale = self.current_atom_scale()
        geometry_key = (atom_style, round(atom_scale, 4))
        geometry = self.molecule_geometry_cache.get(geometry_key)
        if geometry is None:
            geometry = build_molecule_geometry(
                self.wf,
                self.bonds,
                atom_style,
                atom_scale,
            )
            self.molecule_geometry_cache[geometry_key] = geometry
        self.add_molecule_item(
            scene_slot,
            molecule_mesh_item(
                self.wf,
                self.bonds,
                atom_style,
                atom_scale,
                geometry,
            ),
        )
        labels = atom_label_item(
            self.wf,
            self.current_atom_label_mode(),
            self.current_label_size(),
            placement=self.current_atom_label_placement(),
            atom_style=atom_style,
            atom_scale=atom_scale,
        )
        if labels is not None:
            self.add_molecule_item(scene_slot, labels)

    def phase_legend_html(self) -> str:
        positive = self.current_positive_color()
        negative = self.current_negative_color()
        positive_name = html.escape(positive.name)
        negative_name = html.escape(negative.name)
        return (
            f'<span style="color:{positive.hex_color};">&#9679;</span> '
            f"{positive_name} +psi &nbsp;&middot;&nbsp; "
            f'<span style="color:{negative.hex_color};">&#9679;</span> '
            f"{negative_name} -psi"
        )

    def scene_metadata_html(self, grid: OrbitalGrid, level: float | None) -> str:
        iso_text = "--" if level is None else f"{level:.5g}"
        shape_text = " &times; ".join(str(value) for value in grid.shape)
        return (
            f"Isovalue <b>&plusmn;{iso_text}</b> &nbsp;&middot;&nbsp; "
            f"Grid <b>{shape_text}</b> &nbsp;&middot;&nbsp; {self.phase_legend_html()}"
        )

    def update_scene_header(
        self,
        grid: OrbitalGrid | None = None,
        level: float | None = None,
        comparison_count: int = 1,
    ) -> None:
        if self.wf is None:
            self.scene_title.setText("Open a wavefunction file")
            self.scene_meta_label.setText("")
            return
        if grid is None:
            self.scene_title.setText(f"<b>{html.escape(self.wf.path.name)}</b>")
            self.scene_meta_label.setText(
                f"{len(self.wf.atomic_numbers)} atoms &nbsp;&middot;&nbsp; "
                f"{self.wf.n_basis} basis functions"
            )
            return

        spin = html.escape(grid.spin)
        energy = self.wf.energies(grid.spin)[grid.orbital_index0]
        occupation = self.wf.occupation(grid.spin, grid.orbital_index0)
        if comparison_count > 1:
            primary = f"<b>Comparing {comparison_count} {spin} orbitals</b>"
        else:
            energy_text = (
                "Energy unavailable"
                if not np.isfinite(energy)
                else f"Energy <b>{energy:.6f} Eh / {energy * HARTREE_TO_EV:.3f} eV</b>"
            )
            primary = (
                f"<b>{spin} MO {grid.orbital_index0 + 1}</b> &nbsp;&middot;&nbsp; "
                f"Occupancy <b>{occupation:g}</b> &nbsp;&middot;&nbsp; {energy_text}"
            )
        self.scene_title.setText(primary)
        self.scene_meta_label.setText(self.scene_metadata_html(grid, level))

    def refresh_scene_header(self) -> None:
        primary = self.primary_slot
        if primary is None or primary.grid is None:
            self.update_scene_header()
            return
        self.update_scene_header(
            primary.grid,
            primary.level,
            comparison_count=self.visible_slot_count,
        )

    def comparison_slot_title(self, grid: OrbitalGrid) -> str:
        assert self.wf is not None
        energy = self.wf.energies(grid.spin)[grid.orbital_index0]
        occupation = self.wf.occupation(grid.spin, grid.orbital_index0)
        energy_text = "--" if not np.isfinite(energy) else f"{energy:.5f} Eh"
        return (
            f"<b>{html.escape(grid.spin)} MO {grid.orbital_index0 + 1}</b> "
            f"&middot; occ {occupation:g} &middot; {energy_text}"
        )

    def draw_empty(self) -> None:
        self.configure_scene_slots(1)
        self.clear_scene()
        self.update_scene_header()
        for slot in self.visible_slots():
            slot.title_label.setText("")
            slot.scene_limit_arrays = []
            self.update_camera(slot)
        self.update_corner_axes()

    def draw_scene(
        self,
        pos: SurfaceMesh | None = None,
        neg: SurfaceMesh | None = None,
        grid: OrbitalGrid | None = None,
        level: float | None = None,
        slot: SceneSlot | None = None,
    ) -> None:
        scene_slot = self.slot_or_primary(slot)
        self.clear_surfaces(scene_slot)
        scene_slot.positive_mesh = pos
        scene_slot.negative_mesh = neg
        self.rebuild_surface_items(scene_slot)
        points_for_limits: list[np.ndarray] = []

        if self.wf is not None:
            coords = self.wf.coordinates_angstrom
            points_for_limits.append(grid_box_points(grid) if grid is not None else coords)
            self.ensure_molecule_model(scene_slot)

        if self.wf is None or grid is None:
            scene_slot.title_label.setText("")
        else:
            scene_slot.title_label.setText(self.comparison_slot_title(grid))
        if scene_slot is self.primary_slot:
            self.update_scene_header(grid, level)
        scene_slot.scene_limit_arrays = [arr.copy() for arr in points_for_limits]
        if points_for_limits:
            self.update_base_view(np.vstack([arr.reshape(-1, 3) for arr in points_for_limits if arr.size]), scene_slot)
        scene_slot.grid = grid
        scene_slot.level = level
        self.update_scene_transform(scene_slot)

    def rotation_center(self, slot: SceneSlot | None = None) -> np.ndarray:
        scene_slot = self.slot_or_primary(slot)
        if scene_slot.view_center is not None:
            return scene_slot.view_center
        if scene_slot.base_center is not None:
            return scene_slot.base_center
        return np.zeros(3, dtype=np.float64)

    def scene_transform(self, slot: SceneSlot | None = None) -> pg.Transform3D:
        scene_slot = self.slot_or_primary(slot)
        center = self.rotation_center(scene_slot)
        rotation = scene_slot.scene_rotation
        translation = center - rotation @ center
        return pg.Transform3D(
            [
                [rotation[0, 0], rotation[0, 1], rotation[0, 2], translation[0]],
                [rotation[1, 0], rotation[1, 1], rotation[1, 2], translation[1]],
                [rotation[2, 0], rotation[2, 1], rotation[2, 2], translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def transform_points(self, points: np.ndarray, slot: SceneSlot | None = None) -> np.ndarray:
        if points.size == 0:
            return points.reshape((-1, 3)).astype(np.float64, copy=False)
        scene_slot = self.slot_or_primary(slot)
        center = self.rotation_center(scene_slot)
        pts = points.reshape((-1, 3)).astype(np.float64, copy=False)
        return (pts - center) @ scene_slot.scene_rotation.T + center

    def update_scene_transform(self, slot: SceneSlot | None = None) -> None:
        slots = self.visible_slots() if slot is None else [slot]
        for scene_slot in slots:
            transform = self.scene_transform(scene_slot)
            for item in scene_slot.molecule_items + scene_slot.surface_items:
                try:
                    item.setTransform(transform)
                except Exception:
                    pass
            self.update_camera(scene_slot)
            self.update_corner_axes(scene_slot)
            scene_slot.view.update()

    def apply_scene_rotation(self, angle_deg: float, axis: str, slot: SceneSlot | None = None) -> None:
        if abs(angle_deg) < 1.0e-9:
            return
        matrix = axis_rotation_matrix(axis, angle_deg)
        for scene_slot in self.target_slots(slot):
            scene_slot.scene_rotation = matrix @ scene_slot.scene_rotation

    def apply_mouse_rotation(self, slot: SceneSlot | None, dx: float, dy: float, drag_mode: str) -> None:
        target = self.slot_or_primary(slot)
        if drag_mode == "roll":
            self.apply_scene_rotation(dx * VMD_ROTATE_DEG_PER_PIXEL, "z", target)
        else:
            self.apply_scene_rotation(dy * VMD_ROTATE_DEG_PER_PIXEL, "x", target)
            self.apply_scene_rotation(dx * VMD_ROTATE_DEG_PER_PIXEL, "y", target)
        self.update_scene_transform(None if self.sync_rotation_enabled() else target)

    def update_base_view(self, points: np.ndarray, slot: SceneSlot | None = None) -> None:
        if points.size == 0:
            return
        scene_slot = self.slot_or_primary(slot)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        scene_slot.base_center = 0.5 * (mins + maxs)
        scene_slot.base_radius = max(1.0, 0.56 * float((maxs - mins).max()))
        if scene_slot.view_center is None:
            scene_slot.view_center = scene_slot.base_center.copy()

    def update_camera(self, slot: SceneSlot | None = None) -> None:
        scene_slot = self.slot_or_primary(slot)
        center = scene_slot.view_center if scene_slot.view_center is not None else scene_slot.base_center
        if center is None:
            center = np.zeros(3, dtype=np.float64)
        radius = max(0.1, scene_slot.base_radius / max(scene_slot.view_zoom, 0.05))
        fov = 18.0
        distance = max(4.0, radius / math.tan(math.radians(fov) * 0.5) * 1.22)
        qcenter = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))
        update_fog_shader_params(distance, radius)
        scene_slot.view.setCameraParams(center=qcenter, distance=distance, elevation=90.0, azimuth=-90.0, fov=fov)

    def update_corner_axes(self, slot: SceneSlot | None = None) -> None:
        slots = self.visible_slots() if slot is None else [slot]
        for scene_slot in slots:
            self._update_corner_axes_for_slot(scene_slot)

    def _update_corner_axes_for_slot(self, slot: SceneSlot) -> None:
        slot.corner_view.setVisible(self.corner_check.isChecked())
        slot.corner_view.update()

    def reset_view(self) -> None:
        for slot in self.visible_slots():
            slot.scene_rotation = default_scene_rotation()
            slot.view_zoom = self.default_zoom
            slot.view_center = None if slot.base_center is None else slot.base_center.copy()
            slot.center_atom_idx = None
        self.zoom_control.setValueQuietly(int(round(self.default_zoom * 100.0)))
        self.update_scene_transform()

    def zoom_view(self, factor: float, slot: SceneSlot | None = None) -> None:
        targets = self.target_slots(slot)
        for scene_slot in targets:
            scene_slot.view_zoom = max(0.35, min(3.0, scene_slot.view_zoom * factor))
        self.zoom_control.setValueQuietly(int(round(targets[0].view_zoom * 100.0)))
        for scene_slot in targets:
            self.update_camera(scene_slot)

    def on_zoom_slider(self, value: int) -> None:
        zoom = max(0.35, min(3.0, value / 100.0))
        for slot in self.target_slots():
            slot.view_zoom = zoom
            self.update_camera(slot)

    def pick_rotation_center(self, slot: SceneSlot | None, pos: QtCore.QPointF) -> None:
        self.center_pick_mode = False
        if self.wf is None:
            return
        scene_slot = self.slot_or_primary(slot)
        idx = self.nearest_atom_at_pos(scene_slot, pos)
        if idx is None:
            self.set_ready("No atom near click. Press C and try again.")
            return
        scene_slot.center_atom_idx = idx
        scene_slot.view_center = self.wf.coordinates_angstrom[idx].astype(np.float64, copy=True)
        if self.sync_rotation_enabled():
            for target in self.visible_slots():
                target.center_atom_idx = idx
                target.view_center = scene_slot.view_center.copy()
        symbol = atom_symbol(int(self.wf.atomic_numbers[idx]))
        self.set_ready(f"Rotation center: {symbol}{idx + 1}")
        self.update_scene_transform(None if self.sync_rotation_enabled() else scene_slot)

    def nearest_atom_at_pos(self, slot: SceneSlot, pos: QtCore.QPointF) -> int | None:
        assert self.wf is not None
        coords = self.transform_points(self.wf.coordinates_angstrom, slot)
        try:
            width = max(1, slot.view.width())
            height = max(1, slot.view.height())
            viewport = slot.view.getViewport()
            projection = slot.view.projectionMatrix((0, 0, width, height), viewport)
            view_matrix = slot.view.viewMatrix()
            matrix = projection * view_matrix
            xy = []
            for coord in coords:
                mapped = matrix.map(QtGui.QVector3D(float(coord[0]), float(coord[1]), float(coord[2])))
                sx = (mapped.x() + 1.0) * 0.5 * width
                sy = (1.0 - mapped.y()) * 0.5 * height
                xy.append((sx, sy))
            xy_arr = np.asarray(xy, dtype=np.float64)
        except Exception:
            center = slot.view_center if slot.view_center is not None else np.zeros(3, dtype=np.float64)
            scale = min(slot.view.width(), slot.view.height()) / max(slot.base_radius * 2.4, 1.0)
            xy_arr = np.column_stack(
                (
                    slot.view.width() * 0.5 + (coords[:, 0] - center[0]) * scale,
                    slot.view.height() * 0.5 - (coords[:, 1] - center[1]) * scale,
                )
            )
        target = np.array([float(pos.x()), float(pos.y())], dtype=np.float64)
        distances = np.linalg.norm(xy_arr - target, axis=1)
        idx = int(np.argmin(distances))
        return idx if float(distances[idx]) <= 36.0 else None

    def enable_center_pick(self) -> None:
        self.center_pick_mode = True
        self.set_ready("Center pick: click an atom in the preview.")

    def select_relative_orbital(self, delta: int) -> None:
        if self.loading or self.wf is None:
            return
        idx = self.selected_orbital()
        if idx is None:
            idx = self.wf.default_orbital(self.spin_combo.currentText())
        self.select_orbital(idx + delta)
        if self.orbital_timer is not None:
            self.orbital_timer.stop()
        self.orbital_timer = QtCore.QTimer(self)
        self.orbital_timer.setSingleShot(True)
        self.orbital_timer.timeout.connect(self.render_selected)
        self.orbital_timer.start(140)
        self.slot_or_primary().view.setFocus()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.focus_accepts_text():
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == QtCore.Qt.Key.Key_A:
            self.select_relative_orbital(-1)
        elif key == QtCore.Qt.Key.Key_D:
            self.select_relative_orbital(1)
        elif key == QtCore.Qt.Key.Key_C:
            self.enable_center_pick()
        elif key == QtCore.Qt.Key.Key_Left:
            self.apply_scene_rotation(-7.0, "y")
            self.update_scene_transform()
        elif key == QtCore.Qt.Key.Key_Right:
            self.apply_scene_rotation(7.0, "y")
            self.update_scene_transform()
        elif key == QtCore.Qt.Key.Key_Up:
            self.apply_scene_rotation(6.0, "x")
            self.update_scene_transform()
        elif key == QtCore.Qt.Key.Key_Down:
            self.apply_scene_rotation(-6.0, "x")
            self.update_scene_transform()
        elif key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self.zoom_view(1.15)
        elif key == QtCore.Qt.Key.Key_Minus:
            self.zoom_view(1.0 / 1.15)
        else:
            super().keyPressEvent(event)

    def set_ready(self, text: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setFormat("")
        self.progress.setValue(0)
        self.status_label.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.job_token += 1
        self.prefetch_token += 1
        if self.iso_timer is not None:
            self.iso_timer.stop()
        if self.orbital_timer is not None:
            self.orbital_timer.stop()
        if self.appearance_timer is not None:
            self.appearance_timer.stop()
        self.prefetch_restart_timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.prefetch_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def run_gui(
    input_path: str | None,
    default_grid: int,
    default_iso: float,
    default_margin: float,
    prefetch_workers: int,
    auto_render: bool,
    file_format: str | None = None,
    app_config: AppConfig | None = None,
) -> int:
    _require_gui_dependencies()
    app = QtWidgets.QApplication.instance()
    owned_app = app is None
    if app is None:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        app = QtWidgets.QApplication(sys.argv[:1])
    window = OpenGLViewer(
        input_path,
        default_grid,
        default_iso,
        default_margin,
        prefetch_workers,
        auto_render,
        file_format=file_format,
        app_config=app_config,
    )
    window.show()
    return int(app.exec()) if owned_app else 0
