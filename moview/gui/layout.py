from __future__ import annotations

import math

from ..constants import DEFAULT_ISOVALUE
from .gl_view import QtCore, QtGui, QtWidgets
from .presentation import (
    ATOM_LABEL_OPTIONS,
    ATOM_LABEL_PLACEMENT_OPTIONS,
    ATOM_STYLE_OPTIONS,
    QT_GRID_MAXIMUM,
    SURFACE_STYLE_OPTIONS,
)
from .styles import APP_STYLE_SHEET
from .widgets import ValueSlider


class MainWindowLayoutMixin:
    """Build the Qt widget tree while the main window owns behavior and state."""

    def _build_ui(self, default_grid: int, default_iso: float, default_margin: float) -> None:
        render_defaults = self.app_config.render
        view_defaults = self.app_config.view
        self.setStyleSheet(APP_STYLE_SHEET)
        shell = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(shell)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(shell)

        side = QtWidgets.QFrame(objectName="side")
        side.setFixedWidth(388)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(18, 14, 18, 14)
        side_layout.setSpacing(9)
        root_layout.addWidget(side)

        header = QtWidgets.QLabel("MOview")
        header.setObjectName("header")
        side_layout.addWidget(header)
        subtitle = QtWidgets.QLabel("Molecular orbital explorer")
        subtitle.setObjectName("subtitle")
        side_layout.addWidget(subtitle)
        self.file_label = QtWidgets.QLabel("No wavefunction file loaded")
        self.file_label.setProperty("class", "muted")
        self.file_label.setWordWrap(True)
        side_layout.addWidget(self.file_label)

        button_row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open file")
        self.open_button.setObjectName("accent")
        self.open_button.clicked.connect(self.open_file)
        self.render_button = QtWidgets.QPushButton("Render")
        self.render_button.setEnabled(False)
        self.render_button.clicked.connect(self.render_selected)
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.render_button)
        side_layout.addLayout(button_row)

        meta_row = QtWidgets.QHBoxLayout()
        self.atom_label = QtWidgets.QLabel("Atoms: -")
        self.basis_label = QtWidgets.QLabel("Basis: -")
        self.atom_label.setProperty("class", "muted")
        self.basis_label.setProperty("class", "muted")
        meta_row.addWidget(self.atom_label)
        meta_row.addSpacing(18)
        meta_row.addWidget(self.basis_label)
        meta_row.addStretch(1)
        side_layout.addLayout(meta_row)

        self.settings_tabs = QtWidgets.QTabWidget()
        self.settings_tabs.setDocumentMode(True)
        self.settings_tabs.setMaximumHeight(310)
        self.settings_tabs.tabBar().setExpanding(True)
        self.settings_tabs.tabBar().setUsesScrollButtons(False)

        compute_tab = QtWidgets.QWidget()
        compute_layout = QtWidgets.QVBoxLayout(compute_tab)
        compute_layout.setContentsMargins(12, 10, 12, 10)
        compute_layout.setSpacing(8)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(7)
        self.spin_combo = QtWidgets.QComboBox()
        self.spin_combo.addItem("alpha")
        self.spin_combo.currentTextChanged.connect(lambda _text: self.populate_orbitals())
        self.grid_spin = QtWidgets.QSpinBox()
        self.grid_spin.setRange(8, QT_GRID_MAXIMUM)
        self.grid_spin.setSingleStep(8)
        self.grid_spin.setValue(max(8, min(QT_GRID_MAXIMUM, int(default_grid))))
        self.grid_spin.setToolTip("Points along the longest grid axis")
        self.grid_spin.valueChanged.connect(self.schedule_prefetch_for_compute_change)
        self.margin_spin = QtWidgets.QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 30.0)
        self.margin_spin.setSingleStep(0.5)
        self.margin_spin.setDecimals(2)
        self.margin_spin.setValue(max(0.0, default_margin))
        self.margin_spin.valueChanged.connect(self.schedule_prefetch_for_compute_change)
        form.addRow("Spin", self.spin_combo)
        form.addRow("Grid", self.grid_spin)
        form.addRow("Margin / bohr", self.margin_spin)
        compute_layout.addLayout(form)

        iso_header = QtWidgets.QHBoxLayout()
        iso_title = QtWidgets.QLabel("Isovalue")
        iso_title.setObjectName("sectionTitle")
        iso_header.addWidget(iso_title)
        iso_header.addStretch(1)
        compute_layout.addLayout(iso_header)
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.iso_slider.setRange(1, 1000)
        self.iso_slider.valueChanged.connect(self.on_iso_drag)
        compute_layout.addWidget(self.iso_slider)
        iso_entry_row = QtWidgets.QHBoxLayout()
        initial_iso = float(default_iso)
        if not math.isfinite(initial_iso) or initial_iso <= 0.0:
            initial_iso = DEFAULT_ISOVALUE
        self.iso_entry = QtWidgets.QLineEdit(f"{initial_iso:.5f}")
        self.iso_entry.returnPressed.connect(self.apply_iso_entry)
        self.apply_iso_button = QtWidgets.QPushButton("Apply")
        self.apply_iso_button.clicked.connect(self.apply_iso_entry)
        iso_entry_row.addWidget(self.iso_entry)
        iso_entry_row.addWidget(self.apply_iso_button)
        compute_layout.addLayout(iso_entry_row)
        compute_layout.addStretch(1)
        self.set_iso_slider_value(initial_iso, update_text=False)

        display_tab = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_tab)
        display_layout.setContentsMargins(12, 10, 12, 10)
        display_layout.setSpacing(6)
        display_form = QtWidgets.QFormLayout()
        display_form.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        display_form.setHorizontalSpacing(14)
        display_form.setVerticalSpacing(4)
        self.surface_style_combo = QtWidgets.QComboBox()
        for label, value in SURFACE_STYLE_OPTIONS:
            self.surface_style_combo.addItem(label, value)
        self.surface_style_combo.setCurrentIndex(
            self.surface_style_combo.findData(render_defaults.surface_style)
        )
        self.surface_style_combo.currentIndexChanged.connect(self.on_surface_style_changed)
        self.positive_color_combo = QtWidgets.QComboBox()
        self.negative_color_combo = QtWidgets.QComboBox()
        for preset in self.app_config.colors:
            swatch = QtGui.QPixmap(14, 14)
            swatch.fill(QtGui.QColor(*preset.rgb))
            icon = QtGui.QIcon(swatch)
            self.positive_color_combo.addItem(icon, preset.name, preset.name)
            self.negative_color_combo.addItem(icon, preset.name, preset.name)
        self.positive_color_combo.setCurrentIndex(
            self.positive_color_combo.findData(render_defaults.positive_color)
        )
        self.negative_color_combo.setCurrentIndex(
            self.negative_color_combo.findData(render_defaults.negative_color)
        )
        self.positive_color_combo.setToolTip("Color for the positive wavefunction phase")
        self.negative_color_combo.setToolTip("Color for the negative wavefunction phase")
        self.positive_color_combo.setAccessibleName("Positive phase color")
        self.negative_color_combo.setAccessibleName("Negative phase color")
        self.positive_color_combo.currentIndexChanged.connect(self.on_surface_color_changed)
        self.negative_color_combo.currentIndexChanged.connect(self.on_surface_color_changed)
        self.atom_style_combo = QtWidgets.QComboBox()
        for label, value in ATOM_STYLE_OPTIONS:
            self.atom_style_combo.addItem(label, value)
        self.atom_style_combo.setCurrentIndex(
            self.atom_style_combo.findData(render_defaults.atom_style)
        )
        self.atom_style_combo.currentIndexChanged.connect(self.schedule_molecule_refresh)
        self.atom_scale_control = ValueSlider(
            35,
            220,
            int(round(render_defaults.atom_scale * 100.0)),
            divisor=100.0,
            decimals=2,
            suffix="x",
        )
        self.atom_scale_control.setToolTip("Scale atomic radii")
        self.atom_scale_control.valueChanged.connect(self.schedule_molecule_refresh)
        self.atom_label_combo = QtWidgets.QComboBox()
        for label, value in ATOM_LABEL_OPTIONS:
            self.atom_label_combo.addItem(label, value)
        self.atom_label_combo.setCurrentIndex(
            self.atom_label_combo.findData(render_defaults.label_mode)
        )
        self.atom_label_combo.currentIndexChanged.connect(self.schedule_molecule_refresh)
        self.atom_label_placement_combo = QtWidgets.QComboBox()
        for label, value in ATOM_LABEL_PLACEMENT_OPTIONS:
            self.atom_label_placement_combo.addItem(label, value)
        self.atom_label_placement_combo.setCurrentIndex(
            self.atom_label_placement_combo.findData(render_defaults.label_placement)
        )
        self.atom_label_placement_combo.setToolTip(
            "Attach labels in 3D or use collision-avoiding floating labels"
        )
        self.atom_label_placement_combo.setAccessibleName("Atom label placement")
        self.atom_label_placement_combo.currentIndexChanged.connect(
            self.schedule_molecule_refresh
        )
        self.label_size_control = ValueSlider(
            8,
            28,
            render_defaults.label_size,
            suffix=" pt",
        )
        self.label_size_control.valueChanged.connect(self.schedule_molecule_refresh)
        display_form.addRow("Surface", self.surface_style_combo)
        display_form.addRow("+ phase", self.positive_color_combo)
        display_form.addRow("- phase", self.negative_color_combo)
        display_form.addRow("Atoms", self.atom_style_combo)
        display_form.addRow("Atom size", self.atom_scale_control)
        label_options = QtWidgets.QHBoxLayout()
        label_options.setContentsMargins(0, 0, 0, 0)
        label_options.setSpacing(6)
        label_options.addWidget(self.atom_label_combo, stretch=3)
        label_options.addWidget(self.atom_label_placement_combo, stretch=2)
        display_form.addRow("Labels", label_options)
        display_form.addRow("Label size", self.label_size_control)
        display_layout.addLayout(display_form)

        self.zoom_control = ValueSlider(
            35,
            300,
            int(round(view_defaults.zoom * 100.0)),
            divisor=100.0,
            decimals=2,
            suffix="x",
        )
        self.zoom_slider = self.zoom_control.slider
        self.zoom_control.valueChanged.connect(self.on_zoom_slider)
        zoom_row = QtWidgets.QFormLayout()
        zoom_row.addRow("Zoom", self.zoom_control)
        display_layout.addLayout(zoom_row)
        view_flags = QtWidgets.QHBoxLayout()
        view_flags.setSpacing(18)
        self.sync_views_check = QtWidgets.QCheckBox("Sync views")
        self.sync_views_check.setChecked(view_defaults.sync_views)
        self.sync_views_check.stateChanged.connect(self.on_sync_views_changed)
        self.corner_check = QtWidgets.QCheckBox("Axes")
        self.corner_check.setChecked(view_defaults.show_axes)
        self.corner_check.stateChanged.connect(lambda _state: self.update_corner_axes())
        view_flags.addWidget(self.sync_views_check)
        view_flags.addWidget(self.corner_check)
        view_flags.addStretch(1)
        display_layout.addLayout(view_flags)
        display_layout.addStretch(1)

        self.settings_tabs.addTab(compute_tab, "Compute")
        self.settings_tabs.addTab(display_tab, "Display")
        side_layout.addWidget(self.settings_tabs)

        quick_row = QtWidgets.QHBoxLayout()
        self.homo_button = QtWidgets.QPushButton("HOMO")
        self.homo_button.setEnabled(False)
        self.homo_button.clicked.connect(lambda: self.select_frontier("homo"))
        self.lumo_button = QtWidgets.QPushButton("LUMO")
        self.lumo_button.setEnabled(False)
        self.lumo_button.clicked.connect(lambda: self.select_frontier("lumo"))
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        quick_row.addWidget(self.homo_button)
        quick_row.addWidget(self.lumo_button)
        quick_row.addWidget(self.reset_button)
        side_layout.addLayout(quick_row)

        compare_row = QtWidgets.QHBoxLayout()
        compare_row.addWidget(QtWidgets.QLabel("Compare"))
        self.compare_entry = QtWidgets.QLineEdit()
        self.compare_entry.setEnabled(False)
        self.compare_entry.setPlaceholderText("1-based MOs, e.g. 45,46,47")
        self.compare_entry.returnPressed.connect(self.render_selected)
        compare_row.addWidget(self.compare_entry, stretch=1)
        side_layout.addLayout(compare_row)

        self.orbital_info_label = QtWidgets.QLabel("Select an orbital")
        self.orbital_info_label.setWordWrap(True)
        self.orbital_info_label.setProperty("class", "muted")
        side_layout.addWidget(self.orbital_info_label)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setEnabled(False)
        self.tree.setHeaderLabels(["Orb", "Occ", "Energy / Eh", "eV"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self.update_orbital_info)
        self.tree.itemDoubleClicked.connect(lambda _item, _col: self.render_selected())
        self.tree.setColumnWidth(0, 58)
        self.tree.setColumnWidth(1, 58)
        self.tree.setColumnWidth(2, 112)
        side_layout.addWidget(self.tree, stretch=1)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("class", "muted")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        side_layout.addWidget(self.status_label)
        side_layout.addWidget(self.progress)

        canvas_panel = QtWidgets.QWidget()
        canvas_layout = QtWidgets.QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(8, 8, 10, 8)
        canvas_layout.setSpacing(6)
        scene_header = QtWidgets.QWidget()
        scene_header.setMinimumHeight(38)
        scene_header.setMaximumHeight(42)
        scene_header_layout = QtWidgets.QVBoxLayout(scene_header)
        scene_header_layout.setContentsMargins(0, 0, 0, 0)
        scene_header_layout.setSpacing(0)
        self.scene_title = QtWidgets.QLabel("Open a wavefunction file")
        self.scene_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scene_title.setObjectName("scenePrimary")
        self.scene_title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.scene_meta_label = QtWidgets.QLabel("")
        self.scene_meta_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scene_meta_label.setObjectName("sceneMeta")
        self.scene_meta_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        scene_header_layout.addWidget(self.scene_title)
        scene_header_layout.addWidget(self.scene_meta_label)
        canvas_layout.addWidget(scene_header)
        self.scene_grid_widget = QtWidgets.QWidget()
        self.scene_grid = QtWidgets.QGridLayout(self.scene_grid_widget)
        self.scene_grid.setContentsMargins(0, 0, 0, 0)
        self.scene_grid.setSpacing(8)
        canvas_layout.addWidget(self.scene_grid_widget, stretch=1)
        root_layout.addWidget(canvas_panel, stretch=1)
        self.configure_scene_slots(1)
        self.draw_empty()
