from __future__ import annotations

import os
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtTest, QtWidgets

    from moview.constants import DEFAULT_ISOVALUE
    from moview.config import AppConfig, ColorPreset, DEFAULT_CONFIG
    from moview.grid import OrbitalGrid
    from moview.gui.gl_view import (
        AttachedAtomLabelItem,
        FOG_FLAT_SHADER,
        AtomLabelItem,
        atom_label_item,
        atom_display_radius,
        build_molecule_geometry,
        mesh_item,
    )
    from moview.gui.main_window import OpenGLViewer
    from moview.gui.presentation import GRID_PREFETCH_DEBOUNCE_MS
    from moview.surface import SurfaceMesh, _empty_mesh
    from moview.wavefunction import Shell, Wavefunction

    GUI_AVAILABLE = True
except ModuleNotFoundError:
    GUI_AVAILABLE = False


def _tiny_wavefunction() -> "Wavefunction":
    return Wavefunction(
        path=Path("tiny.fch"),
        title="tiny",
        method_line="test",
        atomic_numbers=np.asarray([1], dtype=np.int32),
        coordinates_bohr=np.zeros((1, 3), dtype=np.float64),
        n_alpha=1,
        n_beta=1,
        n_basis=1,
        shell_types=np.asarray([0], dtype=np.int32),
        shell_to_atom=np.asarray([0], dtype=np.int32),
        shells=[
            Shell(
                shell_type=0,
                center=np.zeros(3, dtype=np.float64),
                exponents=np.asarray([1.0]),
                coefficients=np.asarray([1.0]),
            )
        ],
        alpha_energies=np.asarray([-0.5]),
        beta_energies=None,
        alpha_coefficients=np.asarray([[1.0]]),
        beta_coefficients=None,
        alpha_occupations=np.asarray([2.0]),
    )


@unittest.skipUnless(GUI_AVAILABLE, "PyQt6/pyqtgraph/PyOpenGL are required")
class GuiBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.viewer = OpenGLViewer(None, 81, DEFAULT_ISOVALUE, 4.0, 1, False)

    def tearDown(self) -> None:
        self.viewer.close()
        self.app.processEvents()

    def test_defaults_and_unlimited_grid_control(self) -> None:
        self.assertEqual(self.viewer.grid_spin.maximum(), 2_147_483_647)
        self.viewer.grid_spin.setValue(257)
        self.assertEqual(self.viewer.display_grid_for_orbital("alpha", 0), 257)
        self.assertEqual(self.viewer.surface_style_combo.currentData(), "glass")
        self.assertEqual(self.viewer.atom_style_combo.currentData(), "ball_stick")
        self.assertEqual(self.viewer.atom_label_combo.currentData(), "off")
        self.assertEqual(self.viewer.atom_label_placement_combo.currentData(), "attached")
        self.assertEqual(self.viewer.positive_color_combo.currentData(), "Red")
        self.assertEqual(self.viewer.negative_color_combo.currentData(), "Blue")
        self.assertAlmostEqual(float(self.viewer.iso_entry.text()), DEFAULT_ISOVALUE)
        self.assertFalse(hasattr(self.viewer, "auto_iso_check"))
        self.viewer.set_iso_slider_value(0.051234, update_text=True)
        self.assertEqual(self.viewer.iso_value(), 0.051234)

    def test_configured_resources_and_display_defaults_are_applied(self) -> None:
        colors = DEFAULT_CONFIG.colors + (ColorPreset("Mint", (40, 210, 160)),)
        config = AppConfig(
            resources=replace(
                DEFAULT_CONFIG.resources,
                background_jobs=2,
                render_cache_mib=96,
                render_cache_entries=32,
                grid_chunk_points=32_768,
                surface_face_limit=80_000,
            ),
            render=replace(
                DEFAULT_CONFIG.render,
                surface_style="solid_edges",
                atom_style="licorice",
                atom_scale=1.25,
                label_mode="number_element",
                label_placement="floating",
                label_size=15,
                positive_color="Mint",
                negative_color="Violet",
            ),
            view=replace(
                DEFAULT_CONFIG.view,
                zoom=1.2,
                sync_views=False,
                show_axes=False,
            ),
            colors=colors,
        )
        viewer = OpenGLViewer(None, 81, DEFAULT_ISOVALUE, 4.0, 2, False, app_config=config)
        try:
            self.assertEqual(viewer.background_jobs, 2)
            self.assertEqual(viewer.render_cache_limit_bytes, 96 * 1024**2)
            self.assertEqual(viewer.cache_limit, 32)
            self.assertEqual(viewer.grid_chunk_points, 32_768)
            self.assertEqual(viewer.surface_face_limit, 80_000)
            self.assertEqual(viewer.surface_style_combo.currentData(), "solid_edges")
            self.assertEqual(viewer.atom_style_combo.currentData(), "licorice")
            self.assertEqual(viewer.atom_scale_control.value(), 125)
            self.assertEqual(viewer.atom_label_combo.currentData(), "number_element")
            self.assertEqual(viewer.atom_label_placement_combo.currentData(), "floating")
            self.assertEqual(viewer.label_size_control.value(), 15)
            self.assertEqual(viewer.positive_color_combo.currentData(), "Mint")
            self.assertEqual(viewer.negative_color_combo.currentData(), "Violet")
            self.assertEqual(viewer.zoom_control.value(), 120)
            self.assertFalse(viewer.sync_views_check.isChecked())
            self.assertFalse(viewer.corner_check.isChecked())
        finally:
            viewer.close()

    def test_high_grid_warning_is_acknowledged_per_value(self) -> None:
        self.viewer.wf = SimpleNamespace(
            coordinates_bohr=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        )
        yes = QtWidgets.QMessageBox.StandardButton.Yes
        with patch.object(QtWidgets.QMessageBox, "warning", return_value=yes) as warning:
            self.assertTrue(self.viewer.confirm_grid_performance(257))
            self.assertTrue(self.viewer.confirm_grid_performance(257))
        warning.assert_called_once()

    def test_explicit_isovalue_does_not_reuse_different_cached_surface(self) -> None:
        wavefunction = _tiny_wavefunction()
        self.viewer.wf = wavefunction
        self.viewer.bonds = []
        self.viewer.grid_spin.setValue(8)
        self.viewer.margin_spin.setValue(4.0)
        values = np.linspace(-0.5, 0.5, 8**3, dtype=np.float32).reshape((8, 8, 8))
        grid = OrbitalGrid(
            spin="alpha",
            orbital_index0=0,
            grid_size=8,
            margin_bohr=4.0,
            values=values,
            shape=values.shape,
            spacing=np.ones(3),
            origin=np.zeros(3),
        )
        key = self.viewer.cache_key("alpha", 0, 8, 4.0)
        self.viewer.render_cache[key] = (grid, _empty_mesh(), _empty_mesh(), 0.2)
        with patch.object(self.viewer, "submit_job") as submit_job:
            self.viewer.render_orbitals([0])
        submit_job.assert_called_once()

    def test_grid_change_restarts_prefetch_after_debounce(self) -> None:
        self.viewer.wf = _tiny_wavefunction()
        with patch.object(
            self.viewer,
            "restart_prefetch_after_settings_change",
        ) as restart:
            self.viewer.grid_spin.setValue(82)
            QtTest.QTest.qWait(GRID_PREFETCH_DEBOUNCE_MS + 100)
            self.app.processEvents()
        restart.assert_called_once()

    def test_prefetch_restart_uses_current_grid_and_isovalue(self) -> None:
        self.viewer.wf = _tiny_wavefunction()
        self.viewer.grid_spin.setValue(96)
        self.viewer.prefetch_restart_timer.stop()
        self.viewer.set_iso_slider_value(0.07, update_text=True)
        with patch.object(self.viewer, "pump_prefetch") as pump:
            self.viewer.start_prefetch_common_orbitals()
        self.assertEqual(self.viewer.prefetch_grid_size, 96)
        self.assertEqual(self.viewer.prefetch_iso, 0.07)
        self.assertEqual(self.viewer.prefetch_total, 1)
        pump.assert_called_once()

    def test_isovalue_change_before_first_grid_still_restarts_prefetch(self) -> None:
        self.viewer.wf = _tiny_wavefunction()
        self.assertIsNone(self.viewer.grid_cache)
        self.viewer.iso_slider.setValue(self.viewer.iso_slider.value() + 1)
        self.assertTrue(self.viewer.prefetch_restart_timer.isActive())

    def test_prefetch_count_adapts_to_grid_memory(self) -> None:
        self.viewer.wf = _tiny_wavefunction()
        low_grid_limit = self.viewer.prefetch_task_limit(81, 4.0)
        high_grid_limit = self.viewer.prefetch_task_limit(257, 4.0)
        oversized_grid_limit = self.viewer.prefetch_task_limit(512, 4.0)
        self.assertGreater(low_grid_limit, high_grid_limit)
        self.assertGreater(high_grid_limit, 0)
        self.assertEqual(oversized_grid_limit, 0)

    def test_prefetch_reuses_cached_scalar_grid_for_new_isovalue(self) -> None:
        self.viewer.wf = _tiny_wavefunction()
        values = np.zeros((8, 8, 8), dtype=np.float32)
        grid = OrbitalGrid(
            spin="alpha",
            orbital_index0=0,
            grid_size=8,
            margin_bohr=4.0,
            values=values,
            shape=values.shape,
            spacing=np.ones(3),
            origin=np.zeros(3),
        )
        with (
            patch("moview.gui.main_window.compute_orbital_grids_from_basis") as from_basis,
            patch("moview.gui.main_window.compute_orbital_grids_float32") as chunked,
        ):
            results = self.viewer.prefetch_batch(
                "alpha",
                [(0, 8)],
                4.0,
                DEFAULT_ISOVALUE,
                {(0, 8): grid},
            )
        from_basis.assert_not_called()
        chunked.assert_not_called()
        self.assertEqual(results[0][2][3], DEFAULT_ISOVALUE)

    def test_stale_wavefunction_generation_cannot_populate_basis_cache(self) -> None:
        wavefunction = _tiny_wavefunction()
        self.viewer.wf = wavefunction
        self.viewer.wavefunction_generation = 2
        basis_grid = SimpleNamespace(nbytes=4)
        with patch("moview.gui.main_window.compute_basis_grid", return_value=basis_grid):
            stale_result = self.viewer.get_basis_grid(8, 4.0, wavefunction, 1)
            current_result = self.viewer.get_basis_grid(8, 4.0, wavefunction, 2)
        self.assertIs(stale_result, basis_grid)
        self.assertIs(current_result, basis_grid)
        self.assertEqual(list(self.viewer.basis_cache), [(2, 8, 4.0)])

    def test_standard_controls_keep_their_keyboard_input(self) -> None:
        self.assertTrue(self.viewer.focus_accepts_text(self.viewer.grid_spin.lineEdit()))
        self.assertTrue(self.viewer.focus_accepts_text(self.viewer.tree))
        self.assertTrue(self.viewer.focus_accepts_text(self.viewer.zoom_slider))
        self.assertFalse(self.viewer.focus_accepts_text(self.viewer.view))

    def test_loading_runs_asynchronously_and_restores_controls(self) -> None:
        wavefunction = _tiny_wavefunction()

        def delayed_parse(*_args, **_kwargs):
            time.sleep(0.05)
            return wavefunction

        with patch("moview.gui.main_window.parse_wavefunction", side_effect=delayed_parse):
            self.viewer.load_wavefunction("tiny.fch", auto_render=False)
            self.assertFalse(self.viewer.open_button.isEnabled())
            deadline = time.monotonic() + 2.0
            while self.viewer.wf is not wavefunction and time.monotonic() < deadline:
                QtTest.QTest.qWait(20)
                self.app.processEvents()
        self.assertIs(self.viewer.wf, wavefunction)
        self.assertTrue(self.viewer.open_button.isEnabled())
        self.assertTrue(self.viewer.render_button.isEnabled())

    def test_rendering_style_helpers_preserve_defaults(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32),
            faces=np.asarray(((0, 1, 2),), dtype=np.uint32),
        )
        glass = mesh_item(mesh, "#ef3b2c")
        solid = mesh_item(mesh, "#ef3b2c", "solid")
        wire = mesh_item(mesh, "#ef3b2c", "wireframe")
        self.assertFalse(glass.opts["smooth"])
        self.assertFalse(glass.opts["drawEdges"])
        self.assertEqual(glass.opts["shader"], FOG_FLAT_SHADER)
        self.assertTrue(solid.opts["smooth"])
        self.assertFalse(wire.opts["drawFaces"])
        self.assertTrue(wire.opts["drawEdges"])

        wavefunction = _tiny_wavefunction()
        ball = build_molecule_geometry(wavefunction, [], "ball_stick", 1.0)
        space = build_molecule_geometry(wavefunction, [], "space_filling", 1.0)
        licorice = build_molecule_geometry(wavefunction, [], "licorice", 1.0)
        self.assertGreater(ball.faces.shape[0], 0)
        self.assertGreater(space.faces.shape[0], ball.faces.shape[0])
        self.assertGreater(licorice.faces.shape[0], 0)
        self.assertAlmostEqual(
            atom_display_radius(6, "ball_stick", 1.5),
            atom_display_radius(6, "ball_stick", 1.0) * 1.5,
        )

    def test_attached_atom_labels_use_one_perspective_batch(self) -> None:
        wavefunction = _tiny_wavefunction()
        attached = atom_label_item(
            wavefunction,
            "number_element",
            12,
            placement="attached",
            atom_style="ball_stick",
            atom_scale=1.0,
        )
        self.assertIsInstance(attached, AttachedAtomLabelItem)
        assert isinstance(attached, AttachedAtomLabelItem)
        self.assertEqual(attached.label_count, 1)
        self.assertEqual(attached.vertex_data.shape, (6, 8))
        np.testing.assert_allclose(attached.vertex_data[:, :3], np.zeros((6, 3)))
        self.assertGreater(int(attached.atlas_rgba[..., 3].max()), 0)
        self.assertGreater(attached.depth_offsets[0], atom_display_radius(1))
        self.assertGreater(float(np.ptp(attached.vertex_data[:, 3])), 0.0)
        self.assertGreater(float(np.ptp(attached.vertex_data[:, 4])), 0.0)
        second_attached = atom_label_item(
            wavefunction,
            "number_element",
            16,
            placement="attached",
        )
        assert isinstance(second_attached, AttachedAtomLabelItem)
        self.assertIs(attached.atlas_rgba, second_attached.atlas_rgba)
        self.assertGreater(
            float(np.ptp(second_attached.vertex_data[:, 4])),
            float(np.ptp(attached.vertex_data[:, 4])),
        )
        space_filling = atom_label_item(
            wavefunction,
            "number_element",
            12,
            placement="attached",
            atom_style="space_filling",
            atom_scale=1.5,
        )
        assert isinstance(space_filling, AttachedAtomLabelItem)
        self.assertGreater(space_filling.depth_offsets[0], attached.depth_offsets[0])

        floating = atom_label_item(
            wavefunction,
            "number_element",
            12,
            placement="floating",
        )
        self.assertIsInstance(floating, AtomLabelItem)
        self.assertIsNone(atom_label_item(wavefunction, "off", 12))

    def test_appearance_changes_do_not_submit_calculation(self) -> None:
        self.viewer.wf = _tiny_wavefunction()
        self.viewer.bonds = []
        self.viewer.draw_scene()
        mesh = SurfaceMesh(
            vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32),
            faces=np.asarray(((0, 1, 2),), dtype=np.uint32),
        )
        slot = self.viewer.slot_or_primary()
        slot.positive_mesh = mesh
        slot.negative_mesh = mesh
        with patch.object(self.viewer, "submit_job") as submit_job:
            self.viewer.surface_style_combo.setCurrentIndex(
                self.viewer.surface_style_combo.findData("solid")
            )
            self.viewer.atom_label_combo.setCurrentIndex(
                self.viewer.atom_label_combo.findData("number_element")
            )
            self.viewer.atom_label_placement_combo.setCurrentIndex(
                self.viewer.atom_label_placement_combo.findData("floating")
            )
            self.viewer.atom_label_placement_combo.setCurrentIndex(
                self.viewer.atom_label_placement_combo.findData("attached")
            )
            self.viewer.atom_scale_control.setValue(135)
            self.viewer.positive_color_combo.setCurrentIndex(
                self.viewer.positive_color_combo.findData("Violet")
            )
            self.viewer.negative_color_combo.setCurrentIndex(
                self.viewer.negative_color_combo.findData("Cyan")
            )
            self.viewer.refresh_molecule_models()
        submit_job.assert_not_called()
        self.assertEqual(slot.surface_items[0].opts["shader"], "orbitalFogShaded")
        self.assertEqual(len(slot.surface_items), 2)
        np.testing.assert_allclose(
            slot.surface_items[0].opts["color"][:3],
            np.asarray((139, 92, 246), dtype=np.float64) / 255.0,
        )
        np.testing.assert_allclose(
            slot.surface_items[1].opts["color"][:3],
            np.asarray((6, 182, 212), dtype=np.float64) / 255.0,
        )
        self.assertEqual(len(slot.molecule_items), 2)
        self.assertIsInstance(slot.molecule_items[1], AttachedAtomLabelItem)

    def test_color_names_are_synchronized_with_scene_header(self) -> None:
        wavefunction = _tiny_wavefunction()
        self.viewer.wf = wavefunction
        self.viewer.bonds = []
        values = np.zeros((8, 8, 8), dtype=np.float32)
        grid = OrbitalGrid(
            spin="alpha",
            orbital_index0=0,
            grid_size=8,
            margin_bohr=4.0,
            values=values,
            shape=values.shape,
            spacing=np.ones(3),
            origin=np.zeros(3),
        )
        self.viewer.draw_scene(grid=grid, level=DEFAULT_ISOVALUE)
        self.viewer.positive_color_combo.setCurrentIndex(
            self.viewer.positive_color_combo.findData("Violet")
        )
        self.viewer.negative_color_combo.setCurrentIndex(
            self.viewer.negative_color_combo.findData("Cyan")
        )
        metadata = self.viewer.scene_meta_label.text()
        self.assertIn("Violet +psi", metadata)
        self.assertIn("Cyan -psi", metadata)
        self.assertIn("Grid <b>8 &times; 8 &times; 8</b>", metadata)
        self.assertNotIn("|", self.viewer.scene_title.text() + metadata)


if __name__ == "__main__":
    unittest.main()
