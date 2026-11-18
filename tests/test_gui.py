"""GUI behaviour, driven headlessly (QT_QPA_PLATFORM=offscreen, window never shown)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from moview.gui.qt import QtWidgets
except ImportError:  # pragma: no cover - GUI extras not installed
    QtWidgets = None

from moview.config import DEFAULT_CONFIG

from helpers import make_wavefunction, write_molden


@unittest.skipIf(QtWidgets is None, "PyQt6 / pyqtgraph / PyOpenGL are not installed")
class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.tmp = tempfile.TemporaryDirectory()
        directory = Path(cls.tmp.name)
        cls.restricted = write_molden(directory / "rhf.molden", make_wavefunction())
        cls.unrestricted = write_molden(directory / "uhf.molden", make_wavefunction(unrestricted=True), orca=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        from moview.gui.main_window import MainWindow

        self.window = MainWindow(DEFAULT_CONFIG, grid=20, isovalue=0.05, margin=3.0)

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_for(self, predicate, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("timed out waiting for the GUI")

    def load(self, path: Path) -> None:
        self.window.open_path(path, auto_render=True)
        self.wait_for(lambda: self.window.wf is not None and self.window.viewports and self.window.viewports[0].result is not None)

    def test_load_render_and_navigate(self) -> None:
        w = self.window
        self.load(self.restricted)
        self.assertEqual(w.orbitals.model.rowCount(), 34)
        self.assertEqual(w.displayed, [4])
        self.assertIn("HOMO", w.viewports[0].title.text())
        self.assertIn("standard conventions", w.file_card.notes.text())
        self.assertTrue(w.actions["render"].isEnabled())

        field = w.viewports[0].result.grid
        w.inspector.iso_box.setValue(0.02)
        self.wait_for(lambda: w.viewports[0].result.level == 0.02)
        self.assertIs(w.viewports[0].result.grid, field)  # isovalue change reuses the field

        w.step_orbital(1)
        self.wait_for(lambda: w.viewports[0].result.index == 5)
        self.assertIn("LUMO", w.viewports[0].title.text())

        w.orbitals.compare.setText("3 - 5")
        w.orbitals._compare()
        self.wait_for(lambda: w.visible_count == 3 and all(v.result is not None and v.result.index == v.orbital for v in w.visible()))
        self.assertEqual([v.orbital for v in w.visible()], [2, 3, 4])

    def test_appearance_changes_do_not_recompute(self) -> None:
        w = self.window
        self.load(self.restricted)
        result = w.viewports[0].result
        w.inspector.surface_style.setCurrentIndex(w.inspector.surface_style.findData("solid_edges"))
        self.assertTrue(all(item.mode == "solid_edges" for item in w.viewports[0].surface_items))
        w.inspector.label_mode.setCurrentIndex(w.inspector.label_mode.findData("number_element"))
        w.inspector.atom_style.setCurrentIndex(w.inspector.atom_style.findData("licorice"))
        self.wait_for(lambda: len(w.viewports[0].molecule_items) == 2)
        self.assertEqual(type(w.viewports[0].molecule_items[1]).__name__, "AttachedLabelItem")
        w.set_theme("dark")
        self.assertTrue(w.theme.dark)
        self.assertIs(w.viewports[0].result, result)
        self.assertFalse(w.foreground.busy)

    def test_unrestricted_spin_switch(self) -> None:
        w = self.window
        self.load(self.unrestricted)
        self.assertTrue(w.orbitals.spin.buttons["beta"].isEnabled())
        w.orbitals.spin.buttons["beta"].click()
        self.wait_for(lambda: w.spin == "beta" and w.viewports[0].result is not None and w.viewports[0].result.spin == "beta")
        self.assertEqual(w.displayed, [3])

    def test_stepping_keeps_keyboard_focus_on_the_view(self) -> None:
        """Regression: re-laying out the views moved focus into the orbital text box."""
        w = self.window
        w.show()
        self.load(self.restricted)
        view = w.viewports[0].view
        view.setFocus()
        self.app.processEvents()
        for expected in (5, 6, 7):
            w.step_orbital(1)
            self.wait_for(lambda: w.displayed == [expected] and w.viewports[0].result is not None and w.viewports[0].result.index == expected)
        self.assertEqual(w.displayed, [7])
        self.assertIsNot(QtWidgets.QApplication.focusWidget(), w.orbitals.jump)
        self.assertEqual(w.orbitals.jump.text(), "")
        self.assertEqual(w.orbitals.jump.focusPolicy(), w.orbitals.jump.focusPolicy().ClickFocus)

    def test_single_theme_control_in_toolbar(self) -> None:
        w = self.window
        self.assertFalse(hasattr(w.inspector, "theme"))
        w.theme_actions["dark"].trigger()
        self.assertTrue(w.theme.dark)
        self.assertTrue(w.theme_actions["dark"].isChecked())
        w.theme_actions["light"].trigger()
        self.assertFalse(w.theme.dark)

    def test_orbital_list_parsing(self) -> None:
        from moview.gui.panels import parse_orbital_list

        self.assertEqual(parse_orbital_list("45,46 47", 100), [44, 45, 46])
        self.assertEqual(parse_orbital_list("49-45", 100), [48, 47, 46, 45, 44])
        self.assertEqual(parse_orbital_list("2 - 4, 3", 100), [1, 2, 3])
        for text in ("0", "101", "a", "1-20"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_orbital_list(text, 100)


if __name__ == "__main__":
    unittest.main()
