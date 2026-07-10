from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import moview
import numpy as np
from moview.basis.gaussian import n_shell_functions
from moview.constants import BOHR_TO_ANG, DEFAULT_ISOVALUE, HARTREE_TO_EV, atom_symbol
from moview.grid import OrbitalGrid, make_grid_spec
from moview.gui.native_stderr import _should_suppress_native_stderr, filter_macos_gui_warnings
from moview.gui.presentation import atom_label_texts, high_grid_warning_text
from moview.parsers.molden import MoldenParser
from moview.surface import _empty_mesh, extract_isosurfaces


class SmokeTests(unittest.TestCase):
    def test_constants_and_shell_counts(self) -> None:
        self.assertAlmostEqual(BOHR_TO_ANG, 0.529177210903)
        self.assertAlmostEqual(HARTREE_TO_EV, 27.211386245988)
        self.assertEqual(n_shell_functions(-2), 5)
        self.assertEqual(n_shell_functions(3), 10)

    def test_public_api_imports(self) -> None:
        self.assertEqual(atom_symbol(6), "C")
        self.assertTrue(hasattr(moview, "parse_wavefunction"))
        self.assertTrue(hasattr(moview, "surface_for_orbital"))

    def test_empty_surface_mesh(self) -> None:
        self.assertEqual(_empty_mesh().n_faces, 0)
        self.assertEqual(_empty_mesh().faces.dtype, np.uint32)

    def test_isovalue_is_explicit_and_positive(self) -> None:
        self.assertEqual(DEFAULT_ISOVALUE, 0.05)
        values = np.zeros((2, 2, 2), dtype=np.float32)
        grid = OrbitalGrid(
            spin="alpha",
            orbital_index0=0,
            grid_size=8,
            margin_bohr=1.0,
            values=values,
            shape=values.shape,
            spacing=np.ones(3),
            origin=np.zeros(3),
        )
        for invalid in (0.0, -0.05, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    extract_isosurfaces(grid, invalid)

    def test_grid_spec_has_no_hidden_point_cap(self) -> None:
        wavefunction = SimpleNamespace(
            coordinates_bohr=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        )
        spec = make_grid_spec(wavefunction, 257, 0.0)
        self.assertEqual(spec.shape, (257, 257, 257))
        self.assertGreater(spec.n_points, 900_000)
        self.assertIn("Grid 257", high_grid_warning_text(spec))

    def test_atom_label_modes(self) -> None:
        atomic_numbers = [20, 26, 0]
        self.assertEqual(atom_label_texts(atomic_numbers, "number"), ["1", "2", ""])
        self.assertEqual(atom_label_texts(atomic_numbers, "element"), ["Ca", "Fe", ""])
        self.assertEqual(
            atom_label_texts(atomic_numbers, "number_element"),
            ["1Ca", "2Fe", ""],
        )

    def test_molden_rejects_out_of_range_mo_coefficients(self) -> None:
        parser = MoldenParser(Path("unused.molden"))
        with self.assertRaisesRegex(ValueError, "outside 1..1"):
            parser._finish_mo_block(
                {"coefficients": [(2, 0.5)]},
                1,
                [],
                [],
                [],
                [],
                [],
                [],
            )

    def test_macos_gui_warning_filter_predicate(self) -> None:
        suppressed = (
            "TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED(-1)",
            "2026-07-10 python[1:2] error messaging the mach port "
            "for IMKCFRunLoopWakeUpReliable",
            "qt.qpa.keymapper: Mismatch between Cocoa 'd' and Carbon '\\x0' "
            "for virtual key 2 with QFlags<Qt::KeyboardModifier>(NoModifier)",
        )
        for line in suppressed:
            with self.subTest(line=line):
                self.assertTrue(_should_suppress_native_stderr(line))
        preserved = (
            "Traceback (most recent call last):",
            "qt.qpa.keymapper: keyboard initialization failed",
            "Mismatch between Cocoa and Carbon without the Qt category",
            "OpenGL shader compilation failed",
        )
        for line in preserved:
            with self.subTest(line=line):
                self.assertFalse(_should_suppress_native_stderr(line))

    def test_macos_gui_warning_filter_preserves_other_stderr(self) -> None:
        capture_read, capture_write = os.pipe()
        saved_stderr = os.dup(2)
        restored = False
        try:
            os.dup2(capture_write, 2)
            os.close(capture_write)
            with patch.object(sys, "platform", "darwin"):
                with filter_macos_gui_warnings():
                    os.write(2, b"TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED(-1)\n")
                    os.write(2, b"error messaging the mach port for IMKCFRunLoopWakeUpReliable\n")
                    os.write(
                        2,
                        b"qt.qpa.keymapper: Mismatch between Cocoa 'd' and Carbon '\\x0' "
                        b"for virtual key 2 with QFlags<Qt::KeyboardModifier>(NoModifier)\n",
                    )
                    os.write(2, b"keep this stderr line\n")
                    os.write(2, b"Traceback (most recent call last):\n")
            os.dup2(saved_stderr, 2)
            restored = True
            with os.fdopen(capture_read, "rb") as reader:
                output = reader.read().decode("utf-8", errors="replace")
        finally:
            if not restored:
                os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        self.assertNotIn("TSMSendMessageToUIServer", output)
        self.assertNotIn("IMKCFRunLoopWakeUpReliable", output)
        self.assertNotIn("qt.qpa.keymapper", output)
        self.assertIn("keep this stderr line", output)
        self.assertIn("Traceback (most recent call last):", output)

    def test_path_launcher_works_outside_project_directory(self) -> None:
        launcher = Path(__file__).resolve().parents[1] / "bin" / "moview"
        self.assertTrue(os.access(launcher, os.X_OK))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(launcher), "--help"],
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: moview", result.stdout)


if __name__ == "__main__":
    unittest.main()
