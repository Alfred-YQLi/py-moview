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
from moview.gui.linux_qt import (
    _missing_shared_libraries,
    _parse_missing_libraries,
    linux_qt_platform_issue,
)
from moview.gui.native_stderr import _should_suppress_native_stderr, filter_macos_gui_warnings
from moview.gui.presentation import atom_label_texts, high_grid_warning_text
from moview.parsers import parse_wavefunction
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

    def test_molden_rejects_unknown_spin(self) -> None:
        parser = MoldenParser(Path("unused.molden"))
        with self.assertRaisesRegex(ValueError, "Invalid Molden MO spin"):
            parser._finish_mo_block(
                {"coefficients": [(1, 0.5)], "spin": "Gamma"},
                1,
                [],
                [],
                [],
                [],
                [],
                [],
            )

    def test_molden_spherical_markers_scale_and_compact_spin(self) -> None:
        cases = (
            ("5D", (-2, -3), 12),
            ("5D7F", (-2, -3), 12),
            ("5D10F", (-2, 3), 15),
            ("7F", (2, -3), 13),
        )
        with tempfile.TemporaryDirectory() as directory:
            for marker, shell_types, n_basis in cases:
                with self.subTest(marker=marker):
                    alpha_coefficients = "\n".join(
                        f"{index} {1.0 if index == 1 else 0.0}"
                        for index in range(1, n_basis + 1)
                    )
                    beta_coefficients = "\n".join(
                        f"{index} {-1.0 if index == 1 else 0.0}"
                        for index in range(1, n_basis + 1)
                    )
                    path = Path(directory) / f"{marker.lower()}.molden"
                    path.write_text(
                        "\n".join(
                            (
                                "[Molden Format]",
                                "[Atoms] AU",
                                "H 1 1 0.0 0.0 0.0",
                                "[GTO]",
                                "1 0",
                                "d 1 2.0",
                                "1.0 0.5",
                                "f 1 3.0",
                                "0.8 0.25",
                                f"[{marker}]",
                                "[MO]",
                                "Sym= A",
                                "Ene= -0.1",
                                "Spin= Alpha",
                                "Occup= 1.0",
                                alpha_coefficients,
                                "Sym= A",
                                "Ene= 0.2",
                                "Spin=Beta",
                                "Occup= 0.0",
                                beta_coefficients,
                            )
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    wavefunction = parse_wavefunction(path)

                    self.assertEqual(tuple(wavefunction.shell_types), shell_types)
                    self.assertEqual(wavefunction.n_basis, n_basis)
                    self.assertEqual(wavefunction.alpha_coefficients.shape, (1, n_basis))
                    self.assertEqual(wavefunction.beta_coefficients.shape, (1, n_basis))
                    self.assertEqual(wavefunction.n_alpha, 1)
                    self.assertEqual(wavefunction.n_beta, 0)
                    self.assertTrue(wavefunction.is_unrestricted)
                    np.testing.assert_array_equal(wavefunction.shell_to_atom, (0, 0))
                    np.testing.assert_allclose(
                        [
                            wavefunction.shells[0].coefficients[0],
                            wavefunction.shells[1].coefficients[0],
                        ],
                        (0.5, 0.25),
                    )
                    self.assertEqual(wavefunction.alpha_coefficients[0, 0], 1.0)
                    self.assertEqual(wavefunction.beta_coefficients[0, 0], -1.0)

    def test_molden_sp_shell_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sp-shell.molden"
            path.write_text(
                "\n".join(
                    (
                        "[Molden Format]",
                        "[Atoms] Angs",
                        f"H 1 1 {BOHR_TO_ANG} 0.0 0.0",
                        "[GTO]",
                        "1",
                        "sp 2 1.0",
                        "1.0D+00 4.0D-01 5.0D-01",
                        "5.0D-01 6.0D-01 7.0D-01",
                        "[MO]",
                        "Sym= A1",
                        "Ene= -0.5",
                        "Spin= Alpha",
                        "Occup= 2.0",
                        "1 1.0",
                        "2 0.0",
                        "3 0.0",
                        "4 0.0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            wavefunction = parse_wavefunction(path)

        self.assertEqual(wavefunction.n_basis, 4)
        self.assertEqual(tuple(wavefunction.shell_types), (-1,))
        self.assertEqual(wavefunction.alpha_coefficients.shape, (1, 4))
        np.testing.assert_allclose(wavefunction.coordinates_bohr[0], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(wavefunction.shells[0].exponents, (1.0, 0.5))
        np.testing.assert_allclose(wavefunction.shells[0].coefficients, (0.4, 0.6))
        np.testing.assert_allclose(wavefunction.shells[0].sp_coefficients, (0.5, 0.7))

    def test_molden_cartesian_g_coefficients_follow_format_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cartesian-g.molden"
            coefficients = "\n".join(f"{index} {index}" for index in range(1, 16))
            path.write_text(
                "\n".join(
                    (
                        "[Molden Format]",
                        "[Atoms] AU",
                        "H 1 1 0.0 0.0 0.0",
                        "[GTO]",
                        "1 0",
                        "g 1 1.0",
                        "1.0 1.0",
                        "[MO]",
                        "Sym= A1",
                        "Ene= -0.5",
                        "Spin= Alpha",
                        "Occup= 2.0",
                        coefficients,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            wavefunction = parse_wavefunction(path)

        self.assertEqual(tuple(wavefunction.shell_types), (4,))
        np.testing.assert_array_equal(
            wavefunction.alpha_coefficients[0],
            (3, 9, 12, 7, 2, 8, 15, 14, 6, 11, 13, 10, 5, 4, 1),
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


class LinuxQtPreflightTests(unittest.TestCase):
    def test_missing_ldd_libraries_are_parsed_once(self) -> None:
        output = """
            libxcb-cursor.so.0 => not found
            libxcb-image.so.0 => /lib64/libxcb-image.so.0 (0x0001)
            libxcb-cursor.so.0 => not found
            libxkbcommon-x11.so.0 => not found
        """
        self.assertEqual(
            _parse_missing_libraries(output),
            ("libxcb-cursor.so.0", "libxkbcommon-x11.so.0"),
        )

    def test_ldd_runs_with_stable_locale(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ldd", "libqxcb.so"],
            returncode=0,
            stdout="libxcb-cursor.so.0 => not found\n",
            stderr="",
        )
        with (
            patch("moview.gui.linux_qt.shutil.which", return_value="/usr/bin/ldd"),
            patch("moview.gui.linux_qt.subprocess.run", return_value=completed) as run,
        ):
            missing = _missing_shared_libraries(
                Path("/qt/plugins/libqxcb.so"),
                {"LD_LIBRARY_PATH": "/custom/lib"},
            )

        self.assertEqual(missing, ("libxcb-cursor.so.0",))
        self.assertEqual(run.call_args.args[0], ["/usr/bin/ldd", "/qt/plugins/libqxcb.so"])
        self.assertEqual(run.call_args.kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(run.call_args.kwargs["env"]["LD_LIBRARY_PATH"], "/custom/lib")
        self.assertFalse(run.call_args.kwargs["check"])

    def test_non_linux_and_non_xcb_platforms_skip_preflight(self) -> None:
        with patch("moview.gui.linux_qt._find_xcb_plugin") as find_plugin:
            self.assertIsNone(linux_qt_platform_issue(environ={}, host_platform="darwin"))
            self.assertIsNone(
                linux_qt_platform_issue(
                    environ={"QT_QPA_PLATFORM": "offscreen"},
                    host_platform="linux",
                )
            )
        find_plugin.assert_not_called()

    def test_complete_xcb_dependencies_allow_startup(self) -> None:
        with (
            patch(
                "moview.gui.linux_qt._find_xcb_plugin",
                return_value=Path("/site-packages/PyQt6/Qt6/plugins/platforms/libqxcb.so"),
            ),
            patch("moview.gui.linux_qt._missing_shared_libraries", return_value=()),
        ):
            issue = linux_qt_platform_issue(environ={}, host_platform="linux")

        self.assertIsNone(issue)

    def test_centos_8_issue_names_native_package(self) -> None:
        with (
            patch(
                "moview.gui.linux_qt._find_xcb_plugin",
                return_value=Path("/site-packages/PyQt6/Qt6/plugins/platforms/libqxcb.so"),
            ),
            patch(
                "moview.gui.linux_qt._missing_shared_libraries",
                return_value=("libxcb-cursor.so.0",),
            ),
            patch(
                "moview.gui.linux_qt._os_release",
                return_value={"ID": "centos", "ID_LIKE": "rhel fedora", "VERSION_ID": "8"},
            ),
        ):
            issue = linux_qt_platform_issue(environ={}, host_platform="linux")

        self.assertIsNotNone(issue)
        self.assertIn("libxcb-cursor.so.0", issue)
        self.assertIn("sudo dnf install epel-release", issue)
        self.assertIn("sudo dnf install xcb-util-cursor", issue)
        self.assertIn("cannot be installed by pip", issue)


if __name__ == "__main__":
    unittest.main()
