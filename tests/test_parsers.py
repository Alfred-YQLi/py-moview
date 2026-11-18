"""FCHK and Molden readers, including writer-specific Molden conventions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from moview.core.evaluate import evaluate_orbitals
from moview.core.overlap import orthonormality
from moview.parsers import detect_wavefunction_format, parse_wavefunction
from moview.parsers.molden import parse_molden

from helpers import make_wavefunction, write_fchk, write_molden

RNG = np.random.default_rng(7)
POINTS = RNG.normal(scale=1.6, size=(400, 3))


def assert_same_orbitals(test: unittest.TestCase, parsed, reference, spin: str = "alpha") -> None:
    """Parsed and reference wavefunctions must give the same MO values everywhere."""
    indices = list(range(reference.n_orbitals(spin)))
    expected = evaluate_orbitals(reference, spin, indices, POINTS)
    actual = evaluate_orbitals(parsed, spin, indices, POINTS)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    test.assertLess(max(r.worst for r in orthonormality(parsed)), 1e-8)


class MoldenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_standard_molden_round_trip(self) -> None:
        reference = make_wavefunction()
        parsed = parse_molden(write_molden(self.dir / "std.molden", reference))
        self.assertEqual(parsed.source_format, "molden")
        self.assertEqual(tuple(parsed.shell_types), tuple(reference.shell_types))
        self.assertIn("standard conventions", parsed.notes[0])
        assert_same_orbitals(self, parsed, reference)
        self.assertEqual(parsed.homo_index("alpha"), 4)

    def test_orca_molden_normalization_and_signs_are_undone(self) -> None:
        """ORCA bakes primitive norms into coefficients and flips f/g |m|>=3 signs."""
        reference = make_wavefunction()
        path = write_molden(self.dir / "orca.molden.input", reference, orca=True)
        parsed = parse_molden(path)
        self.assertIn("ORCA conventions", parsed.notes[0])
        assert_same_orbitals(self, parsed, reference)
        # Read naively (the old behavior) the same file is far from orthonormal.
        naive = parse_molden(path, resolve_conventions=False)
        self.assertGreater(max(r.worst for r in orthonormality(naive)), 1e-2)

    def test_orca_conventions_are_detected_without_title_hint(self) -> None:
        reference = make_wavefunction(seed=3)
        path = write_molden(self.dir / "orca.molden", reference, orca=True)
        path.write_text(path.read_text().replace("Molden file created by orca_2mkl", "renamed"))
        assert_same_orbitals(self, parse_molden(path), reference)

    def test_mo_blocks_without_sym_lines(self) -> None:
        reference = make_wavefunction(unrestricted=True)
        parsed = parse_molden(write_molden(self.dir / "nosym.molden", reference, with_sym=False, angstrom=True))
        self.assertEqual(parsed.alpha_coefficients.shape, reference.alpha_coefficients.shape)
        self.assertEqual(parsed.beta_coefficients.shape, reference.beta_coefficients.shape)
        assert_same_orbitals(self, parsed, reference, "alpha")
        assert_same_orbitals(self, parsed, reference, "beta")
        self.assertEqual((parsed.homo_index("alpha"), parsed.homo_index("beta")), (4, 3))

    def test_cartesian_molden_including_15g_order(self) -> None:
        reference = make_wavefunction(cartesian=True)
        parsed = parse_molden(write_molden(self.dir / "cart.molden", reference))
        self.assertIn(4, set(parsed.shell_types))
        assert_same_orbitals(self, parsed, reference)

    def test_axis_normalized_cartesian_writer_is_recognized(self) -> None:
        reference = make_wavefunction(cartesian=True, seed=5)
        parsed = parse_molden(write_molden(self.dir / "axis.molden", reference, axis_cartesian=True))
        self.assertIn("axis-normalized Cartesian", parsed.notes[0])
        assert_same_orbitals(self, parsed, reference)

    def test_invalid_files_raise_clear_errors(self) -> None:
        reference = make_wavefunction()
        text = write_molden(self.dir / "ok.molden", reference).read_text()
        cases = {
            "no_mo": (text.split("[MO]")[0], "missing \\[MO\\]"),
            "bad_spin": (text.replace("Spin= Alpha", "Spin= Gamma", 1), "Invalid Molden MO spin"),
            "bad_index": (text.replace("\n    1  ", "\n  999  ", 1), "outside 1..34"),
            "bad_label": (text.replace("d 1 1.00", "k 1 1.00"), "Unsupported Molden shell label"),
        }
        for name, (content, message) in cases.items():
            with self.subTest(name=name):
                path = self.dir / f"{name}.molden"
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    parse_molden(path)


class FchkTests(unittest.TestCase):
    def test_fchk_round_trip_with_character_array_and_fortran_exponent(self) -> None:
        reference = make_wavefunction()
        with tempfile.TemporaryDirectory() as directory:
            path = write_fchk(Path(directory) / "test.fchk", reference)
            self.assertEqual(detect_wavefunction_format(path), "fchk")
            parsed = parse_wavefunction(path)
        self.assertEqual(parsed.title, "synthetic fchk")
        self.assertEqual(parsed.method_line, "SP RHF test")
        self.assertEqual(parsed.alpha_coefficients[0, 0], 1e-100)  # written as "1.00000000-100"
        parsed.alpha_coefficients[0, 0] = reference.alpha_coefficients[0, 0]
        assert_same_orbitals(self, parsed, reference)
        self.assertEqual((parsed.homo_index("alpha"), parsed.lumo_index("alpha")), (4, 5))
        self.assertFalse(parsed.notes)

    def test_fchk_header_glued_to_previous_data_line(self) -> None:
        """A missing newline must not swallow the next field (seen in concatenated files)."""
        reference = make_wavefunction()
        with tempfile.TemporaryDirectory() as directory:
            path = write_fchk(Path(directory) / "glued.fchk", reference)
            text = path.read_text().replace("\nAlpha MO coefficients", "Alpha MO coefficients", 1)
            text = text.replace("\nAlpha Orbital Energies", "\nTotal SCF Density                          R   N=           1\n  1.0E+00Alpha Orbital Energies", 1)
            path.write_text(text)
            parsed = parse_wavefunction(path)
        np.testing.assert_allclose(parsed.alpha_energies, reference.alpha_energies, rtol=1e-7)
        self.assertEqual(parsed.alpha_coefficients.shape, reference.alpha_coefficients.shape)

    def test_fchk_basis_mismatch_is_rejected(self) -> None:
        reference = make_wavefunction()
        with tempfile.TemporaryDirectory() as directory:
            path = write_fchk(Path(directory) / "bad.fchk", reference)
            path.write_text(path.read_text().replace("I               34", "I               35", 1))
            with self.assertRaisesRegex(ValueError, "Basis count mismatch|divisible"):
                parse_wavefunction(path)

    def test_format_detection_by_content_and_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            molden = write_molden(Path(directory) / "file.txt", make_wavefunction())
            self.assertEqual(detect_wavefunction_format(molden), "molden")
            unknown = Path(directory) / "unknown.dat"
            unknown.write_text("nothing to see\n")
            with self.assertRaises(ValueError):
                detect_wavefunction_format(unknown)


if __name__ == "__main__":
    unittest.main()
