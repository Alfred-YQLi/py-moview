from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from moview.basis.gaussian import n_shell_functions
from moview.parsers import detect_wavefunction_format, parse_wavefunction
from moview.surface import surface_for_orbital


WAVEFUNCTION_DIR = Path(__file__).with_name("wavefunctions")
MOLDEN_SAMPLE = WAVEFUNCTION_DIR / "save.molden.input"
FCHK_SAMPLES = (
    WAVEFUNCTION_DIR / "save_uks.fch",
    WAVEFUNCTION_DIR / "save_uks_UNO.fch",
    WAVEFUNCTION_DIR / "save_unouno.fch",
)


def require_samples(paths: tuple[Path, ...]) -> None:
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise unittest.SkipTest(f"Local wavefunction fixtures not available: {', '.join(missing)}")


class MoldenSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        require_samples((MOLDEN_SAMPLE,))
        cls.wavefunction = parse_wavefunction(MOLDEN_SAMPLE)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.wavefunction

    def test_molden_metadata_basis_and_orbitals(self) -> None:
        wavefunction = self.wavefunction
        self.assertEqual(detect_wavefunction_format(MOLDEN_SAMPLE), "molden")
        self.assertEqual(wavefunction.source_format, "molden")
        self.assertEqual(
            wavefunction.title,
            "Molden file created by orca_2mkl for BaseName=5ucw_8A-IM1_orca",
        )

        self.assertEqual(wavefunction.atomic_numbers.size, 94)
        self.assertEqual(int(wavefunction.atomic_numbers[0]), 7)
        self.assertEqual(int(wavefunction.atomic_numbers[38]), 26)
        self.assertEqual(int(wavefunction.atomic_numbers[-1]), 1)
        np.testing.assert_allclose(
            wavefunction.coordinates_bohr[0],
            (98.2471015603, 92.7373687229, 90.2897459338),
            rtol=0.0,
            atol=1.0e-10,
        )

        self.assertEqual(len(wavefunction.shells), 454)
        self.assertEqual(wavefunction.n_basis, 986)
        self.assertEqual(
            Counter(wavefunction.shell_types.tolist()),
            {0: 246, 1: 151, -2: 56, -3: 1},
        )
        self.assertEqual(sum(shell.exponents.size for shell in wavefunction.shells), 878)
        self.assertEqual(
            sum(n_shell_functions(int(shell_type)) for shell_type in wavefunction.shell_types),
            wavefunction.n_basis,
        )
        self.assertEqual(wavefunction.shell_to_atom.shape, (454,))
        self.assertEqual(int(wavefunction.shell_to_atom.min()), 0)
        self.assertEqual(int(wavefunction.shell_to_atom.max()), 93)

        self.assertTrue(wavefunction.is_unrestricted)
        self.assertEqual(wavefunction.alpha_coefficients.shape, (986, 986))
        self.assertIsNotNone(wavefunction.beta_coefficients)
        assert wavefunction.beta_coefficients is not None
        self.assertEqual(wavefunction.beta_coefficients.shape, (986, 986))
        self.assertEqual(wavefunction.alpha_energies.shape, (986,))
        self.assertIsNotNone(wavefunction.beta_energies)
        assert wavefunction.beta_energies is not None
        self.assertEqual(wavefunction.beta_energies.shape, (986,))
        self.assertTrue(np.isfinite(wavefunction.alpha_energies).all())
        self.assertTrue(np.isfinite(wavefunction.beta_energies).all())

        self.assertEqual(wavefunction.n_alpha, 212)
        self.assertEqual(wavefunction.n_beta, 208)
        self.assertIsNotNone(wavefunction.alpha_occupations)
        self.assertIsNotNone(wavefunction.beta_occupations)
        assert wavefunction.alpha_occupations is not None
        assert wavefunction.beta_occupations is not None
        self.assertEqual(float(wavefunction.alpha_occupations.sum()), 212.0)
        self.assertEqual(float(wavefunction.beta_occupations.sum()), 208.0)
        self.assertEqual(int(np.count_nonzero(wavefunction.alpha_occupations)), 212)
        self.assertEqual(int(np.count_nonzero(wavefunction.beta_occupations)), 208)
        self.assertEqual(wavefunction.default_orbital("alpha"), 211)
        self.assertEqual(wavefunction.default_orbital("beta"), 207)
        self.assertEqual(wavefunction.lumo_orbital("alpha"), 212)
        self.assertEqual(wavefunction.lumo_orbital("beta"), 208)

        self.assertAlmostEqual(wavefunction.alpha_energies[0], -256.388734660997, places=12)
        self.assertAlmostEqual(wavefunction.alpha_energies[211], -0.158367063985074, places=12)
        self.assertAlmostEqual(wavefunction.beta_energies[207], -0.158053247792225, places=12)
        self.assertAlmostEqual(wavefunction.beta_energies[-1], 4.34302510445390, places=12)
        self.assertAlmostEqual(wavefunction.alpha_coefficients[0, 0], -0.000002125672, places=12)
        self.assertAlmostEqual(
            wavefunction.alpha_coefficients[211, 985],
            -0.000055113355,
            places=12,
        )
        self.assertAlmostEqual(wavefunction.beta_coefficients[0, 0], -0.000002126815, places=12)
        self.assertAlmostEqual(wavefunction.beta_coefficients[-1, -1], 0.002111956070, places=12)

    def test_molden_alpha_and_beta_homo_surfaces(self) -> None:
        for spin in ("alpha", "beta"):
            with self.subTest(spin=spin):
                orbital_index0 = self.wavefunction.default_orbital(spin)
                positive, negative, level, shape = surface_for_orbital(
                    self.wavefunction,
                    spin,
                    orbital_index0,
                    grid_size=16,
                    iso=0.05,
                    margin_bohr=2.0,
                )
                self.assertEqual(level, 0.05)
                self.assertEqual(len(shape), 3)
                self.assertGreater(positive.n_faces, 0)
                self.assertGreater(negative.n_faces, 0)


class FCHKSampleTests(unittest.TestCase):
    def test_parse_fchk_samples(self) -> None:
        require_samples(FCHK_SAMPLES)
        for path in FCHK_SAMPLES:
            with self.subTest(path=path.name):
                wavefunction = parse_wavefunction(path)
                self.assertEqual(wavefunction.atomic_numbers.size, 94)
                self.assertGreater(wavefunction.n_basis, 0)
                self.assertEqual(wavefunction.alpha_coefficients.shape[1], wavefunction.n_basis)
                self.assertEqual(detect_wavefunction_format(path), "fchk")
                self.assertEqual(wavefunction.source_format, "fchk")
                with self.assertRaises(ValueError):
                    wavefunction.energies("invalid")
                with self.assertRaises(IndexError):
                    wavefunction.occupation("alpha", -1)

    def test_fchk_representative_surface(self) -> None:
        path = FCHK_SAMPLES[0]
        require_samples((path,))
        wavefunction = parse_wavefunction(path)
        orbital_index0 = wavefunction.default_orbital("alpha")
        positive, negative, level, shape = surface_for_orbital(
            wavefunction,
            "alpha",
            orbital_index0,
            grid_size=16,
            iso=0.05,
            margin_bohr=2.0,
        )
        self.assertEqual(level, 0.05)
        self.assertEqual(len(shape), 3)
        self.assertGreater(positive.n_faces + negative.n_faces, 0)


class FixtureBehaviorTests(unittest.TestCase):
    def test_missing_local_fixture_is_reported_as_skip(self) -> None:
        missing = WAVEFUNCTION_DIR / "fixture-that-does-not-exist.fch"
        with self.assertRaises(unittest.SkipTest):
            require_samples((missing,))


if __name__ == "__main__":
    unittest.main()
