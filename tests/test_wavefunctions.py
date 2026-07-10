from __future__ import annotations

import unittest
from pathlib import Path

from moview.parsers import detect_wavefunction_format, parse_wavefunction
from moview.surface import surface_for_orbital


WAVEFUNCTION_DIR = Path(__file__).with_name("wavefunctions")
SAMPLES = [
    WAVEFUNCTION_DIR / "save.molden.input",
    WAVEFUNCTION_DIR / "save_uks.fch",
    WAVEFUNCTION_DIR / "save_uks_UNO.fch",
    WAVEFUNCTION_DIR / "save_unouno.fch",
]


class WavefunctionSampleTests(unittest.TestCase):
    def require_samples(self, paths: list[Path]) -> None:
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            self.skipTest(f"Local wavefunction fixtures not available: {', '.join(missing)}")

    def test_missing_local_fixture_is_reported_as_skip(self) -> None:
        missing = WAVEFUNCTION_DIR / "fixture-that-does-not-exist.fch"
        with self.assertRaises(unittest.SkipTest):
            self.require_samples([missing])

    def test_parse_all_sample_wavefunctions(self) -> None:
        self.require_samples(SAMPLES)
        for path in SAMPLES:
            with self.subTest(path=path.name):
                wavefunction = parse_wavefunction(path)
                self.assertEqual(wavefunction.atomic_numbers.size, 94)
                self.assertGreater(wavefunction.n_basis, 0)
                self.assertEqual(wavefunction.alpha_coefficients.shape[1], wavefunction.n_basis)
                self.assertEqual(detect_wavefunction_format(path), wavefunction.source_format)
                with self.assertRaises(ValueError):
                    wavefunction.energies("invalid")
                with self.assertRaises(IndexError):
                    wavefunction.occupation("alpha", -1)

    def test_surface_for_representative_formats(self) -> None:
        try:
            import skimage  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("scikit-image is required for marching cubes")

        representative = [SAMPLES[0], SAMPLES[1]]
        self.require_samples(representative)
        for path in representative:
            with self.subTest(path=path.name):
                wavefunction = parse_wavefunction(path)
                orbital_index0 = wavefunction.default_orbital("alpha")
                pos, neg, level, shape = surface_for_orbital(
                    wavefunction,
                    "alpha",
                    orbital_index0,
                    grid_size=16,
                    iso=0.05,
                    margin_bohr=2.0,
                )
                self.assertEqual(level, 0.05)
                self.assertEqual(len(shape), 3)
                self.assertGreater(pos.n_faces + neg.n_faces, 0)


if __name__ == "__main__":
    unittest.main()
