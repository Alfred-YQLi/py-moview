"""Optional checks on real wavefunction files.

Large files are not committed.  Put them in ``tests/wavefunctions/`` or point
``MOVIEW_FIXTURES`` at a directory; every ``.fch``/``.fchk``/Molden file found is
parsed, checked for MO orthonormality and rendered once at a coarse grid.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from moview import orthonormality, parse_wavefunction, surface_for_orbital

FIXTURE_DIR = Path(os.environ.get("MOVIEW_FIXTURES") or Path(__file__).with_name("wavefunctions"))
FILES = sorted(
    p for p in FIXTURE_DIR.glob("*") if p.is_file() and not p.name.startswith("._") and (p.suffix in (".fch", ".fchk") or "molden" in p.name)
) if FIXTURE_DIR.is_dir() else []


@unittest.skipUnless(FILES, f"no wavefunction fixtures in {FIXTURE_DIR}")
class FixtureTests(unittest.TestCase):
    def test_every_fixture_is_orthonormal_and_renders(self) -> None:
        for path in FILES:
            with self.subTest(file=path.name):
                wf = parse_wavefunction(path)
                worst = max(report.worst for report in orthonormality(wf))
                # Natural-orbital files written with limited precision reach ~3e-3.
                self.assertLess(worst, 1e-2, f"{path.name}: MOs are not orthonormal ({worst:.1e})")
                if wf.source_format == "molden":
                    self.assertTrue(any("conventions" in note for note in wf.notes))
                positive, negative, _grid = surface_for_orbital(wf, "alpha", wf.homo_index("alpha"), 24, 0.05, 4.0)
                self.assertGreater(positive.n_faces + negative.n_faces, 0)


if __name__ == "__main__":
    unittest.main()
