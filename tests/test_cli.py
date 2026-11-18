"""Command line: batch mode, orbital names, the --check report and the PATH launcher."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from moview.cli import main, resolve_orbital

from helpers import make_wavefunction, write_molden


def run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(args)
        except SystemExit as exc:  # argparse errors
            code = int(exc.code)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wf = make_wavefunction(unrestricted=True)
        cls.path = write_molden(Path(cls.tmp.name) / "uhf.molden", cls.wf, orca=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_batch_defaults_to_homo_and_reports_check(self) -> None:
        code, out, _ = run([str(self.path), "--batch", "--grid", "20", "--check"])
        self.assertEqual(code, 0)
        self.assertIn("orbital: alpha 5 (HOMO)", out)
        self.assertIn("ORCA conventions", out)
        self.assertEqual(out.count("[OK]"), 2)
        self.assertIn("surface:", out)

    def test_named_orbitals(self) -> None:
        self.assertEqual(resolve_orbital(self.wf, "beta", "homo"), 3)
        self.assertEqual(resolve_orbital(self.wf, "alpha", "lumo+2"), 7)
        self.assertEqual(resolve_orbital(self.wf, "alpha", "12"), 11)
        code, out, _ = run([str(self.path), "--batch", "--grid", "16", "--spin", "beta", "--orbital", "LUMO"])
        self.assertEqual(code, 0)
        self.assertIn("orbital: beta 5 (LUMO)", out)

    def test_invalid_requests_fail_cleanly(self) -> None:
        self.assertEqual(run([str(self.path), "--batch", "--orbital", "999"])[0], 2)
        self.assertEqual(run([str(self.path), "--batch", "--orbital", "middle"])[0], 2)
        self.assertEqual(run([str(self.path), "--batch", "--grid", "4"])[0], 2)
        self.assertEqual(run([str(self.path), "--batch", "--iso", "-1"])[0], 2)
        code, _, err = run([str(Path(self.tmp.name) / "missing.fchk"), "--batch"])
        self.assertEqual(code, 1)
        self.assertIn("error", err)

    def test_path_launcher_works_outside_project_directory(self) -> None:
        launcher = Path(__file__).resolve().parents[1] / "bin" / "moview"
        self.assertTrue(os.access(launcher, os.X_OK))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(launcher), "--help"], cwd=directory, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: moview", result.stdout)


if __name__ == "__main__":
    unittest.main()
