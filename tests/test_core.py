"""Numerical core: basis conventions, evaluation, grids, surfaces, bonds, overlaps."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from moview.constants import BOHR_TO_ANG
from moview.core.basis import CARTESIAN_POWERS, SPHERICAL_TO_CARTESIAN, cartesian_factor, n_shell_functions
from moview.core.bonds import compute_bonds
from moview.core.evaluate import evaluate_orbitals, evaluate_shell
from moview.core.grid import compute_orbital_grid, compute_orbital_grids, make_grid_spec
from moview.core.overlap import orthonormality, overlap_matrix
from moview.core.surface import extract_isosurfaces
from moview.core.wavefunction import Shell, Wavefunction
from moview.parsers.numbers import fortran_float, parse_floats

from helpers import make_wavefunction


def single_shell_wavefunction(shell_type: int, center=(0.0, 0.0, 0.0), alpha: float = 0.8) -> Wavefunction:
    center = np.asarray(center, dtype=np.float64)
    n = n_shell_functions(shell_type)
    shell = Shell(shell_type, center, np.array([alpha]), np.array([1.0]))
    return Wavefunction(
        Path("x"), "", "", np.array([6]), center[None, :], 1, 1, n, np.array([shell_type]), np.array([0]),
        [shell], np.zeros(n), None, np.eye(n), None,
    )


class BasisConventionTests(unittest.TestCase):
    def test_shell_function_counts(self) -> None:
        self.assertEqual([n_shell_functions(t) for t in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)], [11, 9, 7, 5, 4, 1, 3, 6, 10, 15, 21])
        self.assertEqual(n_shell_functions(9), 0)

    def test_pure_functions_match_real_solid_harmonics(self) -> None:
        """Columns are m = 0, +1, -1, +2, -2, ... with one common positive scale."""
        from scipy.special import sph_harm_y

        rng = np.random.default_rng(0)
        v = rng.normal(size=(300, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        theta, phi = np.arccos(v[:, 2]), np.arctan2(v[:, 1], v[:, 0])
        for l in (2, 3, 4, 5):
            cart = np.stack(
                [v[:, 0] ** a * v[:, 1] ** b * v[:, 2] ** c * cartesian_factor(a, b, c) for a, b, c in CARTESIAN_POWERS[l]], 1
            )
            ours = cart @ SPHERICAL_TO_CARTESIAN[l]
            ms = [0] + [s * k for k in range(1, l + 1) for s in (1, -1)]
            scales = []
            for column, m in enumerate(ms):
                y = sph_harm_y(l, abs(m), theta, phi) * (-1) ** abs(m)  # remove Condon-Shortley phase
                ref = y.real if m == 0 else math.sqrt(2.0) * (y.real if m > 0 else y.imag)
                scale = float(ours[:, column] @ ref / (ref @ ref))
                np.testing.assert_allclose(ours[:, column], scale * ref, atol=1e-12, err_msg=f"l={l} m={m}")
                scales.append(scale)
            self.assertGreater(min(scales), 0.0)
            np.testing.assert_allclose(scales, scales[0], rtol=1e-12)

    def test_pure_shells_are_orthonormal_and_cartesian_shells_normalized(self) -> None:
        for shell_type in (0, 1, 2, -2, 3, -3, 4, -4, 5, -5):
            with self.subTest(shell_type=shell_type):
                overlap = overlap_matrix(single_shell_wavefunction(shell_type))
                np.testing.assert_allclose(np.diag(overlap), 1.0, atol=1e-12)
                if shell_type < 0 or shell_type <= 1:
                    np.testing.assert_allclose(overlap, np.eye(overlap.shape[0]), atol=1e-12)

    def test_two_center_s_overlap_matches_closed_form(self) -> None:
        a, b, r = 0.7, 1.9, 1.3
        wf = single_shell_wavefunction(0, alpha=a)
        wf.shells.append(Shell(0, np.array([0.0, 0.0, r]), np.array([b]), np.array([1.0])))
        wf.n_basis = 2
        wf.alpha_coefficients = np.eye(2)
        expected = (2.0 * math.sqrt(a * b) / (a + b)) ** 1.5 * math.exp(-a * b / (a + b) * r * r)
        self.assertAlmostEqual(overlap_matrix(wf)[0, 1], expected, places=13)

    def test_shell_evaluation_matches_analytic_p_function(self) -> None:
        alpha = 0.6
        shell = Shell(1, np.zeros(3), np.array([alpha]), np.array([1.0]))
        points = np.array([[0.3, -0.2, 0.5], [1.0, 1.0, 1.0]])
        values = evaluate_shell(shell, points)
        norm = (2 * alpha / math.pi) ** 0.75 * 2 * math.sqrt(alpha)
        r2 = (points**2).sum(1)
        np.testing.assert_allclose(values, (norm * points * np.exp(-alpha * r2)[:, None]).T, rtol=1e-12)

    def test_synthetic_mos_are_orthonormal(self) -> None:
        for kwargs in ({}, {"cartesian": True}, {"unrestricted": True}):
            with self.subTest(**kwargs):
                reports = orthonormality(make_wavefunction(**kwargs))
                self.assertLess(max(report.worst for report in reports), 1e-10)


class GridAndSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = np.array([1.0, 2.0, 3.0])
        self.px = single_shell_wavefunction(1, self.center, alpha=0.5)

    def test_grid_has_cubic_voxels_and_is_centered(self) -> None:
        spec = make_grid_spec(self.px, 40, 5.0)
        self.assertEqual(spec.shape, (40, 40, 40))
        np.testing.assert_allclose(0.5 * (spec.origin + spec.high), self.center)
        with self.assertRaises(ValueError):
            make_grid_spec(self.px, 4, 1.0)
        with self.assertRaises(ValueError):
            make_grid_spec(self.px, 16, -1.0)

    def test_separable_grid_evaluation_is_exact(self) -> None:
        """Tensor-product evaluation equals point-wise evaluation (pure, Cartesian, all l)."""
        for kwargs in ({}, {"cartesian": True}):
            wf = make_wavefunction(**kwargs)
            spec = make_grid_spec(wf, 24, 3.0)
            reference = compute_orbital_grids(wf, "alpha", [3, 4], spec, dtype=np.float64, chunk_points=10**7)
            sliced = compute_orbital_grids(wf, "alpha", [3, 4], spec, dtype=np.float64, chunk_points=512)
            points = np.column_stack([axis.ravel() for axis in np.meshgrid(*[spec.axis(d) for d in range(3)], indexing="ij")])
            direct = evaluate_orbitals(wf, "alpha", [3, 4], points).reshape((2,) + spec.shape)
            for row in range(2):
                np.testing.assert_allclose(sliced[row].values, reference[row].values, atol=1e-12)
                np.testing.assert_allclose(reference[row].values, direct[row], atol=1e-10)
            single = compute_orbital_grids(wf, "alpha", [4], spec, dtype=np.float32)[0]
            np.testing.assert_allclose(single.values, direct[1], atol=1e-5)

    def test_px_lobes_are_placed_and_oriented_outwards(self) -> None:
        grid = compute_orbital_grid(self.px, "alpha", 0, 48, 5.0)
        positive, negative, level = extract_isosurfaces(grid, 0.05)
        self.assertEqual(level, 0.05)
        atom = self.center * BOHR_TO_ANG
        for mesh, sign in ((positive, 1.0), (negative, -1.0)):
            centroid = mesh.vertices.mean(axis=0)
            self.assertGreater(sign * (centroid - atom)[0], 0.3)  # + lobe at +x, - lobe at -x
            np.testing.assert_allclose((centroid - atom)[1:], 0.0, atol=1e-3)
            triangles = mesh.vertices[mesh.faces]
            face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            outward = triangles.mean(axis=1) - centroid
            self.assertTrue(np.all(np.einsum("ij,ij->i", face_normals, outward) > 0), "winding must face outwards")
            self.assertGreater(np.mean(np.einsum("ij,ij->i", mesh.normals, mesh.vertices - centroid) > 0), 0.99)
            np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-4)

    def test_isovalue_outside_field_gives_empty_meshes(self) -> None:
        grid = compute_orbital_grid(self.px, "alpha", 0, 16, 2.0)
        positive, negative, _ = extract_isosurfaces(grid, 50.0)
        self.assertEqual((positive.n_faces, negative.n_faces), (0, 0))
        with self.assertRaises(ValueError):
            extract_isosurfaces(grid, 0.0)


class WavefunctionModelTests(unittest.TestCase):
    def test_frontier_indices_and_labels(self) -> None:
        wf = make_wavefunction()
        self.assertEqual((wf.homo_index("alpha"), wf.lumo_index("alpha")), (4, 5))
        self.assertEqual(wf.occupation("alpha", 0), 2.0)
        self.assertEqual([wf.frontier_label("alpha", i) for i in (3, 4, 5, 7)], ["HOMO-1", "HOMO", "LUMO", "LUMO+2"])
        self.assertEqual(wf.spins, ("alpha",))
        uwf = make_wavefunction(unrestricted=True)
        self.assertEqual((uwf.homo_index("beta"), uwf.lumo_index("beta")), (3, 4))
        self.assertEqual(uwf.occupation("alpha", 4), 1.0)

    def test_explicit_occupations_win(self) -> None:
        wf = make_wavefunction()
        wf.alpha_occupations = np.zeros(wf.n_basis)
        wf.alpha_occupations[[0, 1, 3]] = 2.0  # a hole below the highest occupied orbital
        self.assertEqual((wf.homo_index("alpha"), wf.lumo_index("alpha")), (3, 2))


class BondTests(unittest.TestCase):
    def test_bonds_use_full_covalent_radius_table(self) -> None:
        # Cu-N at 2.0 A was missed by the old 13-element table (default 0.75 A radius).
        numbers = np.array([29, 7, 1, 0])
        coords = np.array([[0, 0, 0], [2.0, 0, 0], [2.6, 0.75, 0], [0, 0, 0.3]], dtype=float)
        bonds = compute_bonds(numbers, coords)
        self.assertEqual([tuple(map(int, b)) for b in bonds], [(0, 1), (1, 2)])  # ghost atom 3 never bonds

    def test_no_bonds_for_single_atom(self) -> None:
        self.assertEqual(compute_bonds(np.array([6]), np.zeros((1, 3))).shape, (0, 2))


class NumberParsingTests(unittest.TestCase):
    def test_fortran_numbers(self) -> None:
        self.assertEqual(fortran_float("1.5D+02"), 150.0)
        self.assertEqual(fortran_float("-2.0d-3"), -0.002)
        self.assertEqual(fortran_float("1.00000000-100"), 1e-100)
        np.testing.assert_allclose(parse_floats(" 1.0D0  2.5-101\n3.0E+00 ", 3), [1.0, 2.5e-101, 3.0])
        with self.assertRaises(ValueError):
            parse_floats("1.0 2.0", 3)


if __name__ == "__main__":
    unittest.main()
