from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moview.cli import main
from moview.config import ColorPreset, ConfigError, DEFAULT_CONFIG, load_config


class ConfigTests(unittest.TestCase):
    def write_config(self, directory: str, contents: str, name: str = "custom.ini") -> Path:
        path = Path(directory) / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_builtin_defaults_are_stable(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.render.label_placement, "attached")
        self.assertEqual(DEFAULT_CONFIG.render.isovalue, 0.05)
        self.assertEqual(DEFAULT_CONFIG.color("red").hex_color, "#ef3b2c")
        self.assertEqual(DEFAULT_CONFIG.color("BLUE").hex_color, "#2563eb")

    def test_example_config_matches_builtin_defaults(self) -> None:
        example_path = Path(__file__).resolve().parents[1] / "moview.example.ini"
        example = load_config(example_path)

        self.assertEqual(example.resources, DEFAULT_CONFIG.resources)
        self.assertEqual(example.render, DEFAULT_CONFIG.render)
        self.assertEqual(example.view, DEFAULT_CONFIG.view)
        self.assertEqual(example.colors, DEFAULT_CONFIG.colors)

    def test_custom_config_overrides_and_adds_named_colors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                """
[resources]
basis_workers = 4
background_jobs = 2
grid_chunk_points = 65536
surface_face_limit = 90000

[render]
grid = 96
surface_style = solid_edges
atom_style = licorice
label_placement = floating
positive_color = Mint
negative_color = red

[view]
zoom = 1.25
sync_views = false
show_axes = false
auto_render = false

[colors]
Red = 255, 10, 20
Mint = 40, 210, 160
""",
            )
            config = load_config(path)

        self.assertEqual(config.resources.basis_workers, 4)
        self.assertEqual(config.resources.background_jobs, 2)
        self.assertEqual(config.resources.grid_chunk_points, 65_536)
        self.assertEqual(config.resources.surface_face_limit, 90_000)
        self.assertEqual(config.render.grid, 96)
        self.assertEqual(config.render.surface_style, "solid_edges")
        self.assertEqual(config.render.atom_style, "licorice")
        self.assertEqual(config.render.label_placement, "floating")
        self.assertEqual(config.render.positive_color, "Mint")
        self.assertEqual(config.render.negative_color, "Red")
        self.assertEqual(config.color("Red").rgb, (255, 10, 20))
        self.assertEqual(config.color("Mint").hex_color, "#28d2a0")
        self.assertEqual(config.view.zoom, 1.25)
        self.assertFalse(config.view.sync_views)
        self.assertFalse(config.view.show_axes)
        self.assertFalse(config.view.auto_render)

    def test_explicit_path_precedes_environment_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = self.write_config(directory, "[render]\ngrid = 72\n", "explicit.ini")
            environment = self.write_config(directory, "[render]\ngrid = 88\n", "env.ini")
            config = load_config(
                explicit,
                environ={"MOVIEW_CONFIG": str(environment), "HOME": directory},
                cwd=Path(directory),
            )
        self.assertEqual(config.render.grid, 72)

    def test_invalid_color_and_unknown_setting_are_rejected(self) -> None:
        cases = (
            ("[colors]\nBad = 256, 0, 0\n", "between 0 and 255"),
            ("[render]\nunknown = value\n", "unknown setting"),
            ("[render]\npositive_color = Missing\n", "unknown color"),
            ("[render]\nlabel_placement = detached\n", "must be one of"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (contents, message) in enumerate(cases):
                with self.subTest(message=message):
                    path = self.write_config(directory, contents, f"invalid-{index}.ini")
                    with self.assertRaisesRegex(ConfigError, message):
                        load_config(path)

    def test_invalid_memory_relationship_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                "[resources]\nbasis_cache_mib = 128\nmax_basis_cache_entry_mib = 256\n",
            )
            with self.assertRaisesRegex(ConfigError, "cannot exceed"):
                load_config(path)

    def test_programmatic_color_presets_are_validated(self) -> None:
        with self.assertRaises(ConfigError):
            ColorPreset("Bad", (256, 0, 0))
        with self.assertRaises(ConfigError):
            ColorPreset(" bad ", (1, 2, 3))

    def test_cli_uses_config_defaults_and_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                """
[resources]
basis_workers = 3
grid_chunk_points = 32768

[render]
grid = 72
margin_bohr = 2.5
isovalue = 0.04
""",
            )
            with patch("moview.cli.run_batch", return_value=0) as run_batch:
                result = main(["--config", str(path), "sample.fch", "--batch", "--grid", "80"])

        self.assertEqual(result, 0)
        args, config = run_batch.call_args.args
        self.assertEqual(args.grid, 80)
        self.assertEqual(args.margin, 2.5)
        self.assertEqual(args.iso, 0.04)
        self.assertEqual(args.prefetch_workers, 3)
        self.assertEqual(config.resources.grid_chunk_points, 32_768)


if __name__ == "__main__":
    unittest.main()
