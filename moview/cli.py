from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from pathlib import Path

from .config import AppConfig, ConfigError, DEFAULT_CONFIG, load_config
from .constants import HARTREE_TO_EV
from .grid import MIN_GRID_SIZE, make_grid_spec
from .parsers import parse_wavefunction
from .surface import surface_for_orbital


def run_batch(args: argparse.Namespace, app_config: AppConfig = DEFAULT_CONFIG) -> int:
    wf = parse_wavefunction(Path(args.input), args.file_format)
    if args.grid > 256:
        spec = make_grid_spec(wf, args.grid, args.margin)
        scalar_gib = spec.scalar_nbytes(float) / 1024**3
        print(
            f"warning: Grid {args.grid} creates {spec.n_points:,} points "
            f"({scalar_gib:.2f} GiB per float64 scalar field); computation may be slow.",
            file=sys.stderr,
        )
    orbital_index0 = args.orbital - 1
    pos, neg, level, shape = surface_for_orbital(
        wf,
        args.spin,
        orbital_index0,
        args.grid,
        args.iso,
        args.margin,
        chunk_points=app_config.resources.grid_chunk_points,
    )
    energy = wf.energies(args.spin)[orbital_index0]
    occ = wf.occupation(args.spin, orbital_index0)
    print(f"file: {wf.path}")
    print(f"format: {wf.source_format}")
    print(f"atoms: {len(wf.atomic_numbers)}  basis: {wf.n_basis}")
    print(f"spin: {args.spin}  orbital: {args.orbital}  occupation: {occ:g}")
    print(f"energy: {energy:.8f} Eh  {energy * HARTREE_TO_EV:.4f} eV")
    print(f"grid: {shape}  isovalue: {level:.6g}")
    print(f"positive triangles: {pos.n_faces}")
    print(f"negative triangles: {neg.n_faces}")
    return 0


def build_parser(app_config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moview",
        description="Read fchk/Molden files and view molecular orbital isosurfaces with OpenGL.",
    )
    parser.add_argument("input", nargs="?", help="Path to .fchk/.fch/Molden file")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Configuration file; overrides automatic config discovery",
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--fchk",
        dest="file_format",
        action="store_const",
        const="fchk",
        help="Treat input as Gaussian fchk",
    )
    format_group.add_argument(
        "--molden",
        dest="file_format",
        action="store_const",
        const="molden",
        help="Treat input as Molden",
    )
    parser.add_argument("--batch", action="store_true", help="Run a non-GUI parse/evaluate/surface test")
    parser.add_argument("--spin", choices=["alpha", "beta"], default="alpha")
    parser.add_argument("--orbital", type=int, default=1, help="1-based orbital index for --batch")
    parser.add_argument(
        "--grid",
        type=int,
        default=app_config.render.grid,
        help=f"Grid points along the longest axis (default: {app_config.render.grid})",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=app_config.render.isovalue,
        help=f"Positive isovalue (default: {app_config.render.isovalue:g})",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=app_config.render.margin_bohr,
        help=f"Box margin in bohr (default: {app_config.render.margin_bohr:g})",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=app_config.resources.basis_workers,
        help=(
            "Worker threads for cached BasisGrid construction "
            f"(default: {app_config.resources.basis_workers})"
        ),
    )
    auto_render_group = parser.add_mutually_exclusive_group()
    auto_render_group.add_argument(
        "--auto-render",
        dest="auto_render",
        action="store_true",
        help="Automatically render the default orbital after opening",
    )
    auto_render_group.add_argument(
        "--no-auto-render",
        dest="auto_render",
        action="store_false",
        help="Do not automatically render HOMO after opening the GUI",
    )
    parser.set_defaults(auto_render=app_config.view.auto_render)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else None
    config_parser = argparse.ArgumentParser(prog="moview", add_help=False)
    config_parser.add_argument("--config")
    config_args, _remaining = config_parser.parse_known_args(arguments)
    try:
        app_config = load_config(config_args.config)
    except ConfigError as exc:
        config_parser.error(str(exc))

    parser = build_parser(app_config)
    args = parser.parse_args(arguments)
    if args.grid < MIN_GRID_SIZE:
        parser.error(f"--grid must be at least {MIN_GRID_SIZE}")
    if not math.isfinite(args.margin) or args.margin < 0.0:
        parser.error("--margin must be finite and non-negative")
    if not math.isfinite(args.iso) or args.iso <= 0.0:
        parser.error("--iso must be a finite positive value")
    if args.orbital < 1:
        parser.error("--orbital must be at least 1")
    if args.prefetch_workers < 1 or args.prefetch_workers > 256:
        parser.error("--prefetch-workers must be between 1 and 256")
    if args.batch:
        if not args.input:
            parser.error("--batch requires an input path")
        return run_batch(args, app_config)
    from .gui.native_stderr import filter_macos_gui_warnings

    with filter_macos_gui_warnings():
        from .gui.main_window import run_gui

        return run_gui(
            args.input,
            args.grid,
            args.iso,
            args.margin,
            args.prefetch_workers,
            auto_render=args.auto_render,
            file_format=args.file_format,
            app_config=app_config,
        )
