"""Command line entry point: ``moview [FILE] [options]``.

Without ``--batch`` the Qt GUI is started (imported lazily so batch mode works
without PyQt/OpenGL).  Configuration is loaded first so that its values become
argument defaults; explicit options always win.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from . import __version__
from .config import AppConfig, ConfigError, load_config
from .constants import HARTREE_TO_EV, HIGH_GRID_WARNING_THRESHOLD, MIN_GRID_SIZE
from .core.grid import make_grid_spec
from .core.surface import surface_for_orbital
from .core.wavefunction import Wavefunction
from .parsers import parse_wavefunction

_ORBITAL_NAME = re.compile(r"^(homo|lumo)\s*([+-]\s*\d+)?$", re.IGNORECASE)


def _orbital_spec(text: str) -> str:
    text = text.strip()
    if text.isdigit() and int(text) >= 1:
        return text
    if _ORBITAL_NAME.match(text):
        return text.lower().replace(" ", "")
    raise argparse.ArgumentTypeError("use a 1-based number or homo, lumo, homo-1, lumo+2, ...")


def resolve_orbital(wavefunction: Wavefunction, spin: str, spec: str | None) -> int:
    """Translate ``"12"``, ``"homo"``, ``"lumo+1"`` ... into a 0-based index."""
    n_orbitals = wavefunction.n_orbitals(spin)
    if spec is None:
        index = wavefunction.homo_index(spin)
    elif spec.isdigit():
        index = int(spec) - 1
    else:
        match = _ORBITAL_NAME.match(spec)
        assert match is not None
        base = wavefunction.homo_index(spin) if match.group(1).lower() == "homo" else wavefunction.lumo_index(spin)
        index = base + (int(match.group(2).replace(" ", "")) if match.group(2) else 0)
    if not 0 <= index < n_orbitals:
        raise ValueError(f"orbital {spec or 'HOMO'} resolves to {index + 1}, outside 1..{n_orbitals}")
    return index


def _describe(wavefunction: Wavefunction) -> list[str]:
    lines = [
        f"file:    {wavefunction.path}",
        f"format:  {wavefunction.source_format}   {wavefunction.method_line}".rstrip(),
        f"atoms:   {wavefunction.n_atoms}   basis functions: {wavefunction.n_basis}   "
        f"{'unrestricted' if wavefunction.is_unrestricted else 'restricted'}",
    ]
    for spin in wavefunction.spins:
        homo, lumo = wavefunction.homo_index(spin), wavefunction.lumo_index(spin)
        lines.append(f"{spin:<7}  HOMO {homo + 1}   LUMO {lumo + 1}   orbitals {wavefunction.n_orbitals(spin)}")
    lines.extend(f"note:    {note}" for note in wavefunction.notes)
    return lines


def run_batch(args: argparse.Namespace, app_config: AppConfig) -> int:
    wavefunction = parse_wavefunction(Path(args.input), args.file_format)
    spin = args.spin if args.spin in wavefunction.spins else "alpha"
    if spin != args.spin:
        print(f"warning: restricted wavefunction; using alpha orbitals for --spin {args.spin}", file=sys.stderr)
    try:
        index = resolve_orbital(wavefunction, spin, args.orbital)
    except ValueError as exc:
        print(f"moview: error: {exc}", file=sys.stderr)
        return 2

    for line in _describe(wavefunction):
        print(line)

    if args.check:
        from .core.overlap import orthonormality

        for report in orthonormality(wavefunction):
            status = "OK" if report.worst < 1.0e-3 else "SUSPICIOUS"
            print(
                f"check:   {report.spin} MOs orthonormality  max|<i|i>-1| {report.max_norm_error:.1e}"
                f"   max|<i|j>| {report.max_overlap:.1e}   [{status}]"
            )

    if args.grid > HIGH_GRID_WARNING_THRESHOLD:
        spec = make_grid_spec(wavefunction, args.grid, args.margin)
        print(
            f"warning: grid {args.grid} has {spec.n_points:,} points "
            f"({spec.scalar_nbytes(float) / 1024**3:.2f} GiB per float64 field); this may be slow.",
            file=sys.stderr,
        )
    positive, negative, grid = surface_for_orbital(
        wavefunction,
        spin,
        index,
        args.grid,
        args.iso,
        args.margin,
        chunk_points=app_config.resources.grid_chunk_points,
    )
    energy = float(wavefunction.energies(spin)[index])
    label = wavefunction.frontier_label(spin, index)
    print(f"orbital: {spin} {index + 1}{f' ({label})' if label else ''}   occupation {wavefunction.occupation(spin, index):g}")
    if math.isfinite(energy):
        print(f"energy:  {energy:.8f} Eh   {energy * HARTREE_TO_EV:.4f} eV")
    print(f"grid:    {grid.shape}   isovalue +/-{args.iso:g}")
    print(f"surface: {positive.n_faces} positive and {negative.n_faces} negative triangles")
    return 0


def build_parser(app_config: AppConfig) -> argparse.ArgumentParser:
    render = app_config.render
    parser = argparse.ArgumentParser(
        prog="moview",
        description="View molecular orbitals from Gaussian FCHK and Molden files.",
        epilog="Without --batch the graphical viewer is opened.",
    )
    parser.add_argument("input", nargs="?", help="wavefunction file (.fchk, .fch, .molden, .molden.input)")
    parser.add_argument("--version", action="version", version=f"MOview {__version__}")
    parser.add_argument("--config", metavar="PATH", help="configuration file (overrides automatic discovery)")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--fchk", dest="file_format", action="store_const", const="fchk", help="force Gaussian FCHK")
    fmt.add_argument("--molden", dest="file_format", action="store_const", const="molden", help="force Molden")

    compute = parser.add_argument_group("computation")
    compute.add_argument("--grid", type=int, default=render.grid, help=f"points along the longest box edge (default {render.grid})")
    compute.add_argument("--iso", type=float, default=render.isovalue, help=f"positive isovalue (default {render.isovalue:g})")
    compute.add_argument("--margin", type=float, default=render.margin_bohr, help=f"box margin in bohr (default {render.margin_bohr:g})")
    # Accepted for compatibility with 0.1/0.2 command lines; evaluation now uses BLAS threads.
    compute.add_argument("--prefetch-workers", type=int, default=None, help=argparse.SUPPRESS)

    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--batch", action="store_true", help="parse, evaluate one orbital and print statistics (no GUI)")
    batch.add_argument("--spin", choices=["alpha", "beta"], default="alpha")
    batch.add_argument("--orbital", type=_orbital_spec, default=None, help="1-based number or homo/lumo[+-n] (default: HOMO)")
    batch.add_argument("--check", action="store_true", help="verify MO orthonormality with analytic overlaps")

    gui = parser.add_argument_group("GUI")
    auto = gui.add_mutually_exclusive_group()
    auto.add_argument("--auto-render", dest="auto_render", action="store_true", help="render the HOMO after opening")
    auto.add_argument("--no-auto-render", dest="auto_render", action="store_false", help="do not render automatically")
    parser.set_defaults(auto_render=app_config.view.auto_render)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else None
    pre = argparse.ArgumentParser(prog="moview", add_help=False)
    pre.add_argument("--config")
    known, _rest = pre.parse_known_args(arguments)
    try:
        app_config = load_config(known.config)
    except ConfigError as exc:
        pre.error(str(exc))

    parser = build_parser(app_config)
    args = parser.parse_args(arguments)
    if args.grid < MIN_GRID_SIZE:
        parser.error(f"--grid must be at least {MIN_GRID_SIZE}")
    if not math.isfinite(args.margin) or args.margin < 0.0:
        parser.error("--margin must be finite and non-negative")
    if not math.isfinite(args.iso) or args.iso <= 0.0:
        parser.error("--iso must be a finite positive value")

    if args.batch:
        if not args.input:
            parser.error("--batch requires an input file")
        try:
            return run_batch(args, app_config)
        except (OSError, ValueError) as exc:
            print(f"moview: error: {exc}", file=sys.stderr)
            return 1

    from .gui.platform.macos_stderr import filter_macos_gui_warnings

    with filter_macos_gui_warnings():
        from .gui.app import run_gui

        return run_gui(
            args.input,
            grid=args.grid,
            isovalue=args.iso,
            margin=args.margin,
            auto_render=args.auto_render,
            file_format=args.file_format,
            app_config=app_config,
        )
