# MOview

**English** | [简体中文](README.zh-CN.md)

MOview is a Python molecular-orbital wavefunction reader and OpenGL viewer. It
reads Gaussian formatted checkpoint (`.fchk`/`.fch`) and Molden files, evaluates
selected molecular orbitals on a three-dimensional grid, extracts positive and
negative isosurfaces, and displays the molecular structure, orbital surfaces,
atom labels, and coordinate axes.

The package provides an interactive GUI and a non-GUI batch mode. The GUI is
intended for orbital inspection and comparison; batch mode is useful for parser
checks, automation, and isosurface statistics.

## Features

- Read Gaussian FCHK/FCH and Molden wavefunctions.
- Display alpha/beta orbitals, energies, occupations, HOMO, and LUMO.
- Compare up to nine orbitals with synchronized or independent views.
- Use an explicit positive isovalue for the two wavefunction phases; the default
  is `0.05`.
- Adjust atom size, with a default scale of `1.00x`.
- Show atom numbers, element symbols, or both (`1`, `Ca`, `1Ca`) with adjustable
  label size.
- Use depth-tested `Attached` labels or collision-avoiding `Floating` labels.
- Choose Ball & stick, Space filling, or Licorice atom styles.
- Choose Glass, Solid, Wireframe, or Solid + edges surface styles.
- Select independent named colors for positive and negative phases. Presets can
  be changed or extended with RGB values in the configuration file.
- The default appearance uses Glass surfaces, Ball & stick atoms, and disabled
  labels. Attached placement is selected when labels are enabled.
- Request any Grid supported by the machine. Grid values above 256 produce a
  performance and memory warning instead of being rejected.
- Restart background pre-rendering after a 650 ms debounce when Grid, Margin,
  or Isovalue changes.
- Parse large files in a worker thread so the Qt interface remains responsive.
- Bound basis-grid, scalar-field, surface, and prefetch caches by configurable
  resource budgets.

## Installation

Python 3.10 or newer is required. Python 3.12 is the primary tested version.

```bash
git clone https://github.com/Alfred-YQLi/py-moview.git
cd py-moview
python -m pip install -e ".[gui]"
```

Core batch operation requires `numpy` and `scikit-image`. The GUI additionally
requires `PyQt6`, `PyOpenGL`, and `pyqtgraph`. The `gui` extra constrains PyQt6
to `>=6.7.1,<6.10` so that Linux x86-64 installations retain a binary wheel
compatible with glibc 2.28, as used by CentOS/RHEL 8. A normal installation
selects PyQt6 6.9.1 and its compatible Qt 6.9 runtime; no Qt SDK or `qmake` is
required.

Confirm the installed Qt versions with:

```bash
python -c 'from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR; print("PyQt", PYQT_VERSION_STR, "Qt", QT_VERSION_STR)'
```

If an installation attempts to download a `PyQt6-*.tar.gz` source archive,
update the repository and repeat the `.[gui]` installation. Do not install an
unconstrained PyQt6 separately on CentOS/RHEL 8: PyQt6 6.10 and newer Linux
x86-64 wheels require a newer glibc baseline.

### CentOS/RHEL 8 native GUI libraries

The PyQt6 wheel includes Qt, but the Qt X11 `xcb` platform plugin still uses
native libraries supplied by the operating system. On CentOS/RHEL 8,
`libxcb-cursor.so.0` is provided by the EPEL `xcb-util-cursor` package:

```bash
sudo dnf install epel-release
sudo dnf install xcb-util-cursor
rpm -q xcb-util-cursor
```

These are system packages and cannot be installed by `pip`. MOview checks the
linked libraries of Qt's `libqxcb.so` before creating `QApplication`. If any
are missing, it exits normally and prints the missing library names and the
appropriate `dnf` or `apt` command instead of allowing Qt to abort.

### Linux OpenGL

The GUI requires a desktop OpenGL context. Before creating `QApplication`,
MOview requests desktop OpenGL 2.1 with 8-bit RGBA, a 24-bit depth buffer, an
8-bit stencil buffer, and double buffering. This format works with hardware
drivers and Mesa software rendering such as llvmpipe.

For an X11 session, verify the display and renderer with:

```bash
echo "$DISPLAY"
glxinfo -B
```

`Qt: Session management error` is an independent ICE session warning, not an
OpenGL failure. It can be omitted for one launch without changing rendering:

```bash
env -u SESSION_MANAGER moview /path/to/wavefunction.fch
```

To request Mesa software rendering explicitly, use:

```bash
env -u SESSION_MANAGER LIBGL_ALWAYS_SOFTWARE=1 moview /path/to/wavefunction.fch
```

If Qt still reports a GLX configuration error, enable its OpenGL diagnostics
with `QT_LOGGING_RULES="qt.qpa.gl=true"`. This usually indicates that the Qt
installation is loading a different GLX/Mesa library stack from `glxinfo`.
Batch mode remains available without Qt or OpenGL.

## Running MOview

### Python module

Without installing the package, run it from the repository root, where
`pyproject.toml` and the `moview/` package directory are located:

```bash
cd /path/to/py-moview
python -m moview /path/to/wavefunction.fch
```

Do not enter the inner `moview/` package directory before running
`python -m moview`. The wavefunction path is a normal argument after the module
name; it is not itself a Python module.

### Installed console command

An editable or normal package installation provides the `moview` command from
any directory:

```bash
moview /path/to/wavefunction.fch
```

### Source-tree PATH launcher

The executable `bin/moview` resolves the repository from its own real path, so
it can be used without an editable install and can also be symlinked elsewhere:

```bash
export PATH="/path/to/py-moview/bin:$PATH"
moview /path/to/wavefunction.fch
```

For zsh, place the `export` line in `~/.zshrc`. Activate the Conda environment
that contains the GUI dependencies before launching; the script uses the
`python3` found in the active PATH.

Common commands:

```bash
moview --help
moview file.fch --fchk
moview file.molden.input --molden
moview file.fch --grid 96 --margin 4
moview file.fch --prefetch-workers 4
moview file.fch --config /path/to/custom.ini
moview file.fch --no-auto-render
```

## Batch Mode

Batch mode does not import PyQt or OpenGL. It prints the detected format, atom
and basis counts, orbital metadata, actual grid shape, isovalue, and positive
and negative triangle counts.

```bash
moview file.fch --batch --grid 16 --orbital 1
moview file.molden.input --batch --grid 16 --orbital 1
moview file.fch --batch --spin beta --orbital 52 --grid 64 --iso 0.03 --margin 4
```

Batch orbital numbers are one-based. Grid must be at least 8, Margin must be
non-negative, and Isovalue must be finite and positive. Grid values above 256
print a warning to stderr but continue as requested.

## Molden Compatibility

MOview reads the `[Atoms]`, `[GTO]`, and `[MO]` sections used by the
[official Molden format](https://www.theochem.ru.nl/molden/molden_format.html).
It supports atomic coordinates in atomic units or angstrom, `S` through `G`
shells, combined `SP` shells, Cartesian and spherical D/F/G conventions, and
both alpha and beta molecular orbitals. Indexed MO coefficients may be sparse.

The `[MO]` section is streamed into coefficient matrices, so large Molden files
are not retained in memory as millions of source lines. Molden files produced
by tools such as ORCA's `orca_2mkl` can be opened directly.

## Configuration

The repository includes `moview.example.ini`. Copy it to the user-specific
`moview.ini` before changing core counts, memory budgets, styles, or colors:

```bash
cp moview.example.ini moview.ini
```

`moview.ini` is ignored by Git so machine-specific settings are not committed
accidentally. MOview uses the first configuration found in this order:

1. `--config PATH`.
2. `MOVIEW_CONFIG` environment variable.
3. `moview.ini` in the current working directory.
4. `~/Library/Application Support/MOview/config.ini` on macOS.
5. `$XDG_CONFIG_HOME/moview/config.ini`, or `~/.config/moview/config.ini`.
6. `moview.ini` beside the source package.

Explicit command-line values override configuration values. For example,
`--grid 96` overrides `[render] grid`. If no file is found, the package uses
the same validated built-in defaults. Restart MOview after editing a config.

### Resource settings

Memory values in `[resources]` are expressed in MiB.

| Setting | Purpose |
| --- | --- |
| `basis_workers` | Threads used to construct a cached BasisGrid. |
| `background_jobs` | Concurrent background pre-render batches. |
| `basis_cache_mib` | Total BasisGrid cache budget. |
| `max_basis_cache_entry_mib` | Largest individual BasisGrid eligible for caching; cannot exceed the total budget. |
| `render_cache_mib` | Total scalar-field and isosurface cache budget. |
| `render_cache_entries` | Additional limit on render-cache entries. |
| `prefetch_field_budget_mib` | Scalar-field budget available to background pre-render planning. |
| `max_prefetch_orbitals` | Maximum orbitals queued by one pre-render cycle. |
| `grid_chunk_points` | Points evaluated per chunk; smaller values reduce temporary memory. |
| `surface_face_limit` | Maximum triangles uploaded for each positive or negative OpenGL surface. |

Increasing `background_jobs` changes scheduling concurrency but does not bypass
`prefetch_field_budget_mib`. Physical CPU, RAM, and GPU memory remain the final
limits.

### Render, view, and color defaults

`[render]` controls Grid, Margin, Isovalue, surface/atom styles, atom scale,
label content, `label_placement = attached|floating`, label size, and phase
colors. `[view]` controls Zoom, synchronized views, axes, and initial automatic
rendering.

`[colors]` defines named presets as 0-255 RGB values. Reusing a built-in name
changes that preset; a new name is appended to the GUI selectors.

```ini
[render]
positive_color = Mint
negative_color = Violet

[colors]
Red = 255, 64, 64
Mint = 40, 210, 160
```

Color references are case-insensitive, while configured spelling is preserved
in the GUI and scene legend. Invalid RGB values, unknown settings, unknown
colors, and contradictory memory budgets produce actionable startup errors.

## GUI Controls

The left panel keeps file metadata, HOMO/LUMO actions, comparison input, and the
orbital table available while settings are divided into Compute and Display
tabs.

### Compute

- `Spin`: choose alpha or beta for unrestricted wavefunctions.
- `Grid`: target points along the longest axis; the other dimensions follow the
  molecular bounding-box proportions.
- `Margin / bohr`: expand the molecular bounding box.
- `Isovalue`: set the positive isovalue; the negative phase uses its negative.

Changing Grid, Margin, or Isovalue cancels affected background work and
restarts pre-rendering after 650 ms. Grid values above 256 require performance
confirmation before GUI pre-rendering begins.

### Display

- `Surface`: Glass, Solid, Wireframe, or Solid + edges.
- `+ phase` / `- phase`: named positive/negative color selectors with swatches.
- `Atoms`: Ball & stick, Space filling, or Licorice.
- `Atom size`: scale atom radii without changing bond radii.
- Left `Labels` selector: Off, Number, Element, or Number + element.
- Right `Labels` selector: Attached or Floating placement.
- `Label size`: adjust label size.
- `Zoom`: change camera zoom.
- `Sync views`: synchronize rotation and zoom across comparison views.
- `Axes`: show or hide the lower-right coordinate axes.

Attached labels are depth-tested billboards placed at the atom front surface.
They move with the molecule and scale with perspective while remaining oriented
toward the camera for legibility. Floating labels use screen-space collision
avoidance and leader lines, which is useful for dense structures.

Appearance controls only rebuild lightweight display objects. Changing atom
style/size, labels, surface style, or phase colors does not parse the file,
evaluate an orbital grid, or run marching cubes. All Attached labels share one
texture atlas and one OpenGL draw call; labels Off allocates no label resources.

The scene header uses two compact lines: orbital identity, occupation, and
energy on the first; Isovalue, Grid, and named phase colors on the second.

### Orbital comparison

Comparison input uses one-based orbital numbers and accepts commas, spaces, and
ranges in either direction:

```text
45,46,47
45 46 47
45-49
49-45
```

### Keyboard shortcuts

- `A` / `D`: select and render the adjacent orbital after a short debounce.
- Arrow keys: rotate the active view.
- `+` / `-`: zoom.
- `C`, then click an atom: set the rotation center.

Global shortcuts do not take over while focus is in the orbital table, a text
field, combo box, spin box, or slider.

## Grid and Performance

Grid is the requested number of points along the molecular bounding box's
longest axis. Work and scalar-field storage grow approximately with Grid cubed,
so no application cap does not imply every value fits the current machine.

- GUI Grid values above 256 show actual point count and estimated float32 field
  memory before proceeding.
- A confirmed Grid value is remembered for the current file session.
- BasisGrid caching is used only when the estimated entry fits the configured
  per-entry threshold.
- Larger cases use chunked float32 multi-orbital evaluation rather than a full
  `n_basis x n_points` matrix.
- Built-in defaults provide 768 MiB for BasisGrid cache, 512 MiB for render
  cache, and 192 MiB for background scalar fields.
- Background pre-rendering has no fixed Grid cutoff. It queues at most 48
  frontier orbitals and reduces that count according to the field budget.
- If one float32 field already exceeds the prefetch budget, pre-rendering is
  skipped while foreground Render remains available.
- A changed Isovalue reuses cached scalar fields and reruns only marching cubes.
- Async jobs are tied to a wavefunction generation, preventing stale results
  from populating a newly loaded file's caches.
- Labels are Off by default, so label rendering has no default rotation cost.

## Program Flow

### GUI

1. `moview/__main__.py` calls `moview.cli.main()`.
2. `moview.cli` loads config first, builds argument defaults, and applies CLI
   overrides.
3. GUI mode imports `moview.gui.main_window.run_gui()` lazily; Linux checks the
   native xcb plugin dependencies before `QApplication`, while macOS enables
   the narrow native-log filter.
4. `moview.gui.opengl_context` installs one explicit surface format before
   `run_gui()` creates `QApplication` and `OpenGLViewer` with `AppConfig`.
5. `moview.gui.layout` builds the controls, orbital table, and OpenGL views.
6. `load_wavefunction()` submits parsing to a worker thread.
7. `moview.parsers` detects FCHK or Molden and dispatches the parser.
8. The parser creates a `Wavefunction` containing atoms, shells, energies,
   occupations, and MO coefficients.
9. `moview.analysis.compute_bonds()` builds bonds from covalent radii.
10. The GUI populates orbitals and selects the default HOMO.
11. Render checks high-Grid confirmation and the orbital cache.
12. Small grids use cached BasisGrid multiplication; large grids use chunked
    float32 single/multi-orbital evaluation.
13. `moview.surface.extract_isosurfaces()` runs marching cubes for both phases.
14. `moview.gui.gl_view` creates surface, molecule, and batched label items.
15. `SceneSlot` stores each comparison view's geometry, camera, and transform.
16. Budgeted background jobs pre-render frontier orbitals and restart after
    debounced Compute changes.

### Batch

1. CLI loads config and validates arguments.
2. `run_batch()` parses a `Wavefunction`.
3. `moview.grid.compute_orbital_grid()` performs configured chunked evaluation.
4. `extract_isosurfaces()` creates positive and negative meshes.
5. Batch mode prints orbital, grid, and triangle statistics.

## Project Layout

| Path | Responsibility |
| --- | --- |
| `bin/moview` | Source-tree executable suitable for PATH or symlinking. |
| `moview/__main__.py` | `python -m moview` package entry point. |
| `moview/cli.py` | Config-first argument parsing, batch mode, and lazy GUI launch. |
| `moview/config.py` | Discovery, validation, immutable settings, and named RGB colors. |
| `moview.example.ini` | Conservative resource/render/view/color configuration template. |
| `moview/wavefunction.py` | Shell and Wavefunction data models. |
| `moview/constants.py` | Units, symbols, element colors, and radii. |
| `moview/cache.py` | Writable runtime cache locations. |
| `moview/parsers/` | Format detection plus FCHK and Molden parsers; direct ORCA input is reserved. |
| `moview/basis/` | Gaussian normalization, spherical transforms, and evaluators. |
| `moview/grid.py` | Grid specifications, BasisGrid, and chunked orbital evaluation. |
| `moview/surface.py` | SurfaceMesh and marching-cubes extraction. |
| `moview/analysis/geometry.py` | Covalent-radius bond detection. |
| `moview/gui/main_window.py` | Async jobs, caches, scene state, comparison, and camera. |
| `moview/gui/layout.py` | Compute/Display controls, orbital table, header, and canvas. |
| `moview/gui/gl_view.py` | OpenGL materials, molecule geometry, labels, axes, and input. |
| `moview/gui/opengl_context.py` | Portable desktop Qt OpenGL surface format. |
| `moview/gui/linux_qt.py` | Linux xcb shared-library preflight and package-manager guidance. |
| `moview/gui/native_stderr.py` | Exact harmless macOS TSM/IMK/Qt keymapper filtering while preserving other stderr. |
| `tests/test_config.py` | Config discovery, validation, colors, and CLI precedence. |
| `tests/test_smoke.py` | Core API, Linux Qt preflight, labels, Grid, launcher, parser errors, and stderr filtering. |
| `tests/test_gui.py` | GUI defaults, caches, pre-rendering, async behavior, and appearance. |
| `tests/test_wavefunctions.py` | Optional FCHK/Molden fixture integration and low-Grid surface tests. |

Large wavefunction fixtures are local-only and are not part of the repository.
Fixture-dependent integration tests skip explicitly when those files are
unavailable.

## Development Checks

```bash
python -m compileall -q -x '(^|/)\._' moview tests
python -m unittest discover -s tests -v
python -m moview file.fch --batch --grid 16 --orbital 1
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

Offscreen Qt commonly reports that `QOpenGLWidget` cannot create a real context;
behavior tests still run, but visual OpenGL checks require a normal desktop
session. External macOS volumes may create AppleDouble `._*` metadata, which is
why compile checks exclude those names.

Editable installation and the source launcher work from such volumes. For a
release wheel, build from a clean Git checkout on APFS or another filesystem
that does not create `._*` files inside generated setuptools directories.

MOview filters only exact, known-harmless macOS input-method diagnostics. Python
tracebacks, OpenGL failures, and unrelated Qt/native stderr remain visible.

## Current Limitations

- ORCA wavefunction parsing is reserved but not implemented.
- Extremely high Grid values can still fail when physical memory is exhausted.
- Marching cubes requires `scikit-image`.
- The GUI requires desktop OpenGL; headless SSH/offscreen platforms may not
  create a usable context.
