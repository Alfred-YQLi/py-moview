# MOview

**English** | [简体中文](README.zh-CN.md)

MOview reads molecular-orbital wavefunctions from Gaussian formatted checkpoint
(`.fchk`/`.fch`) and Molden files, evaluates orbitals on a 3D grid and shows
their positive and negative isosurfaces together with the molecule in an
interactive OpenGL viewer. A batch mode does the same without a GUI and can
verify that a file was interpreted correctly.

## Features

- Gaussian FCHK/FCH and Molden (S–H shells, SP shells, Cartesian and pure
  functions, restricted and unrestricted).
- **Automatic Molden convention detection.** Programs disagree on details the
  Molden format leaves open (ORCA's `orca_2mkl` bakes primitive normalization
  into the contraction coefficients and flips the sign of f/g functions with
  |m| ≥ 3). MOview checks MO orthonormality (`C S Cᵀ = I`) with analytic
  overlap integrals and keeps the convention that is correct. The result is
  shown in the file panel and by `--batch`.
- Orbital table with HOMO/LUMO labels (HOMO−n, LUMO+n), occupations and
  energies in Eh and eV; click to render, ⌘/Ctrl- or Shift-click to compare up
  to nine orbitals side by side, or type `210-214`.
- Glass, Solid, Wireframe or Solid + edges surfaces with smooth normals,
  two-sided lighting and depth cueing; named phase colors (configurable).
- Ball and stick, Space filling or Licorice atoms; bonds from a full
  covalent-radius table; atom labels attached in 3D or floating.
- Light and dark themes: the toolbar's sun/moon menu chooses System (default,
  follows macOS/Windows live), Light or Dark.
- Fast and lean: orbitals are evaluated with the exact separable
  (tensor-product) form of Gaussians, so a 3D field is a single matrix product
  (about 10 ms at grid 64 and 45 ms at grid 128 for a 1000-function basis).
  Orbitals render on demand; only the two neighbours of the displayed orbital
  are pre-rendered so `A`/`D` is instant, and changing only the isovalue
  reruns marching cubes on the cached field.
- Save the view as a PNG; drag and drop files onto the window.

## Installation

Python 3.10 or newer. From the repository root, inside the environment you use
(for example `conda activate wavefunction`):

```bash
python -m pip install ".[gui]"
```

Use `python -m pip install -e ".[gui]"` instead for development (edits take
effect immediately). Core/batch mode needs `numpy`, `scipy` and
`scikit-image`; the GUI adds `PyQt6`, `PyOpenGL` and `pyqtgraph`.

On Linux the `gui` extra pins PyQt6 to `>=6.7.1,<6.10`, whose x86-64 wheels
still support glibc 2.28 (CentOS/RHEL 8). Other platforms take the latest
PyQt6. Check the installed versions with:

```bash
python -c 'from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR; print("PyQt", PYQT_VERSION_STR, "Qt", QT_VERSION_STR)'
```

### Linux notes

The PyQt6 wheel ships Qt, but the X11 `xcb` plugin needs a few system
libraries. MOview checks them before starting Qt and prints the missing names
with the right `dnf`/`apt` command instead of letting Qt abort. On CentOS/RHEL 8
`libxcb-cursor.so.0` comes from EPEL:

```bash
sudo dnf install epel-release
sudo dnf install xcb-util-cursor
```

The GUI needs desktop OpenGL 2.1 or newer (hardware drivers or Mesa llvmpipe).
Useful environment tweaks:

```bash
env -u SESSION_MANAGER moview file.fch                         # silence the ICE session warning
env -u SESSION_MANAGER LIBGL_ALWAYS_SOFTWARE=1 moview file.fch # force Mesa software rendering
QT_LOGGING_RULES="qt.qpa.gl=true" moview file.fch              # OpenGL diagnostics
```

Batch mode never imports Qt or OpenGL.

## Usage

```bash
moview                                   # open the window, then click Open or drag a file in
moview path/to/wavefunction.fch          # open and render the HOMO
moview file.molden.input --grid 96 --iso 0.03
moview file.fch --no-auto-render
moview file.fch --config my.ini
```

Without installing, run `python -m moview ...` from the repository root, or put
`bin/` on your `PATH` (`bin/moview` finds the package from its own location).

### Mouse and keyboard

| Action | Control |
| --- | --- |
| Rotate | left-drag, arrow keys |
| Roll | right-drag |
| Pan | Shift + left-drag, middle-drag |
| Zoom | wheel, `+` / `-` |
| Previous / next orbital | `A` / `D`, ‹ › toolbar buttons |
| Rotation centre | press `C`, then click an atom (`Esc` cancels) |
| Reset view | `R` |
| Open / render / save image | ⌘/Ctrl `O`, ⌘/Ctrl `R`, ⌘/Ctrl `S` |

Typing in a text field never triggers these shortcuts.

### Settings panel

- **Isosurface** – isovalue (logarithmic slider or exact value; surfaces at
  ±value), grid points along the longest box edge, and box margin in bohr. The
  resulting grid shape, point count and spacing are shown below. Grid values
  above 256 ask for confirmation with a memory estimate.
- **Appearance** – surface style, +ψ/−ψ colors, atom style and size, labels
  (number, element, both) and their placement and size. Appearance changes
  never recompute orbitals.
- **View** – rotate comparison views together, show the axes triad.

## Batch mode

```bash
moview file.fch --batch                          # HOMO, default grid and isovalue
moview file.molden --batch --check               # also verify MO orthonormality
moview file.fch --batch --spin beta --orbital lumo+1 --grid 64 --iso 0.03
```

`--orbital` accepts a 1-based number or `homo`, `lumo`, `homo-2`, `lumo+1`, ….
The output lists the file summary, HOMO/LUMO per spin, parser notes, the
orbital's occupation and energy, grid shape and triangle counts. `--check`
reports `max|<i|i>-1|` and `max|<i|j>|`; values around `1e-6` or smaller mean
the file was read correctly.

## Configuration

Copy `moview.example.ini` to `moview.ini` and edit it. The first file found is
used: `--config PATH`, `$MOVIEW_CONFIG`, `./moview.ini`,
`~/Library/Application Support/MOview/config.ini` (macOS),
`$XDG_CONFIG_HOME/moview/config.ini` (or `~/.config/moview/config.ini`), and
`moview.ini` next to the package. Command-line options override the file.

| Section | Keys |
| --- | --- |
| `[resources]` | `render_cache_mib`, `render_cache_entries` (surface cache), `grid_chunk_points` (points per matrix product), `surface_face_limit` |
| `[render]` | `grid`, `margin_bohr`, `isovalue`, `surface_style` (glass, solid, wireframe, solid_edges), `atom_style` (ball_stick, space_filling, licorice), `atom_scale`, `label_mode` (off, number, element, number_element), `label_placement` (attached, floating), `label_size`, `positive_color`, `negative_color` |
| `[view]` | `zoom`, `sync_views`, `show_axes`, `auto_render`, `theme` (system, light, dark) |
| `[colors]` | `Name = R, G, B` to change a preset or add one |

Memory values are MiB. Unknown keys and invalid values are reported with the
file and key at startup. Keys of the old pre-render pipeline (`basis_workers`,
`background_jobs`, `basis_cache_mib`, `max_basis_cache_entry_mib`,
`prefetch_field_budget_mib`, `max_prefetch_orbitals`) are still accepted and
ignored, so older `moview.ini` files keep working.

## How it works

```
parsers/ (FCHK, Molden) ──► core.Wavefunction ──► core.grid (orbital values on a grid)
                                                        │
                       gui.render_service (caches) ◄────┤──► core.surface (marching cubes)
                                │
              gui.main_window ──┴──► gui.gl_items / gui.viewport (OpenGL)
```

1. The parser builds a `Wavefunction`: atoms, contracted shells, MO
   coefficients, energies and occupations. Molden files are checked against
   the candidate conventions with `core.overlap`.
2. `core.grid` places a box around the molecule (plus the margin) with cubic
   voxels. Every Cartesian Gaussian primitive factorizes as
   `X(x)·Y(y)·Z(z)`, so an orbital on the grid is
   `ψ[i,j,k] = Σ_t c_t X_t[i] Y_t[j] Z_t[k]`: `exp` is evaluated only on the
   three axes and the 3D sum is the matrix product `(X·c) @ (Y⊙Z)ᵀ`, done by
   BLAS in slabs of `grid_chunk_points`. This is exact (equal to point-wise
   evaluation to 1e-13) and needs no stored basis-function values.
3. `core.surface` extracts the ±isovalue surfaces with outward-facing
   triangles and gradient normals.
4. The GUI renders on demand. Surfaces are cached per isovalue (small);
   full scalar fields only for the last 12 orbitals shown, which keeps
   isovalue changes instant without holding many large grids. The two
   neighbours of the displayed orbital are pre-rendered (surfaces only). All
   work is tagged with the loaded file's generation so stale results are
   dropped.

## Project layout

| Path | Contents |
| --- | --- |
| `moview/cli.py` | Command line, batch mode, lazy GUI start |
| `moview/config.py` | Configuration discovery and validation |
| `moview/constants.py`, `moview/elements.py` | Units, defaults; element symbols, colors, covalent radii |
| `moview/core/wavefunction.py` | `Shell`, `Wavefunction` (occupations, HOMO/LUMO, labels) |
| `moview/core/basis.py` | Normalization and Cartesian/pure conventions |
| `moview/core/evaluate.py` | Point-wise basis-function and MO evaluation |
| `moview/core/grid.py`, `primitives.py` | `GridSpec`, separable orbital-grid evaluation |
| `moview/core/surface.py` | Marching-cubes isosurfaces |
| `moview/core/overlap.py` | Analytic overlaps, orthonormality check |
| `moview/core/bonds.py` | Bond detection |
| `moview/parsers/` | Format detection, FCHK and Molden readers |
| `moview/gui/app.py`, `main_window.py` | Application start-up and window wiring |
| `moview/gui/render_service.py`, `jobs.py` | Qt-free render pipeline and caches; worker threads |
| `moview/gui/panels.py`, `widgets.py`, `theme.py`, `icons.py` | Panels, controls and look |
| `moview/gui/viewport.py`, `gl_items.py`, `molecule.py` | Views, camera, shaders, meshes, labels |
| `moview/gui/platform/` | Linux xcb preflight, macOS stderr filter, GL surface format |
| `tests/` | Unit and GUI tests (`helpers.py` builds synthetic orthonormal wavefunctions) |

## Development

```bash
python -m unittest discover -s tests -v
MOVIEW_FIXTURES=/path/to/real/files python -m unittest discover -s tests   # also test real files
```

GUI tests run headless (`QT_QPA_PLATFORM=offscreen`); Qt may print
"QOpenGLWidget is not supported on this platform", which is expected there.
Large wavefunction files are not committed: put them in `tests/wavefunctions/`
or point `MOVIEW_FIXTURES` at them.

External macOS volumes create AppleDouble `._*` files; they are git-ignored.
Build release wheels from a checkout on APFS or another filesystem that does
not create them.

## Limitations

- ORCA `.gbw` and other native formats are not read; convert with `orca_2mkl
  name -molden`.
- Very high grids are limited by memory; the GUI estimates the cost first.
- The GUI needs desktop OpenGL; headless SSH sessions should use batch mode.
