# Changelog

## 1.0.0

### Rendering pipeline

- Orbitals are evaluated with the exact separable (tensor-product) form of
  Gaussian primitives: `psi[i,j,k] = sum_t c_t X_t[i] Y_t[j] Z_t[k]`, one BLAS
  matrix product per slab. 13-24x faster than 0.2 (identical to 1e-13 in
  float64), with no stored basis-function grids.
- On-demand rendering replaces the 48-orbital background pre-render and the
  basis-grid cache: on the 94-atom test system a step to the next orbital
  takes ~90 ms instead of 0.6 s (grid 64) / 1.5 s (grid 128), and orbital
  caches stay small and bounded (surfaces cached, at most 12 full fields).
- Only the two neighbours of the displayed orbital are pre-rendered
  (surfaces only). The status bar shows the render time.
- Molden coefficients are collected in compact arrays while reading.
- `[resources]` keys of the old pipeline are accepted and ignored;
  `--prefetch-workers` is accepted and ignored.

### Fixed

- Pressing `A`/`D` after clicking the 3D view moved keyboard focus into the
  "Go to orbital" box, so further key presses were typed there. Re-laying out
  the views no longer hides the focused view, and the orbital text boxes only
  take focus when clicked.
- A timer firing after the window closed could submit work to a stopped
  thread pool.

### Changed

- One theme control: the toolbar sun/moon button opens System / Light / Dark
  (the duplicate selector in the settings panel was removed). System follows
  the OS appearance live.

## 0.2.0

### Fixed (logic errors found in the audit)

- **ORCA Molden files gave wrong orbitals.** `orca_2mkl` multiplies every
  contraction coefficient by the primitive normalization `(2a/pi)^(3/4) (4a)^(l/2)`
  and flips the sign of the pure f(+3), f(-3), g(+3), g(-3), g(+4), g(-4)
  functions. Both were ignored, so basis functions had norms between 0.003 and
  26,000 and every rendered orbital was distorted (MO orthonormality error
  2.6e4 on the bundled test file; now 8e-9). Molden conventions are now
  detected by testing `C S C^T = I` with analytic overlaps, and contracted
  functions are renormalized.
- **Molden MO blocks were only split on `Sym=` lines.** Files whose writers omit
  `Sym=` collapsed into a single orbital. A new orbital now starts at any
  keyword that follows coefficient lines.
- **Positive and negative lobes had opposite triangle winding**, so any
  normal-based lighting shaded the two phases differently. Both lobes are now
  outward-facing with outward gradient normals.
- **Large surfaces got holes.** Above `surface_face_limit` every n-th triangle
  was dropped before drawing. Surfaces are now always drawn complete; the
  setting only triggers a hint to use a coarser grid.
- **Bonds used a 13-element covalent-radius table** with 0.75 Å for every other
  element (missed metal-ligand bonds such as Cu-N). Replaced with the full
  Cordero et al. table and vectorized.
- **Corner axes did not match the view.** They were drawn with a fixed oblique
  projection instead of the scene rotation.
- **Glass lighting rotated with the molecule** because shading was baked into
  per-face colors in model space. Lighting is now computed per pixel in eye
  space.
- **FCHK character arrays were not skipped**, so their text could be parsed as
  field headers; Fortran reals with three-digit exponents (`1.0-100`) failed;
  a header glued to the previous data line (missing newline, seen in
  `save_unouno.fch`) silently dropped that field (beta energies became NaN).
- **HOMO with explicit occupations** was "number of occupied orbitals - 1",
  wrong when occupations are not ordered; now the highest occupied index.
- Batch mode crashed with a traceback for out-of-range orbitals.

### Changed

- Code reorganized into `moview.core` (numerics), `moview.parsers` and
  `moview.gui` (render service, panels, viewport, GL items, theme).
- New GUI: toolbar, orbital table with HOMO/LUMO labels and multi-select
  comparison, settings inspector, light/dark themes, smooth-shaded surfaces,
  two-color bonds, PNG export, drag and drop, pan.
- Faster evaluation: shared radial parts, distance screening per grid block,
  threaded single-orbital evaluation (about 3x faster on the test system).
- Batch mode: `--check`, `--orbital homo|lumo±n`, default orbital is the HOMO.
- Combined atom labels are written element first (`Fe38`).
- `[view] theme = system|light|dark`; default `grid_chunk_points` is 32768.
- PyQt6 `<6.10` pin applies to Linux only.
