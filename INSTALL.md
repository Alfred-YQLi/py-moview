# Installing MOview 1.0.0

## 1. Requirements

- Python 3.10 or newer (3.12 recommended)
- For the GUI: a desktop with OpenGL 2.1+ (any Mac; Linux with X11)

## 2. Unpack

```bash
tar -xJf moview-1.0.0.tar.xz
cd moview-1.0.0
```

## 3. Create an environment (skip if you already have one)

```bash
conda create -n wavefunction python=3.12
conda activate wavefunction
```

## 4. Install

```bash
python -m pip install ".[gui]"
```

This installs NumPy, SciPy, scikit-image, PyQt6, PyOpenGL and pyqtgraph.
For batch mode only (no GUI), use `python -m pip install .` instead.

## 5. Check

```bash
moview --version
python -m unittest discover -s tests
```

## 6. Use

```bash
moview                               # open the window, then click Open or drag a file in
moview molecule.fchk                 # open a file and show the HOMO
moview molecule.molden --batch --check   # no GUI: summary + orthonormality check
```

Keys: `A` / `D` previous / next orbital, drag to rotate, wheel to zoom,
`C` then click an atom to set the rotation centre.

## Linux notes

On CentOS/RHEL 8 the Qt window needs one system library:

```bash
sudo dnf install epel-release
sudo dnf install xcb-util-cursor
```

MOview tells you the exact package if anything else is missing. Over SSH
without a display, use `--batch`.

## Uninstall

```bash
python -m pip uninstall moview
```

See `README.md` for configuration (`moview.example.ini`) and all options.
