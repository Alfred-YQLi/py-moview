# Local wavefunction fixtures

Large wavefunction files are intentionally excluded from Git. To run the local
format-integration tests, place these files in this directory:

- `save.molden.input`
- `save_uks.fch`
- `save_uks_UNO.fch`
- `save_unouno.fch`

`tests/test_wavefunctions.py` skips the affected tests when the required files
are absent. Do not publish wavefunctions that contain confidential structures
or calculation data.
