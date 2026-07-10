from __future__ import annotations

import os
import tempfile
from pathlib import Path


CACHE_DIR = Path(
    os.environ.get("MOVIEW_CACHE_DIR") or Path(tempfile.gettempdir()) / "fchk_orbital_viewer_cache"
).expanduser()


def configure_runtime_cache(cache_dir: Path = CACHE_DIR) -> Path:
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    return cache_dir


configure_runtime_cache()
