"""GUI entry point."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from ..config import DEFAULT_CONFIG, AppConfig


def _writable_cache_dirs() -> None:
    """Point Qt/font caches at a writable place when HOME is read-only (clusters)."""
    cache_root = Path(os.environ.get("MOVIEW_CACHE_DIR") or Path(tempfile.gettempdir()) / "moview-cache")
    home_cache = Path.home() / ".cache"
    if os.access(home_cache if home_cache.exists() else Path.home(), os.W_OK):
        return
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def run_gui(
    input_path: str | None = None,
    *,
    grid: int | None = None,
    isovalue: float | None = None,
    margin: float | None = None,
    auto_render: bool | None = None,
    file_format: str | None = None,
    app_config: AppConfig | None = None,
) -> int:
    config = app_config or DEFAULT_CONFIG
    _writable_cache_dirs()
    try:
        from .qt import QtWidgets
    except ModuleNotFoundError as exc:
        print(
            f"Missing GUI dependency: {exc.name}\n"
            'Install the GUI extras from the repository root with: python -m pip install ".[gui]"',
            file=sys.stderr,
        )
        return 1
    from .platform.linux_qt import linux_qt_platform_issue

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        issue = linux_qt_platform_issue()
        if issue is not None:
            print(issue, file=sys.stderr)
            return 1
        app = QtWidgets.QApplication(sys.argv[:1])
        app.setApplicationName("MOview")
        app.setApplicationDisplayName("MOview")
        # Fusion + MOview's style sheet looks identical on macOS, Linux and Windows.
        app.setStyle("Fusion")

    from .main_window import MainWindow

    window = MainWindow(
        config,
        grid=config.render.grid if grid is None else grid,
        isovalue=config.render.isovalue if isovalue is None else isovalue,
        margin=config.render.margin_bohr if margin is None else margin,
    )
    window.show()
    if input_path:
        window.open_path(input_path, file_format=file_format, auto_render=config.view.auto_render if auto_render is None else auto_render)
    return int(app.exec()) if owns_app else 0
