"""PyQt6/OpenGL user interface (imported lazily by the CLI).

Modules:
    app             run_gui(): QApplication setup and platform checks
    main_window     MainWindow: wiring of panels, viewports and jobs
    render_service  Qt-free orbital rendering pipeline and caches
    jobs            thread-pool jobs delivered back on the GUI thread
    panels          file card, orbital table, settings inspector
    viewport        OpenGL view widget and per-view camera state
    gl_items        shaders, shaded meshes and atom labels
    molecule        atom/bond triangle meshes (NumPy only)
    theme, icons, widgets   look and feel
    platform/       Linux xcb preflight, macOS stderr filter, GL surface format
"""

from __future__ import annotations

__all__ = ["run_gui"]


def __getattr__(name: str):
    if name == "run_gui":
        from .app import run_gui

        return run_gui
    raise AttributeError(name)
