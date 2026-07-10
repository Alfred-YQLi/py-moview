from __future__ import annotations


__all__ = ["OpenGLViewer", "run_gui"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from .main_window import OpenGLViewer, run_gui

    return {"OpenGLViewer": OpenGLViewer, "run_gui": run_gui}[name]
