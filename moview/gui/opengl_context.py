from __future__ import annotations

from PyQt6 import QtCore, QtGui


OPENGL_VERSION = (2, 1)
COLOR_BUFFER_BITS = 8
DEPTH_BUFFER_BITS = 24
STENCIL_BUFFER_BITS = 8


def create_opengl_surface_format() -> QtGui.QSurfaceFormat:
    """Return the portable desktop format used by every MOview GL surface."""
    surface_format = QtGui.QSurfaceFormat()
    surface_format.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setVersion(*OPENGL_VERSION)
    surface_format.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.NoProfile)
    surface_format.setRedBufferSize(COLOR_BUFFER_BITS)
    surface_format.setGreenBufferSize(COLOR_BUFFER_BITS)
    surface_format.setBlueBufferSize(COLOR_BUFFER_BITS)
    surface_format.setAlphaBufferSize(COLOR_BUFFER_BITS)
    surface_format.setDepthBufferSize(DEPTH_BUFFER_BITS)
    surface_format.setStencilBufferSize(STENCIL_BUFFER_BITS)
    surface_format.setSamples(0)
    surface_format.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
    surface_format.setSwapInterval(1)
    return surface_format


def configure_default_opengl_surface_format() -> bool:
    """Configure Qt before QApplication creates its shared GL resources."""
    if QtCore.QCoreApplication.instance() is not None:
        return False
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts,
        True,
    )
    QtGui.QSurfaceFormat.setDefaultFormat(create_opengl_surface_format())
    return True
