"""The one place that imports Qt, PyOpenGL and pyqtgraph.

Importing this module also installs MOview's portable OpenGL surface format,
which must happen before ``QApplication`` exists.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtOpenGL, QtSvg, QtWidgets  # noqa: F401

from .platform.opengl_context import (  # noqa: F401
    configure_default_opengl_surface_format,
    create_opengl_surface_format,
)

configure_default_opengl_surface_format()

import pyqtgraph as pg  # noqa: E402,F401
import pyqtgraph.opengl as gl  # noqa: E402,F401
from OpenGL import GL  # noqa: E402,F401
from OpenGL.GL import shaders as gl_shaders  # noqa: E402,F401
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem  # noqa: E402,F401

Signal = QtCore.pyqtSignal
Qt = QtCore.Qt
