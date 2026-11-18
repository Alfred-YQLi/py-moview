"""Small line icons drawn from inline SVG in the current theme color."""

from __future__ import annotations

from .qt import QtCore, QtGui, QtSvg

_STROKE = 'fill="none" stroke="{c}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'

_PATHS = {
    "open": f'<path {_STROKE} d="M3 7.5V18a1.5 1.5 0 0 0 1.5 1.5h15A1.5 1.5 0 0 0 21 18V9.5A1.5 1.5 0 0 0 19.5 8H12l-2-2.5H4.5A1.5 1.5 0 0 0 3 7z"/>',
    "render": f'<path {_STROKE} d="M7 5.5v13l11-6.5z"/>',
    "prev": f'<path {_STROKE} d="M14.5 6l-6 6 6 6"/>',
    "next": f'<path {_STROKE} d="M9.5 6l6 6-6 6"/>',
    "homo": f'<path {_STROKE} d="M5 15h14M8 11.5h8"/><circle cx="12" cy="7.5" r="1.6" fill="{{c}}"/>',
    "lumo": f'<path {_STROKE} d="M5 9h14M8 12.5h8"/><circle cx="12" cy="16.5" r="1.6" {_STROKE}/>',
    "reset": f'<path {_STROKE} d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v3.5H8"/>',
    "camera": f'<path {_STROKE} d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2l1.5-2h6l1.5 2h2A1.5 1.5 0 0 1 20 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/><circle cx="12" cy="13" r="3.2" {_STROKE}/>',
    "sun": f'<circle cx="12" cy="12" r="4" {_STROKE}/><path {_STROKE} d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/>',
    "moon": f'<path {_STROKE} d="M19.5 14.5A7.5 7.5 0 0 1 9.5 4.5a7.5 7.5 0 1 0 10 10z"/>',
    "panel-left": f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_STROKE}/><path {_STROKE} d="M9 4.5v15"/>',
    "panel-right": f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_STROKE}/><path {_STROKE} d="M15 4.5v15"/>',
}


def icon(name: str, color: str, size: int = 18) -> QtGui.QIcon:
    body = _PATHS[name].replace("{c}", color)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    result = QtGui.QIcon()
    for scale in (1, 2, 3):
        pixmap = QtGui.QPixmap(size * scale, size * scale)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    return result


def dot_icon(color: str, filled: bool, size: int = 10) -> QtGui.QIcon:
    result = QtGui.QIcon()
    for scale in (1, 2, 3):
        pixmap = QtGui.QPixmap(size * scale, size * scale)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(color), 1.4 * scale)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(color) if filled else QtCore.Qt.BrushStyle.NoBrush)
        inset = 1.6 * scale
        painter.drawEllipse(QtCore.QRectF(inset, inset, size * scale - 2 * inset, size * scale - 2 * inset))
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    return result


def swatch_icon(rgb: tuple[int, int, int], size: int = 14) -> QtGui.QIcon:
    result = QtGui.QIcon()
    for scale in (1, 2, 3):
        pixmap = QtGui.QPixmap(size * scale, size * scale)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(*rgb))
        painter.drawRoundedRect(QtCore.QRectF(0, 0, size * scale, size * scale), 3.5 * scale, 3.5 * scale)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    return result
