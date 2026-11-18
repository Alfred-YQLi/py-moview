"""Light and dark themes: palette, Qt style sheet and small generated icons."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .qt import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool
    window: str  # application background
    panel: str  # sidebars
    field: str  # inputs, cards
    field_hover: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_hover: str
    accent_text: str
    accent_soft: str  # selection background
    occupied: str
    warning: str
    viewport: str  # OpenGL background
    label_text: str
    label_halo: str
    bond: tuple[float, float, float]

    def rgb(self, attribute: str) -> tuple[float, float, float]:
        color = QtGui.QColor(getattr(self, attribute))
        return (color.redF(), color.greenF(), color.blueF())


LIGHT = Theme(
    name="light",
    dark=False,
    window="#eef0f4",
    panel="#ffffff",
    field="#f6f7f9",
    field_hover="#eceef2",
    border="#e3e6eb",
    border_strong="#cfd4dc",
    text="#16181d",
    text_muted="#5f6673",
    text_faint="#9aa1ad",
    accent="#4f5bd5",
    accent_hover="#4350c4",
    accent_text="#ffffff",
    accent_soft="#e4e7fb",
    occupied="#1f9d74",
    warning="#b45309",
    viewport="#fbfbfc",
    label_text="#15171c",
    label_halo="#ffffff",
    bond=(0.56, 0.59, 0.64),
)

DARK = Theme(
    name="dark",
    dark=True,
    window="#0f1115",
    panel="#171a20",
    field="#20242c",
    field_hover="#282d36",
    border="#272b33",
    border_strong="#363c47",
    text="#e7e9ee",
    text_muted="#a0a7b4",
    text_faint="#6c7380",
    accent="#7c87f6",
    accent_hover="#919bf8",
    accent_text="#0f1115",
    accent_soft="#2c3160",
    occupied="#3ccf9c",
    warning="#f2a33a",
    viewport="#14171c",
    label_text="#f1f3f7",
    label_halo="#0d0f13",
    bond=(0.50, 0.53, 0.58),
)

THEMES = {"light": LIGHT, "dark": DARK}


def system_prefers_dark() -> bool:
    app = QtGui.QGuiApplication.instance()
    if app is None:
        return False
    hints = app.styleHints()
    if hasattr(hints, "colorScheme"):
        return hints.colorScheme() == QtCore.Qt.ColorScheme.Dark
    return app.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128


def resolve_theme(setting: str) -> Theme:
    if setting in THEMES:
        return THEMES[setting]
    return DARK if system_prefers_dark() else LIGHT


# ------------------------------------------------------------------- icons
_ICON_DIR = Path(tempfile.gettempdir()) / "moview-theme-icons"

_SVG = {
    "chevron-down": '<path d="M4 6l4 4 4-4" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>',
    "chevron-up": '<path d="M4 10l4-4 4 4" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>',
    "check": '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
}


def _icon_file(name: str, color: str) -> str:
    _ICON_DIR.mkdir(parents=True, exist_ok=True)
    path = _ICON_DIR / f"{name}-{color.lstrip('#')}.svg"
    if not path.exists():
        body = _SVG[name].format(c=color)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">{body}</svg>')
    return path.as_posix()


# -------------------------------------------------------------- style sheet
def stylesheet(t: Theme) -> str:
    chevron = _icon_file("chevron-down", t.text_muted)
    chevron_up = _icon_file("chevron-up", t.text_muted)
    check = _icon_file("check", t.accent_text)
    return f"""
* {{ outline: 0; }}
QMainWindow, QDialog {{ background: {t.window}; }}
QWidget {{ color: {t.text}; font-size: 13px; }}
QToolTip {{ color: {t.text}; background: {t.panel}; border: 1px solid {t.border_strong}; padding: 6px 8px; border-radius: 6px; }}

QFrame#sidebar, QScrollArea#inspector, QWidget#inspectorBody {{ background: {t.panel}; }}
QFrame#sidebar {{ border-right: 1px solid {t.border}; }}
QScrollArea#inspector {{ border: 0; border-left: 1px solid {t.border}; }}
QFrame#card {{ background: {t.field}; border: 1px solid {t.border}; border-radius: 10px; }}
QFrame#viewportFrame {{ background: {t.viewport}; border: 1px solid {t.border}; border-radius: 10px; }}
QFrame#viewportFrame[active="true"] {{ border: 1px solid {t.accent}; }}
QWidget#viewportHeader {{ background: transparent; }}

QLabel#appTitle {{ font-size: 15px; font-weight: 700; }}
QLabel#sectionTitle {{ color: {t.text_faint}; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; }}
QLabel#fileName {{ font-size: 14px; font-weight: 650; }}
QLabel#muted, QLabel[muted="true"] {{ color: {t.text_muted}; }}
QLabel#faint {{ color: {t.text_faint}; font-size: 12px; }}
QLabel#pill {{ color: {t.accent}; background: {t.accent_soft}; border-radius: 8px; padding: 1px 8px; font-size: 11px; font-weight: 700; }}
QLabel#note {{ color: {t.text_muted}; font-size: 12px; }}
QLabel#warning {{ color: {t.warning}; font-size: 12px; }}
QLabel#viewTitle {{ font-size: 13px; }}
QLabel#viewMeta {{ color: {t.text_muted}; font-size: 12px; }}
QLabel#emptyTitle {{ font-size: 20px; font-weight: 650; }}
QLabel#emptyHint {{ color: {t.text_muted}; font-size: 13px; }}

QToolBar {{ background: {t.panel}; border: 0; border-bottom: 1px solid {t.border}; spacing: 2px; padding: 5px 8px; }}
QToolBar::separator {{ background: {t.border}; width: 1px; margin: 6px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 9px; color: {t.text}; }}
QToolButton:hover {{ background: {t.field_hover}; }}
QToolButton:pressed, QToolButton:checked {{ background: {t.accent_soft}; color: {t.accent}; }}
QToolButton:disabled {{ color: {t.text_faint}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#primary {{ background: {t.accent}; color: {t.accent_text}; font-weight: 650; }}
QToolButton#primary:hover {{ background: {t.accent_hover}; }}
QToolButton#primary:disabled {{ background: {t.field_hover}; color: {t.text_faint}; }}

QPushButton {{ background: {t.field}; border: 1px solid {t.border_strong}; border-radius: 7px; padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {t.field_hover}; }}
QPushButton:pressed {{ background: {t.accent_soft}; }}
QPushButton:disabled {{ color: {t.text_faint}; border-color: {t.border}; }}
QPushButton#primary {{ background: {t.accent}; border-color: {t.accent}; color: {t.accent_text}; font-weight: 650; }}
QPushButton#primary:hover {{ background: {t.accent_hover}; border-color: {t.accent_hover}; }}
QPushButton#segment {{ border-radius: 0; padding: 3px 12px; min-width: 22px; }}
QPushButton#segment[first="true"] {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton#segment[last="true"] {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}
QPushButton#segment:checked {{ background: {t.accent}; border-color: {t.accent}; color: {t.accent_text}; font-weight: 650; }}
QPushButton#stepper {{ padding: 0; min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; font-size: 15px; font-weight: 600; color: {t.text_muted}; }}

QLineEdit, QAbstractSpinBox, QComboBox {{
    background: {t.field}; border: 1px solid {t.border_strong}; border-radius: 7px;
    padding: 3px 8px; min-height: 20px; selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border-color: {t.accent}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {t.text_faint}; }}
QAbstractSpinBox {{ padding-right: 8px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0; border: 0; }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{ border: 0; width: 22px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({chevron}); width: 14px; height: 14px; }}
QComboBox::down-arrow:on {{ image: url({chevron_up}); }}
QComboBox QAbstractItemView {{
    background: {t.panel}; border: 1px solid {t.border_strong}; border-radius: 6px; padding: 4px;
    selection-background-color: {t.accent_soft}; selection-color: {t.text};
}}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {t.border_strong}; background: {t.field}; }}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url({check}); }}

QSlider {{ min-height: 20px; }}
QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {t.border_strong}; }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {t.accent}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; background: {t.panel}; border: 2px solid {t.accent}; }}
QSlider::handle:horizontal:hover {{ background: {t.accent_soft}; }}
QSlider::groove:horizontal:disabled, QSlider::sub-page:horizontal:disabled {{ background: {t.border}; }}

QTableView {{
    background: {t.panel}; alternate-background-color: {t.field}; border: 0; font-size: 12px;
    selection-background-color: {t.accent_soft}; selection-color: {t.text}; gridline-color: transparent;
}}
QTableView::item {{ padding: 0 6px; border: 0; }}
QTableView::item:selected {{ background: {t.accent_soft}; color: {t.text}; }}
QHeaderView {{ background: {t.panel}; }}
QHeaderView::section {{
    background: {t.panel}; color: {t.text_faint}; border: 0; border-bottom: 1px solid {t.border};
    padding: 6px 6px; font-size: 11px; font-weight: 700;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.border_strong}; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {t.text_faint}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t.border_strong}; border-radius: 3px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ width: 0; height: 0; background: none; }}

QSplitter::handle {{ background: {t.border}; }}
QStatusBar {{ background: {t.panel}; border-top: 1px solid {t.border}; color: {t.text_muted}; }}
QStatusBar QLabel {{ color: {t.text_muted}; font-size: 12px; }}
QStatusBar::item {{ border: 0; }}
QProgressBar {{ background: {t.field_hover}; border: 0; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}
QMenu {{ background: {t.panel}; border: 1px solid {t.border_strong}; padding: 4px; border-radius: 8px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t.accent_soft}; }}
QMessageBox {{ background: {t.panel}; }}
"""


def apply_theme(app: QtWidgets.QApplication, theme: Theme) -> None:
    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    palette.setColor(role.Window, QtGui.QColor(theme.window))
    palette.setColor(role.WindowText, QtGui.QColor(theme.text))
    palette.setColor(role.Base, QtGui.QColor(theme.panel))
    palette.setColor(role.AlternateBase, QtGui.QColor(theme.field))
    palette.setColor(role.Text, QtGui.QColor(theme.text))
    palette.setColor(role.Button, QtGui.QColor(theme.field))
    palette.setColor(role.ButtonText, QtGui.QColor(theme.text))
    palette.setColor(role.Highlight, QtGui.QColor(theme.accent))
    palette.setColor(role.HighlightedText, QtGui.QColor(theme.accent_text))
    palette.setColor(role.ToolTipBase, QtGui.QColor(theme.panel))
    palette.setColor(role.ToolTipText, QtGui.QColor(theme.text))
    palette.setColor(role.PlaceholderText, QtGui.QColor(theme.text_faint))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(theme))
