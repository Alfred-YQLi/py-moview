from __future__ import annotations


APP_STYLE_SHEET = """
QMainWindow {
    background: #f4f4f5;
}
QWidget {
    color: #111114;
    font-size: 11pt;
}
QFrame#side {
    background: #ffffff;
    border-right: 1px solid #e4e4e7;
}
QLabel#header {
    color: #8b5cf6;
    font-size: 22pt;
    font-weight: 700;
}
QLabel#subtitle, QLabel.muted {
    color: #6b6b73;
}
QLabel#sectionTitle {
    color: #27272a;
    font-size: 10pt;
    font-weight: 600;
}
QLabel#scenePrimary {
    color: #27272a;
    font-size: 10pt;
}
QLabel#sceneMeta {
    color: #71717a;
    font-size: 9pt;
}
QPushButton {
    min-height: 30px;
    padding: 2px 12px;
    border: 1px solid #d4d4d8;
    border-radius: 4px;
    background: #ffffff;
}
QPushButton:hover {
    background: #f4f4f5;
    border-color: #a1a1aa;
}
QPushButton:pressed {
    background: #ede9fe;
}
QPushButton:focus {
    border-color: #8b5cf6;
}
QPushButton:disabled {
    color: #a1a1aa;
    background: #f4f4f5;
}
QPushButton#accent {
    color: #ffffff;
    background: #8b5cf6;
    border-color: #8b5cf6;
    font-weight: 600;
}
QPushButton#accent:hover {
    background: #7c3aed;
    border-color: #7c3aed;
}
QPushButton#accent:pressed {
    background: #6d28d9;
    border-color: #6d28d9;
}
QTabWidget::pane {
    border: 1px solid #e4e4e7;
    border-radius: 4px;
    background: #ffffff;
    top: -1px;
}
QTabBar {
    background: #f4f4f5;
}
QTabBar::tab {
    min-width: 92px;
    padding: 7px 14px;
    color: #71717a;
    background: #f4f4f5;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #7c3aed;
    background: #ffffff;
    border-bottom-color: #8b5cf6;
    font-weight: 600;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 28px;
    padding: 1px 7px;
    background: #ffffff;
    border: 1px solid #d4d4d8;
    border-radius: 4px;
    selection-background-color: #8b5cf6;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #8b5cf6;
}
QComboBox::drop-down {
    border: 0;
    width: 22px;
}
QComboBox QAbstractItemView {
    color: #111114;
    background: #ffffff;
    border: 1px solid #d4d4d8;
    selection-color: #ffffff;
    selection-background-color: #8b5cf6;
}
QCheckBox {
    spacing: 7px;
}
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #e4e4e7;
}
QSlider::sub-page:horizontal {
    border-radius: 2px;
    background: #a78bfa;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    border: 2px solid #ffffff;
    border-radius: 7px;
    background: #8b5cf6;
}
QTreeWidget {
    background: #ffffff;
    alternate-background-color: #fafafa;
    border: 1px solid #e4e4e7;
    border-radius: 4px;
    outline: 0;
}
QTreeWidget::item {
    min-height: 24px;
}
QTreeWidget::item:selected {
    color: #ffffff;
    background: #8b5cf6;
}
QHeaderView::section {
    color: #3f3f46;
    background: #f4f4f5;
    padding: 5px;
    border: 0;
    border-bottom: 1px solid #e4e4e7;
    font-weight: 600;
}
QProgressBar {
    min-height: 7px;
    max-height: 7px;
    border: 0;
    border-radius: 3px;
    background: #e4e4e7;
}
QProgressBar::chunk {
    border-radius: 3px;
    background: #8b5cf6;
}
QToolTip {
    color: #ffffff;
    background: #18181b;
    border: 1px solid #18181b;
    padding: 5px;
}
"""
