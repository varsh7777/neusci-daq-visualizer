"""The app's flat, quiet look: one stylesheet on top of Qt's Fusion style.

Fusion is used everywhere so the app looks the same on Windows and macOS
(the native Windows style draws gradient buttons and boxed drop-down arrows).
The plots have their own dark/light colours (theme.py); this is the rest of
the window, which is always light.
"""
from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

ICON_DIR = Path(__file__).with_name("icons")

INK = "#1f2328"          # text
MUTED = "#4a5058"        # labels, arrows
SUBTLE = "#5c6370"       # disabled text, small notes
GROUND = "#f3f4f6"       # window and toolbar background
BUTTON = "#e3e6ea"
BUTTON_HOVER = "#d9dde3"
BUTTON_DOWN = "#c9ced6"  # pressed, or a toggle that is on
FIELD_BORDER = "#c9ced6"
LINE = "#d9dce1"
DISABLED = "#eceef1"
FOCUS = "#1a5fd0"
RECORDING = "#c62828"


def _url(name: str) -> str:
    return (ICON_DIR / name).as_posix()


def stylesheet() -> str:
    down, up = _url("chevron-down.svg"), _url("chevron-up.svg")
    down_off, up_off = _url("chevron-down-disabled.svg"), _url("chevron-up-disabled.svg")
    return f"""
QMainWindow, QToolBar, QStatusBar, QMenuBar {{ background: {GROUND}; }}
QToolBar {{ border: none; border-bottom: 1px solid {LINE}; padding: 6px 8px; spacing: 8px; }}
QToolBar::separator {{ background: #d0d4da; width: 1px; margin: 6px 4px; }}
QStatusBar {{ border-top: 1px solid {LINE}; color: {MUTED}; }}
QLabel {{ color: {MUTED}; }}

QPushButton {{
    background: {BUTTON}; color: {INK}; border: none; border-radius: 6px;
    padding: 6px 14px; min-height: 18px;
}}
QPushButton:hover {{ background: {BUTTON_HOVER}; }}
QPushButton:pressed, QPushButton:checked {{ background: {BUTTON_DOWN}; }}
QPushButton:disabled {{ background: {DISABLED}; color: #6b7280; }}
QPushButton#start, QPushButton#record {{ font-weight: 600; }}
QPushButton#record:checked {{ background: {RECORDING}; color: #ffffff; }}

QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px; color: {INK}; }}
QToolButton:hover {{ background: {BUTTON}; }}
QToolButton#remove {{ background: {BUTTON}; color: {MUTED}; }}
QToolButton#remove:hover {{ background: {BUTTON_HOVER}; }}
QToolButton#remove:disabled {{ background: {DISABLED}; color: #9aa0a8; }}

QLineEdit, QComboBox, QAbstractSpinBox {{
    background: #ffffff; color: {INK}; border: 1px solid {FIELD_BORDER}; border-radius: 6px;
    padding: 4px 8px; min-height: 18px;
    selection-background-color: {FOCUS}; selection-color: #ffffff;
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {FOCUS}; }}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    background: {DISABLED}; border-color: {LINE}; color: {SUBTLE};
}}
QComboBox {{ padding-right: 24px; }}
QComboBox QLineEdit {{ border: none; padding: 0; background: transparent; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 22px; border: none; background: transparent;
}}
QComboBox::down-arrow {{ image: url({down}); width: 10px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url({down_off}); }}
QComboBox QAbstractItemView {{
    background: #ffffff; color: {INK}; border: 1px solid {FIELD_BORDER};
    selection-background-color: {BUTTON}; selection-color: {INK}; outline: none;
}}
QAbstractSpinBox {{ padding-right: 22px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 20px; border: none; background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin-top: 3px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 3px; }}
QAbstractSpinBox::up-arrow {{ image: url({up}); width: 9px; height: 5px; }}
QAbstractSpinBox::down-arrow {{ image: url({down}); width: 9px; height: 5px; }}
QAbstractSpinBox::up-arrow:disabled {{ image: url({up_off}); }}
QAbstractSpinBox::down-arrow:disabled {{ image: url({down_off}); }}

QTreeWidget#comments {{
    background: #ffffff; border: none; color: {INK}; outline: none;
    selection-background-color: #eef3fb; selection-color: {INK};
}}
QTreeWidget#comments::item {{ padding: 4px 2px; }}
QWidget#commentsPanel {{ background: #ffffff; border-left: 1px solid {LINE}; }}
QLabel#commentsTitle {{ color: {INK}; font-weight: 600; }}
QFrame#commentPopup {{ background: #ffffff; border: 1px solid {FIELD_BORDER}; border-radius: 8px; }}
"""


def light_palette() -> QtGui.QPalette:
    """A fixed light palette, so the window doesn't turn half-dark when the OS uses dark mode."""
    p = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    for role, c in ((R.Window, GROUND), (R.WindowText, INK), (R.Base, "#ffffff"),
                    (R.AlternateBase, "#f7f8fa"), (R.Text, INK), (R.Button, BUTTON),
                    (R.ButtonText, INK), (R.ToolTipBase, "#ffffff"), (R.ToolTipText, INK),
                    (R.Highlight, FOCUS), (R.HighlightedText, "#ffffff"),
                    (R.PlaceholderText, SUBTLE)):
        p.setColor(role, QtGui.QColor(c))
    p.setColor(QtGui.QPalette.ColorGroup.Disabled, R.Text, QtGui.QColor(SUBTLE))
    p.setColor(QtGui.QPalette.ColorGroup.Disabled, R.ButtonText, QtGui.QColor("#6b7280"))
    return p


def apply(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setStyleSheet(stylesheet())


def dot_icon(color: str, square: bool = False, size: int = 10) -> QtGui.QIcon:
    """The red dot on Record (or the white square on Stop recording)."""
    pix = QtGui.QPixmap(size * 2, size * 2)     # 2x, so it stays sharp on scaled displays
    pix.setDevicePixelRatio(2.0)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QColor(color))
    r = QtCore.QRectF(1, 1, size - 2, size - 2)
    p.drawRect(r) if square else p.drawEllipse(r)
    p.end()
    return QtGui.QIcon(pix)
