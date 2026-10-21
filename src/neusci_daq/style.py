"""The app's flat look, in light and dark: one stylesheet on top of Qt's Fusion style.

Fusion is used everywhere so the app looks the same on Windows and macOS
(the native Windows style draws gradient buttons and boxed drop-down arrows).
The moon/sun button switches the whole app, plots included (see theme.py for
the plot colours).
"""
from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

ICON_DIR = Path(__file__).with_name("icons")
LIGHT, DARK = "light", "dark"

# Colours for each mode. Start (blue) and Record (red) are the same in both.
TOKENS = {
    LIGHT: dict(
        ink="#1f2328", muted="#4a5058", subtle="#5c6370", ground="#f3f4f6", panel="#ffffff",
        tabbar="#eceef1", tab_on="#ffffff", button="#e3e6ea", button_hover="#d9dde3",
        button_down="#c9ced6", field="#ffffff", field_border="#c9ced6", line="#d9dce1",
        disabled="#eceef1", disabled_ink="#6b7280", focus="#1a5fd0", link="#1a5fb4",
        selection="#eef3fb", arrows="", error="#b42318",
    ),
    DARK: dict(
        ink="#e6e8eb", muted="#a9b0b9", subtle="#8b929c", ground="#1b1d22", panel="#202328",
        tabbar="#1f2126", tab_on="#2b2e34", button="#2f333a", button_hover="#3a3f47",
        button_down="#474c55", field="#24272d", field_border="#3f444c", line="#3a3f47",
        disabled="#222529", disabled_ink="#7d848e", focus="#4d8dff", link="#8ab4ff",
        selection="#2c3a52", arrows="-dark", error="#ff8a80",
    ),
}
START = "#1a5fd0"
RECORDING = "#c62828"
RECORD = "#b42318"

_applied = {"mode": None}     # set once the style is in use (by run(), not by tests)


def tokens(mode: str) -> dict:
    return TOKENS[mode if mode in TOKENS else LIGHT]


def _url(name: str) -> str:
    return (ICON_DIR / name).as_posix()


def stylesheet(mode: str = LIGHT) -> str:
    t = tokens(mode)
    a = t["arrows"]
    down, up = _url(f"chevron-down{a}.svg"), _url(f"chevron-up{a}.svg")
    down_off, up_off = _url(f"chevron-down-disabled{a}.svg"), _url(f"chevron-up-disabled{a}.svg")
    check, check_off = _url("check.svg"), _url("check-disabled.svg")
    return f"""
QWidget {{ color: {t['ink']}; }}
QMainWindow, QToolBar, QStatusBar, QMenuBar, QDialog {{ background: {t['ground']}; }}
QMenuBar::item {{ background: transparent; color: {t['ink']}; padding: 3px 8px; }}
QMenuBar::item:selected {{ background: {t['button']}; }}
QMenu {{ background: {t['panel']}; color: {t['ink']}; border: 1px solid {t['line']}; }}
QMenu::item:selected {{ background: {t['button']}; }}
QMenu::item:disabled {{ color: {t['disabled_ink']}; }}
QMenu::separator {{ height: 1px; background: {t['line']}; margin: 4px 8px; }}
QToolTip {{ background: {t['panel']}; color: {t['ink']}; border: 1px solid {t['line']}; }}
QToolBar {{ border: none; border-bottom: 1px solid {t['line']}; padding: 6px 8px; spacing: 6px; }}
QToolBar::separator {{ background: {t['line']}; width: 1px; margin: 6px 4px; }}
QStatusBar {{ border-top: 1px solid {t['line']}; color: {t['muted']}; }}
QStatusBar QLabel {{ color: {t['muted']}; }}
QLabel {{ color: {t['muted']}; background: transparent; }}
QCheckBox {{ color: {t['ink']}; background: transparent; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {t['field_border']}; border-radius: 3px;
    background: {t['field']};
}}
QCheckBox::indicator:hover {{ border-color: {t['focus']}; }}
QCheckBox::indicator:checked {{ background: {t['focus']}; border-color: {t['focus']}; image: url({check}); }}
QCheckBox::indicator:disabled {{ background: {t['disabled']}; border-color: {t['line']}; }}
QCheckBox::indicator:checked:disabled {{ image: url({check_off}); }}

QPushButton {{
    background: {t['button']}; color: {t['ink']}; border: none; border-radius: 6px;
    padding: 6px 14px; min-height: 18px;
}}
QPushButton:hover {{ background: {t['button_hover']}; }}
QPushButton:pressed, QPushButton:checked {{ background: {t['button_down']}; }}
QPushButton:disabled {{ background: {t['disabled']}; color: {t['disabled_ink']}; }}
QPushButton#start, QPushButton#record {{ font-weight: 600; color: #ffffff; }}
QPushButton#start {{ background: {START}; }}
QPushButton#start:hover {{ background: #1752b5; }}
QPushButton#start:checked {{ background: {START}; }}
QPushButton#record {{ background: {RECORD}; }}
QPushButton#record:hover {{ background: #9e1f15; }}
QPushButton#record:checked {{ background: {RECORDING}; }}
QPushButton#start:disabled, QPushButton#record:disabled {{ background: {t['disabled']}; color: {t['disabled_ink']}; }}

QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px; color: {t['ink']}; }}
QToolButton:hover {{ background: {t['button']}; }}
QToolButton:checked {{ background: {t['button_down']}; }}
QToolButton#remove {{ background: {t['button']}; color: {t['muted']}; }}
QToolButton#remove:hover {{ background: {t['button_hover']}; }}
QToolButton#remove:disabled {{ background: {t['disabled']}; color: {t['disabled_ink']}; }}
QWidget#splitGroup {{ background: {t['field']}; border: 1px solid {t['field_border']}; border-radius: 6px; }}
QWidget#splitGroup QToolButton {{ padding: 2px 6px; border-radius: 4px; }}

QLineEdit, QComboBox, QAbstractSpinBox {{
    background: {t['field']}; color: {t['ink']}; border: 1px solid {t['field_border']}; border-radius: 6px;
    padding: 4px 8px; min-height: 18px;
    selection-background-color: {t['focus']}; selection-color: #ffffff;
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {t['focus']}; }}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    background: {t['disabled']}; border-color: {t['line']}; color: {t['subtle']};
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
    background: {t['panel']}; color: {t['ink']}; border: 1px solid {t['field_border']};
    selection-background-color: {t['button']}; selection-color: {t['ink']}; outline: none;
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

QTableWidget, QTreeWidget {{
    background: {t['panel']}; color: {t['ink']}; gridline-color: {t['line']};
    border: 1px solid {t['line']}; selection-background-color: {t['selection']}; selection-color: {t['ink']};
}}
QHeaderView::section {{ background: {t['tabbar']}; color: {t['muted']}; border: none;
    border-right: 1px solid {t['line']}; border-bottom: 1px solid {t['line']}; padding: 4px; }}
QTreeWidget#comments {{ border: none; outline: none; }}
QTreeWidget#comments::item {{ padding: 4px 2px; }}
QWidget#commentsPanel {{ background: {t['panel']}; border-left: 1px solid {t['line']}; }}
QLabel#commentsTitle {{ color: {t['ink']}; font-weight: 600; }}
QFrame#commentPopup {{ background: {t['panel']}; border: 1px solid {t['field_border']}; border-radius: 8px; }}

QTabBar {{ background: {t['tabbar']}; }}
QTabBar::tab {{
    background: transparent; color: {t['muted']}; padding: 7px 16px; margin: 4px 2px 0 2px;
    border: none; border-bottom: 2px solid transparent;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: {t['tab_on']}; color: {t['ink']}; font-weight: 600;
    border-bottom: 2px solid {t['focus']}; }}
QTabBar::tab:hover:!selected {{ color: {t['ink']}; }}
QWidget#paneHead {{ background: {t['tabbar']}; border-bottom: 1px solid {t['line']}; }}
QSplitter::handle {{ background: {t['tab_on']}; }}
QWidget#stimChip {{ background: {t['button']}; border-radius: 6px; }}
QWidget#stimChip QPushButton {{ background: {t['button_hover']}; padding: 3px 9px; min-height: 16px; }}
QWidget#stimChip QPushButton:hover {{ background: {t['button_down']}; }}
QLabel#stimChipText {{ color: {t['ink']}; }}
QScrollBar {{ background: {t['ground']}; }}
"""


def palette(mode: str = LIGHT) -> QtGui.QPalette:
    """A fixed palette per mode, so the OS's own light/dark setting can't mix in."""
    t = tokens(mode)
    p = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    for role, c in ((R.Window, t["ground"]), (R.WindowText, t["ink"]), (R.Base, t["field"]),
                    (R.AlternateBase, t["panel"]), (R.Text, t["ink"]), (R.Button, t["button"]),
                    (R.ButtonText, t["ink"]), (R.ToolTipBase, t["panel"]), (R.ToolTipText, t["ink"]),
                    (R.Highlight, t["focus"]), (R.HighlightedText, "#ffffff"),
                    (R.PlaceholderText, t["subtle"]), (R.Link, t["link"])):
        p.setColor(role, QtGui.QColor(c))
    for role in (R.Text, R.ButtonText, R.WindowText):
        p.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(t["disabled_ink"]))
    return p


def light_palette() -> QtGui.QPalette:   # kept for older callers
    return palette(LIGHT)


def apply(app: QtWidgets.QApplication, mode: str = LIGHT) -> None:
    """Use the flat style (called at start-up, and again when the mode changes)."""
    app.setStyle("Fusion")
    app.setPalette(palette(mode))
    app.setStyleSheet(stylesheet(mode))
    _applied["mode"] = mode


def restyle(mode: str) -> None:
    """Switch an already styled app to `mode`. Does nothing if the style isn't in use."""
    app = QtWidgets.QApplication.instance()
    if app is not None and _applied["mode"] is not None:
        apply(app, mode)


def dot_icon(color: str, square: bool = False, size: int = 10) -> QtGui.QIcon:
    """The dot on Record (or the square on Stop recording)."""
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
