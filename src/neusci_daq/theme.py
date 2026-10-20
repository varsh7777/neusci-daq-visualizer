"""Dark or light plot backgrounds, and the moon/sun toggle icon.

Only the plots change; the rest of the window follows the operating system.
Trace colours are stored as chosen, but very light ones (white, yellow) are
drawn darker on a white background so they stay visible.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui

DARK, LIGHT = "dark", "light"


@dataclass(frozen=True)
class Theme:
    background: str
    foreground: str     # axes, tick labels, axis titles
    marker: str         # comment lines and their labels


THEMES = {
    DARK: Theme(background="#000000", foreground="#c8c8c8", marker="#ffff00"),
    LIGHT: Theme(background="#ffffff", foreground="#303030", marker="#c05000"),
}


def _luminance(c: QtGui.QColor) -> float:
    """Relative luminance (0 = black, 1 = white), as used for contrast ratios."""
    def lin(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(c.red()) + 0.7152 * lin(c.green()) + 0.0722 * lin(c.blue())


def display_color(color: str, mode: str) -> str:
    """The colour to draw a trace in. On white, too-light colours are darkened
    until they reach about 3:1 contrast; on black, too-dark ones are lightened."""
    c = QtGui.QColor(color)
    if mode == LIGHT:
        while _luminance(c) > 0.30:            # (1.05)/(L+0.05) >= 3
            c = c.darker(115)
    else:
        while _luminance(c) < 0.02 and c.lightness() < 250:   # pure black etc.
            c = c.lighter(150) if c.lightness() else QtGui.QColor("#606060")
    return c.name()


def _draw_symbol(p: QtGui.QPainter, rect: QtCore.QRectF, mode: str, ink: QtGui.QColor) -> None:
    """Draw the moon (dark mode) or sun (light mode) to fill `rect`."""
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    s = min(rect.width(), rect.height())
    x0 = rect.x() + (rect.width() - s) / 2
    y0 = rect.y() + (rect.height() - s) / 2
    if mode == DARK:
        moon = QtGui.QPainterPath()
        moon.addEllipse(QtCore.QRectF(x0 + s * 0.12, y0 + s * 0.12, s * 0.76, s * 0.76))
        bite = QtGui.QPainterPath()
        bite.addEllipse(QtCore.QRectF(x0 + s * 0.38, y0 + s * 0.02, s * 0.64, s * 0.64))
        p.fillPath(moon.subtracted(bite), ink)
    else:
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(ink)
        p.drawEllipse(QtCore.QRectF(x0 + s * 0.3, y0 + s * 0.3, s * 0.4, s * 0.4))
        p.setPen(QtGui.QPen(ink, max(1.5, s / 12), cap=QtCore.Qt.PenCapStyle.RoundCap))
        c = QtCore.QPointF(x0 + s / 2, y0 + s / 2)
        for i in range(8):
            t = QtGui.QTransform().rotate(45 * i)
            p.drawLine(c + t.map(QtCore.QPointF(0, -s * 0.31)), c + t.map(QtCore.QPointF(0, -s * 0.46)))


class _SymbolEngine(QtGui.QIconEngine):
    """Draws the symbol at whatever pixel size is needed, so it stays sharp at any
    display scaling (a fixed bitmap gets enlarged and looks blurry at 125–150%)."""

    def __init__(self, mode: str, ink: QtGui.QColor):
        super().__init__()
        self.mode, self.ink = mode, QtGui.QColor(ink)

    def paint(self, painter, rect, mode, state):
        _draw_symbol(painter, QtCore.QRectF(rect), self.mode, self.ink)

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        w, h = max(1, round(size.width() * scale)), max(1, round(size.height() * scale))
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pix)
        _draw_symbol(p, QtCore.QRectF(0, 0, w, h), self.mode, self.ink)
        p.end()
        pix.setDevicePixelRatio(scale)
        return pix

    def clone(self):
        return _SymbolEngine(self.mode, self.ink)


def toggle_icon(mode: str, color: QtGui.QColor | str = "#000000") -> QtGui.QIcon:
    """A moon in dark mode, a sun in light mode, in one plain colour (black on a light
    window, white on a dark one). Drawn at the needed resolution, so never blurry."""
    return QtGui.QIcon(_SymbolEngine(mode, QtGui.QColor(color)))
