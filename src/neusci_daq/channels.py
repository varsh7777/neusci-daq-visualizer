"""Per-channel controls shown to the left of each live plot.

Each row is one channel: switch it on or off, rename it, pick which DAQ input
(ai0, ai1, ...) it reads, change its trace colour, or remove it. Only
switched-on rows are acquired and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .config import PALETTE, PALETTE_NAMES, Channel, normalize_color

STRIP_WIDTH = 180
OFF_COLOR = "#9aa0a6"


@dataclass
class ChannelRow:
    ch: Channel     # ch.enabled is the on/off switch; ch.color is the trace colour


class ColorButton(QtWidgets.QToolButton):
    """A coloured bar. Click it to pick a palette colour or a custom one.

    Calls on_pick(hex) with the chosen colour. Used next to live plots (in the
    channel strip) and review plots.
    """

    def __init__(self, color: str, on_pick, tooltip: str = "Change this channel's trace colour",
                 enabled_look: bool = True):
        super().__init__()
        self._on_pick = on_pick
        self.setToolTip(tooltip)
        self.setFixedWidth(10)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.set_color(color, enabled_look)
        self.clicked.connect(self._show_menu)

    def set_color(self, color: str, enabled_look: bool = True) -> None:
        self.color = color
        self.setStyleSheet(f"QToolButton {{ background: {color if enabled_look else OFF_COLOR}; "
                           "border: none; }")

    def _show_menu(self):
        menu = QtWidgets.QMenu(self)
        for c, name in zip(PALETTE, PALETTE_NAMES):
            pix = QtGui.QPixmap(14, 14)
            pix.fill(QtGui.QColor(c))
            painter = QtGui.QPainter(pix)          # border, so white shows on a white menu
            painter.setPen(QtGui.QColor("#808080"))
            painter.drawRect(0, 0, 13, 13)
            painter.end()
            a = menu.addAction(QtGui.QIcon(pix), name)
            a.setCheckable(True)
            a.setChecked(normalize_color(c) == self.color)
            a.triggered.connect(lambda _=False, c=c: self._pick(c))
        menu.addSeparator()
        menu.addAction("Custom…", self._custom)
        menu.exec(self.mapToGlobal(QtCore.QPoint(self.width(), 0)))

    def _custom(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color), self, "Trace colour")
        if c.isValid():
            self._pick(c.name())

    def _pick(self, c: str):
        c = normalize_color(c)
        if c != self.color:
            # Deferred: the caller may rebuild this button.
            QtCore.QTimer.singleShot(0, lambda: self._on_pick(c))


class ChannelStrip(QtWidgets.QFrame):
    """On/off, name, input, colour and remove controls for one channel."""

    def __init__(self, row: ChannelRow, inputs: list[str], diff: bool,
                 on_change, on_remove, on_color, can_remove: bool):
        super().__init__()
        self.row = row
        self.setFixedWidth(STRIP_WIDTH)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # Colour stays changeable while recording: it's display-only.
        self.color_btn = ColorButton(row.ch.color, lambda c: on_color(self, c),
                                     enabled_look=row.ch.enabled)
        outer.addWidget(self.color_btn)
        body = QtWidgets.QWidget()
        outer.addWidget(body, 1)
        lay = QtWidgets.QGridLayout(body)
        lay.setContentsMargins(8, 4, 4, 4)
        lay.setVerticalSpacing(4)

        self.on = QtWidgets.QCheckBox()
        self.on.setChecked(row.ch.enabled)
        self.on.setToolTip("Record and show this channel")
        self.name = QtWidgets.QLineEdit(row.ch.name)
        self.name.setToolTip("Channel name")
        self.input = QtWidgets.QComboBox()
        self.input.addItems(inputs)
        if row.ch.physical not in inputs:          # e.g. ai5 after switching to DIFF
            self.input.addItem(row.ch.physical)
        self.input.setCurrentText(row.ch.physical)
        self.input.setToolTip("Which DAQ input this channel reads"
                              + (" (DIFF: aiN is measured against ai(N+4))" if diff else ""))
        self.remove = QtWidgets.QToolButton(text="✕")
        self.remove.setToolTip("Remove this channel")
        self._can_remove = can_remove
        self.remove.setEnabled(can_remove)

        lay.addWidget(self.on, 0, 0)
        lay.addWidget(self.name, 0, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Input"), 1, 0, 1, 1)
        lay.addWidget(self.input, 1, 1)
        lay.addWidget(self.remove, 1, 2)
        info = QtWidgets.QLabel(f"{row.ch.units} · scale ×{row.ch.scale:g}")
        info.setToolTip("Units and scale come from the config file")
        info.setStyleSheet("color: gray")
        lay.addWidget(info, 2, 1, 1, 2)
        lay.setRowStretch(3, 1)

        # Defer: the strip is rebuilt after a change, and must not be deleted
        # while its own signal is still running.
        later = lambda f: (lambda *_: QtCore.QTimer.singleShot(0, f))
        self.on.toggled.connect(later(lambda: on_change(self, "enabled", self.on.isChecked())))
        self.input.activated.connect(later(lambda: on_change(self, "physical", self.input.currentText())))
        self.name.editingFinished.connect(later(lambda: self.name.text().strip() != row.ch.name
                                                and on_change(self, "name", self.name.text().strip())))
        self.remove.clicked.connect(later(lambda: on_remove(self)))

    def set_locked(self, locked: bool) -> None:
        """While recording, lock everything except the colour."""
        for w in (self.on, self.name, self.input):
            w.setEnabled(not locked)
        self.remove.setEnabled(self._can_remove and not locked)
