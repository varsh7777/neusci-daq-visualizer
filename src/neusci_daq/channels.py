"""Per-channel controls shown to the left of each live plot.

Each row is one channel: switch it on or off, rename it, pick which DAQ input
(ai0, ai1, ...) it reads, or remove it. Only switched-on rows are acquired and
recorded.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .config import Channel

STRIP_WIDTH = 180


@dataclass
class ChannelRow:
    ch: Channel     # ch.enabled is the on/off switch
    color: int      # trace colour index; stays with the channel when others change


def color_hex(index: int, n: int) -> str:
    return pg.intColor(index, hues=max(3, n)).name()


class ChannelStrip(QtWidgets.QFrame):
    """On/off, name, input and remove controls for one channel."""

    def __init__(self, row: ChannelRow, inputs: list[str], n_colors: int, diff: bool,
                 on_change, on_remove, can_remove: bool):
        super().__init__()
        self.row = row
        self.setFixedWidth(STRIP_WIDTH)
        color = color_hex(row.color, n_colors) if row.ch.enabled else "#9aa0a6"
        self.setObjectName("strip")
        self.setStyleSheet(f"#strip {{ border-left: 5px solid {color}; }}")
        lay = QtWidgets.QGridLayout(self)
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
