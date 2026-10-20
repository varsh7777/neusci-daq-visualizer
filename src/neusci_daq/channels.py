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
        self.setFixedWidth(8)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.set_color(color, enabled_look)
        self.clicked.connect(self._show_menu)

    def set_color(self, color: str, enabled_look: bool = True) -> None:
        self.color = color
        self.setStyleSheet(f"QToolButton {{ background: {color if enabled_look else OFF_COLOR}; "
                           "border: none; border-radius: 0; padding: 0; }")

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


COMPACT_WIDTH = 54
LINE_MIN, LINE_MAX = 0.5, 20.0


class ChannelStrip(QtWidgets.QFrame):
    """On/off, name, input, line thickness, colour and remove controls for one channel.

    In split view the strip is compact (colour bar and name) and shows its full
    controls while the mouse is over it, or while you're editing one of them.
    """

    hover_changed = QtCore.Signal(bool)

    def __init__(self, row: ChannelRow, inputs: list[str], diff: bool,
                 on_change, on_remove, on_color, can_remove: bool, on_width=None):
        super().__init__()
        self.row = row
        self.setObjectName("channelStrip")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(STRIP_WIDTH)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # Colour stays changeable while recording: it's display-only.
        self.color_btn = ColorButton(row.ch.color, lambda c: on_color(self, c),
                                     enabled_look=row.ch.enabled)
        outer.addWidget(self.color_btn)
        # Compact view: just the name.
        self.mini = QtWidgets.QLabel(row.ch.name)
        self.mini.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.mini.setWordWrap(True)
        self.mini.setToolTip("Point at this to show the channel's controls")
        self.mini.hide()
        outer.addWidget(self.mini, 1)
        self.body = QtWidgets.QWidget()
        outer.addWidget(self.body, 1)
        lay = QtWidgets.QGridLayout(self.body)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.setVerticalSpacing(6)

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
        self.remove.setObjectName("remove")
        self.remove.setFixedSize(26, 26)
        self.remove.setToolTip("Remove this channel")
        self._can_remove = can_remove
        self.remove.setEnabled(can_remove)
        # Line thickness: typed, 0.5-20 px. Display only, so it works while recording.
        self.line_width = QtWidgets.QDoubleSpinBox(minimum=LINE_MIN, maximum=LINE_MAX, decimals=1, singleStep=0.5)
        self.line_width.setSuffix(" px")
        self.line_width.setValue(float(getattr(row.ch, "width", 1.0)))
        self.line_width.setKeyboardTracking(False)
        self.line_width.setToolTip(f"Line thickness ({LINE_MIN:g}–{LINE_MAX:g} px)")

        lay.addWidget(self.on, 0, 0)
        lay.addWidget(self.name, 0, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Input"), 1, 0, 1, 1)
        lay.addWidget(self.input, 1, 1)
        lay.addWidget(self.remove, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Line"), 2, 0, 1, 1)
        lay.addWidget(self.line_width, 2, 1)
        info = QtWidgets.QLabel(f"{row.ch.units} · scale ×{row.ch.scale:g}")
        info.setToolTip("Units and scale come from the config file")
        info.setStyleSheet("color: gray")
        lay.addWidget(info, 3, 1, 1, 2)
        lay.setRowStretch(4, 1)

        # Defer: the strip is rebuilt after a change, and must not be deleted
        # while its own signal is still running.
        later = lambda f: (lambda *_: QtCore.QTimer.singleShot(0, f))
        self.on.toggled.connect(later(lambda: on_change(self, "enabled", self.on.isChecked())))
        self.input.activated.connect(later(lambda: on_change(self, "physical", self.input.currentText())))
        self.name.editingFinished.connect(later(lambda: self.name.text().strip() != row.ch.name
                                                and on_change(self, "name", self.name.text().strip())))
        self.remove.clicked.connect(later(lambda: on_remove(self)))
        if on_width is not None:
            self.line_width.valueChanged.connect(lambda v: on_width(self, float(v)))

        self.compact = False
        self.expanded = True
        self._collapse_timer = QtCore.QTimer(self, interval=250, timeout=self._maybe_collapse)

    def set_locked(self, locked: bool) -> None:
        """While recording, lock everything except the colour and line thickness."""
        for w in (self.on, self.name, self.input):
            w.setEnabled(not locked)
        self.remove.setEnabled(self._can_remove and not locked)

    # -- compact (split view) ----------------------------------------------------
    def set_compact(self, compact: bool) -> None:
        self.compact = compact
        self._set_expanded(not compact)

    def _set_expanded(self, on: bool) -> None:
        self.expanded = on
        self.body.setVisible(on)
        self.mini.setVisible(not on)
        self.setFixedWidth(STRIP_WIDTH if on else COMPACT_WIDTH)
        if self.compact and on:
            self._collapse_timer.start()
        else:
            self._collapse_timer.stop()
        self.hover_changed.emit(on)

    def enterEvent(self, ev):
        if self.compact and not self.expanded:
            self._set_expanded(True)
        super().enterEvent(ev)

    def _busy(self) -> bool:
        """Still in use: under the mouse, editing a field, or a drop-down is open."""
        if self.underMouse():
            return True
        fw = QtWidgets.QApplication.focusWidget()
        if fw is not None and (fw is self or self.isAncestorOf(fw)) and fw is not self.color_btn:
            return True
        popup = QtWidgets.QApplication.activePopupWidget()
        return popup is not None

    def _maybe_collapse(self):
        if self.compact and self.expanded and not self._busy():
            self._set_expanded(False)


class ChannelRowWidget(QtWidgets.QWidget):
    """One channel row: its strip and its plot (or an "off" label).

    In split view the strip is compact, and when it expands it slides over the
    plot instead of squeezing it, so the trace doesn't jump.
    """

    def __init__(self, strip: ChannelStrip, content: QtWidgets.QWidget):
        super().__init__()
        self.strip, self.content = strip, content
        strip.setParent(self)
        content.setParent(self)
        strip.raise_()
        strip.hover_changed.connect(lambda _: self._relayout())

    def set_compact(self, compact: bool) -> None:
        self.strip.set_compact(compact)
        self._relayout()

    def sizeHint(self):
        s = self.content.sizeHint()
        return QtCore.QSize(STRIP_WIDTH + s.width(), max(s.height(), self.strip.sizeHint().height()))

    def minimumSizeHint(self):
        return QtCore.QSize(COMPACT_WIDTH + 50, self.strip.minimumSizeHint().height())

    def resizeEvent(self, ev):
        self._relayout()
        super().resizeEvent(ev)

    def _relayout(self):
        w, h = self.width(), self.height()
        left = COMPACT_WIDTH if self.strip.compact else STRIP_WIDTH
        self.content.setGeometry(left, 0, max(0, w - left), h)
        self.strip.setGeometry(0, 0, STRIP_WIDTH if self.strip.expanded else COMPACT_WIDTH, h)
        self.strip.raise_()
