"""Tabs across the top (Acquisition, Stimulator, Scope view, Analysis), with an
optional split: two tabs side by side or one above the other.

Each page is a single widget, so it can only be in one half at a time. Picking
the tab that's already open in the other half goes back to one tab, showing
that page full screen.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

ONE, SIDE, STACKED = "off", "h", "v"


class _Pane(QtWidgets.QWidget):
    picked = QtCore.Signal(int, str)        # pane index, page id

    def __init__(self, index: int, ids: list[str], names: dict[str, str]):
        super().__init__()
        self.index, self.ids = index, ids
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QtWidgets.QWidget()
        head.setObjectName("paneHead")
        head.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(8, 0, 8, 0)
        self.bar = QtWidgets.QTabBar()
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        # No ◀ ▶ scroll arrows when the tabs don't fit; shorten the names ("Analy…") instead.
        self.bar.setUsesScrollButtons(False)
        self.bar.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.bar.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for i in ids:
            self.bar.addTab(names[i])
        self.bar.tabBarClicked.connect(lambda k: k >= 0 and self.picked.emit(self.index, self.ids[k]))
        hl.addWidget(self.bar)
        hl.addStretch(1)
        self.head_layout = hl
        lay.addWidget(head)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.body, 1)
        self.page_id: str | None = None

    def show_page(self, pid: str, widget: QtWidgets.QWidget):
        self.bar.blockSignals(True)
        self.bar.setCurrentIndex(self.ids.index(pid))
        self.bar.blockSignals(False)
        if self.page_id == pid and widget.parent() is self:
            return
        while self.body.count():
            w = self.body.takeAt(0).widget()
            if w is not None:
                w.hide()
        self.body.addWidget(widget)
        widget.show()
        self.page_id = pid


class TabArea(QtWidgets.QWidget):
    """The pages under the tab bars. `changed` fires whenever what's visible changes."""

    changed = QtCore.Signal()

    def __init__(self, pages: list[tuple[str, str, QtWidgets.QWidget]]):
        super().__init__()
        self.ids = [pid for pid, _, _ in pages]
        self.names = {pid: name for pid, name, _ in pages}
        self.pages = {pid: w for pid, _, w in pages}
        self.layout_mode = ONE
        self.shown = [self.ids[0], self.ids[1]]
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(7)
        self.panes = [_Pane(i, self.ids, self.names) for i in range(2)]
        for p in self.panes:
            p.picked.connect(self.pick)
            self.splitter.addWidget(p)
        lay.addWidget(self.splitter)
        self._corner: QtWidgets.QWidget | None = None
        self._hold = QtWidgets.QWidget()          # parks pages that aren't shown
        self._hold.hide()
        for w in self.pages.values():
            w.setParent(self._hold)
        self._place()

    def set_corner(self, widget: QtWidgets.QWidget) -> None:
        """Controls shown at the right end of the (last visible) tab bar."""
        self._corner = widget
        self._place()

    # -- what's visible --------------------------------------------------------
    @property
    def split(self) -> bool:
        return self.layout_mode != ONE

    def visible(self) -> list[str]:
        return self.shown[:2] if self.split else self.shown[:1]

    def is_visible(self, pid: str) -> bool:
        return pid in self.visible()

    def shares_screen(self, pid: str) -> bool:
        """True if `pid` is showing in one half of a split (not full screen)."""
        return self.split and self.is_visible(pid)

    # -- changing it -------------------------------------------------------------
    def pick(self, pane: int, pid: str) -> None:
        """A tab was clicked in a pane."""
        if self.split and self.shown[1 - pane] == pid:
            # Already open in the other half: show it full screen instead.
            other = self.shown[pane]
            self.shown = [pid, other]
            self.layout_mode = ONE
        else:
            self.shown[pane] = pid
        self._place()

    def set_layout(self, mode: str) -> None:
        if mode not in (ONE, SIDE, STACKED):
            raise ValueError(mode)
        if mode != ONE and self.shown[0] == self.shown[1]:
            self.shown[1] = next(i for i in self.ids if i != self.shown[0])
        self.layout_mode = mode
        self._place()

    def show_tab(self, pid: str) -> None:
        """Make `pid` visible (used by shortcuts and menu items)."""
        if self.is_visible(pid):
            return
        if self.split:
            self.shown[1] = pid
        else:
            self.shown[0] = pid
        self._place()

    def _place(self):
        self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical if self.layout_mode == STACKED
                                     else QtCore.Qt.Orientation.Horizontal)
        vis = self.visible()
        for pid, w in self.pages.items():
            if pid not in vis:
                w.hide()
                w.setParent(self._hold)
        for i, pane in enumerate(self.panes):
            if i < len(vis):
                pane.show_page(vis[i], self.pages[vis[i]])
                pane.show()
            else:
                pane.hide()
                pane.page_id = None
        if self._corner is not None:
            last = self.panes[len(vis) - 1]
            if self._corner.parent() is not last.head_layout.parentWidget():
                last.head_layout.addWidget(self._corner)
            self._corner.show()
        if self.split:
            total = max(2, sum(self.splitter.sizes()) or 2)
            self.splitter.setSizes([total // 2, total - total // 2])
        self.changed.emit()


def layout_icon(mode: str, ink: str, size: int = 16) -> QtGui.QIcon:
    """One box (one tab), a box split left/right, or a box split top/bottom."""
    scale = 2.0
    pix = QtGui.QPixmap(int(size * scale), int(size * scale))
    pix.setDevicePixelRatio(scale)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor(ink), 1.3))
    r = QtCore.QRectF(1.5, 2.5, size - 3, size - 5)
    p.drawRoundedRect(r, 1.5, 1.5)
    if mode == SIDE:
        p.drawLine(QtCore.QPointF(r.center().x(), r.top()), QtCore.QPointF(r.center().x(), r.bottom()))
    elif mode == STACKED:
        p.drawLine(QtCore.QPointF(r.left(), r.center().y()), QtCore.QPointF(r.right(), r.center().y()))
    p.end()
    return QtGui.QIcon(pix)
