"""The Comments panel: the shown block's comments, editable in place.

Click a time to jump to it, click the text to edit it, and press Delete (or
right-click → Delete) to remove one. The window does the saving; this panel
only shows the list and reports what the user did.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

PANEL_WIDTH = 240


class CommentsPanel(QtWidgets.QWidget):
    edited = QtCore.Signal(int, str)      # row, new text
    deleted = QtCore.Signal(int)          # row
    jump = QtCore.Signal(float)           # time (s, from block start)
    closed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("commentsPanel")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(PANEL_WIDTH)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 8, 10)
        lay.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Comments")
        self.title.setObjectName("commentsTitle")
        close = QtWidgets.QToolButton(text="✕")
        close.setToolTip("Hide comments")
        close.clicked.connect(self.closed.emit)
        head.addWidget(self.title, 1)
        head.addWidget(close)
        lay.addLayout(head)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setObjectName("comments")
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setStretchLastSection(True)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemChanged.connect(self._changed)
        self.tree.customContextMenuRequested.connect(self._menu)
        for key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.tree)
            sc.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(self._delete_selected)
        lay.addWidget(self.tree, 1)
        self.empty = QtWidgets.QLabel("No comments in this block")
        self.empty.setStyleSheet("color: #5c6370")
        lay.addWidget(self.empty)
        lay.addStretch(0)
        self._filling = False
        self._texts: list[str] = []
        self.editable = True

    # -- filling -------------------------------------------------------
    def set_comments(self, title: str, comments: list[tuple[float, str]], editable: bool = True) -> None:
        self._filling = True
        self.title.setText(title)
        self.editable = editable
        self.tree.clear()
        self._texts = [text for _, text in comments]
        for t, text in comments:
            it = QtWidgets.QTreeWidgetItem([f"{t:.1f} s", text])
            it.setForeground(0, QtGui.QBrush(QtGui.QColor("#1a5fb4")))
            it.setToolTip(0, "Click to jump here")
            if editable:
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                it.setToolTip(1, "Click to edit · Right-click to delete")
            self.tree.addTopLevelItem(it)
        self.tree.setVisible(bool(comments))
        self.empty.setVisible(not comments)
        self._filling = False

    # -- user actions --------------------------------------------------
    def _row(self, item) -> int:
        return self.tree.indexOfTopLevelItem(item)

    def _clicked(self, item, column: int):
        if column == 0:
            self.jump.emit(float(item.text(0).split()[0]))
        elif self.editable:
            self.tree.editItem(item, 1)

    def _changed(self, item, column: int):
        if self._filling or column != 1:
            return
        row, text = self._row(item), item.text(1).strip()
        if row < 0 or text == self._texts[row]:
            return
        if not text:                                   # cleared: treat as delete
            QtCore.QTimer.singleShot(0, lambda: self.deleted.emit(row))
        else:
            QtCore.QTimer.singleShot(0, lambda: self.edited.emit(row, text))

    def _delete_selected(self):
        item = self.tree.currentItem()
        if item is not None and self.editable:
            self.deleted.emit(self._row(item))

    def _menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None or not self.editable:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Edit", lambda: self.tree.editItem(item, 1))
        menu.addAction("Delete", lambda: self.deleted.emit(self._row(item)))
        menu.exec(self.tree.viewport().mapToGlobal(pos))


class CommentPopup(QtWidgets.QFrame):
    """The small box that appears when you double-click a trace in review."""

    submitted = QtCore.Signal(float, str)

    def __init__(self, parent, t: float):
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setObjectName("commentPopup")
        self.t = t
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        lay.addWidget(QtWidgets.QLabel(f"New comment at {t:.1f} s"))
        self.edit = QtWidgets.QLineEdit(placeholderText="Press Enter to add comment")
        self.edit.setMinimumWidth(230)
        self.edit.returnPressed.connect(self._submit)
        lay.addWidget(self.edit)

    def _submit(self):
        text = self.edit.text().strip()
        self.close()
        if text:
            self.submitted.emit(self.t, text)

    def show_at(self, global_pos: QtCore.QPoint):
        self.adjustSize()
        self.move(global_pos + QtCore.QPoint(12, 8))
        self.show()
        self.edit.setFocus()
