"""The Trials tab: run a stimulus many times while stepping one or two of its
settings, and see the responses lined up, overlaid, averaged and measured.

Left: the protocol (what to vary, repeats, order, timing, window).
Middle: Plan (the order of trials and the stimulus at each value), Responses
(one panel per value of b, coloured by a) and Curve (the measurement against a).
Right: the measurement and a table of every trial; untick a trial to leave it out.

The protocol logic lives in trials.py; this file is the window and the runner.
"""
from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import stim as stimmod
from . import style, theme
from . import trials as T

ROW_GAP = 8          # px between the parts of a settings row (box, button, label)
B_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#F0E442", "#999999")


class _Spin(QtWidgets.QDoubleSpinBox):
    """Shows 0.5 rather than 0.500, so values fit in narrow boxes."""

    def textFromValue(self, v: float) -> str:
        return f"{v:.{self.decimals()}f}".rstrip("0").rstrip(".") or "0"


def _spin(lo, hi, decimals=3, suffix="", step=1.0, width=78):
    s = _Spin(minimum=lo, maximum=hi, decimals=decimals, singleStep=step)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    s.setMinimumWidth(width)
    return s


class _Segmented(QtWidgets.QWidget):
    """A row of buttons where exactly one is on."""

    picked = QtCore.Signal(str)

    def __init__(self, items: list[tuple[str, str]], tip: str = ""):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        for key, label in items:
            b = QtWidgets.QPushButton(label, checkable=True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, key=key: self.picked.emit(key))
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[key] = b

    def set(self, key: str) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def value(self) -> str:
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return next(iter(self.buttons))


class VarRow(QtWidgets.QWidget):
    """One variable: which setting, and its values as a Range or a List."""

    changed = QtCore.Signal()
    removed = QtCore.Signal()

    def __init__(self, name: str, removable: bool = False):
        super().__init__()
        self.name = name
        self._shape = "pulse"
        self._loading = False
        g = QtWidgets.QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)
        tag = QtWidgets.QLabel(f"Vary {name}")
        tag.setMinimumWidth(56)
        self.setting = QtWidgets.QComboBox()
        self.setting.setToolTip(f"Which setting of the stimulus {name} steps through")
        self.mode = _Segmented([("range", "Range"), ("list", "List")],
                               "Range: from, to and step. List: any values, in the order typed.")
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(self.setting, 1)
        top.addWidget(self.mode)
        self.btn_remove = None
        if removable:
            self.btn_remove = QtWidgets.QToolButton()
            self.btn_remove.setObjectName("remove")
            self.btn_remove.setText("✕")
            self.btn_remove.setToolTip(f"Don't vary a second setting")
            self.btn_remove.setAccessibleName(f"Remove {name}")
            self.btn_remove.clicked.connect(self.removed.emit)
            top.addWidget(self.btn_remove)
        g.addWidget(tag, 0, 0)
        g.addLayout(top, 0, 1)

        self.range_box = QtWidgets.QWidget()
        r = QtWidgets.QHBoxLayout(self.range_box)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(4)
        self.start = _spin(-1e6, 1e6, width=80)
        self.stop = _spin(-1e6, 1e6, width=80)
        self.step = _spin(0, 1e6, width=80)
        for lab, w in (("from", self.start), ("to", self.stop), ("step", self.step)):
            r.addWidget(QtWidgets.QLabel(lab))
            r.addWidget(w)
        r.addStretch(1)
        self.list_edit = QtWidgets.QLineEdit()
        self.list_edit.setPlaceholderText("e.g. 0.5, 1, 2, 3")
        self.list_edit.setToolTip("Values separated by commas or spaces, in the order you want them")
        self.count = QtWidgets.QLabel()
        vals = QtWidgets.QVBoxLayout()
        vals.setSpacing(3)
        vals.addWidget(self.range_box)
        vals.addWidget(self.list_edit)
        vals.addWidget(self.count)
        g.addLayout(vals, 1, 1)
        g.setColumnStretch(1, 1)

        self.setting.activated.connect(self._setting_picked)
        self.mode.picked.connect(self._mode_picked)
        for w in (self.start, self.stop, self.step):
            w.valueChanged.connect(self._edited)
        self.list_edit.editingFinished.connect(self._edited)
        self._list_text = ""

    # -- in and out ------------------------------------------------------------
    def set_shape(self, shape: str, avoid: str = "") -> None:
        """Offer the settings of `shape`; keep the current one if it still applies."""
        self._shape = shape
        cur = self.setting.currentData()
        self.setting.blockSignals(True)
        self.setting.clear()
        for key, label, unit in T.SETTINGS.get(shape, []):
            self.setting.addItem(f"{label} ({unit})" if unit and unit != "×" else label, key)
        keys = [k for k, _, _ in T.SETTINGS.get(shape, [])]
        if cur not in keys:
            cur = next((k for k in keys if k != avoid), keys[0] if keys else "")
            if cur:
                self._set_range(*T.DEFAULT_RANGES.get(cur, (1, 5, 1)))
        self.setting.setCurrentIndex(max(0, self.setting.findData(cur)))
        self.setting.blockSignals(False)
        self._update_units()

    def set_var(self, v: T.Variable) -> None:
        self._loading = True
        self.setting.setCurrentIndex(max(0, self.setting.findData(v.key)))
        self.mode.set(v.mode)
        self._set_range(v.start, v.stop, v.step)
        self._list_text = v.text
        self.list_edit.setText(v.text)
        self._loading = False
        self._show_mode()
        self._update_units()

    def var(self) -> T.Variable:
        return T.Variable(key=self.setting.currentData() or "", mode=self.mode.value(),
                          start=self.start.value(), stop=self.stop.value(), step=self.step.value(),
                          text=self.list_edit.text())

    def set_locked(self, locked: bool) -> None:
        for w in (self.setting, self.mode, self.start, self.stop, self.step, self.list_edit):
            w.setEnabled(not locked)
        if self.btn_remove is not None:
            self.btn_remove.setEnabled(not locked)

    def show_count(self, text: str, error: bool, mode: str) -> None:
        t = style.tokens(mode)
        self.count.setStyleSheet(f"color: {t['error'] if error else t['subtle']}")
        self.count.setText(text)

    # -- internals ---------------------------------------------------------------
    def _set_range(self, a, b, c):
        for w, v in ((self.start, a), (self.stop, b), (self.step, c)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _setting_picked(self, *_):
        key = self.setting.currentData()
        self._set_range(*T.DEFAULT_RANGES.get(key, (1, 5, 1)))
        self._update_units()
        self.changed.emit()

    def _mode_picked(self, *_):
        self._show_mode()
        self.changed.emit()

    def _show_mode(self):
        is_range = self.mode.value() == "range"
        self.range_box.setVisible(is_range)
        self.list_edit.setVisible(not is_range)

    def _edited(self, *_):
        if not self._loading:
            self.changed.emit()

    def _update_units(self):
        unit = T.setting_info(self._shape, self.setting.currentData() or "")[1]
        tip = f"In {unit}" if unit not in ("", "×") else ""
        for w in (self.start, self.stop, self.step):
            w.setToolTip(tip)      # the unit is in the setting's name, e.g. Amplitude (V)


class TrialsPage(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._mode = style.LIGHT
        self._loading = False
        self._has_b = False
        self.protocol = self._saved_protocol()
        self.chk: T.Check | None = None
        self.plan: list[T.Trial] = []
        self.trials: list[T.Trial] = []
        self.results: dict[int, T.Result] = {}
        self.running = False
        self.paused = False
        self.next_i = 0
        self._next_fire = 0.0
        self.capture: T.Capture | None = None
        self._q = None
        self._run = None                # what the current/last run used: dict
        self.picked: int | None = None
        self.hidden: set[int] = set()
        self.show_mode = "overlay"
        self._items: dict[int, pg.PlotDataItem] = {}
        self._avg_items: dict[tuple[int, int], pg.PlotDataItem] = {}
        self._plots: list[pg.PlotItem] = []
        self._regions: list[pg.LinearRegionItem] = []
        self._plan_row = 0
        self.timer = QtCore.QTimer(self, interval=20, timeout=self._tick)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self._build_left())
        self.split.addWidget(self._build_middle())
        self.split.addWidget(self._build_right())
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setStretchFactor(2, 0)
        outer.addWidget(self.split)
        self._sized = False

        self._load_protocol_into_widgets(self.protocol)
        self.apply_theme(style.LIGHT)
        self.sync()
        self.show_view("plan")

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._sized:
            self._sized = True
            QtCore.QTimer.singleShot(0, self._size_panels)    # once the window has its size

    def _size_panels(self):
        """The side panels get what their contents need; the plots get the rest."""
        total = self.split.width()
        right = 270
        left = max(300, min(440, total - right - 300))
        self.split.setSizes([left, max(200, total - left - right), right])

    # ======================================================================
    # building
    # ======================================================================
    def _build_left(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("trialsSide")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(8)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self.name = QtWidgets.QLineEdit()
        self.name.setAccessibleName("Protocol name")
        self.btn_save = QtWidgets.QPushButton("Save…")
        self.btn_save.setToolTip("Save this protocol to a file")
        self.btn_load = QtWidgets.QPushButton("Load…")
        self.btn_load.setToolTip("Load a saved protocol")
        name_row = QtWidgets.QHBoxLayout()
        name_row.setSpacing(ROW_GAP)
        name_row.addWidget(self.name, 1)
        name_row.addWidget(self.btn_save)
        name_row.addWidget(self.btn_load)
        form.addRow("Protocol", name_row)
        shape_row = QtWidgets.QHBoxLayout()
        shape_row.setSpacing(ROW_GAP)
        self.shape = QtWidgets.QComboBox()
        for key, label in stimmod.SHAPES.items():
            self.shape.addItem(label, key)
        self.shape.setToolTip("The stimulus is the Stimulator tab's; changing the shape here changes it there too")
        self.btn_edit_stim = QtWidgets.QPushButton("Edit…")
        self.btn_edit_stim.setToolTip("Open the Stimulator tab to change the other settings")
        shape_row.addWidget(self.shape, 1)
        shape_row.addWidget(self.btn_edit_stim)
        form.addRow("Shape", shape_row)
        self.stim_info = QtWidgets.QLabel()
        self.stim_info.setWordWrap(True)
        form.addRow("", self.stim_info)
        v.addLayout(form)
        v.addWidget(self._line())

        self.row_a = VarRow("a")
        self.row_b = VarRow("b", removable=True)
        self.btn_add_b = QtWidgets.QPushButton("+ Add a second variable (b)")
        v.addWidget(self.row_a)
        v.addWidget(self.row_b)
        v.addWidget(self.btn_add_b, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        form2 = QtWidgets.QFormLayout()
        form2.setHorizontalSpacing(10)
        form2.setVerticalSpacing(8)
        rep_row = QtWidgets.QHBoxLayout()
        rep_row.setSpacing(ROW_GAP)
        self.repeats = QtWidgets.QSpinBox(minimum=1, maximum=1000)
        self.repeats.setKeyboardTracking(False)
        self.rep_note = QtWidgets.QLabel("of each value")
        rep_row.addWidget(self.repeats)
        rep_row.addWidget(self.rep_note, 1)
        form2.addRow("Repeats", rep_row)
        self.order = _Segmented(list(T.ORDERS.items()),
                                "Blocked: all repeats of one value, then the next. Interleaved: every value "
                                "once per repeat. Random: every value once per repeat, shuffled.")
        form2.addRow("Order", self.order)
        self.gap = _spin(0.05, 3600, decimals=2, suffix=" s", step=0.5)
        self.gap.setToolTip("From the start of one trial to the start of the next")
        form2.addRow("Between", self.gap)
        win_row = QtWidgets.QHBoxLayout()
        win_row.setSpacing(ROW_GAP)
        self.before = _spin(0, 10000, decimals=1, suffix=" ms", step=1, width=70)
        self.after = _spin(0.1, 10000, decimals=1, suffix=" ms", step=5, width=70)
        win_row.addWidget(self.before)
        win_row.addWidget(QtWidgets.QLabel("before,"))
        win_row.addWidget(self.after)
        win_row.addWidget(QtWidgets.QLabel("after"))
        win_row.addStretch(1)
        form2.addRow("Show", win_row)
        align_row = QtWidgets.QHBoxLayout()
        align_row.setSpacing(ROW_GAP)
        self.align = QtWidgets.QLabel()
        self.align.setWordWrap(True)
        self.btn_monitor = QtWidgets.QPushButton("Choose…")
        self.btn_monitor.setToolTip("Pick the input the stimulus is wired to (Stimulator tab › Record on)")
        align_row.addWidget(self.align, 1)
        align_row.addWidget(self.btn_monitor)
        form2.addRow("Line up on", align_row)
        v.addLayout(form2)

        v.addStretch(1)
        # Pinned below the settings, so Run is always on screen.
        foot = QtWidgets.QWidget()
        foot.setObjectName("trialsFoot")
        fv = QtWidgets.QVBoxLayout(foot)
        fv.setContentsMargins(14, 8, 14, 10)
        fv.setSpacing(8)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        fv.addWidget(self.summary)
        run_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run protocol")
        self.btn_run.setObjectName("start")
        self.btn_pause = QtWidgets.QPushButton("Pause", checkable=True)
        self.btn_pause.setToolTip("Hold the next trial; Resume carries on where it left off")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setToolTip("End the protocol and keep the trials so far (Esc)")
        for b in (self.btn_run, self.btn_pause, self.btn_stop):
            run_row.addWidget(b)
        run_row.addStretch(1)
        fv.addLayout(run_row)

        # signals
        self.name.editingFinished.connect(self._settings_changed)
        self.shape.activated.connect(self._shape_picked)
        self.btn_edit_stim.clicked.connect(self.main.open_stim_window)
        self.row_a.changed.connect(self._settings_changed)
        self.row_b.changed.connect(self._settings_changed)
        self.row_b.removed.connect(lambda: self._set_b(False))
        self.btn_add_b.clicked.connect(lambda: self._set_b(True))
        self.repeats.valueChanged.connect(self._settings_changed)
        self.order.picked.connect(self._settings_changed)
        for w in (self.gap, self.before, self.after):
            w.valueChanged.connect(self._settings_changed)
        self.btn_monitor.clicked.connect(self.main.open_stim_window)
        self.btn_run.clicked.connect(self.run_protocol)
        self.btn_pause.toggled.connect(self.set_paused)
        self.btn_stop.clicked.connect(lambda: self.stop_protocol("Protocol stopped"))
        self.btn_save.clicked.connect(self._save)
        self.btn_load.clicked.connect(self._load)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(scroll, 1)
        lv.addWidget(self._line())
        lv.addWidget(foot)
        left.setMinimumWidth(260)
        return left

    def _build_middle(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)
        self.view = _Segmented([("plan", "Plan"), ("responses", "Responses"), ("curve", "Curve")])
        self.view.set("plan")
        bar.addWidget(self.view)
        self.show_seg = _Segmented([("overlay", "Overlay"), ("averages", "Averages"), ("both", "Both")],
                                   "Every trial, the average of each a, or both")
        self.show_seg.set("overlay")
        bar.addStretch(1)
        v.addLayout(bar)
        row2 = QtWidgets.QVBoxLayout()
        row2.setSpacing(6)
        seg_row = QtWidgets.QHBoxLayout()
        seg_row.addWidget(self.show_seg)
        seg_row.addStretch(1)
        row2.addLayout(seg_row)
        legend_inner = QtWidgets.QWidget()
        self.legend = QtWidgets.QHBoxLayout(legend_inner)
        self.legend.setContentsMargins(0, 0, 0, 0)
        self.legend.setSpacing(4)
        self.legend_box = QtWidgets.QScrollArea()
        self.legend_box.setWidget(legend_inner)
        self.legend_box.setWidgetResizable(True)
        self.legend_box.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.legend_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.legend_box.setFixedHeight(46)
        self.legend_box.setToolTip("Click a value of a to show or hide it")
        self.legend_box.setMinimumWidth(60)
        row2.addWidget(self.legend_box)
        self.row2 = QtWidgets.QWidget()
        self.row2.setLayout(row2)
        row2.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.row2)

        self.stack = QtWidgets.QStackedWidget()
        # plan
        plan = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plan)
        pv.setContentsMargins(0, 0, 0, 0)
        self.plan_title = QtWidgets.QLabel()
        self.grid = QtWidgets.QTableWidget()
        self.grid.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.grid.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.grid.setToolTip("The trials each combination runs in. Click a row to preview its stimuli.")
        self.plan_note = QtWidgets.QLabel()
        self.plan_note.setWordWrap(True)
        self.preview = pg.PlotWidget()
        self.preview.getPlotItem().setLabel("left", "Output", units="V")
        self.preview.getPlotItem().setLabel("bottom", "time", units="s")
        self.preview.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.preview_title = QtWidgets.QLabel()
        self.preview_title.setWordWrap(True)
        self.plan_title.setWordWrap(True)
        pv.addWidget(self.plan_title)
        pv.addWidget(self.grid)
        pv.addWidget(self.plan_note)
        pv.addWidget(self.preview_title)
        pv.addWidget(self.preview, 1)
        self.stack.addWidget(plan)
        # responses: one plot per value of b. Separate plot widgets, deleted
        # explicitly when rebuilt: a pyqtgraph GraphicsLayoutWidget left to the
        # garbage collector can crash Qt in a later layout pass.
        self.resp = QtWidgets.QWidget()
        self.resp_lay = QtWidgets.QVBoxLayout(self.resp)
        self.resp_lay.setContentsMargins(0, 0, 0, 0)
        self.resp_lay.setSpacing(2)
        self._resp_widgets: list[pg.PlotWidget] = []
        self.stack.addWidget(self.resp)
        # curve
        self.curve_plot = pg.PlotWidget()
        self.curve_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.stack.addWidget(self.curve_plot)
        v.addWidget(self.stack, 1)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        v.addWidget(self.message)

        self.view.picked.connect(self.show_view)
        self.show_seg.picked.connect(self._show_mode_picked)
        self.grid.itemSelectionChanged.connect(self._plan_row_picked)
        return w

    def _build_right(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setObjectName("trialsSide")
        w.setMinimumWidth(270)
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(8)
        self.m_channel = QtWidgets.QComboBox()
        self.m_kind = QtWidgets.QComboBox()
        for k, label in T.MEASURES.items():
            self.m_kind.addItem(label, k)
        rng = QtWidgets.QHBoxLayout()
        rng.setSpacing(ROW_GAP)
        self.m_from = _spin(-10000, 10000, decimals=1, suffix=" ms", step=1, width=88)
        self.m_to = _spin(-10000, 10000, decimals=1, suffix=" ms", step=1, width=88)
        rng.addWidget(self.m_from)
        rng.addWidget(QtWidgets.QLabel("to"))
        rng.addWidget(self.m_to)
        form.addRow("Channel", self.m_channel)
        form.addRow("Measure", self.m_kind)
        form.addRow("From", rng)
        v.addLayout(form)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "#", "a", "b", "rep", "value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c, mode in enumerate([QtWidgets.QHeaderView.ResizeMode.ResizeToContents] * 5
                                 + [QtWidgets.QHeaderView.ResizeMode.Stretch]):
            hh.setSectionResizeMode(c, mode)
        self.table.setToolTip("Untick a trial to leave it out of the averages and the curve. "
                              "Click a row to highlight its trace.")
        v.addWidget(self.table, 1)
        self.btn_export = QtWidgets.QPushButton("Export CSV…")
        self.btn_export.setToolTip("Every trial's values and measurement, for a spreadsheet")
        v.addWidget(self.btn_export, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.m_channel.activated.connect(self._measure_changed)
        self.m_kind.activated.connect(self._measure_changed)
        self.m_from.valueChanged.connect(self._measure_changed)
        self.m_to.valueChanged.connect(self._measure_changed)
        self.table.itemChanged.connect(self._use_toggled)
        self.table.itemSelectionChanged.connect(self._row_picked)
        self.btn_export.clicked.connect(self._export)
        return w

    @staticmethod
    def _line() -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        return f

    # ======================================================================
    # protocol <-> widgets
    # ======================================================================
    def _saved_protocol(self) -> T.Protocol:
        text = self.main.settings.value("trials_protocol", "") or ""
        if text:
            try:
                return T.Protocol.loads(text)
            except ValueError:
                pass
        shape = self.main.stimulus.shape
        keys = [k for k, _, _ in T.SETTINGS.get(shape, [])]
        key = keys[0] if keys else "amplitude_v"
        f, t, s = T.DEFAULT_RANGES.get(key, (0.5, 3.0, 0.5))
        return T.Protocol(name="Protocol", a=T.Variable(key=key, start=f, stop=t, step=s))

    def _load_protocol_into_widgets(self, p: T.Protocol) -> None:
        self._loading = True
        shape = self.main.stimulus.shape
        self.name.setText(p.name)
        self.row_a.set_shape(shape)
        self.row_a.set_var(p.a)
        self.row_b.set_shape(shape, avoid=p.a.key)
        if p.b is not None:
            self.row_b.set_var(p.b)
        self._show_b(p.b is not None)
        self.repeats.setValue(p.repeats)
        self.order.set(p.order)
        self.gap.setValue(p.gap_s)
        self.before.setValue(p.before_ms)
        self.after.setValue(p.after_ms)
        self.m_kind.setCurrentIndex(max(0, self.m_kind.findData(p.measure)))
        self.m_from.setValue(p.measure_from_ms)
        self.m_to.setValue(p.measure_to_ms)
        self._loading = False
        self._settings_changed()

    def protocol_from_widgets(self) -> T.Protocol:
        return T.Protocol(name=self.name.text().strip() or "Protocol", a=self.row_a.var(),
                          b=self.row_b.var() if self._has_b else None, repeats=self.repeats.value(),
                          order=self.order.value(), gap_s=self.gap.value(),
                          before_ms=self.before.value(), after_ms=self.after.value(),
                          measure=self.m_kind.currentData(), measure_from_ms=self.m_from.value(),
                          measure_to_ms=self.m_to.value(), measure_channel=self.m_channel.currentText())

    def _show_b(self, on: bool) -> None:
        self._has_b = on
        self.row_b.setVisible(on)
        self.btn_add_b.setVisible(not on)
        self.rep_note.setText("of each a × b combination" if on else "of each value")

    def _set_b(self, on: bool) -> None:
        if on:
            self.row_b.set_shape(self.main.stimulus.shape, avoid=self.row_a.var().key)
            if self.row_b.var().key == self.row_a.var().key:
                keys = [k for k, _, _ in T.SETTINGS.get(self.main.stimulus.shape, []) if k != self.row_a.var().key]
                if keys:
                    self.row_b.set_var(T.Variable(keys[0], "range", *T.DEFAULT_RANGES.get(keys[0], (1, 5, 1))))
        self._show_b(on)
        self._settings_changed()

    def _shape_picked(self, *_):
        key = self.shape.currentData()
        if key != self.main.stimulus.shape:
            self.main.set_stimulus(replace(self.main.stimulus, shape=key))   # calls stimulus_changed

    def stimulus_changed(self) -> None:
        """The Stimulator tab's stimulus changed (here or there)."""
        if self.running:
            return
        shape = self.main.stimulus.shape
        self.shape.blockSignals(True)
        self.shape.setCurrentIndex(max(0, self.shape.findData(shape)))
        self.shape.blockSignals(False)
        self.row_a.set_shape(shape)
        self.row_b.set_shape(shape, avoid=self.row_a.var().key)
        self._settings_changed()

    # ======================================================================
    # reacting to settings
    # ======================================================================
    def _settings_changed(self, *_):
        if self._loading:
            return
        self.protocol = self.protocol_from_widgets()
        self.main.settings.setValue("trials_protocol", self.protocol.dumps())
        cfg = self.main.cfg
        s = self.main.stimulus
        self.stim_info.setText(s.describe() + f" · on {cfg.stim_output}")
        self.chk = T.check(self.protocol, s, cfg.stim_rate, cfg.stim_max_v)
        self._show_counts()
        self._update_summary()
        if not self.running:
            self._update_plan()

    def _show_counts(self):
        shape = self.main.stimulus.shape
        for row, var in ((self.row_a, self.protocol.a), (self.row_b, self.protocol.b)):
            if var is None:
                continue
            try:
                vals = var.values()
            except ValueError as e:
                row.show_count(str(e), True, self._mode)
                continue
            unit = T.setting_info(shape, var.key)[1]
            text = f"{len(vals)} value{'s' if len(vals) != 1 else ''}"
            if var.mode == "range":
                text += f", last {T.fmt(vals[-1], unit)}"
            else:
                text += ": " + ", ".join(T.fmt(v, unit) for v in vals[:8]) + (" …" if len(vals) > 8 else "")
            row.show_count(text, False, self._mode)

    def _update_summary(self):
        t = style.tokens(self._mode)
        c, cfg = self.chk, self.main.cfg
        problems = list(c.errors) if c else []
        if not cfg.stim_monitor:
            problems.append("Record the stimulus on a spare input first (Line up on › Choose…), "
                            "so each trial can be lined up on when the stimulus really started.")
        if problems:
            self.summary.setStyleSheet(f"color: {t['error']}")
            self.summary.setText("\n".join(problems))
        else:
            mins, secs = divmod(int(round(c.duration_s)), 60)
            dur = f"{mins} min {secs} s" if mins else f"{secs} s"
            text = (f"{c.n_trials} trials · about {dur} · highest {c.peak_v:g} V "
                    f"(limit {cfg.stim_max_v:g} V)")
            if c.warnings:
                text += "\n" + "\n".join(c.warnings)
            self.summary.setStyleSheet(f"color: {t['muted'] if not c.warnings else t['error']}")
            self.summary.setText(text)
        self._update_buttons()

    def _update_buttons(self):
        idle = not self.running
        can = idle and bool(self.chk and self.chk.ok) and bool(self.main.cfg.stim_monitor)
        self.btn_run.setEnabled(can)
        self.btn_pause.setEnabled(self.running)
        self.btn_stop.setEnabled(self.running)
        for w in (self.name, self.shape, self.btn_edit_stim, self.btn_add_b, self.repeats, self.order,
                  self.gap, self.before, self.after, self.btn_monitor, self.btn_load):
            w.setEnabled(idle)
        self.row_a.set_locked(not idle)
        self.row_b.set_locked(not idle)
        self.btn_export.setEnabled(bool(self.results))

    def sync(self) -> None:
        """Channels or the stimulus input changed in the main window."""
        cfg = self.main.cfg
        t = style.tokens(self._mode)
        mon = next((c for c in cfg.channels if c.physical == cfg.stim_monitor), None) if cfg.stim_monitor else None
        if mon is not None:
            self.align.setStyleSheet("")
            self.align.setText(f"{mon.name} ({mon.physical}), where the stimulus starts")
        else:
            self.align.setStyleSheet(f"color: {t['error']}")
            self.align.setText("Nothing yet: wire the output to a spare input and pick it")
        if not self.running:
            names = [c.name for c in cfg.channels]
            cur = self.m_channel.currentText() or self.protocol.measure_channel
            self.m_channel.blockSignals(True)
            self.m_channel.clear()
            self.m_channel.addItems(names)
            if cur not in names:
                cur = next((c.name for c in cfg.channels if c.physical != cfg.stim_monitor), names[0] if names else "")
            self.m_channel.setCurrentText(cur)
            self.m_channel.blockSignals(False)
        self._settings_changed()

    # ======================================================================
    # plan view
    # ======================================================================
    def _update_plan(self):
        c, p = self.chk, self.protocol
        shape = self.main.stimulus.shape
        self.grid.clear()
        if not c or not c.a_vals:
            self.grid.setRowCount(0)
            self.grid.setColumnCount(0)
            self.plan_title.setText("Fix the settings on the left to see the plan.")
            self.preview.getPlotItem().clear()
            return
        self.plan = T.sequence(p, c.a_vals, c.b_vals, np.random.default_rng(0))
        ua = T.setting_info(shape, p.a.key)[1]
        nb = len(c.b_vals) if c.b_vals else 1
        self.plan_title.setText(
            f"{len(c.a_vals)} a" + (f" × {nb} b" if c.b_vals else "") + f" × {p.repeats} repeats = "
            f"{len(self.plan)} trials" + (" (one random order; each run shuffles anew)" if p.order == "random" else ""))
        self.grid.setColumnCount(len(c.a_vals))
        self.grid.setRowCount(nb)
        self.grid.setHorizontalHeaderLabels([f"a = {T.fmt(a, ua)}" for a in c.a_vals])
        ub = T.setting_info(shape, p.b.key)[1] if p.b else ""
        self.grid.setVerticalHeaderLabels([f"b = {T.fmt(b, ub)}" for b in c.b_vals] if c.b_vals else ["Trials"])
        where: dict[tuple[int, int], list[int]] = {}
        for tr in self.plan:
            where.setdefault((tr.ia, tr.ib), []).append(tr.n)
        colors = self._a_colors(len(c.a_vals))
        for ib in range(nb):
            for ia in range(len(c.a_vals)):
                it = QtWidgets.QTableWidgetItem(", ".join(map(str, where.get((ia, ib), []))))
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                col = QtGui.QColor(colors[ia])
                it.setBackground(col)
                it.setForeground(QtGui.QColor("#0b0d10" if col.lightness() > 120 else "#ffffff"))
                self.grid.setItem(ib, ia, it)
        self.grid.resizeColumnsToContents()
        h = self.grid.horizontalHeader().height() + sum(self.grid.rowHeight(r) for r in range(nb)) + 4
        self.grid.setFixedHeight(min(h, 220))
        notes = {"blocked": "Blocked: all repeats of one value, then the next. Easy to follow, but slow "
                            "changes in the preparation look like an effect of a.",
                 "interleaved": "Interleaved: every value once per repeat, in order. Spreads slow changes "
                                "across the values.",
                 "random": "Random: every value once per repeat, shuffled; a new order each run."}
        self.plan_note.setText(notes.get(p.order, ""))
        self._plan_row = min(self._plan_row, nb - 1)
        self._draw_preview()

    def _plan_row_picked(self):
        rows = {i.row() for i in self.grid.selectedIndexes()}
        if rows:
            self._plan_row = min(rows)
            self._draw_preview()

    def _draw_preview(self):
        c, p, cfg = self.chk, self.protocol, self.main.cfg
        pi = self.preview.getPlotItem()
        pi.clear()
        if not c or not c.a_vals:
            return
        b = c.b_vals[self._plan_row] if c.b_vals else None
        shape = self.main.stimulus.shape
        self.preview_title.setText("Stimulus at each a" + (
            f", for b = {T.fmt(b, T.setting_info(shape, p.b.key)[1])} (click a row above to change)" if b is not None else ""))
        colors = self._a_colors(len(c.a_vals))
        for ia, a in enumerate(c.a_vals):
            try:
                wave, _ = T.make_wave(self.main.stimulus, p, a, b, cfg.stim_rate)
            except ValueError:
                continue
            t = np.arange(len(wave)) / cfg.stim_rate
            pi.plot(np.repeat(t, 2)[1:], np.repeat(wave, 2)[:-1], pen=pg.mkPen(colors[ia], width=2))

    # ======================================================================
    # running
    # ======================================================================
    def run_protocol(self) -> bool:
        main, cfg = self.main, self.main.cfg
        self._settings_changed()
        c = self.chk
        problem = None
        if main.stim_out is None:
            problem = "Trials needs the nidaqmx (or simulated) backend to send stimuli."
        elif main.mode != "live" or not main.backend.running:
            problem = "Press Start first: trials run while acquiring."
        elif not cfg.stim_monitor or not any(ch.physical == cfg.stim_monitor for ch in cfg.channels):
            problem = "Record the stimulus on a spare input first (Line up on › Choose…)."
        elif not c or not c.ok:
            problem = "Fix the protocol first: " + (c.errors[0] if c and c.errors else "")
        elif main.stim_out.busy:
            problem = "A stimulus is still being sent; wait for it to finish."
        if problem:
            self.show_message(problem, error=True)
            return False
        main.set_repeat(False)
        p = self.protocol
        self.trials = T.sequence(p, c.a_vals, c.b_vals)
        rate = main.backend.actual_rate if main.backend.hardware_timed else cfg.sample_rate
        chans = list(cfg.channels)
        mon_row = next(i for i, ch in enumerate(chans) if ch.physical == cfg.stim_monitor)
        to_units = lambda raw: np.vstack([ch.to_units(raw[i]) for i, ch in enumerate(chans)])
        keep = max(10.0, 3 * (p.gap_s + c.longest_s + p.after_ms / 1000.0) + 2)
        self.capture = T.Capture(rate, mon_row, to_units, keep, read_s=cfg.chunk_size / max(rate, 1e-9))
        self._q = main.backend.subscribe()
        self._run = {"protocol": p, "stimulus": main.stimulus, "shape": main.stimulus.shape,
                     "a_vals": c.a_vals, "b_vals": c.b_vals, "channels": [ch.name for ch in chans],
                     "units": [ch.units for ch in chans], "out_rate": cfg.stim_rate,
                     "block": main.recorder.path.name if main.recorder.recording and main.recorder.path else ""}
        self.results = {}
        self.hidden = set()
        self.picked = None
        self.running, self.paused, self.next_i = True, False, 0
        self._next_fire = time.monotonic() + 0.2
        main._lock_channels(True)
        self.btn_pause.blockSignals(True)
        self.btn_pause.setChecked(False)
        self.btn_pause.setText("Pause")
        self.btn_pause.blockSignals(False)
        self._fill_table()
        self._build_response_plots()
        self._rebuild_legend()
        self.show_view("responses")
        self._update_buttons()
        self.timer.start()
        self.show_message(f"Running {p.name}: {len(self.trials)} trials")
        main._update_stim_chip()
        return True

    def set_paused(self, on: bool) -> None:
        if not self.running:
            return
        self.paused = on
        self.btn_pause.setText("Resume" if on else "Pause")
        if not on:
            self._next_fire = time.monotonic()
        self.show_message("Paused: Resume carries on with trial "
                          f"{self.next_i + 1}" if on else "Resumed")
        self.main._update_stim_chip()

    def stop_protocol(self, why: str = "Protocol stopped") -> None:
        if not self.running:
            return
        self.running = False
        if self.main.stim_out is not None:
            self.main.stim_out.stop()          # output to 0 V
        self._drain()
        if self.capture is not None:
            for r in self.capture.resolve(self._run["protocol"], force=True):
                self._add_result(r)
        self._end(why)

    def _end(self, why: str):
        self.running = False
        self.timer.stop()
        if self._q is not None:
            self.main.backend.unsubscribe(self._q)
            self._q = None
        self.main._lock_channels(self.main.recorder.recording)
        self._save_to_block()
        found = sum(1 for r in self.results.values() if r.found)
        missed = len(self.results) - found
        msg = f"{why}: {found} of {len(self.trials)} trials recorded"
        if missed:
            msg += (f"; {missed} couldn't be lined up (the recorded stimulus didn't match what was "
                    "sent: check the wire from the output to the input)")
        self.show_message(msg, error=bool(missed))
        self.main._stim_message(msg, error=bool(missed))
        self._update_buttons()
        self._redraw_curve()
        self.main._update_stim_chip()

    def chip_text(self) -> str:
        if not self.running:
            return ""
        n = min(self.next_i, len(self.trials))
        cur = self.trials[n - 1] if n else None
        text = f"Trial {n} of {len(self.trials)}"
        if cur is not None:
            text += " · " + T.describe(self._run["protocol"], self._run["shape"], cur.a, cur.b)
        return ("Paused · " if self.paused else "") + text

    def _drain(self):
        if self._q is None or self.capture is None:
            return
        while True:
            try:
                c = self._q.get_nowait()
            except Exception:
                break
            self.capture.add(c.t, c.data)

    def _tick(self):
        if self.capture is None:
            return
        self._drain()
        p = self._run["protocol"]
        for r in self.capture.resolve(p):
            self._add_result(r)
        if not self.running:
            return
        if not self.main.backend.running:
            self.stop_protocol("Acquisition stopped")
            return
        now = time.monotonic()
        if self.next_i < len(self.trials):
            if self.paused or now < self._next_fire:
                return
            if self.main.stim_out.busy or not self.capture.can_send(p.gap_s):
                return                        # too soon after the last one; try again shortly
            tr = self.trials[self.next_i]
            try:
                wave, delay = T.make_wave(self._run["stimulus"], p, tr.a, tr.b, self._run["out_rate"])
            except ValueError as e:
                self.stop_protocol(f"Stopped at trial {tr.n}: {e}")
                return
            label = (f"Trial {tr.n}/{len(self.trials)}: "
                     + T.describe(p, self._run["shape"], tr.a, tr.b))
            if not self.main.send_wave(wave, label):
                self.stop_protocol(f"Stopped at trial {tr.n}")
                return
            self.capture.sent(tr, wave, self._run["out_rate"], delay)
            self._set_row_status(tr.n, "sent")
            self.next_i += 1
            self._next_fire = max(self._next_fire + p.gap_s, now + 0.05)
            self.main._update_stim_chip()
        elif not self.capture.pending:
            self._end("Protocol finished")

    # ======================================================================
    # results
    # ======================================================================
    def _value(self, r: T.Result) -> float:
        if not r.found or r.data is None:
            return float("nan")
        p = self.protocol
        row = self._channel_row()
        if row is None:
            return float("nan")
        return T.measure(p.measure, r.t_ms, r.data[row], p.measure_from_ms, p.measure_to_ms)

    def _channel_row(self) -> int | None:
        names = self._run["channels"] if self._run else [c.name for c in self.main.cfg.channels]
        name = self.m_channel.currentText()
        return names.index(name) if name in names else None

    def _add_result(self, r: T.Result):
        self.results[r.trial.n] = r
        self._set_row_status(r.trial.n, "done")
        self._draw_result(r)
        self._update_average(r.trial.ia, r.trial.ib)
        self._rebuild_legend()
        if self.stack.currentIndex() == 2:
            self._redraw_curve()
        self._update_buttons()

    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.trials))
        shape = self._run["shape"] if self._run else self.main.stimulus.shape
        p = self._run["protocol"] if self._run else self.protocol
        ua = T.setting_info(shape, p.a.key)[1]
        ub = T.setting_info(shape, p.b.key)[1] if p.b else ""
        for k, tr in enumerate(self.trials):
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
                         | QtCore.Qt.ItemFlag.ItemIsSelectable)
            use.setCheckState(QtCore.Qt.CheckState.Checked)
            use.setData(QtCore.Qt.ItemDataRole.UserRole, tr.n)
            self.table.setItem(k, 0, use)
            for c, text in enumerate([str(tr.n), T.fmt(tr.a, ua),
                                      T.fmt(tr.b, ub) if tr.b is not None else "", str(tr.rep), "—"], 1):
                self.table.setItem(k, c, QtWidgets.QTableWidgetItem(text))
        self.table.setColumnHidden(3, p.b is None)
        self.table.blockSignals(False)

    def _set_row_status(self, n: int, status: str):
        k = n - 1
        if not (0 <= k < self.table.rowCount()):
            return
        self.table.blockSignals(True)
        item = self.table.item(k, 5)
        r = self.results.get(n)
        if status == "sent":
            item.setText("…")
            self.table.scrollToItem(item)
        elif r is not None:
            if not r.found:
                item.setText("not found")
                item.setToolTip(f"The recorded stimulus didn't match what was sent (match {r.match:.2f}).")
                self.table.item(k, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
                r.use = False
            else:
                v = self._value(r)
                item.setText("—" if not np.isfinite(v) else f"{v:.4g}")
        self.table.blockSignals(False)

    def _use_toggled(self, item: QtWidgets.QTableWidgetItem):
        if item.column() != 0:
            return
        n = item.data(QtCore.Qt.ItemDataRole.UserRole)
        r = self.results.get(n)
        if r is None:
            return
        r.use = item.checkState() == QtCore.Qt.CheckState.Checked and r.found
        self._style_item(r)
        self._update_average(r.trial.ia, r.trial.ib)
        self._redraw_curve()

    def _row_picked(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        self.picked = (min(rows) + 1) if rows else None
        for r in self.results.values():
            self._style_item(r)

    def _measure_changed(self, *_):
        self._settings_changed()
        for reg in self._regions:
            reg.blockSignals(True)
            reg.setRegion((self.m_from.value(), self.m_to.value()))
            reg.blockSignals(False)
        for n in self.results:
            self._set_row_status(n, "done")
        if self._run and self.m_channel.currentText():
            self._build_response_plots()
        self._redraw_curve()

    def _region_moved(self, reg):
        lo, hi = reg.getRegion()
        for w, v in ((self.m_from, lo), (self.m_to, hi)):
            w.blockSignals(True)
            w.setValue(round(v, 1))
            w.blockSignals(False)
        for other in self._regions:
            if other is not reg:
                other.blockSignals(True)
                other.setRegion((lo, hi))
                other.blockSignals(False)
        self._settings_changed()
        for n in self.results:
            self._set_row_status(n, "done")
        self._redraw_curve()

    # ======================================================================
    # responses view
    # ======================================================================
    def _a_colors(self, n: int) -> list[str]:
        cmap = pg.colormap.get("viridis")
        lo, hi = (0.25, 1.0) if self._mode == style.DARK else (0.0, 0.8)
        pos = [0.5 * (lo + hi)] if n == 1 else list(np.linspace(lo, hi, n))
        return [cmap.map(x, mode="qcolor").name() for x in pos]

    def _clear_response_plots(self):
        for pw in self._resp_widgets:
            self.resp_lay.removeWidget(pw)
            pw.setParent(None)
            pw.deleteLater()
        self._resp_widgets = []

    def _build_response_plots(self):
        self._clear_response_plots()
        self._plots, self._regions = [], []
        self._items, self._avg_items = {}, {}
        if not self._run:
            return
        th = self._theme()
        b_vals = self._run["b_vals"] or [None]
        row = self._channel_row()
        name = self.m_channel.currentText()
        unit = self._run["units"][row] if row is not None else ""
        ub = T.setting_info(self._run["shape"], self._run["protocol"].b.key)[1] if self._run["protocol"].b else ""
        for ib, b in enumerate(b_vals):
            pw = pg.PlotWidget()
            pw.setBackground(th.background)
            self.resp_lay.addWidget(pw, 1)
            self._resp_widgets.append(pw)
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True, alpha=0.25)
            pi.setLabel("left", f"{name} ({unit})")
            if b is not None:
                pi.setTitle(f"b = {T.fmt(b, ub)}", color=th.foreground, size="10pt")
            if ib == len(b_vals) - 1:
                pi.setLabel("bottom", "ms after the stimulus")
            for ax in ("left", "bottom"):
                pi.getAxis(ax).setPen(th.foreground)
                pi.getAxis(ax).setTextPen(th.foreground)
                pi.getAxis(ax).enableAutoSIPrefix(False)
            pi.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(th.foreground, style=QtCore.Qt.PenStyle.DashLine)))
            fg = QtGui.QColor(th.foreground)
            fg.setAlpha(150)
            reg = pg.LinearRegionItem((self.m_from.value(), self.m_to.value()),
                                      brush=pg.mkBrush(128, 128, 128, 40), pen=pg.mkPen(fg, width=1))
            reg.setToolTip("Measured between these lines; drag them to change it")
            reg.sigRegionChangeFinished.connect(self._region_moved)
            pi.addItem(reg)
            p = self._run["protocol"]
            pi.setXRange(-p.before_ms, p.after_ms, padding=0.02)
            pi.enableAutoRange(axis="x", enable=False)
            if self._plots:
                pi.setXLink(self._plots[0])
            self._plots.append(pi)
            self._regions.append(reg)
        for r in self.results.values():
            self._draw_result(r)
        for ia in range(len(self._run["a_vals"])):
            for ib in range(len(b_vals)):
                self._update_average(ia, ib)

    def _draw_result(self, r: T.Result):
        if not r.found or r.data is None or not self._plots:
            return
        row = self._channel_row()
        if row is None:
            return
        pi = self._plots[min(r.trial.ib, len(self._plots) - 1)]
        item = self._items.get(r.trial.n)
        if item is None:
            item = pi.plot()
            self._items[r.trial.n] = item
        item.setData(r.t_ms, r.data[row])
        self._style_item(r)

    def _style_item(self, r: T.Result):
        item = self._items.get(r.trial.n)
        if item is None or not self._run:
            return
        color = QtGui.QColor(self._a_colors(len(self._run["a_vals"]))[r.trial.ia])
        sel = self.picked == r.trial.n
        alpha = 255 if sel else (60 if self.show_mode == "both" or (self.picked and not sel) else 200)
        if not r.use:
            alpha = 50
        color.setAlpha(alpha)
        pen = pg.mkPen(color, width=2.6 if sel else 1,
                       style=QtCore.Qt.PenStyle.DotLine if not r.use else QtCore.Qt.PenStyle.SolidLine)
        item.setPen(pen)
        item.setVisible(self.show_mode != "averages" and r.trial.ia not in self.hidden)
        if sel:
            item.setZValue(10)
        else:
            item.setZValue(0)

    def _update_average(self, ia: int, ib: int):
        if not self._plots or not self._run:
            return
        row = self._channel_row()
        mine = [r for r in self.results.values()
                if r.trial.ia == ia and r.trial.ib == ib and r.use and r.found and r.data is not None]
        key = (ia, ib)
        item = self._avg_items.get(key)
        if not mine or row is None:
            if item is not None:
                item.setData([], [])
            return
        if item is None:
            item = self._plots[min(ib, len(self._plots) - 1)].plot()
            item.setZValue(5)
            self._avg_items[key] = item
        n = min(len(r.data[row]) for r in mine)
        avg = np.mean([r.data[row][:n] for r in mine], axis=0)
        item.setData(mine[0].t_ms[:n], avg)
        item.setPen(pg.mkPen(self._a_colors(len(self._run["a_vals"]))[ia], width=2.4))
        item.setVisible(self.show_mode != "overlay" and ia not in self.hidden)

    def _show_mode_picked(self, mode: str):
        self.show_mode = mode
        for r in self.results.values():
            self._style_item(r)
        for (ia, ib) in list(self._avg_items):
            self._update_average(ia, ib)

    def _rebuild_legend(self):
        while self.legend.count():
            w = self.legend.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if not self._run:
            return
        shape, p = self._run["shape"], self._run["protocol"]
        ua = T.setting_info(shape, p.a.key)[1]
        colors = self._a_colors(len(self._run["a_vals"]))
        for ia, a in enumerate(self._run["a_vals"]):
            n = sum(1 for r in self.results.values() if r.trial.ia == ia and r.found)
            b = QtWidgets.QPushButton(f"{T.fmt(a, ua)} · {n}", checkable=True)
            b.setChecked(ia not in self.hidden)
            b.setIcon(style.dot_icon(colors[ia]))
            b.setToolTip("Show or hide this value of a (number = trials so far)")
            b.toggled.connect(lambda on, ia=ia: self._legend_toggled(ia, on))
            self.legend.addWidget(b)
        self.legend.addStretch(1)

    def _legend_toggled(self, ia: int, on: bool):
        (self.hidden.discard if on else self.hidden.add)(ia)
        for r in self.results.values():
            self._style_item(r)
        for (a, b) in list(self._avg_items):
            self._update_average(a, b)

    # ======================================================================
    # curve view
    # ======================================================================
    def _redraw_curve(self):
        pi = self.curve_plot.getPlotItem()
        pi.clear()
        if pi.legend is not None:
            pi.legend.scene().removeItem(pi.legend) if pi.legend.scene() else None
            pi.legend = None
        if not self._run or not self.results:
            return
        th = self._theme()
        p, shape = self.protocol, self._run["shape"]
        la, ua = T.setting_info(shape, p.a.key)
        row = self._channel_row()
        unit = "ms" if p.measure == "latency" else (self._run["units"][row] if row is not None else "")
        pi.setLabel("bottom", f"a: {la}" + (f" ({ua})" if ua and ua != "×" else ""))
        pi.setLabel("left", f"{T.MEASURES[p.measure]} ({unit})")
        for ax in ("left", "bottom"):
            pi.getAxis(ax).setPen(th.foreground)
            pi.getAxis(ax).setTextPen(th.foreground)
            pi.getAxis(ax).enableAutoSIPrefix(False)
        b_vals = self._run["b_vals"] or [None]
        if len(b_vals) > 1:
            pi.addLegend(offset=(10, 10), labelTextColor=th.foreground)
        ub = T.setting_info(shape, self._run["protocol"].b.key)[1] if self._run["protocol"].b else ""
        a_vals = self._run["a_vals"]
        for ib, b in enumerate(b_vals):
            color = B_COLORS[ib % len(B_COLORS)] if len(b_vals) > 1 else ("#4d8dff" if self._mode == style.DARK else "#1a5fd0")
            xs, ys, means, sds = [], [], [], []
            for ia, a in enumerate(a_vals):
                vals = [self._value(r) for r in self.results.values()
                        if r.trial.ia == ia and r.trial.ib == ib and r.use]
                n, m, sd = T.stats(vals)
                xs += [a] * len([v for v in vals if np.isfinite(v)])
                ys += [v for v in vals if np.isfinite(v)]
                means.append(m)
                sds.append(sd)
            ma = np.asarray(a_vals, float)
            mm, ss = np.asarray(means, float), np.asarray(sds, float)
            ok = np.isfinite(mm)
            if xs:
                pi.plot(xs, ys, pen=None, symbol="o", symbolSize=6, symbolPen=None,
                        symbolBrush=pg.mkBrush(QtGui.QColor(color).lighter(100).name() + "88"))
            if ok.any():
                name = f"b = {T.fmt(b, ub)}" if b is not None else None
                pi.plot(ma[ok], mm[ok], pen=pg.mkPen(color, width=2), symbol="s", symbolSize=9,
                        symbolBrush=color, symbolPen=th.background, name=name)
                pi.addItem(pg.ErrorBarItem(x=ma[ok], y=mm[ok], top=ss[ok], bottom=ss[ok], beam=0.0,
                                           pen=pg.mkPen(color, width=1.5)))

    # ======================================================================
    # views, theme, messages
    # ======================================================================
    def show_view(self, key: str):
        self.view.set(key)
        self.stack.setCurrentIndex({"plan": 0, "responses": 1, "curve": 2}[key])
        self.row2.setVisible(key == "responses")
        if key == "curve":
            self._redraw_curve()

    def _theme(self):
        return theme.THEMES[theme.DARK if self._mode == style.DARK else theme.LIGHT]

    def apply_theme(self, mode: str) -> None:
        self._mode = mode
        th = self._theme()
        for pw in (self.preview, self.curve_plot):
            pw.setBackground(th.background)
            for ax in ("left", "bottom"):
                pw.getPlotItem().getAxis(ax).setPen(th.foreground)
                pw.getPlotItem().getAxis(ax).setTextPen(th.foreground)
        self._settings_changed()
        if self._run:
            self._build_response_plots()
            self._rebuild_legend()
            self._redraw_curve()

    def shutdown(self) -> None:
        """Called when the window closes: stop any protocol and delete the plots now.

        Left to Python's garbage collector, pyqtgraph plots can be deleted while Qt
        still has a layout pass queued for them, which crashes Qt. The main window
        deletes its own plots the same way.
        """
        if self.running:
            self.stop_protocol("Window closed")
        self.timer.stop()
        if self._q is not None:
            self.main.backend.unsubscribe(self._q)
            self._q = None
        self._clear_response_plots()
        self._plots, self._regions, self._items, self._avg_items = [], [], {}, {}
        for pw in (self.preview, self.curve_plot):
            pw.setParent(None)
            pw.deleteLater()

    def show_message(self, text: str, error: bool = False):
        t = style.tokens(self._mode)
        self.message.setStyleSheet(f"color: {t['error'] if error else t['muted']}")
        self.message.setText(text)

    # ======================================================================
    # saving
    # ======================================================================
    def _save_to_block(self):
        """Keep the protocol and each trial's values and timing with the recording."""
        rec = self.main.recorder
        if not rec.recording or not self._run:
            return
        import json
        trials = []
        for tr in self.trials:
            r = self.results.get(tr.n)
            trials.append({"n": tr.n, "rep": tr.rep, "a": tr.a, "b": tr.b,
                           "found": bool(r and r.found), "use": bool(r and r.use),
                           "t": (r.t0 - rec.block_start) if (r and r.found) else None})
        rec.set_attr("trials", json.dumps({"protocol": self._run["protocol"].to_dict(),
                                           "stimulus": self._run["stimulus"].to_dict(),
                                           "trials": trials}))

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export trials", "trials.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        p, shape = self.protocol, self._run["shape"]
        la, ua = T.setting_info(shape, p.a.key)
        row = self._channel_row()
        unit = "ms" if p.measure == "latency" else (self._run["units"][row] if row is not None else "")
        head = ["trial", "repeat", f"a: {la} ({ua})"]
        if p.b is not None:
            lb, ub = T.setting_info(shape, p.b.key)
            head.append(f"b: {lb} ({ub})")
        head += ["used", "found",
                 f"{T.MEASURES[p.measure]} of {self.m_channel.currentText()}, "
                 f"{p.measure_from_ms:g} to {p.measure_to_ms:g} ms ({unit})"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            for tr in self.trials:
                r = self.results.get(tr.n)
                v = self._value(r) if r else float("nan")
                line = [tr.n, tr.rep, f"{tr.a:g}"] + ([f"{tr.b:g}"] if p.b is not None else [])
                line += ["yes" if r and r.use else "no", "yes" if r and r.found else "no",
                         "" if not np.isfinite(v) else f"{v:.6g}"]
                w.writerow(line)
        self.show_message(f"Exported {len(self.trials)} trials to {path.name}")
        return path

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save protocol", "protocol.json", "Protocol (*.json)")
        if path:
            Path(path).write_text(self.protocol_from_widgets().dumps(), encoding="utf-8")
            self.show_message(f"Saved to {Path(path).name}")

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load protocol", "", "Protocol (*.json)")
        if not path:
            return
        try:
            p = T.Protocol.loads(Path(path).read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            self.show_message(str(e), error=True)
            return
        self._load_protocol_into_widgets(p)
