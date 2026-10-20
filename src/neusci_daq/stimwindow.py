"""The Stimulus window (Stimulator > Stimulus…): design a stimulus, see it, fire it.

Every change applies straight away to the stimulus that Fire and Repeat send,
so there is no Apply button. The preview shows exactly the samples that will
be written to the output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import stim as stimmod
from .stim import Stimulus


def _spin(lo, hi, decimals=2, suffix="", step=1.0):
    s = QtWidgets.QDoubleSpinBox(minimum=lo, maximum=hi, decimals=decimals, singleStep=step)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)       # update when you finish typing, not on every digit
    return s


class StimWindow(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__(main, QtCore.Qt.WindowType.Window)
        self.main = main
        self.setWindowTitle("Stimulus")
        self.resize(760, 470)
        self._loading = False
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(16)

        # ---- left: shape and its settings -------------------------------
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        top = QtWidgets.QFormLayout()
        self.shape = QtWidgets.QComboBox()
        for key, label in stimmod.SHAPES.items():
            self.shape.addItem(label, key)
        self.delay = _spin(0, 60000, suffix=" ms")
        self.delay.setToolTip("0 V for this long before the stimulus starts")
        top.addRow("Shape", self.shape)
        top.addRow("Delay", self.delay)
        left.addLayout(top)

        self.pages = QtWidgets.QStackedWidget()
        # pulse
        pw = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(pw); f.setContentsMargins(0, 0, 0, 0)
        self.p_amp = _spin(-10, 10, suffix=" V", step=0.1)
        self.p_width = _spin(0, 60000, suffix=" ms", step=0.2)
        self.p_count = QtWidgets.QSpinBox(minimum=1, maximum=10000)
        self.p_interval = _spin(0, 60000, suffix=" ms", step=10)
        self.p_interval.setToolTip("From the start of one pulse to the start of the next")
        self.p_bi = QtWidgets.QCheckBox("Biphasic (+ then −)")
        f.addRow("Amplitude", self.p_amp); f.addRow("Width", self.p_width)
        f.addRow("Pulses", self.p_count); f.addRow("Interval", self.p_interval); f.addRow("", self.p_bi)
        self.pages.addWidget(pw)
        # step
        sw = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(sw); f.setContentsMargins(0, 0, 0, 0)
        self.s_amp = _spin(-10, 10, suffix=" V", step=0.1)
        self.s_dur = _spin(0, 60000, suffix=" ms", step=10)
        f.addRow("Level", self.s_amp); f.addRow("Duration", self.s_dur)
        self.pages.addWidget(sw)
        # ramp
        rw = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(rw); f.setContentsMargins(0, 0, 0, 0)
        self.r_start = _spin(-10, 10, suffix=" V", step=0.1)
        self.r_end = _spin(-10, 10, suffix=" V", step=0.1)
        self.r_dur = _spin(0, 60000, suffix=" ms", step=10)
        f.addRow("From", self.r_start); f.addRow("To", self.r_end); f.addRow("Duration", self.r_dur)
        self.pages.addWidget(rw)
        # sine
        nw = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(nw); f.setContentsMargins(0, 0, 0, 0)
        self.n_amp = _spin(0, 10, suffix=" V", step=0.1)
        self.n_off = _spin(-10, 10, suffix=" V", step=0.1)
        self.n_freq = _spin(0.01, 2500, suffix=" Hz", step=1)
        self.n_dur = _spin(0, 60000, suffix=" ms", step=10)
        f.addRow("Amplitude", self.n_amp); f.addRow("Offset", self.n_off)
        f.addRow("Frequency", self.n_freq); f.addRow("Duration", self.n_dur)
        self.pages.addWidget(nw)
        # custom
        cw = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(cw); v.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Duration (ms)", "Start (V)", "End (V)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip("Each row is a straight line from Start to End. Same Start and End = a flat level.")
        rowbtns = QtWidgets.QHBoxLayout()
        self.btn_add_seg = QtWidgets.QPushButton("Add segment")
        self.btn_del_seg = QtWidgets.QPushButton("Remove segment")
        rowbtns.addWidget(self.btn_add_seg); rowbtns.addWidget(self.btn_del_seg); rowbtns.addStretch(1)
        v.addWidget(self.table, 1); v.addLayout(rowbtns)
        self.pages.addWidget(cw)
        # file
        fw = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(fw); f.setContentsMargins(0, 0, 0, 0)
        frow = QtWidgets.QHBoxLayout()
        self.f_path = QtWidgets.QLineEdit(placeholderText="CSV or text file")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        frow.addWidget(self.f_path, 1); frow.addWidget(self.btn_browse)
        self.f_rate = _spin(0, 1e6, decimals=0, suffix=" Hz", step=100)
        self.f_rate.setSpecialValueText("same as output")
        self.f_rate.setToolTip("Sample rate of a one-column file. Two-column files (time in s, volts) don't need it.")
        f.addRow("File", frow); f.addRow("File's rate", self.f_rate)
        hint = QtWidgets.QLabel("One column: volts per sample. Two columns: time (s), volts.")
        hint.setWordWrap(True)
        f.addRow("", hint)
        self.pages.addWidget(fw)
        left.addWidget(self.pages, 1)

        bottom = QtWidgets.QFormLayout()
        self.repeat = _spin(0.05, 3600, suffix=" s", step=0.5)
        self.repeat.setToolTip("Stimulator > Repeat fires the stimulus this often")
        bottom.addRow("Repeat every", self.repeat)
        left.addLayout(bottom)
        btns = QtWidgets.QHBoxLayout()
        self.btn_fire = QtWidgets.QPushButton("Fire")
        self.btn_fire.setObjectName("start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_save = QtWidgets.QPushButton("Save…")
        self.btn_load = QtWidgets.QPushButton("Load…")
        for b in (self.btn_fire, self.btn_stop, self.btn_save, self.btn_load):
            btns.addWidget(b)
        left.addLayout(btns)
        outer.addLayout(left, 0)

        # ---- right: preview ----------------------------------------------
        right = QtWidgets.QVBoxLayout()
        self.preview = pg.PlotWidget(background="#ffffff")
        p = self.preview.getPlotItem()
        p.setLabel("left", "Output", units="V")
        p.setLabel("bottom", "time", units="s")
        p.showGrid(x=True, y=True, alpha=0.25)
        for ax in ("left", "bottom"):
            p.getAxis(ax).setPen("#303030"); p.getAxis(ax).setTextPen("#303030")
        self.curve = p.plot(pen=pg.mkPen("#1a5fd0", width=2), stepMode=None)
        right.addWidget(self.preview, 1)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        right.addWidget(self.info)
        right.addWidget(self.message)
        outer.addLayout(right, 1)

        # ---- signals -------------------------------------------------------
        self.set_stimulus(main.stimulus)
        self.repeat.setValue(main.stim_repeat_s)
        self.shape.currentIndexChanged.connect(self._shape_changed)
        for w in (self.delay, self.p_amp, self.p_width, self.p_interval, self.s_amp, self.s_dur,
                  self.r_start, self.r_end, self.r_dur, self.n_amp, self.n_off, self.n_freq, self.n_dur,
                  self.f_rate):
            w.valueChanged.connect(self._changed)
        self.p_count.valueChanged.connect(self._changed)
        self.p_bi.toggled.connect(self._changed)
        self.f_path.editingFinished.connect(self._changed)
        self.table.itemChanged.connect(self._changed)
        self.btn_add_seg.clicked.connect(self._add_segment)
        self.btn_del_seg.clicked.connect(self._remove_segment)
        self.btn_browse.clicked.connect(self._browse)
        self.repeat.valueChanged.connect(lambda v: setattr(self.main, "stim_repeat_s", float(v)))
        self.btn_fire.clicked.connect(self.main.fire_stimulus)
        self.btn_stop.clicked.connect(self.main.stop_stimulus)
        self.btn_save.clicked.connect(self._save)
        self.btn_load.clicked.connect(self._load)
        self._update_preview()

    # -- widgets <-> Stimulus ------------------------------------------------
    def set_stimulus(self, s: Stimulus) -> None:
        self._loading = True
        self.shape.setCurrentIndex(max(0, self.shape.findData(s.shape)))
        self.pages.setCurrentIndex(self.shape.currentIndex())
        self.delay.setValue(s.delay_ms)
        self.p_amp.setValue(s.amplitude_v); self.p_width.setValue(s.width_ms)
        self.p_count.setValue(s.count); self.p_interval.setValue(s.interval_ms); self.p_bi.setChecked(s.biphasic)
        self.s_amp.setValue(s.amplitude_v); self.s_dur.setValue(s.duration_ms)
        self.r_start.setValue(s.start_v); self.r_end.setValue(s.end_v); self.r_dur.setValue(s.duration_ms)
        self.n_amp.setValue(abs(s.amplitude_v)); self.n_off.setValue(s.offset_v)
        self.n_freq.setValue(s.freq_hz); self.n_dur.setValue(s.duration_ms)
        self.table.setRowCount(0)
        for seg in s.segments:
            self._append_row(seg["duration_ms"], seg["start_v"], seg["end_v"])
        self.f_path.setText(s.file); self.f_rate.setValue(s.file_rate)
        self._loading = False
        self._changed()          # the stimulus Fire sends and the preview both follow the fields

    def stimulus(self) -> Stimulus:
        """The stimulus the fields describe. Raises ValueError for a bad table cell."""
        shape = self.shape.currentData()
        s = Stimulus(shape=shape, delay_ms=self.delay.value())
        if shape == "pulse":
            s.amplitude_v, s.width_ms = self.p_amp.value(), self.p_width.value()
            s.count, s.interval_ms, s.biphasic = self.p_count.value(), self.p_interval.value(), self.p_bi.isChecked()
        elif shape == "step":
            s.amplitude_v, s.duration_ms = self.s_amp.value(), self.s_dur.value()
        elif shape == "ramp":
            s.start_v, s.end_v, s.duration_ms = self.r_start.value(), self.r_end.value(), self.r_dur.value()
        elif shape == "sine":
            s.amplitude_v, s.offset_v = self.n_amp.value(), self.n_off.value()
            s.freq_hz, s.duration_ms = self.n_freq.value(), self.n_dur.value()
        elif shape == "file":
            s.file, s.file_rate = self.f_path.text().strip(), self.f_rate.value()
        s.segments = self._segments() if shape == "custom" else self.main.stimulus.segments
        return s

    def _segments(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            vals = []
            for c, name in enumerate(("duration", "start", "end")):
                item = self.table.item(r, c)
                try:
                    vals.append(float((item.text() if item else "").strip()))
                except ValueError:
                    raise ValueError(f"Segment {r + 1}: {name} must be a number.") from None
            out.append({"duration_ms": vals[0], "start_v": vals[1], "end_v": vals[2]})
        return out

    # -- reacting to edits ---------------------------------------------------
    def _shape_changed(self, i):
        self.pages.setCurrentIndex(i)
        self._changed()

    def _changed(self, *_):
        if self._loading:
            return
        try:
            s = self.stimulus()
        except ValueError as e:
            self._show_problem(str(e))
            return
        self.main.set_stimulus(s)
        self._update_preview()

    def _update_preview(self):
        cfg, s = self.main.cfg, self.main.stimulus
        self.info.setText(f"Output {cfg.stim_output} · {cfg.stim_rate:,.0f} samples/s · "
                          f"limit ±{cfg.stim_max_v:g} V (set in the config)")
        try:
            wave = s.waveform(cfg.stim_rate)
            stimmod.check_limit(wave, cfg.stim_max_v)
        except ValueError as e:
            self.curve.setData([], [])
            self._show_problem(str(e))
            return
        t = np.arange(len(wave)) / cfg.stim_rate
        # draw each sample as a flat step, which is what the output really does
        self.curve.setData(np.repeat(t, 2)[1:], np.repeat(wave, 2)[:-1])
        peak = float(np.max(np.abs(wave)))
        notes = s.rounding_notes(cfg.stim_rate)
        self.show_message(f"{s.describe()}\nPeak {peak:g} V · {len(wave) / cfg.stim_rate * 1000:g} ms · "
                          f"{len(wave):,} samples" + ("".join("\n" + n for n in notes)))

    def _show_problem(self, text: str):
        self.message.setStyleSheet("color: #b42318")
        self.message.setText(text)

    def show_message(self, text: str, error: bool = False):
        self.message.setStyleSheet("color: #b42318" if error else "color: #4a5058")
        self.message.setText(text)

    # -- custom segments -----------------------------------------------------
    def _append_row(self, dur, start, end):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate((dur, start, end)):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{float(v):g}"))

    def _add_segment(self):
        last = self.table.rowCount() - 1
        end = self.table.item(last, 2).text() if last >= 0 and self.table.item(last, 2) else "0"
        self.table.blockSignals(True)
        self._append_row(10, end, end)
        self.table.blockSignals(False)
        self._changed()

    def _remove_segment(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()} or {self.table.rowCount() - 1})
        self.table.blockSignals(True)
        for r in reversed(rows):
            if r >= 0:
                self.table.removeRow(r)
        self.table.blockSignals(False)
        self._changed()

    # -- files -----------------------------------------------------------------
    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Waveform file", "",
                                                        "Waveform (*.csv *.txt *.tsv);;All files (*)")
        if path:
            self.f_path.setText(path)
            self._changed()

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save stimulus", "stimulus.json",
                                                        "Stimulus (*.json)")
        if path:
            self.main.stimulus.save(path)
            self.show_message(f"Saved to {Path(path).name}")

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load stimulus", "", "Stimulus (*.json)")
        if not path:
            return
        try:
            s = Stimulus.load(path)
        except ValueError as e:
            self.show_message(str(e), error=True)
            return
        self.set_stimulus(s)
