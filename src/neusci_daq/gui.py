"""Scrolling chart-recorder window (pyqtgraph)."""
from __future__ import annotations

import queue

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .backends import Backend, make_backend
from .config import Config
from .recorder import Recorder


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backend: Backend = make_backend(cfg)
        self.display_q: queue.Queue = self.backend.subscribe()
        self.recorder = Recorder(self.backend)
        self.markers: list[tuple[float, list[pg.InfiniteLine]]] = []
        self._build_ui()
        self._alloc_buffers()
        self.timer = QtCore.QTimer(self, interval=33, timeout=self._refresh)

    # -- UI ---------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle(f"neusci-daq-visualizer — {self.cfg.device} ({self.cfg.backend})")
        tb = self.addToolBar("Controls")
        self.btn_run = QtWidgets.QPushButton("Start", checkable=True)
        self.btn_run.toggled.connect(self._toggle_run)
        self.btn_rec = QtWidgets.QPushButton("● Record", checkable=True, enabled=False)
        self.btn_rec.toggled.connect(self._toggle_rec)
        self.window_box = QtWidgets.QDoubleSpinBox(minimum=0.1, maximum=600, value=self.cfg.window_s,
                                                   suffix=" s window", decimals=1)
        self.window_box.valueChanged.connect(self._alloc_buffers)
        self.comment_edit = QtWidgets.QLineEdit(placeholderText="Comment (Enter to add)")
        self.comment_edit.returnPressed.connect(self._add_comment)
        for w in (self.btn_run, self.btn_rec, self.window_box, self.comment_edit):
            tb.addWidget(w)

        self.glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.glw)
        self.plots, self.curves = [], []
        for i, ch in enumerate(self.cfg.channels):
            p = self.glw.addPlot(row=i, col=0)
            p.setLabel("left", ch.name, units=ch.units)
            p.showGrid(x=True, y=True, alpha=0.2)
            # Peak downsampling keeps spikes visible when zoomed out.
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if self.plots:
                p.setXLink(self.plots[0])
            self.plots.append(p)
            self.curves.append(p.plot(pen=pg.intColor(i, hues=max(3, len(self.cfg.channels)))))
        self.plots[-1].setLabel("bottom", "time", units="s")
        self.status = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status)

    def _alloc_buffers(self):
        n = len(self.cfg.channels)
        cap = int(self.window_box.value() * self.cfg.sample_rate)
        old_t = getattr(self, "buf_t", np.empty(0))
        old_y = getattr(self, "buf_y", np.empty((n, 0)))
        self.cap = max(cap, 2)
        self.buf_t, self.buf_y = old_t[-self.cap:], old_y[:, -self.cap:]

    # -- actions ----------------------------------------------------------
    def _toggle_run(self, on: bool):
        if on:
            try:
                self.backend.start()
            except Exception as e:
                self.btn_run.setChecked(False)
                QtWidgets.QMessageBox.critical(self, "Could not start acquisition", repr(e))
                return
            self.buf_t = np.empty(0)
            self.buf_y = np.empty((len(self.cfg.channels), 0))
            self.btn_run.setText("Stop")
            self.btn_rec.setEnabled(True)
            self.timer.start()
        else:
            if self.btn_rec.isChecked():
                self.btn_rec.setChecked(False)
            self.backend.stop()
            self.timer.stop()
            self.btn_run.setText("Start")
            self.btn_rec.setEnabled(False)

    def _toggle_rec(self, on: bool):
        if on:
            path = self.recorder.start()
            self.statusBar().showMessage(f"Recording to {path}")
            self.btn_rec.setText("■ Stop recording")
        else:
            self.recorder.stop()
            self.statusBar().showMessage(f"Saved {self.recorder.path}", 10000)
            self.btn_rec.setText("● Record")

    def _add_comment(self):
        text = self.comment_edit.text().strip()
        if not text or not self.backend.running:
            return
        t = self.recorder.add_comment(text)
        lines = []
        for p in self.plots:
            ln = pg.InfiniteLine(t, angle=90, pen=pg.mkPen("y", style=QtCore.Qt.PenStyle.DashLine),
                                 label=text if p is self.plots[0] else None,
                                 labelOpts={"position": 0.9, "color": "y"})
            p.addItem(ln)
            lines.append(ln)
        self.markers.append((t, lines))
        self.comment_edit.clear()

    # -- refresh loop -----------------------------------------------------
    def _refresh(self):
        if self.backend.error is not None:
            err, self.backend.error = self.backend.error, None
            self.btn_run.setChecked(False)
            QtWidgets.QMessageBox.critical(self, "Acquisition stopped", repr(err))
            return
        ts, ys = [], []
        while True:
            try:
                c = self.display_q.get_nowait()
            except queue.Empty:
                break
            ts.append(c.t)
            ys.append(c.data)
        if not ts:
            return
        self.buf_t = np.concatenate([self.buf_t, *ts])[-self.cap:]
        self.buf_y = np.concatenate([self.buf_y, *ys], axis=1)[:, -self.cap:]
        t_end = self.buf_t[-1]
        for ch, curve, y in zip(self.cfg.channels, self.curves, self.buf_y):
            curve.setData(self.buf_t, ch.to_units(y))
        self.plots[0].setXRange(t_end - self.window_box.value(), t_end, padding=0)
        # drop markers that scrolled off
        t_min = self.buf_t[0]
        while self.markers and self.markers[0][0] < t_min:
            _, lines = self.markers.pop(0)
            for p, ln in zip(self.plots, lines):
                p.removeItem(ln)
        rate = self.backend.actual_rate
        lag = self.display_q.qsize()
        msg = f"rate {rate:,.1f} Hz/ch" + ("" if self.backend.hardware_timed else " (software-timed!)")
        if self.recorder.recording and self.backend.hardware_timed:
            msg += f" | recorded {self.recorder.samples_written / max(rate, 1e-9):,.1f} s"
        if lag > 5:
            msg += f" | display lagging ({lag} chunks queued)"
        self.status.setText(msg)

    def closeEvent(self, ev):
        self.recorder.stop()
        self.backend.stop()
        super().closeEvent(ev)


def run(cfg: Config):
    app = pg.mkQApp("neusci-daq-visualizer")
    w = MainWindow(cfg)
    w.resize(1200, 200 + 180 * len(cfg.channels))
    w.show()
    return app.exec()
