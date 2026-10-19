"""Chart-recorder window (pyqtgraph) with live and review modes.

Live mode: acquire, display, record blocks, add comments.
Review mode: open a saved session and scroll through its blocks.

File menu:
  New        start a blank session (asks first if the current one is unsaved)
  Open…      review a saved session
  Append…    keep recording new blocks into a saved session
  Save       first time: choose a name and location; afterwards: update the .mat
  Save As…   save a copy under a new name and continue with the copy
  Export…    write the session's blocks to a .mat file anywhere
  Recent Sessions

Setup menu:
  Input wiring   RSE (8 inputs vs AI GND) or DIFF (4 inputs, aiN minus ai(N+4))

The toolbar's Sampling rate box sets samples per second per channel. The
status bar always shows the wiring mode.

Recording always writes blocks to disk immediately. Until a session is saved,
its blocks live in <data_dir>/unsaved/<session>/, so nothing is lost if the
program crashes or someone chooses Don't Save.
"""
from __future__ import annotations

import dataclasses
import queue
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import config as cfgmod
from . import session
from .backends import Backend, make_backend
from .channels import ChannelRow, ChannelStrip
from .config import Channel, Config, valid_inputs
from .export import to_labchart_mat
from .recorder import Recorder

LIVE, REVIEW = "live", "review"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Config, settings: QtCore.QSettings | None = None,
                 config_path: str | Path | None = None):
        super().__init__()
        self.cfg = cfg   # shared with the backend and recorder; channel changes update it in place
        self.config_path = Path(config_path) if config_path else None
        self.settings = settings or QtCore.QSettings("neusci", "neusci-daq-visualizer")
        self.backend: Backend = make_backend(cfg)
        self.display_q: queue.Queue = self.backend.subscribe()
        self.recorder = Recorder(self.backend)
        self.markers: list[tuple[float, list[pg.InfiniteLine]]] = []
        self.mode = LIVE
        # Live session: each Record press is one block, like LabChart.
        self.session_dir: Path | None = None
        self.session_saved = False   # False while it lives in data/unsaved/
        self.blocks: list[Path] = []
        self._exported_n = 0
        # Review session.
        self.review_dir: Path | None = None
        self.review_blocks: list[Path] = []
        self.review_block: dict | None = None
        # The channel setup the user edits in the strips next to each plot.
        self.rows = [ChannelRow(c, i) for i, c in enumerate(cfg.channels + cfg.spare_channels)]

        self._build_ui()
        self._build_menu()
        self._build_live_plots()
        self._alloc_buffers()
        self.timer = QtCore.QTimer(self, interval=33, timeout=self._refresh)
        self._rate_busy = False
        self._set_mode(LIVE)
        self._sync_setup_controls()

    # ======================================================================
    # UI construction
    # ======================================================================
    def _build_ui(self):
        tb = self.addToolBar("Controls")
        tb.setMovable(False)
        self.btn_run = QtWidgets.QPushButton("Start", checkable=True)
        self.btn_run.toggled.connect(self._toggle_run)
        self.btn_rec = QtWidgets.QPushButton("● Record", checkable=True, enabled=False)
        self.btn_rec.toggled.connect(self._toggle_rec)
        # Editable: pick a preset or type any rate, then press Enter.
        self.rate_box = QtWidgets.QComboBox(editable=True)
        self.rate_box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.rate_box.addItems([f"{r:,} Hz" for r in cfgmod.RATE_PRESETS])
        self.rate_box.setMinimumContentsLength(9)
        self.rate_box.lineEdit().editingFinished.connect(self._rate_entered)
        self.rate_box.activated.connect(lambda *_: self._rate_entered())
        self.window_box = QtWidgets.QDoubleSpinBox(minimum=0.01, maximum=600, value=self.cfg.window_s,
                                                   suffix=" s window", decimals=2)
        self.window_box.valueChanged.connect(self._window_changed)
        self.comment_edit = QtWidgets.QLineEdit(placeholderText="Comment (Enter to add)")
        self.comment_edit.returnPressed.connect(self._add_comment)
        self.block_combo = QtWidgets.QComboBox()
        self.block_combo.setMinimumContentsLength(18)
        self.block_combo.currentIndexChanged.connect(self._show_review_block)
        self.btn_live = QtWidgets.QPushButton("Back to Live")
        self.btn_live.clicked.connect(self.back_to_live)

        self._live_actions = [tb.addWidget(self.btn_run), tb.addWidget(self.btn_rec),
                              tb.addWidget(QtWidgets.QLabel("  Sampling rate ")),
                              tb.addWidget(self.rate_box), tb.addWidget(QtWidgets.QLabel(" "))]
        self._review_actions = [tb.addWidget(QtWidgets.QLabel(" Block: ")), tb.addWidget(self.block_combo)]
        tb.addWidget(self.window_box)
        self._live_actions.append(tb.addWidget(self.comment_edit))
        self._review_actions.append(tb.addWidget(self.btn_live))

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.plot_area = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_area)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(2)
        self.plots, self.curves, self.strips = [], [], []

        self.channel_bar = QtWidgets.QWidget()
        cb = QtWidgets.QHBoxLayout(self.channel_bar)
        cb.setContentsMargins(4, 2, 4, 2)
        self.btn_add_ch = QtWidgets.QPushButton("+ Add channel")
        self.btn_add_ch.clicked.connect(lambda: QtCore.QTimer.singleShot(0, self.add_channel))
        self.btn_save_setup = QtWidgets.QPushButton("Save channel setup…")
        self.btn_save_setup.setToolTip("Save these channels to a config file to reuse later")
        self.btn_save_setup.clicked.connect(self.save_setup_dialog)
        cb.addWidget(self.btn_add_ch)
        cb.addWidget(self.btn_save_setup)
        cb.addStretch(1)

        self.scroll = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.scroll.valueChanged.connect(self._scrolled)
        lay.addWidget(self.plot_area, 1)
        lay.addWidget(self.channel_bar)
        lay.addWidget(self.scroll)
        self.setCentralWidget(central)

        self.status = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status)
        # Always visible, so a wrong wiring mode can't go unnoticed.
        self.wiring_label = QtWidgets.QLabel()
        self.wiring_label.setToolTip("Input wiring mode. Change it under Setup → Input wiring.")
        self.statusBar().addPermanentWidget(self.wiring_label)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        self.act_new = m.addAction("&New", self.new_session)
        self.act_new.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.act_open = m.addAction("&Open…", self.open_session_dialog)
        self.act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.act_append = m.addAction("&Append…", self.append_session_dialog)
        m.addSeparator()
        self.act_save = m.addAction("&Save", self.save)
        self.act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.act_save_as = m.addAction("Save &As…", self.save_as_dialog)
        self.act_save_as.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        self.act_export = m.addAction("&Export…", self.export_dialog)
        self.act_export.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        m.addSeparator()
        self.recent_menu = m.addMenu("&Recent Sessions")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        m.aboutToShow.connect(self._update_menu_state)

        s = self.menuBar().addMenu("&Setup")
        self.wiring_menu = s.addMenu("Input &wiring")
        group = QtGui.QActionGroup(self)
        self.act_wiring = {}
        for mode, text, tip in (
                ("RSE", "RSE — up to 8 inputs (aiN vs AI GND)",
                 "Each input is measured against AI GND. Every input you use must have a "
                 "connection to AI GND, or it picks up mains hum."),
                ("DIFF", "DIFF — up to 4 inputs (aiN minus ai(N+4))",
                 "Each input is measured against its partner: ai0−ai4, ai1−ai5, ai2−ai6, ai3−ai7. "
                 "Rejects hum common to both wires.")):
            a = self.wiring_menu.addAction(text, lambda mode=mode: self.apply_setup(terminal_config=mode))
            a.setCheckable(True)
            a.setToolTip(tip)
            group.addAction(a)
            self.act_wiring[mode] = a
        self.wiring_menu.setToolTipsVisible(True)

    # -- plots -------------------------------------------------------------
    def _clear_plot_area(self):
        self._clear_markers()
        while self.plot_layout.count():
            w = self.plot_layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.plots, self.curves, self.strips = [], [], []

    def _new_plot(self, name: str, unit: str, color: int, n_colors: int) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        p = pw.getPlotItem()
        p.setLabel("left", name, units=unit)
        p.getAxis("left").setWidth(70)   # same width on every plot, so time axes line up
        p.showGrid(x=True, y=True, alpha=0.2)
        # Peak downsampling keeps spikes visible when zoomed out.
        p.setDownsampling(auto=True, mode="peak")
        p.setClipToView(True)
        p.enableAutoRange(axis="y", enable=True)
        p.setAutoVisible(y=True)
        if self.plots:
            p.setXLink(self.plots[0])
        self.plots.append(p)
        self.curves.append(p.plot(pen=pg.intColor(color, hues=max(3, n_colors))))
        return pw

    def _finish_plots(self):
        if self.plots:
            self.plots[-1].setLabel("bottom", "time", units="s")
            self.plots[0].sigXRangeChanged.connect(self._xrange_changed)

    def _n_colors(self) -> int:
        return max(r.color for r in self.rows) + 1 if self.rows else 3

    def _build_live_plots(self):
        """One row per channel: its controls on the left, its plot on the right.

        A switched-off channel keeps its controls, so it can be switched back on,
        but has no plot.
        """
        self._clear_plot_area()
        inputs = valid_inputs(self.cfg)
        n_colors = self._n_colors()
        locked = self.recorder.recording
        for r in self.rows:
            row_w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row_w)
            h.setContentsMargins(0, 0, 0, 0)
            strip = ChannelStrip(r, inputs, n_colors, self.cfg.terminal_config == "DIFF",
                                 self._channel_edited, self._channel_removed,
                                 can_remove=len(self.rows) > 1)
            strip.setEnabled(not locked)
            self.strips.append(strip)
            h.addWidget(strip, 0)
            if r.ch.enabled:
                h.addWidget(self._new_plot(r.ch.name, r.ch.units, r.color, n_colors), 1)
                self.plot_layout.addWidget(row_w, 1)
            else:
                off = QtWidgets.QLabel("off")
                off.setStyleSheet("color: gray; padding-left: 8px")
                h.addWidget(off, 1)
                strip.setMaximumHeight(strip.sizeHint().height())
                self.plot_layout.addWidget(row_w, 0)
        self._finish_plots()
        self.btn_add_ch.setEnabled(not locked)
        self.rate_box.setEnabled(not locked)

    def _build_review_plots(self, names: list[str], units: list[str]):
        """Plots for a saved session: one per recorded channel, no controls."""
        self._clear_plot_area()
        for i, (name, unit) in enumerate(zip(names, units)):
            self.plot_layout.addWidget(self._new_plot(name, unit, i, len(names)), 1)
        self._finish_plots()

    # -- changing channels -------------------------------------------------
    def _copy_rows(self) -> list[ChannelRow]:
        return [ChannelRow(dataclasses.replace(r.ch), r.color) for r in self.rows]

    def _channel_edited(self, strip: ChannelStrip, what: str, value):
        rows = self._copy_rows()
        setattr(rows[self.rows.index(strip.row)].ch, what, value)
        self.apply_channels(rows)

    def _channel_removed(self, strip: ChannelStrip):
        i = self.rows.index(strip.row)
        self.apply_channels([r for j, r in enumerate(self._copy_rows()) if j != i])

    def add_channel(self) -> bool:
        used = {r.ch.physical for r in self.rows}
        free = [p for p in valid_inputs(self.cfg) if p not in used]
        if not free:
            self._channel_message("All inputs are already in use.")
            return False
        names = {r.ch.name for r in self.rows}
        n = len(self.rows) + 1
        while f"Ch{n}" in names:
            n += 1
        color = max([r.color for r in self.rows] + [-1]) + 1
        return self.apply_channels(self._copy_rows() + [ChannelRow(Channel(free[0], f"Ch{n}"), color)])

    @staticmethod
    def _layout(channels: list[Channel]) -> list[tuple]:
        return [(c.physical, c.name, c.units) for c in channels]

    def apply_channels(self, rows: list[ChannelRow]) -> bool:
        """Switch to a new channel setup (see apply_setup)."""
        return self.apply_setup(rows=rows)

    def apply_setup(self, rows: list[ChannelRow] | None = None, sample_rate: float | None = None,
                    terminal_config: str | None = None) -> bool:
        """Change channels, sampling rate and/or input wiring. Restarts acquisition if running.

        Refused while recording. If the session already has blocks recorded with
        a different setup, a new session is started, because every block in a
        session must match to export as one .mat file.
        """
        what = ("sampling rate" if sample_rate is not None else
                "input wiring" if terminal_config is not None else "channels")

        def refuse(msg):
            self._channel_message(msg)
            self._build_live_plots()   # put the controls back as they were
            self._sync_setup_controls()
            return False

        if self.recorder.recording:
            return refuse(f"Stop recording before changing the {what}.")
        if self.mode != LIVE:
            return refuse(f"Go back to live mode to change the {what}.")
        rows = self._copy_rows() if rows is None else rows
        changes = {}
        if sample_rate is not None:
            changes["sample_rate"] = float(sample_rate)
            changes["chunk_size"] = cfgmod.auto_chunk_size(sample_rate)
        if terminal_config is not None:
            changes["terminal_config"] = terminal_config
            if terminal_config == "DIFF":
                # Say which channel to move, rather than a generic "ai5 isn't available".
                stuck = [r.ch for r in rows if r.ch.physical not in ("ai0", "ai1", "ai2", "ai3")]
                if stuck:
                    names = ", ".join(f"{c.name} ({c.physical})" for c in stuck)
                    return refuse(f"Can't switch to DIFF: {names} must move to ai0–ai3 or be removed "
                                  "first, because DIFF uses ai4–ai7 as the minus inputs.")
        trial = self.cfg.with_channels([r.ch for r in rows], **changes)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfgmod.validate(trial)
        except ValueError as e:
            return refuse(str(e))

        running = self.btn_run.isChecked()
        settings_changed = (trial.sample_rate != self.cfg.sample_rate
                            or trial.terminal_config != self.cfg.terminal_config)
        hardware_changed = settings_changed or (
            [(c.physical, c.enabled) for c in trial.channels]
            != [(c.physical, c.enabled) for c in self.cfg.channels])
        if self.blocks and (settings_changed
                            or self._layout(trial.channels) != self._layout(self.cfg.channels)):
            if not self._confirm_new_session(what):
                return refuse(f"{what.capitalize()} unchanged.")
            if not self.new_session():          # asks Save / Don't Save if unsaved
                return refuse(f"{what.capitalize()} unchanged.")
        if running and hardware_changed:
            self.btn_run.setChecked(False)

        self.rows = rows
        self.cfg.channels, self.cfg.spare_channels = trial.channels, trial.spare_channels
        self.cfg.sample_rate, self.cfg.chunk_size = trial.sample_rate, trial.chunk_size
        self.cfg.terminal_config = trial.terminal_config
        if hardware_changed:
            self.buf_t = np.empty(0)
            self.buf_y = np.empty((len(self.cfg.channels), 0))
            self._alloc_buffers()
        self._build_live_plots()
        self._redraw_live()
        self._sync_setup_controls()
        if running and hardware_changed and not self.btn_run.isChecked():
            self.btn_run.setChecked(True)
        if what == "channels":
            on = ", ".join(f"{c.name} = {c.physical}" for c in self.cfg.channels)
            self.statusBar().showMessage(f"Channels: {on}", 8000)
        elif what == "sampling rate":
            self.statusBar().showMessage(f"Sampling rate: {self.cfg.sample_rate:,g} Hz per channel", 8000)
        else:
            self.statusBar().showMessage(f"Input wiring: {self.cfg.terminal_config}", 8000)
        return True

    def _confirm_new_session(self, what: str) -> bool:
        B = QtWidgets.QMessageBox.StandardButton
        r = QtWidgets.QMessageBox.question(
            self, f"{what.capitalize()} changed",
            f"This session's blocks were recorded with the old {what}. All blocks in a session "
            f"must match, so changing the {what} starts a new session.\n\nContinue?",
            B.Ok | B.Cancel, B.Ok)
        return r == B.Ok

    # -- sampling rate and wiring ------------------------------------------
    @staticmethod
    def _parse_rate(text: str) -> float | None:
        t = text.lower().replace(",", "").replace("hz", "").strip()
        mult = 1.0
        if t.endswith("k"):
            t, mult = t[:-1].strip(), 1000.0
        try:
            v = float(t) * mult
        except ValueError:
            return None
        return v if v > 0 else None

    def _rate_entered(self):
        # Selecting a preset fires both activated and editingFinished, and a
        # dialog taking focus fires editingFinished again: handle only one.
        if self._rate_busy:
            return
        self._rate_busy = True
        try:
            self._apply_entered_rate()
        finally:
            self._rate_busy = False

    def _apply_entered_rate(self):
        rate = self._parse_rate(self.rate_box.currentText())
        if rate is None:
            self._channel_message(f"{self.rate_box.currentText()!r} isn't a sampling rate. "
                                  "Type a number of Hz, e.g. 1000 or 20k.")
            self._sync_setup_controls()
            return
        if rate != self.cfg.sample_rate:
            self.apply_setup(sample_rate=rate)
        else:
            self._sync_setup_controls()   # tidy the text, e.g. "1000" -> "1,000 Hz"

    def _sync_setup_controls(self):
        """Show the current rate and wiring in the toolbar, menu and status bar."""
        self.rate_box.blockSignals(True)
        self.rate_box.setEditText(f"{self.cfg.sample_rate:,g} Hz")
        self.rate_box.blockSignals(False)
        n = len(self.cfg.channels)
        self.rate_box.setToolTip(f"Samples per second, per channel. With {n} channel(s) on, "
                                 f"the USB-6003 allows up to {cfgmod.max_rate(n):,} Hz.")
        self.act_wiring[self.cfg.terminal_config].setChecked(True)
        self._update_wiring_label()

    def _update_wiring_label(self):
        if self.mode == REVIEW:
            tc = (self.review_block or {}).get("terminal_config")
            self.wiring_label.setText(f"  recorded in {tc}  " if tc else "")
        else:
            self.wiring_label.setText(f"  {self.cfg.terminal_config} wiring  ")

    def _channel_message(self, text: str):
        self.statusBar().showMessage(f"⚠ {text}", 10000)

    def save_setup_dialog(self):
        if self.config_path:
            default = self.config_path.with_name(self.config_path.stem + "_setup.toml")
        else:
            default = Path(self._start_dir()) / "channel_setup.toml"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save channel setup", str(default),
                                                        "Config file (*.toml)")
        if path:
            cfgmod.save(self.cfg, path)
            self.statusBar().showMessage(f"Channel setup saved to {path}. "
                                         f"Start with: neusci-daq -c \"{path}\"", 15000)

    def _alloc_buffers(self):
        n = len(self.cfg.channels)
        cap = int(self.window_box.value() * self.cfg.sample_rate)
        old_t = getattr(self, "buf_t", np.empty(0))
        old_y = getattr(self, "buf_y", np.empty((n, 0)))
        self.cap = max(cap, 2)
        self.buf_t, self.buf_y = old_t[-self.cap:], old_y[:, -self.cap:]

    # ======================================================================
    # Modes
    # ======================================================================
    def _set_mode(self, mode: str):
        self.mode = mode
        for a in self._live_actions:
            a.setVisible(mode == LIVE)
        for a in self._review_actions:
            a.setVisible(mode == REVIEW)
        self.scroll.setVisible(mode == REVIEW)
        self.channel_bar.setVisible(mode == LIVE)
        self.wiring_menu.setEnabled(mode == LIVE and not self.recorder.recording)
        self._update_title()
        self._update_wiring_label()

    def _update_title(self):
        if self.mode == REVIEW and self.review_dir is not None:
            where = f"{self.review_dir.name} — Review"
        elif self.session_dir is not None and self.session_saved:
            where = f"{self.session_dir.name} — Live"
        elif self.blocks:
            where = "Untitled session* — Live"
        else:
            where = "Untitled session — Live"
        self.setWindowTitle(f"{where} — neusci-daq-visualizer ({self.cfg.device}, {self.cfg.backend})")

    def _update_menu_state(self):
        recording = self.recorder.recording
        for a in (self.act_new, self.act_open, self.act_append):
            a.setEnabled(not recording)
        self.act_save.setEnabled(self.mode == LIVE and bool(self.blocks) and not recording)
        self.act_save_as.setEnabled(bool(self._export_blocks()) and not recording)
        self.act_export.setEnabled(bool(self._export_blocks()) and not recording)

    def _unsaved_dir(self) -> Path:
        return Path(self.cfg.data_dir) / "unsaved"

    def _has_unsaved(self) -> bool:
        return self.session_dir is not None and not self.session_saved and bool(self.blocks)

    def _ask_unsaved(self) -> str:
        """Ask what to do with an unsaved session: 'save', 'discard' or 'cancel'."""
        B = QtWidgets.QMessageBox.StandardButton
        r = QtWidgets.QMessageBox.question(
            self, "Unsaved session",
            f"This session ({len(self.blocks)} block(s)) hasn't been saved.\n\n"
            f"Save it now? If you don't, its blocks stay in\n{self.session_dir}",
            B.Save | B.Discard | B.Cancel, B.Save)
        return {B.Save: "save", B.Discard: "discard"}.get(r, "cancel")

    def _resolve_unsaved(self) -> bool:
        """True if it's OK to leave the current session."""
        if not self._has_unsaved():
            return True
        choice = self._ask_unsaved()
        if choice == "save":
            return self.save()
        if choice == "discard":
            self.statusBar().showMessage(f"Unsaved blocks left in {self.session_dir}", 10000)
            return True
        return False

    def _leave_live(self) -> bool:
        """Stop acquisition before switching sessions. Refuses while recording."""
        if self.recorder.recording:
            QtWidgets.QMessageBox.information(self, "Recording in progress",
                                              "Stop recording before switching sessions.")
            return False
        if self.btn_run.isChecked():
            self.btn_run.setChecked(False)
        if not self._resolve_unsaved():
            return False
        if self.session_saved:
            self._autoexport_live()
        return True

    def back_to_live(self):
        """Leave review. The next Record starts a new untitled session."""
        self.review_dir, self.review_blocks, self.review_block = None, [], None
        self.session_dir, self.blocks, self._exported_n = None, [], 0
        self.session_saved = False
        self._build_live_plots()
        self._set_mode(LIVE)
        self.status.setText("")

    # ======================================================================
    # File menu: dialogs (thin wrappers around the methods below)
    # ======================================================================
    def _start_dir(self) -> str:
        Path(self.cfg.data_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.cfg.data_dir).resolve())

    def save_as_dialog(self) -> bool:
        if self.recorder.recording:
            QtWidgets.QMessageBox.information(self, "Recording in progress", "Stop recording before saving.")
            return False
        src = self._export_folder()
        if not self._export_blocks():
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Record a block first.")
            return False
        name = src.name if (self.session_saved or self.mode == REVIEW) else \
            datetime.now().strftime("session_%Y%m%d_%H%M%S")
        default = Path(self._start_dir()) / name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session as", str(default), "Session folder (*)",
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return False
        try:
            self.save_as(path)
            return True
        except session.SessionError as e:
            QtWidgets.QMessageBox.critical(self, "Could not save", str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not save", repr(e))
        return False

    def open_session_dialog(self):
        if not self._leave_live():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open session folder", self._start_dir())
        if path:
            self._try(lambda: self.open_session(path), "Could not open session")

    def append_session_dialog(self):
        if not self._leave_live():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Append to session folder", self._start_dir())
        if not path:
            return
        problems, warnings = session.append_check(path, self.cfg)
        if problems:
            QtWidgets.QMessageBox.critical(
                self, "Can't append to this session",
                "New blocks wouldn't match the existing ones:\n\n• " + "\n• ".join(problems)
                + "\n\nUse New instead, or load the config this session was recorded with.")
            return
        if warnings:
            ok = QtWidgets.QMessageBox.question(
                self, "Append anyway?", "\n".join(warnings) + "\n\nAppend with the new settings?")
            if ok != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._try(lambda: self.append_session(path, force=True), "Could not append")

    def export_dialog(self):
        blocks, folder = self._export_blocks(), self._export_folder()
        if not blocks:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Record a block or open a session first.")
            return
        default = folder / f"{folder.name}.mat"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export .mat", str(default),
                                                        "MATLAB file (*.mat)")
        if path:
            self.export(path)

    def _try(self, fn, title):
        try:
            fn()
        except session.SessionError as e:
            QtWidgets.QMessageBox.critical(self, title, str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, title, repr(e))

    # ======================================================================
    # File menu: actions (no dialogs, so they can be tested directly)
    # ======================================================================
    def new_session(self) -> bool:
        """Start a blank, untitled session. Asks first if the current one is unsaved."""
        if not self._leave_live():
            return False
        self.back_to_live()
        self.statusBar().showMessage("New session. Press Record to start; use Save to name it.", 10000)
        return True

    def save(self) -> bool:
        """First save asks for a name (Save As). Afterwards, just update the .mat."""
        if self.recorder.recording:
            QtWidgets.QMessageBox.information(self, "Recording in progress", "Stop recording before saving.")
            return False
        if self.mode != LIVE or not self.blocks:
            return False
        if not self.session_saved:
            return self.save_as_dialog()
        return self.export() is not None

    def save_as(self, path: str | Path) -> Path:
        """Save the current session (live or reviewed) to `path` and continue there.

        An unsaved session is moved out of data/unsaved/; a saved one is copied,
        so the original stays where it was.
        """
        path = Path(path).resolve()
        if self.recorder.recording:
            raise session.SessionError("Stop recording before saving.")
        src = self._export_folder()
        if src is None or not self._export_blocks():
            raise session.SessionError("Nothing recorded yet.")
        src = src.resolve()
        if path == src:
            if self.mode == LIVE:
                self.export()
            return path
        if path.exists() and any(path.iterdir()):
            raise session.SessionError(f"{path} already exists and isn't empty. Choose another name.")
        if path.exists():
            path.rmdir()
        path.parent.mkdir(parents=True, exist_ok=True)
        moving = self.mode == LIVE and not self.session_saved
        if moving:
            shutil.move(str(src), str(path))
        else:
            shutil.copytree(src, path)
        stale = path / f"{src.name}.mat"  # .mat named after the old folder
        if stale.exists():
            stale.unlink()

        if self.mode == REVIEW:
            self.open_session(path)
        else:
            self.session_dir = path
            self.blocks = session.list_blocks(path)
            self.session_saved = True
            self._exported_n = 0
            self.export()
            self._remember(path)
            self._update_title()
        self.statusBar().showMessage(f"Saved to {path}", 10000)
        return path

    def open_session(self, path: str | Path):
        """Review a saved session."""
        path = Path(path)
        blocks = session.list_blocks(path)
        if not blocks:
            raise session.SessionError(f"{path} has no block_NNN.h5 files, so it isn't a session.")
        if self.btn_run.isChecked():
            self.btn_run.setChecked(False)
        self.review_dir, self.review_blocks = path, blocks
        self._set_mode(REVIEW)
        self.block_combo.blockSignals(True)
        self.block_combo.clear()
        for b in blocks:
            try:
                info = session.block_info(b)
                self.block_combo.addItem(f"{b.stem}  ({info['duration_s']:.1f} s)")
            except Exception:
                self.block_combo.addItem(f"{b.stem}  (unreadable)")
        self.block_combo.blockSignals(False)
        self._remember(path)
        self.statusBar().showMessage(f"Opened {path.name} ({len(blocks)} block(s)) for review", 10000)
        self._show_review_block(0)

    def append_session(self, path: str | Path, force: bool = False) -> Path:
        """Continue recording new blocks into an existing session."""
        path = Path(path)
        problems, warnings = session.append_check(path, self.cfg)
        if problems or (warnings and not force):
            raise session.SessionError("\n".join(problems + warnings))
        self.back_to_live()
        self.session_dir = path
        self.session_saved = True
        self.blocks = session.list_blocks(path)
        self._exported_n = len(self.blocks)  # re-export only if new blocks are added
        self._remember(path)
        self._update_title()
        self.statusBar().showMessage(
            f"Appending to {path.name}: next block is {session.next_block_path(path).name}", 10000)
        return path

    def export(self, out_path: str | Path | None = None) -> Path | None:
        """Write the current session's blocks (live or review) to a .mat file."""
        if self.recorder.recording:
            self.btn_rec.setChecked(False)
        blocks, folder = self._export_blocks(), self._export_folder()
        if not blocks:
            return None
        out = Path(out_path) if out_path else folder / f"{folder.name}.mat"
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            to_labchart_mat(blocks, out)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", repr(e))
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if self.mode == LIVE:
            self._exported_n = len(self.blocks)
        self.statusBar().showMessage(f"Exported {len(blocks)} block(s) to {out}", 10000)
        return out

    def _export_blocks(self) -> list[Path]:
        return self.review_blocks if self.mode == REVIEW else self.blocks

    def _export_folder(self) -> Path | None:
        return self.review_dir if self.mode == REVIEW else self.session_dir

    def _autoexport_live(self):
        if len(self.blocks) > self._exported_n:   # don't lose the .mat if they forgot
            prev_mode, self.mode = self.mode, LIVE
            self.export()
            self.mode = prev_mode

    # -- recent sessions ---------------------------------------------------
    def recent_sessions(self) -> list[str]:
        v = self.settings.value("recent_sessions", [])
        if isinstance(v, str):  # QSettings returns a bare string for 1-item lists
            v = [v]
        return list(v or [])

    def _remember(self, path: Path):
        self.settings.setValue("recent_sessions", session.add_recent(self.recent_sessions(), path))

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        recent = self.recent_sessions()
        if not recent:
            a = self.recent_menu.addAction("(none yet)")
            a.setEnabled(False)
            return
        for i, p in enumerate(recent):
            self.recent_menu.addAction(f"&{i + 1}  {p}", lambda p=p: self.open_recent(p))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear List", lambda: self.settings.setValue("recent_sessions", []))

    def open_recent(self, path: str):
        if not session.is_session(path):
            self.settings.setValue("recent_sessions", [r for r in self.recent_sessions() if r != path])
            QtWidgets.QMessageBox.warning(self, "Session not found",
                                          f"{path} no longer exists or has no blocks. "
                                          "It has been removed from the list.")
            return
        if self._leave_live():
            self._try(lambda: self.open_session(path), "Could not open session")

    # ======================================================================
    # Review mode
    # ======================================================================
    def _show_review_block(self, index: int):
        if self.mode != REVIEW or not (0 <= index < len(self.review_blocks)):
            return
        try:
            blk = session.read_block(self.review_blocks[index])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not read block", repr(e))
            return
        self.review_block = blk
        names = [p.getAxis("left").labelText for p in self.plots]
        if self.strips or names != blk["channel_names"] or len(self.plots) != len(blk["channel_names"]):
            self._build_review_plots(blk["channel_names"], blk["units"])
        self._clear_markers()
        for curve, y in zip(self.curves, blk["data"]):
            curve.setData(blk["t"], y)
        for t, text in blk["comments"]:
            self._add_marker(t, text)
        self._update_scroll_range()
        self._update_wiring_label()
        self.plots[0].setXRange(0, min(self.window_box.value(), max(blk["duration_s"], 1e-3)), padding=0)
        self.status.setText(
            f"Reviewing {self.review_blocks[index].stem} ({index + 1}/{len(self.review_blocks)}) · "
            f"{blk['duration_s']:.2f} s · {blk['sample_rate']:,.0f} Hz/ch · "
            f"{len(blk['comments'])} comment(s)")

    def _update_scroll_range(self):
        if self.review_block is None:
            return
        win = self.window_box.value()
        span = max(0.0, self.review_block["duration_s"] - win)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, int(span * 1000))
        self.scroll.setPageStep(int(win * 1000))
        self.scroll.setSingleStep(max(1, int(win * 100)))
        self.scroll.blockSignals(False)

    def _scrolled(self, value_ms: int):
        if self.mode != REVIEW:
            return
        start = value_ms / 1000
        self.plots[0].setXRange(start, start + self.window_box.value(), padding=0)

    def _xrange_changed(self, _vb, rng):
        """Keep the scrollbar in step when the user pans or zooms with the mouse."""
        if self.mode != REVIEW:
            return
        self.scroll.blockSignals(True)
        self.scroll.setValue(int(max(0.0, rng[0]) * 1000))
        self.scroll.blockSignals(False)

    def _window_changed(self):
        if self.mode == REVIEW:
            self._update_scroll_range()
            self._scrolled(self.scroll.value())
        else:
            self._alloc_buffers()

    # ======================================================================
    # Live mode
    # ======================================================================
    def _toggle_run(self, on: bool):
        if on:
            # Drop chunks left over from the previous run; their timestamps
            # restart from 0, so drawing them would join old and new data.
            while True:
                try:
                    self.display_q.get_nowait()
                except queue.Empty:
                    break
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
            if self.session_dir is None:
                # Untitled until saved; recording still goes straight to disk.
                self.session_dir = self._unsaved_dir() / datetime.now().strftime("session_%Y%m%d_%H%M%S")
                self.session_dir.mkdir(parents=True, exist_ok=True)
                self.session_saved = False
            path = session.next_block_path(self.session_dir)
            path = self.recorder.start(path)
            self.blocks.append(path)
            self._update_title()
            self.statusBar().showMessage(f"Recording {path.name} in {self.session_dir}")
            self.btn_rec.setText("■ Stop recording")
            self._lock_channels(True)
        else:
            self.recorder.stop()
            self._lock_channels(False)
            self.statusBar().showMessage(f"Saved {self.recorder.path.name}", 10000)
            self.btn_rec.setText("● Record")

    def _lock_channels(self, locked: bool):
        for st in self.strips:
            st.setEnabled(not locked)
        self.btn_add_ch.setEnabled(not locked)
        self.rate_box.setEnabled(not locked)
        self.wiring_menu.setEnabled(not locked and self.mode == LIVE)

    def _redraw_live(self):
        if self.buf_t.size and self.buf_y.shape[0] == len(self.curves):
            for ch, curve, y in zip(self.cfg.channels, self.curves, self.buf_y):
                curve.setData(self.buf_t, ch.to_units(y))

    def _add_comment(self):
        text = self.comment_edit.text().strip()
        if not text or not self.backend.running:
            return
        t = self.recorder.add_comment(text)
        self._add_marker(t, text)
        self.comment_edit.clear()

    def _add_marker(self, t: float, text: str):
        lines = []
        for p in self.plots:
            ln = pg.InfiniteLine(t, angle=90, pen=pg.mkPen("y", style=QtCore.Qt.PenStyle.DashLine),
                                 label=text if p is self.plots[0] else None,
                                 labelOpts={"position": 0.9, "color": "y"})
            p.addItem(ln)
            lines.append(ln)
        self.markers.append((t, lines))

    def _clear_markers(self):
        for _, lines in self.markers:
            for p, ln in zip(getattr(self, "plots", []), lines):
                p.removeItem(ln)
        self.markers = []

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
        if not ts or self.mode != LIVE:
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
        if self.btn_rec.isChecked():
            self.btn_rec.setChecked(False)
        if not self._resolve_unsaved():
            ev.ignore()
            return
        self.recorder.stop()
        self.backend.stop()
        if self.session_saved:
            self._autoexport_live()
        self.recorder.close()   # end the writer thread before Qt and h5py shut down
        super().closeEvent(ev)


ICON_DIR = Path(__file__).with_name("icons")


def app_icon() -> QtGui.QIcon:
    """The app's icon, in several sizes so it stays sharp from taskbar to title bar."""
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        f = ICON_DIR / f"icon_{size}.png"
        if f.exists():
            icon.addFile(str(f), QtCore.QSize(size, size))
    return icon


def run(cfg: Config, config_path: str | Path | None = None):
    app = pg.mkQApp("neusci-daq-visualizer")
    # pyqtgraph sets its own icon and Windows taskbar ID in mkQApp; replace both
    # before the window opens, or the taskbar keeps showing pyqtgraph's icon.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("neusci.daq.visualizer")
    app.setWindowIcon(app_icon())
    w = MainWindow(cfg, config_path=config_path)
    w.resize(1300, 200 + 180 * len(cfg.channels))
    w.show()
    return app.exec()
