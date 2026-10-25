"""Chart-recorder window (pyqtgraph) with live and review modes.

Live mode: acquire, display, record blocks, add comments.
Review mode: open a saved session and scroll through its blocks.

File menu:
  New        start a blank session (asks first if the current one is unsaved)
  Open…      review a saved session
  Open .mat File…  review a LabChart export (or one of ours); the file isn't changed
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
from . import matfile
from . import session
from . import stim as stimmod
from . import style, theme
from .backends import Backend, make_backend
from .channels import ChannelRow, ChannelRowWidget, ChannelStrip, ColorButton
from .comments import CommentPopup, CommentsPanel
from .config import Channel, Config, valid_inputs
from .export import to_labchart_mat
from .recorder import Recorder
from .stim import Stimulus
from .stimout import make_output
from .stimwindow import StimPage
from .trialspage import TrialsPage
from .tabs import ONE, SIDE, STACKED, TabArea, layout_icon

LIVE, REVIEW = "live", "review"
_SI_BASE = {"V", "A", "s", "Hz", "Ω", "ohm", "N", "Pa", "W", "m", "g", "T"}


class WindowBox(QtWidgets.QDoubleSpinBox):
    """Window length in seconds, shown as "10 s" rather than "10.00 s"."""

    def textFromValue(self, v: float) -> str:
        return f"{v:g}"

    def sizeHint(self) -> QtCore.QSize:
        # Qt sizes the box for "600.00 s" (the maximum, with every decimal); size it for
        # what's actually typed, like "600 s", so the toolbar fits on a laptop.
        h = super().sizeHint()
        return QtCore.QSize(min(h.width(), 76), h.height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()


class _CommentEdit(QtWidgets.QLineEdit):
    """Asks the toolbar for little width, so nothing gets pushed into its overflow
    menu on a narrow screen; it still grows into any free space."""

    def sizeHint(self) -> QtCore.QSize:
        h = super().sizeHint()
        return QtCore.QSize(90, h.height())


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
        self._review_source: Path | None = None   # the .mat an imported session came from
        self.review_blocks: list[Path] = []
        self.review_block: dict | None = None
        self.review_color_btns: list[ColorButton] = []
        # "Recolour recorded blocks?" answer remembered for this session: None (ask), True, False.
        self._recolor_policy: bool | None = None
        self._live_comments_dirty = False    # a saved session's .mat needs re-exporting
        self._review_mat_stale = False
        self._shown_comments: list[dict] = []
        # Live time axis: stream time shown as 0. It's the start of the block
        # being (or last) recorded, so live times match the saved comment times.
        self._axis_zero = 0.0
        self._block_on_screen = False      # is the last block's data in this run?
        self._popup = None
        # The channel setup the user edits in the strips next to each plot.
        cfgmod.fill_colors(cfg.channels + cfg.spare_channels)
        self.rows = [ChannelRow(c) for c in cfg.channels + cfg.spare_channels]

        # Stimulator
        self.stim_out = make_output(cfg, self.backend)
        self.stimulus = self._saved_stimulus()
        self.stim_repeat_s = float(self.settings.value("stim_repeat_s", 1.0))
        self.repeating = False
        self.stim_timer = QtCore.QTimer(self, timeout=self._repeat_fire)
        self._compact_strips = False     # True while Acquisition shares the screen (split view)

        mode = self.settings.value("plot_theme", theme.DARK)
        self.plot_mode = mode if mode in theme.THEMES else theme.DARK
        self._build_ui()
        self._build_menu()
        self._build_live_plots()
        self._apply_app_theme()
        self._alloc_buffers()
        self.timer = QtCore.QTimer(self, interval=33, timeout=self._refresh)
        self._rate_busy = False
        self._set_mode(LIVE)
        self._sync_setup_controls()
        open_ = self.settings.value("comments_open", True, type=bool)
        self.btn_comments.setChecked(open_)
        self.comments_panel.setVisible(open_)
        self.refresh_comments()

    # ======================================================================
    # UI construction
    # ======================================================================
    def _build_ui(self):
        tb = self.addToolBar("Controls")
        tb.setMovable(False)
        self.btn_run = QtWidgets.QPushButton("Start", checkable=True)
        self.btn_run.setObjectName("start")
        self.btn_run.toggled.connect(self._toggle_run)
        self.btn_rec = QtWidgets.QPushButton("Record", checkable=True, enabled=False)
        self.btn_rec.setObjectName("record")
        self.btn_rec.setIcon(style.dot_icon(style.RECORDING))
        self.btn_rec.toggled.connect(self._toggle_rec)
        # Editable: pick a preset or type any rate, then press Enter.
        self.rate_box = QtWidgets.QComboBox(editable=True)
        self.rate_box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.rate_box.addItems([f"{r:,} Hz" for r in cfgmod.RATE_PRESETS])
        self.rate_box.setMinimumContentsLength(8)
        self.rate_box.lineEdit().editingFinished.connect(self._rate_entered)
        self.rate_box.activated.connect(lambda *_: self._rate_entered())
        self.window_box = WindowBox(minimum=0.01, maximum=600, value=self.cfg.window_s,
                                    suffix=" s", decimals=2)
        self.window_box.setToolTip("How many seconds the plots show")
        self.window_box.setMinimumWidth(76)
        self.window_box.valueChanged.connect(self._window_changed)
        # Only usable while recording, because that's when a comment can be saved.
        self.comment_edit = _CommentEdit()
        self.comment_edit.setAccessibleName("Comment")
        self.comment_edit.setToolTip("Shortcut: press C to jump here")
        self.comment_edit.setMinimumWidth(90)
        self.comment_edit.returnPressed.connect(self._add_comment)
        self._set_comment_box(False)
        self.block_combo = QtWidgets.QComboBox()
        self.block_combo.setMinimumContentsLength(18)
        self.block_combo.currentIndexChanged.connect(self._show_review_block)
        self.btn_live = QtWidgets.QPushButton("Back to Live")
        self.btn_live.clicked.connect(self.back_to_live)
        self.btn_comments = QtWidgets.QPushButton("Comments (0)", checkable=True)
        self.btn_comments.setToolTip("Show or hide this block's comments")
        self.btn_comments.toggled.connect(self._toggle_comments)

        # Toolbar groups: shown per mode (live/review) and per visible tab.
        A = lambda w: tb.addWidget(w)
        self._tb_live = [A(self.btn_run), A(self.btn_rec), tb.addSeparator(),
                         A(QtWidgets.QLabel("Sampling rate")), A(self.rate_box)]
        self._tb_review = [A(QtWidgets.QLabel("Block")), A(self.block_combo)]
        self._tb_acq = [A(QtWidgets.QLabel("Window")), A(self.window_box)]
        self._tb_acq_live = [tb.addSeparator(), A(self.comment_edit)]
        self._tb_review.append(A(self.btn_live))
        spacer = QtWidgets.QWidget()            # fills the space when the comment box is hidden
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self._tb_spacer = A(spacer)
        # Shown on every tab while a stimulus is going, so it can always be stopped.
        self.stim_chip = QtWidgets.QWidget()
        self.stim_chip.setObjectName("stimChip")
        self.stim_chip.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        cl = QtWidgets.QHBoxLayout(self.stim_chip)
        cl.setContentsMargins(10, 2, 4, 2)
        cl.setSpacing(6)
        dot = QtWidgets.QLabel()
        dot.setPixmap(style.dot_icon(style.START).pixmap(10, 10))
        self.stim_chip_text = QtWidgets.QLabel("Stimulating")
        self.stim_chip_text.setObjectName("stimChipText")
        self.btn_chip_stop = QtWidgets.QPushButton("Stop")
        self.btn_chip_stop.setToolTip("Stop the stimulus and set the output to 0 V (Esc)")
        self.btn_chip_stop.clicked.connect(self.stop_stimulus)
        cl.addWidget(dot); cl.addWidget(self.stim_chip_text); cl.addWidget(self.btn_chip_stop)
        self._tb_chip = A(self.stim_chip)
        # Split view: one tab, side by side, or top and bottom.
        self.split_group = QtWidgets.QWidget()
        self.split_group.setObjectName("splitGroup")
        self.split_group.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        sl = QtWidgets.QHBoxLayout(self.split_group)
        sl.setContentsMargins(2, 2, 2, 2)
        sl.setSpacing(2)
        self.split_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for mode, tip in ((ONE, "One tab"), (SIDE, "Split: side by side"), (STACKED, "Split: top and bottom")):
            b = QtWidgets.QToolButton(checkable=True)
            b.setToolTip(tip)
            b.setAccessibleName(tip)
            b.clicked.connect(lambda _=False, mode=mode: self.tabs.set_layout(mode))
            group.addButton(b)
            sl.addWidget(b)
            self.split_buttons[mode] = b
        self.split_buttons[ONE].setChecked(True)
        self.btn_theme = QtWidgets.QToolButton()
        self.btn_theme.setAutoRaise(True)
        self.btn_theme.setIconSize(QtCore.QSize(20, 20))
        self.btn_theme.clicked.connect(self.toggle_plot_theme)
        # Split buttons and the light/dark toggle sit at the right end of the toolbar,
        # on the same row as Start and Record, and show on every tab.
        self.tab_corner = QtWidgets.QWidget()
        cl2 = QtWidgets.QHBoxLayout(self.tab_corner)
        cl2.setContentsMargins(0, 0, 0, 0)
        cl2.setSpacing(6)
        cl2.addWidget(self.split_group)
        cl2.addWidget(self.btn_theme)
        self._tb_corner = A(self.tab_corner)
        self._update_theme_button()
        sc = QtGui.QShortcut(QtGui.QKeySequence("C"), self)
        sc.activated.connect(self._focus_comment)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self, activated=lambda: self.fire_stimulus())
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self._escape)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.plot_area = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_area)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(2)
        self.plots, self.curves, self.strips = [], [], []
        self.plot_widgets, self.curve_colors, self.curve_widths = [], [], []
        self.row_widgets = []

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
        # Bottom of the Acquisition page: channel buttons (live only) on the left,
        # Comments on the right (live and review). It lives here rather than in the
        # toolbar so it's on screen whenever Acquisition is, even in a split.
        self.bottom_bar = QtWidgets.QWidget()
        bb = QtWidgets.QHBoxLayout(self.bottom_bar)
        bb.setContentsMargins(0, 0, 4, 0)
        bb.setSpacing(0)
        bb.addWidget(self.channel_bar, 1)
        bb.addStretch(0)
        bb.addWidget(self.btn_comments)

        self.scroll = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.scroll.valueChanged.connect(self._scrolled)
        self.comments_panel = CommentsPanel()
        self.comments_panel.edited.connect(self.edit_comment)
        self.comments_panel.deleted.connect(self.delete_comment)
        self.comments_panel.jump.connect(self.jump_to)
        self.comments_panel.closed.connect(lambda: self.btn_comments.setChecked(False))
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.plot_area, 1)
        row.addWidget(self.comments_panel)
        lay.addLayout(row, 1)
        lay.addWidget(self.bottom_bar)
        lay.addWidget(self.scroll)
        self.stim_page = StimPage(self)
        self.trials_page = TrialsPage(self)
        placeholder = lambda title, text: self._placeholder(title, text)
        self.tabs = TabArea([
            ("acq", "Acquisition", central),
            ("stim", "Stimulator", self.stim_page),
            ("trials", "Trials", self.trials_page),
            ("analysis", "Analysis", placeholder("Analysis", "Coming soon.")),
        ])
        self.tabs.changed.connect(self._tabs_changed)
        self.setCentralWidget(self.tabs)

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
        self.act_open_mat = m.addAction("Open .&mat File…", self.open_mat_dialog)
        self.act_open_mat.setToolTip("Review a LabChart export, or a .mat this app exported. "
                                     "The file itself is never changed.")
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
        self.plot_widgets, self.curve_colors, self.curve_widths = [], [], []
        self.row_widgets = []

    def _new_plot(self, name: str, unit: str, color: str, width: float = 1.0) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        p = pw.getPlotItem()
        self.plot_widgets.append(pw)
        if unit in _SI_BASE:
            p.setLabel("left", name, units=unit)       # pyqtgraph adds m, µ, k as needed
        else:
            # A prefix on a unit that already has one (mV, µA) or isn't SI (mmHg)
            # would read "kmV", so show it as it is.
            p.setLabel("left", f"{name} ({unit})" if unit else name)
            p.getAxis("left").enableAutoSIPrefix(False)
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
        self.curves.append(p.plot())
        self.curve_colors.append(color)
        self.curve_widths.append(float(width))
        self._set_curve_color(len(self.curves) - 1, color)
        self._style_plot(pw)
        return pw

    def _finish_plots(self):
        if self.plots:
            self.plots[-1].setLabel("bottom", "time", units="s")
            self.plots[0].sigXRangeChanged.connect(self._xrange_changed)

    def _build_live_plots(self):
        """One row per channel: its controls on the left, its plot on the right.

        A switched-off channel keeps its controls, so it can be switched back on,
        but has no plot.
        """
        self._clear_plot_area()
        inputs = valid_inputs(self.cfg)
        locked = self.recorder.recording or (hasattr(self, "trials_page") and self.trials_page.running)
        for r in self.rows:
            strip = ChannelStrip(r, inputs, self.cfg.terminal_config == "DIFF",
                                 self._channel_edited, self._channel_removed, self._channel_recolored,
                                 can_remove=len(self.rows) > 1, on_width=self._channel_width)
            strip.set_locked(locked)
            self.strips.append(strip)
            if r.ch.enabled:
                content = self._new_plot(r.ch.name, r.ch.units, r.ch.color, r.ch.width)
            else:
                content = QtWidgets.QLabel("off")
                content.setStyleSheet("color: gray; padding-left: 8px")
            row_w = ChannelRowWidget(strip, content)
            row_w.set_compact(self._compact_strips)
            self.row_widgets.append(row_w)
            if r.ch.enabled:
                self.plot_layout.addWidget(row_w, 1)
            else:
                row_w.setFixedHeight(strip.sizeHint().height())
                self.plot_layout.addWidget(row_w, 0)
        self._finish_plots()
        self.btn_add_ch.setEnabled(not locked)
        self.rate_box.setEnabled(not locked)
        if hasattr(self, "trials_page"):
            self.trials_page.sync()

    def _build_review_plots(self, names: list[str], units: list[str], colors: list[str]):
        self._review_names = list(names)
        """Plots for a saved session: one per recorded channel, with only a colour bar."""
        self._clear_plot_area()
        self.review_color_btns = []
        for i, (name, unit, color) in enumerate(zip(names, units, colors)):
            row_w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row_w)
            h.setContentsMargins(0, 0, 0, 0)
            btn = ColorButton(color, lambda c, i=i: self.set_review_color(i, c),
                              tooltip="Change this channel's colour in every block of this session")
            self.review_color_btns.append(btn)
            h.addWidget(btn, 0)
            widths = {r.ch.name: r.ch.width for r in self.rows}
            pw = self._new_plot(name, unit, color, widths.get(name, 1.0))
            pw.setToolTip("Double-click to add a comment")
            pw.scene().sigMouseClicked.connect(lambda ev, pw=pw: self._plot_clicked(pw, ev))
            h.addWidget(pw, 1)
            self.plot_layout.addWidget(row_w, 1)
        self._finish_plots()

    # -- dark / light plots ------------------------------------------------
    def _set_curve_color(self, k: int, color: str) -> None:
        """Draw trace k in `color` (adjusted for contrast on the current background)."""
        self.curve_colors[k] = color
        self.curves[k].setPen(pg.mkPen(theme.display_color(color, self.plot_mode), width=self.curve_widths[k]))

    def _channel_width(self, strip: ChannelStrip, width: float) -> None:
        self.set_channel_width(self.rows.index(strip.row), width)

    def set_channel_width(self, index: int, width: float) -> None:
        """Change a channel's line thickness (display only; allowed while recording)."""
        from .channels import LINE_MAX, LINE_MIN
        width = min(LINE_MAX, max(LINE_MIN, float(width)))
        ch = self.rows[index].ch
        ch.width = width
        if ch in self.cfg.channels:
            k = self.cfg.channels.index(ch)
            if self.mode == LIVE and k < len(self.curves):
                self.curve_widths[k] = width
                self._set_curve_color(k, self.curve_colors[k])

    def _style_plot(self, pw: pg.PlotWidget) -> None:
        th = theme.THEMES[self.plot_mode]
        pw.setBackground(th.background)
        for name in ("left", "bottom"):
            ax = pw.getPlotItem().getAxis(name)
            ax.setPen(th.foreground)
            ax.setTextPen(th.foreground)

    def _update_theme_button(self) -> None:
        ink = style.tokens(self._app_mode())["ink"]
        self.btn_theme.setIcon(theme.toggle_icon(self.plot_mode, ink))
        self.btn_theme.setToolTip("Dark mode: click for light mode" if self.plot_mode == theme.DARK
                                  else "Light mode: click for dark mode")
        for mode, b in getattr(self, "split_buttons", {}).items():
            b.setIcon(layout_icon(mode, ink))

    def _app_mode(self) -> str:
        return style.DARK if self.plot_mode == theme.DARK else style.LIGHT

    def _apply_app_theme(self) -> None:
        """The window's own colours (the plots are styled separately)."""
        mode = self._app_mode()
        style.restyle(mode)
        self.comments_panel.link_color = style.tokens(mode)["link"]
        self.stim_page.apply_theme(mode)
        if hasattr(self, "trials_page"):
            self.trials_page.apply_theme(mode)
        self._update_theme_button()
        self.refresh_comments(redraw_markers=False)

    def toggle_plot_theme(self) -> None:
        self.set_plot_theme(theme.LIGHT if self.plot_mode == theme.DARK else theme.DARK)

    def set_plot_theme(self, mode: str) -> None:
        """Switch the whole app, plots included, between dark and light. Remembered on this computer."""
        self.plot_mode = mode
        self.settings.setValue("plot_theme", mode)
        for pw in self.plot_widgets:
            self._style_plot(pw)
        for k, color in enumerate(self.curve_colors):
            self._set_curve_color(k, color)
        mc = theme.THEMES[mode].marker
        for _, lines in self.markers:
            for ln in lines:
                ln.setPen(pg.mkPen(mc, style=QtCore.Qt.PenStyle.DashLine))
                if getattr(ln, "label", None) is not None:   # only the top plot's line has one
                    ln.label.setColor(mc)
        self._apply_app_theme()

    # -- tabs ----------------------------------------------------------------
    def _placeholder(self, title: str, text: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addStretch(1)
        for t, big in ((title, True), (text, False)):
            lab = QtWidgets.QLabel(t)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if big:
                f = lab.font(); f.setPointSizeF(f.pointSizeF() * 1.4); f.setBold(True); lab.setFont(f)
            v.addWidget(lab)
        v.addStretch(1)
        return w

    def _tabs_changed(self):
        for mode, b in self.split_buttons.items():
            b.setChecked(mode == self.tabs.layout_mode)
        compact = self.tabs.shares_screen("acq")
        if compact != self._compact_strips:
            self._compact_strips = compact
            for rw in self.row_widgets:
                rw.set_compact(compact)
        self._update_toolbar()

    def _update_toolbar(self):
        """Show the controls that apply to what's on screen."""
        if not hasattr(self, "tabs"):
            return
        live, acq = self.mode == LIVE, self.tabs.is_visible("acq")
        for a in self._tb_live:
            a.setVisible(live)
        for a in self._tb_review:
            a.setVisible(not live and acq)
        for a in self._tb_acq:
            a.setVisible(acq)
        for a in self._tb_acq_live:
            a.setVisible(live and acq)
        self._tb_spacer.setVisible(not (live and acq))
        self._update_stim_chip()

    def _update_stim_chip(self):
        busy = self.stim_out is not None and self.stim_out.busy
        protocol = hasattr(self, "trials_page") and self.trials_page.running
        on = self.repeating or busy or protocol
        if protocol:
            self.stim_chip_text.setText(self.trials_page.chip_text())
            self.stim_chip.setToolTip("A trial protocol is running (Trials tab). Stop ends it.")
        elif on:
            self.stim_chip_text.setText("Stimulating")      # the interval is in the tooltip
            self.stim_chip.setToolTip(f"Repeating every {self.stim_repeat_s:g} s" if self.repeating
                                      else "A stimulus is being sent")
        if self._tb_chip.isVisible() != on:
            self._tb_chip.setVisible(on)

    # -- changing channels -------------------------------------------------
    def _copy_rows(self) -> list[ChannelRow]:
        return [ChannelRow(dataclasses.replace(r.ch)) for r in self.rows]

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
        color = cfgmod.next_color(r.ch.color for r in self.rows)
        return self.apply_channels(self._copy_rows() + [ChannelRow(Channel(free[0], f"Ch{n}", color=color))])

    # -- colours (display only: allowed while recording, never restarts) ----
    def _channel_recolored(self, strip: ChannelStrip, color: str):
        self.set_channel_color(self.rows.index(strip.row), color)

    def set_channel_color(self, index: int, color: str, recolor_recorded: bool | None = None) -> None:
        """Change a live channel's trace colour.

        If the session already has recorded blocks with a different colour for
        this channel, asks whether to recolour them too (or uses
        `recolor_recorded` if given). The block being recorded always follows.
        """
        color = cfgmod.normalize_color(color)
        ch = self.rows[index].ch
        ch.color = color
        strip = self.strips[index] if index < len(self.strips) else None
        if strip is not None:
            strip.color_btn.set_color(color, ch.enabled)
        if ch in self.cfg.channels:
            k = self.cfg.channels.index(ch)
            if k < len(self.curves):
                self._set_curve_color(k, color)
        if self.recorder.recording:
            self.recorder.set_colors([c.color for c in self.cfg.channels])
        current = self.recorder.path if self.recorder.recording else None
        older = [b for b in self.blocks if b != current and self._block_color(b, ch.name) not in (None, color)]
        if older:
            if recolor_recorded is None:
                recolor_recorded = self._recolor_policy
            if recolor_recorded is None:
                recolor_recorded, remember = self._ask_recolor(ch.name, len(older))
                if remember:
                    self._recolor_policy = recolor_recorded
            if recolor_recorded:
                self._recolor(older, {ch.name: color})
                return
        self.statusBar().showMessage(f"{ch.name}: colour {cfgmod.color_label(color)}", 6000)

    @staticmethod
    def _block_color(block: Path, name: str) -> str | None:
        """Colour of channel `name` in a recorded block, or None if it isn't there."""
        try:
            info = session.block_info(block)
        except Exception:
            return None
        names = info["channel_names"]
        return info["colors"][names.index(name)] if name in names else None

    def _ask_recolor(self, name: str, n: int) -> tuple[bool, bool]:
        """Ask whether to recolour recorded blocks. Returns (recolour, remember for this session)."""
        B = QtWidgets.QMessageBox.StandardButton
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Question, "Recolour recorded blocks?",
            f"{n} block(s) already recorded in this session show {name} in another colour.\n\n"
            "Use the new colour for them too? (Only the display colour changes, not the data.)",
            B.Yes | B.No, self)
        box.setDefaultButton(B.Yes)
        remember = QtWidgets.QCheckBox("Do this for every colour change in this session")
        box.setCheckBox(remember)
        return box.exec() == B.Yes, remember.isChecked()

    def _recolor(self, blocks: list[Path], colors: dict[str, str]) -> None:
        failed = session.recolor_blocks(blocks, colors)
        names = ", ".join(colors)
        if failed:
            self._channel_message(f"Couldn't update the colour in {len(failed)} block(s) "
                                  f"(read-only or in use?): {', '.join(f.name for f in failed)}")
        else:
            self.statusBar().showMessage(f"{names}: colour updated in {len(blocks)} recorded block(s)", 8000)

    def set_review_color(self, index: int, color: str) -> None:
        """Recolour one channel in every block of the session being reviewed."""
        if self.mode != REVIEW or self.review_block is None:
            return
        color = cfgmod.normalize_color(color)
        name = self.review_block["channel_names"][index]
        self.review_block["colors"][index] = color
        self._set_curve_color(index, color)
        self.review_color_btns[index].set_color(color)
        self._recolor(self.review_blocks, {name: color})

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
        if hasattr(self, "trials_page") and self.trials_page.running:
            return refuse(f"Stop the trial protocol before changing the {what}.")
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
        self._update_toolbar()
        self.scroll.setVisible(mode == REVIEW)
        self.channel_bar.setVisible(mode == LIVE)
        self.wiring_menu.setEnabled(mode == LIVE and not self.recorder.recording)
        fw = self.focusWidget()
        if fw is not None:      # don't leave a box (e.g. Window) highlighted after switching
            fw.clearFocus()
        self._update_title()
        self._update_wiring_label()
        if hasattr(self, "comments_panel"):
            self.refresh_comments()

    def _update_title(self):
        if self.mode == REVIEW and self._review_source is not None:
            where = f"{self._review_source.name} (imported copy) — Review"
        elif self.mode == REVIEW and self.review_dir is not None:
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
        for a in (self.act_new, self.act_open, self.act_open_mat, self.act_append):
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
        self._flush_review_mat()
        self._live_comments_dirty = False
        self.review_dir, self.review_blocks, self.review_block = None, [], None
        self._review_source = None
        self.session_dir, self.blocks, self._exported_n = None, [], 0
        self.session_saved = False
        self._recolor_policy = None          # a new session asks again
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

    def open_mat_dialog(self):
        if not self._leave_live():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open .mat file", self._start_dir(), "LabChart / MATLAB export (*.mat);;All files (*)")
        if path:
            self._try(lambda: self.open_mat(path), "Could not open the .mat file")

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
        if not self._confirm_append(Path(path)):
            return
        self._try(lambda: self.append_session(path, force=True), "Could not append")

    def _append_notes(self, path: Path) -> list[str]:
        """What happens on Append, including colours that will change."""
        notes = [
            "New blocks must use the same channel names, inputs, units, sampling rate and "
            "wiring as the blocks already in this session. (They do now; if you change any of "
            "them after appending, a new session is started instead.)",
            "Channels are matched by name, so don't rename a channel while appending.",
        ]
        used = session.session_colors(path)
        changed = [f"{r.ch.name} → {cfgmod.color_label(used[r.ch.name])}" for r in self.rows
                   if r.ch.name in used and used[r.ch.name] != r.ch.color]
        if changed:
            notes.append("To match the session, these channels will switch colour: " + ", ".join(changed) + ".")
        return notes

    def _confirm_append(self, path: Path) -> bool:
        """Explain the rules for appending before doing it (can be turned off)."""
        if self.settings.value("skip_append_notice", False, type=bool):
            return True
        B = QtWidgets.QMessageBox.StandardButton
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Information, "Append to session",
            f"Appending to {path.name}.\n\n• " + "\n• ".join(self._append_notes(path)),
            B.Ok | B.Cancel, self)
        box.setDefaultButton(B.Ok)
        skip = QtWidgets.QCheckBox("Don't show this again")
        box.setCheckBox(skip)
        ok = box.exec() == B.Ok
        if ok and skip.isChecked():
            self.settings.setValue("skip_append_notice", True)
        return ok

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
        self._flush_review_mat()
        self._review_source = None
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

    def open_mat(self, path: str | Path) -> Path:
        """Review a LabChart-style .mat. It's turned into a session folder under
        <data_dir>/imported (reused if the .mat hasn't changed), so comments and
        colours can be edited without touching the original file."""
        path = Path(path)
        try:
            folder, notes = matfile.import_mat(path, Path(self.cfg.data_dir) / "imported")
        except matfile.MatError as e:
            raise session.SessionError(str(e)) from None
        self.open_session(folder)
        self._review_source = path
        recent = [r for r in self.recent_sessions() if r != str(folder.resolve())]
        self.settings.setValue("recent_sessions", session.add_recent(recent, path))
        self._update_title()
        msg = (f"Opened {path.name} ({len(self.review_blocks)} block(s)). The .mat itself isn't changed: "
               f"comments and colours are kept in its imported copy, {folder}.")
        if notes:
            msg += " " + " ".join(notes)
        self.statusBar().showMessage(msg, 20000)
        return folder

    def append_session(self, path: str | Path, force: bool = False) -> Path:
        """Continue recording new blocks into an existing session."""
        path = Path(path)
        problems, warnings = session.append_check(path, self.cfg)
        if problems or (warnings and not force):
            raise session.SessionError("\n".join(problems + warnings))
        self.back_to_live()
        # New blocks match the colours already used in this session.
        used = session.session_colors(path)
        for r in self.rows:
            if r.ch.name in used:
                r.ch.color = used[r.ch.name]
        self._build_live_plots()
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
            self._live_comments_dirty = False
        else:
            self._review_mat_stale = False
        self.statusBar().showMessage(f"Exported {len(blocks)} block(s) to {out}", 10000)
        return out

    def _export_blocks(self) -> list[Path]:
        return self.review_blocks if self.mode == REVIEW else self.blocks

    def _export_folder(self) -> Path | None:
        return self.review_dir if self.mode == REVIEW else self.session_dir

    def _autoexport_live(self):
        # Don't lose the .mat if they forgot: new blocks, or comments edited since.
        if len(self.blocks) > self._exported_n or self._live_comments_dirty:
            prev_mode, self.mode = self.mode, LIVE
            if self.export() is not None:
                self._live_comments_dirty = False
            self.mode = prev_mode

    def _flush_review_mat(self):
        """After editing comments in review, update the session's .mat if it has one."""
        if self.mode == REVIEW and self._review_mat_stale and self.review_dir is not None:
            if (self.review_dir / f"{self.review_dir.name}.mat").exists():
                self.export()
        self._review_mat_stale = False

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
        if path.lower().endswith(".mat"):
            if not Path(path).is_file():
                self.settings.setValue("recent_sessions", [r for r in self.recent_sessions() if r != path])
                QtWidgets.QMessageBox.warning(self, "File not found",
                                              f"{path} no longer exists. It has been removed from the list.")
            elif self._leave_live():
                self._try(lambda: self.open_mat(path), "Could not open the .mat file")
            return
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
        names = getattr(self, "_review_names", None)
        if self.strips or names != blk["channel_names"] or len(self.plots) != len(blk["channel_names"]):
            self._build_review_plots(blk["channel_names"], blk["units"], blk["colors"])
        for k, (btn, color) in enumerate(zip(self.review_color_btns, blk["colors"])):
            if k < len(self.curves):
                self._set_curve_color(k, color)
            btn.set_color(color)
        for curve, y in zip(self.curves, blk["data"]):
            curve.setData(blk["t"], y)
        self.refresh_comments()
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
            self._clear_markers()
            self._axis_zero = 0.0
            self._block_on_screen = False
            self.btn_run.setText("Stop")
            self.btn_rec.setEnabled(True)
            self.timer.start()
        else:
            if self.btn_rec.isChecked():
                self.btn_rec.setChecked(False)
            self.stop_stimulus()
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
            # Time 0 = this block's first sample. Until it arrives, use the latest
            # sample; _refresh corrects it (by at most one read) once it's known.
            self._set_axis_zero(self.recorder.last_t)
            self._block_on_screen = True
            self.blocks.append(path)
            self._update_title()
            self.statusBar().showMessage(f"Recording {path.name} in {self.session_dir}")
            self.btn_rec.setText("Stop recording · 0:00")
            self.btn_rec.setIcon(style.dot_icon("#ffffff", square=True))
            self._lock_channels(True)
            self._set_comment_box(True)
        else:
            self.recorder.stop()
            self._lock_channels(False)
            self.statusBar().showMessage(f"Saved {self.recorder.path.name}", 10000)
            self.btn_rec.setText("Record")
            self.btn_rec.setIcon(style.dot_icon(style.RECORDING))
            self._set_comment_box(False)
        # Starting a block keeps the previous block's comment lines: its data is
        # still on screen (now at negative times), and they scroll off with it.
        self.refresh_comments(redraw_markers=not on)

    def _lock_channels(self, locked: bool):
        for st in self.strips:
            st.set_locked(locked)
        self.btn_add_ch.setEnabled(not locked)
        self.rate_box.setEnabled(not locked)
        self.wiring_menu.setEnabled(not locked and self.mode == LIVE)

    def _redraw_live(self):
        if self.buf_t.size and self.buf_y.shape[0] == len(self.curves):
            for ch, curve, y in zip(self.cfg.channels, self.curves, self.buf_y):
                curve.setData(self.buf_t - self._axis_zero, ch.to_units(y))

    def _set_axis_zero(self, zero: float) -> None:
        """Change which stream time the live axis shows as 0; comment lines move with it."""
        delta = self._axis_zero - zero
        if delta:
            for i, (t, lines) in enumerate(self.markers):
                for ln in lines:
                    ln.setValue(ln.value() + delta)
                self.markers[i] = (t + delta, lines)
        self._axis_zero = zero

    def _add_comment(self):
        text = self.comment_edit.text().strip()
        if not text or not self.recorder.recording:
            return
        self._log_comment(text)
        self.comment_edit.clear()

    def _log_comment(self, text: str) -> None:
        t = self.recorder.add_comment(text)      # saved to the block straight away
        self._add_marker(t - self._axis_zero, text)
        self.refresh_comments(redraw_markers=False)

    # -- stimulator ----------------------------------------------------------
    def _saved_stimulus(self) -> Stimulus:
        import json
        try:
            return Stimulus.from_dict(json.loads(self.settings.value("stimulus", "") or "{}"))
        except Exception:
            return Stimulus()

    def set_stimulus(self, s: Stimulus) -> None:
        """Use stimulus `s` for Fire and Repeat, and show it in the Stimulator tab."""
        self._stimulus_edited(s)
        if hasattr(self, "stim_page"):
            self.stim_page.set_stimulus(s)

    def _stimulus_edited(self, s: Stimulus) -> None:
        """The stimulus that Fire and Repeat send (remembered between runs)."""
        import json
        self.stimulus = s
        self.settings.setValue("stimulus", json.dumps(s.to_dict()))
        if hasattr(self, "trials_page"):
            self.trials_page.stimulus_changed()

    def open_stim_window(self) -> StimPage:
        """Show the Stimulator tab."""
        self.tabs.show_tab("stim")
        return self.stim_page

    def _stim_message(self, text: str, error: bool = False) -> None:
        self.statusBar().showMessage(("⚠ " if error else "") + text, 10000)
        self.stim_page.show_message(text, error)

    def _escape(self):
        if self.trials_page.running:
            self.stop_stimulus()
            return
        if self.repeating or (self.stim_out is not None and self.stim_out.busy):
            self.stop_stimulus()
            self._stim_message("Stimulus stopped")

    def fire_stimulus(self) -> bool:
        """Send the current stimulus once. Returns False (with a message) if it can't."""
        if self.stim_out is None:
            self._stim_message("The stimulator needs the nidaqmx (or simulated) backend.", error=True)
            return False
        if self.trials_page.running:
            self._stim_message("A trial protocol is running (Trials tab); stop it first.", error=True)
            return False
        try:
            wave = self.stimulus.waveform(self.cfg.stim_rate)
        except ValueError as e:
            self._stim_message(str(e), error=True)
            return False
        return self.send_wave(wave, self.stimulus.describe())

    def send_wave(self, wave, desc: str) -> bool:
        """Send `wave` (volts at the output rate) and note it as a comment while recording."""
        if self.stim_out is None:
            self._stim_message("The stimulator needs the nidaqmx (or simulated) backend.", error=True)
            return False
        if self.mode != LIVE or not self.backend.running:
            self._stim_message("Press Start first: stimuli are sent while acquiring.", error=True)
            return False
        try:
            stimmod.check_limit(wave, self.cfg.stim_max_v)
            self.stim_out.fire(wave, self.cfg.stim_rate)
        except (ValueError, RuntimeError) as e:
            self._stim_message(str(e), error=True)
            return False
        if self.recorder.recording:
            self._log_comment(f"Stim: {desc}" if not desc.startswith("Trial") else desc)
        self._stim_message(f"Stimulus sent on {self.cfg.stim_output}: {desc}")
        self._update_stim_chip()
        return True

    def stop_stimulus(self) -> None:
        if hasattr(self, "trials_page") and self.trials_page.running:
            self.trials_page.stop_protocol("Protocol stopped")
        self.set_repeat(False)
        if self.stim_out is not None:
            self.stim_out.stop()
        self._update_stim_chip()

    def set_repeat(self, on: bool) -> None:
        if on and hasattr(self, "trials_page") and self.trials_page.running:
            self._stim_message("A trial protocol is running (Trials tab); stop it first.", error=True)
            on = False
        if on and not self.repeating:
            self.repeating = True          # set first, so the chip shows "every N s"
            if not self.fire_stimulus():
                self.repeating = False
            else:
                self.stim_timer.start(int(self.stim_repeat_s * 1000))
                self._stim_message(f"Repeating every {self.stim_repeat_s:g} s")
        elif not on:
            self.repeating = False
            self.stim_timer.stop()
        self.settings.setValue("stim_repeat_s", self.stim_repeat_s)
        b = self.stim_page.btn_repeat
        if b.isChecked() != self.repeating:
            b.blockSignals(True)
            b.setChecked(self.repeating)
            b.blockSignals(False)
        self._update_stim_chip()

    def _repeat_fire(self):
        if self.stim_out is not None and self.stim_out.busy:
            self._stim_message("Skipped one repeat: the previous stimulus was still running. "
                               "Use a longer repeat time.", error=True)
            return
        if not self.fire_stimulus():
            self.set_repeat(False)
        elif self.stim_timer.interval() != int(self.stim_repeat_s * 1000):
            self.stim_timer.setInterval(int(self.stim_repeat_s * 1000))

    def set_stim_output(self, ch: str) -> None:
        if self.stim_out is not None and self.stim_out.busy:
            self.stop_stimulus()
        self.cfg.stim_output = ch
        self.stim_page.sync()
        self.statusBar().showMessage(f"Stimulator output: {ch}", 6000)

    def monitor_choices(self) -> list[tuple[str, str, bool, str]]:
        """Inputs the stimulus can be recorded on: (label, input, usable, tooltip)."""
        used = {r.ch.physical: r.ch.name for r in self.rows
                if not (r.ch.physical == self.cfg.stim_monitor and r.ch.name == "Stim")}
        out = [("None (don't record it)", "", True, "")]
        for phys in valid_inputs(self.cfg):
            if phys in used:
                out.append((f"{phys} (used by {used[phys]})", phys, False, f"Used by {used[phys]}"))
            else:
                pair = f" (minus input ai{int(phys[2:]) + 4})" if self.cfg.terminal_config == "DIFF" else ""
                out.append((phys, phys, True, f"Wire {self.cfg.stim_output} to {phys}{pair}, and AO GND to AI GND"))
        return out

    def set_stim_monitor(self, phys: str) -> bool:
        """Record the stimulus on input `phys` (a channel named Stim), or stop recording it ("")."""
        old = self.cfg.stim_monitor
        if phys == old:
            return True
        rows = [r for r in self._copy_rows() if not (old and r.ch.physical == old and r.ch.name == "Stim")]
        if phys:
            taken = [r.ch.name for r in rows if r.ch.physical == phys]
            if taken:
                self._channel_message(f"{phys} is already used by {taken[0]}.")
                return False
            if any(r.ch.name == "Stim" for r in rows):
                self._channel_message("Another channel is already named Stim; rename it first.")
                return False
            color = cfgmod.next_color(r.ch.color for r in rows)
            rows.append(ChannelRow(Channel(phys, "Stim", units="V", color=color)))
        if not self.apply_channels(rows):
            return False
        self.cfg.stim_monitor = phys
        self.stim_page.sync()
        self.trials_page.sync()
        self.statusBar().showMessage(f"Recording the stimulus on {phys}" if phys
                                     else "Not recording the stimulus", 8000)
        return True

    def _set_comment_box(self, recording: bool):
        self.comment_edit.setEnabled(recording)
        self.comment_edit.setPlaceholderText("Press Enter to add comment" if recording
                                             else "Press Record to add comments")

    def _focus_comment(self):
        if self.mode == LIVE and self.comment_edit.isEnabled():
            self.comment_edit.setFocus()
            self.comment_edit.selectAll()

    # -- comments panel ----------------------------------------------------
    def _toggle_comments(self, on: bool):
        self.comments_panel.setVisible(on)
        self.settings.setValue("comments_open", on)

    def _comment_block(self) -> Path | None:
        """The block whose comments the panel shows: the reviewed block in review;
        in live, the block being recorded, else the last one recorded."""
        if self.mode == REVIEW:
            i = self.block_combo.currentIndex()
            return self.review_blocks[i] if 0 <= i < len(self.review_blocks) else None
        if self.recorder.recording:
            return self.recorder.path
        return self.blocks[-1] if self.blocks else None

    def _read_block_comments(self, block: Path) -> list[dict]:
        if self.recorder.recording and block == self.recorder.path:
            return self.recorder.get_comments()
        try:
            return session.read_comments(block)
        except Exception:
            return []

    def refresh_comments(self, redraw_markers: bool = True) -> None:
        """Reload the panel (and the comment lines on the plots) from the block file."""
        block = self._comment_block()
        self._shown_comments = self._read_block_comments(block) if block else []
        self.comments_panel.set_comments(f"Comments · {block.stem}" if block else "Comments",
                                         [(c["t"], c["text"]) for c in self._shown_comments],
                                         editable=block is not None)
        self.btn_comments.setText(f"Comments ({len(self._shown_comments)})")
        if redraw_markers:
            self._redraw_comment_markers()

    def _redraw_comment_markers(self):
        self._clear_markers()
        if self.mode == REVIEW:
            offset = 0.0
        elif (self._block_on_screen and self.recorder.path is not None
              and self._comment_block() == self.recorder.path):
            offset = self.recorder.block_start - self._axis_zero
        else:
            return                               # that block isn't on screen
        for c in self._shown_comments:
            self._add_marker(c["t"] + offset, c["text"])

    def _save_comments(self, comments: list[dict]) -> None:
        block = self._comment_block()
        if block is None:
            return
        try:
            if self.recorder.recording and block == self.recorder.path:
                self.recorder.set_comments(comments)
            else:
                session.write_comments(block, comments)
        except Exception as e:
            self._channel_message(f"Couldn't save the comment change in {block.name} "
                                  f"(read-only or open elsewhere?): {e}")
        if self.mode == REVIEW:
            self._review_mat_stale = True
        elif self.session_saved:
            self._live_comments_dirty = True
        self.refresh_comments()

    def edit_comment(self, row: int, text: str) -> None:
        comments = [dict(c) for c in self._shown_comments]
        if 0 <= row < len(comments) and text.strip():
            comments[row]["text"] = text.strip()
            self._save_comments(comments)

    def delete_comment(self, row: int) -> None:
        if 0 <= row < len(self._shown_comments):
            self._save_comments([c for i, c in enumerate(self._shown_comments) if i != row])

    def add_comment_at(self, t: float, text: str) -> None:
        """Add a comment at time t (s from the block's start), e.g. from a double-click in review."""
        if not text.strip() or self._comment_block() is None:
            return
        fs = (self.review_block or {}).get("sample_rate") if self.mode == REVIEW else self.backend.actual_rate
        fs = fs or self.cfg.sample_rate
        self._save_comments(self._shown_comments + [
            {"t": float(t), "sample_index": int(round(t * fs)), "text": text.strip()}])

    def jump_to(self, t: float) -> None:
        """Centre the review plots on a comment."""
        if self.mode != REVIEW or not self.plots:
            return
        win = self.window_box.value()
        start = max(0.0, t - win / 2)
        self.plots[0].setXRange(start, start + win, padding=0)

    def _plot_clicked(self, pw, ev):
        if self.mode != REVIEW or not ev.double() or self.review_block is None:
            return
        vb = pw.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        t = min(max(0.0, vb.mapSceneToView(ev.scenePos()).x()), self.review_block["duration_s"])
        self._popup = CommentPopup(self, t)
        self._popup.submitted.connect(self.add_comment_at)
        self._popup.show_at(QtGui.QCursor.pos())

    def _add_marker(self, t: float, text: str):
        lines = []
        # Stagger labels over three heights so nearby comments don't print on top of each
        # other, and keep them short on the plot (the Comments list has the full text).
        self._marker_count = getattr(self, "_marker_count", 0) + 1
        label_y = 0.92 - 0.09 * ((self._marker_count - 1) % 3)
        short = text if len(text) <= 32 else text[:31].rstrip() + "…"
        for p in self.plots:
            mc = theme.THEMES[self.plot_mode].marker
            ln = pg.InfiniteLine(t, angle=90, pen=pg.mkPen(mc, style=QtCore.Qt.PenStyle.DashLine),
                                 label=short if p is self.plots[0] else None,
                                 labelOpts={"position": label_y, "color": mc})
            p.addItem(ln)
            lines.append(ln)
        self.markers.append((t, lines))

    def _clear_markers(self):
        self._marker_count = 0
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
        if self.stim_out is not None and self.stim_out.error is not None:
            err, self.stim_out.error = self.stim_out.error, None
            self.set_repeat(False)
            self._stim_message(f"Stimulus failed: {err}", error=True)
        self._update_stim_chip()
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
        if self.recorder.recording and self.recorder.block_started:
            self._set_axis_zero(self.recorder.block_start)   # exact block start
        x = self.buf_t - self._axis_zero
        t_end = x[-1]
        for ch, curve, y in zip(self.cfg.channels, self.curves, self.buf_y):
            curve.setData(x, ch.to_units(y))
        self.plots[0].setXRange(t_end - self.window_box.value(), t_end, padding=0)
        # drop markers that scrolled off
        t_min = x[0]
        while self.markers and self.markers[0][0] < t_min:
            _, lines = self.markers.pop(0)
            for p, ln in zip(self.plots, lines):
                p.removeItem(ln)
        rate = self.backend.actual_rate
        lag = self.display_q.qsize()
        msg = f"rate {rate:,.1f} Hz/ch" + ("" if self.backend.hardware_timed else " (software-timed!)")
        if self.recorder.recording and self.backend.hardware_timed:
            secs = self.recorder.samples_written / max(rate, 1e-9)
            msg += f" | recorded {secs:,.1f} s"
            self.btn_rec.setText(f"Stop recording · {int(secs) // 60}:{int(secs) % 60:02d}")
        if lag > 5:
            msg += f" | display lagging ({lag} chunks queued)"
        self.status.setText(msg)

    def closeEvent(self, ev):
        if self.btn_rec.isChecked():
            self.btn_rec.setChecked(False)
        if not self._resolve_unsaved():
            ev.ignore()
            return
        self.trials_page.shutdown()          # stops a protocol; deletes its plots now
        self.recorder.stop()
        self.stim_timer.stop()
        if self.stim_out is not None:
            self.stim_out.stop()             # output to 0 V
        self.backend.stop()
        self._flush_review_mat()
        if self.session_saved:
            self._autoexport_live()
        self.recorder.close()   # end the writer thread before Qt and h5py shut down
        # Delete the plots now, in order. Left to Python's garbage collector, a
        # pyqtgraph plot can be freed while Qt still has a layout pass queued for
        # it, which crashes Qt (seen when windows are opened and closed repeatedly).
        self._clear_plot_area()
        self.stim_page.preview.setParent(None)
        self.stim_page.preview.deleteLater()
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
    saved = QtCore.QSettings("neusci", "neusci-daq-visualizer").value("plot_theme", theme.DARK)
    style.apply(app, style.DARK if saved == theme.DARK else style.LIGHT)
    w = MainWindow(cfg, config_path=config_path)
    w.resize(1300, 200 + 180 * len(cfg.channels))
    w.show()
    return app.exec()
