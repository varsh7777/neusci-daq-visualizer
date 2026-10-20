"""Tabs, split view, hover strips, line thickness, whole-app light/dark, stimulator chip."""
import pytest
from pyqtgraph.Qt import QtGui, QtWidgets

from neusci_daq import config, style, theme
from neusci_daq.stim import Stimulus
from neusci_daq.tabs import ONE, SIDE, STACKED

from test_channels import app, make_cfg, pump, win  # noqa: F401  (fixtures)


def shown(w):
    return w.tabs.visible()


# ---------------------------------------------------------------- tabs and split
def test_starts_on_acquisition_with_four_tabs(win):
    assert shown(win) == ["acq"]
    assert [win.tabs.panes[0].bar.tabText(i) for i in range(4)] == \
        ["Acquisition", "Stimulator", "Scope view", "Analysis"]


def test_pick_tabs_and_split(win):
    win.tabs.pick(0, "stim")
    assert shown(win) == ["stim"]
    win.tabs.set_layout(SIDE)
    assert shown(win) == ["stim", "acq"] and win.split_buttons[SIDE].isChecked()
    win.tabs.pick(1, "scope")
    assert shown(win) == ["stim", "scope"]
    win.tabs.set_layout(STACKED)
    assert win.tabs.splitter.orientation().name == "Vertical"
    win.tabs.set_layout(ONE)
    assert shown(win) == ["stim"] and win.split_buttons[ONE].isChecked()


def test_same_tab_in_both_halves_goes_back_to_one_tab(win):
    win.tabs.set_layout(SIDE)
    assert shown(win) == ["acq", "stim"]
    win.tabs.pick(1, "acq")                                  # already open on the left
    assert win.tabs.layout_mode == ONE and shown(win) == ["acq"]


def test_open_stim_window_shows_the_tab(win):
    win.open_stim_window()
    assert win.tabs.is_visible("stim")
    win.tabs.pick(0, "acq")
    win.tabs.set_layout(SIDE)
    win.tabs.pick(1, "scope")
    win.open_stim_window()                                   # puts it in the second half
    assert shown(win) == ["acq", "stim"]


def test_toolbar_follows_visible_tabs(app, win):
    win.resize(1300, 600)       # a narrow window moves toolbar items into its overflow menu
    win.show()
    pump(app, 0.05)
    assert win.comment_edit.isVisible() and win.window_box.isVisible() and win.btn_comments.isVisible()
    win.tabs.pick(0, "stim")
    pump(app, 0.05)
    assert not win.comment_edit.isVisible() and not win.window_box.isVisible()
    assert not win.btn_comments.isVisible()                  # it's part of the Acquisition page
    assert win.btn_run.isVisible() and win.btn_rec.isVisible() and win.rate_box.isVisible()
    win.tabs.set_layout(SIDE)                                # stim | acq
    pump(app, 0.05)
    assert win.comment_edit.isVisible()


# ---------------------------------------------------------------- hover strips
def test_strips_compact_only_when_acquisition_shares_the_screen(win):
    assert not any(s.compact for s in win.strips)
    win.tabs.set_layout(SIDE)
    assert all(s.compact and not s.expanded for s in win.strips)
    assert all(s.width() == 54 for s in win.strips)
    win.tabs.set_layout(ONE)
    assert not any(s.compact for s in win.strips) and all(s.expanded for s in win.strips)


def test_hover_expands_over_the_plot_without_squeezing_it(app, win):
    win.resize(1300, 600)
    win.show()
    win.tabs.set_layout(SIDE)
    pump(app, 0.1)
    rw, strip = win.row_widgets[0], win.strips[0]
    plot_geom = rw.content.geometry()
    strip._set_expanded(True)                                 # what entering the strip does
    assert strip.expanded and strip.width() == 180
    assert rw.content.geometry() == plot_geom                 # plot didn't move
    strip._maybe_collapse()                                    # mouse isn't over it
    assert not strip.expanded and strip.width() == 54


def test_hover_strip_stays_open_while_editing(app, win):
    win.show()
    win.tabs.set_layout(SIDE)
    pump(app, 0.05)
    strip = win.strips[0]
    strip._set_expanded(True)
    strip.name.setFocus()
    pump(app, 0.05)
    strip._maybe_collapse()
    assert strip.expanded
    win.setFocus()
    strip.clearFocus()
    pump(app, 0.05)
    strip._maybe_collapse()
    assert not strip.expanded


def test_rebuilt_strips_keep_compact_mode(win):
    win.tabs.set_layout(SIDE)
    win.add_channel()
    assert all(s.compact for s in win.strips)


# ---------------------------------------------------------------- line thickness
def test_line_thickness_typed_and_clamped(app, win):
    win.strips[1].line_width.setValue(6)
    assert win.cfg.channels[1].width == 6
    assert win.curves[1].opts["pen"].widthF() == 6
    win.set_channel_width(0, 50)
    assert win.cfg.channels[0].width == 20                   # max 20 px
    assert win.strips[0].line_width.maximum() == 20


def test_line_thickness_works_while_recording_and_is_saved(app, win, tmp_path):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    assert win.strips[0].line_width.isEnabled()
    win.set_channel_width(0, 3)
    assert win.recorder.recording and win.backend.running   # no restart
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    back = config.load(config.save(win.cfg, tmp_path / "c.toml"))
    assert back.channels[0].width == 3 and back.channels[1].width == 1.0
    cfg = make_cfg(tmp_path)
    cfg.channels[0].width = 25
    with pytest.raises(ValueError, match="0.5 and 20"):
        config.validate(cfg)


def test_light_mode_keeps_thickness(win):
    win.set_channel_width(0, 4)
    win.set_plot_theme(theme.LIGHT)
    assert win.curves[0].opts["pen"].widthF() == 4


# ---------------------------------------------------------------- whole-app light/dark
def test_one_toggle_switches_window_and_plots(app, win):
    old_sheet, old_pal = app.styleSheet(), app.palette()
    try:
        style.apply(app, style.DARK)
        win.set_plot_theme(theme.DARK)
        assert app.palette().color(QtGui.QPalette.ColorRole.Window).name() == style.TOKENS["dark"]["ground"]
        assert win.plot_widgets[0].backgroundBrush().color().name() == "#000000"
        win.toggle_plot_theme()
        assert app.palette().color(QtGui.QPalette.ColorRole.Window).name() == style.TOKENS["light"]["ground"]
        assert win.plot_widgets[0].backgroundBrush().color().name() == "#ffffff"
        assert win.stim_page.preview.backgroundBrush().color().name() == "#ffffff"
        assert "click for dark mode" in win.btn_theme.toolTip()
    finally:
        style._applied["mode"] = None
        app.setStyleSheet(old_sheet)
        app.setPalette(old_pal)


def test_both_stylesheets_reference_existing_images():
    for mode in (style.LIGHT, style.DARK):
        sheet = style.stylesheet(mode)
        suffix = "-dark" if mode == style.DARK else ""
        for name in ("chevron-down", "chevron-up", "chevron-down-disabled", "chevron-up-disabled"):
            assert (style.ICON_DIR / f"{name}{suffix}.svg").exists()
            assert f"{name}{suffix}.svg" in sheet
        for name in ("check", "check-disabled"):
            assert (style.ICON_DIR / f"{name}.svg").exists() and f"{name}.svg" in sheet


# ---------------------------------------------------------------- stimulator tab and chip
def test_no_stimulator_menu_any_more(win):
    assert [a.text().replace("&", "") for a in win.menuBar().actions()] == ["File", "Setup"]


def test_chip_shows_on_every_tab_while_repeating_and_stops_it(app, win):
    win.resize(1300, 600)
    win.show()
    win.stim_repeat_s = 0.2
    win.set_stimulus(Stimulus(amplitude_v=1, width_ms=1))
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    assert not win.stim_chip.isVisible()
    win.stim_page.btn_repeat.click()
    pump(app, 0.05)
    assert win.stim_chip.isVisible() and "0.2 s" in win.stim_chip_text.text()
    assert "every 0.2 s" in win.stim_chip.toolTip()
    win.tabs.pick(0, "scope")                                # still visible on another tab
    pump(app, 0.05)
    assert win.stim_chip.isVisible()
    win.btn_chip_stop.click()
    pump(app, 0.1)
    assert not win.repeating and not win.stim_chip.isVisible()
    assert not win.stim_page.btn_repeat.isChecked()
    win.btn_run.setChecked(False)


def test_escape_stops_and_ctrl_f_fires(app, win):
    win.set_stimulus(Stimulus(shape="step", amplitude_v=1, duration_ms=2000))
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    assert win.fire_stimulus() and win.stim_out.busy
    win._escape()
    assert not win.stim_out.busy
    win.btn_run.setChecked(False)


def test_stim_tab_output_and_record_on(app, win):
    page = win.open_stim_window()
    page.output.setCurrentText("ao1")
    page.output.activated.emit(page.output.currentIndex())
    assert win.cfg.stim_output == "ao1"
    labels = [page.monitor.itemText(i) for i in range(page.monitor.count())]
    assert labels[0].startswith("None") and "ai0 (used by Ch1)" in labels
    page.monitor.setCurrentIndex(page.monitor.findData("ai3"))
    page._monitor_picked()
    assert win.cfg.stim_monitor == "ai3" and [c.name for c in win.cfg.channels][-1] == "Stim"
    assert page.monitor.currentData() == "ai3"


def test_stimulator_tab_shows_the_stimulus_that_fires(win):
    """Setting the stimulus from outside the tab must update the tab, or editing a
    field would silently swap back to the old one."""
    win.set_stimulus(Stimulus(shape="pulse", amplitude_v=2, width_ms=5, count=4, interval_ms=60))
    page = win.stim_page
    assert page.p_amp.value() == 2 and page.p_count.value() == 4
    page.p_width.setValue(4)                                   # edit one field
    s = win.stimulus
    assert (s.amplitude_v, s.width_ms, s.count, s.interval_ms) == (2, 4, 4, 60)


def test_everything_fits_on_a_1024_px_wide_window(app, win):
    """A 1280 px laptop at 125% scaling is about 1024 px wide."""
    win.resize(1024, 640)
    win.show()
    win.set_stimulus(Stimulus(amplitude_v=1, width_ms=1))
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    win.stim_repeat_s = 12.5                                   # longest chip text
    win.set_repeat(True)                                       # chip showing too
    win.btn_comments.setText("Comments (88)")                  # longest button text
    pump(app, 0.2)
    for w in (win.btn_run, win.btn_rec, win.rate_box, win.window_box, win.comment_edit,
              win.stim_chip, win.btn_comments, win.btn_theme, win.split_group):
        assert w.isVisible(), w                                # not in the overflow menu
        right = w.mapTo(win, w.rect().topRight()).x()
        assert right < win.width(), w
    win.stop_stimulus()
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


def test_split_and_theme_controls_follow_the_last_tab_bar(win):
    corner = win.tab_corner
    assert corner.parent() is win.tabs.panes[0].head_layout.parentWidget()
    win.tabs.set_layout(SIDE)
    assert corner.parent() is win.tabs.panes[1].head_layout.parentWidget()
    win.tabs.set_layout(ONE)
    assert corner.parent() is win.tabs.panes[0].head_layout.parentWidget()


def test_long_comment_labels_are_shortened_on_the_plot(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    long = "Stim: 10 ms delay, Pulse 2 V, 5 ms ×4 every 60 ms"
    win.comment_edit.setText(long)
    win._add_comment()
    label = win.markers[0][1][0].label.format
    assert len(label) <= 32 and label.endswith("…")
    assert win._shown_comments[-1]["text"] == long             # full text is kept
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
