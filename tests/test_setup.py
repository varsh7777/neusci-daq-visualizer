"""Changing the sampling rate and input wiring from inside the app."""
import pytest

from neusci_daq import config, session
from neusci_daq.config import Channel

from test_channels import app, edit, make_cfg, pump, win  # noqa: F401  (fixtures)


def record_block(app, w):
    w.btn_run.setChecked(True)
    pump(app, 0.2)
    w.btn_rec.setChecked(True)
    pump(app, 0.3)
    w.btn_rec.setChecked(False)
    w.btn_run.setChecked(False)


def type_rate(w, text):
    w.rate_box.setEditText(text)
    w._rate_entered()


# ---------------------------------------------------------------- sampling rate
def test_rate_text_is_parsed():
    from neusci_daq.gui import MainWindow
    p = MainWindow._parse_rate
    assert p("1,000 Hz") == 1000 and p("20k") == 20000 and p(" 500 ") == 500 and p("2.5 kHz") == 2500
    assert p("abc") is None and p("0") is None and p("-5") is None


def test_rate_change_updates_config_read_size_and_display(win):
    type_rate(win, "5000")
    assert win.cfg.sample_rate == 5000
    assert win.cfg.chunk_size == 250          # ~50 ms per read
    assert win.cap == int(win.window_box.value() * 5000)
    assert win.rate_box.currentText() == "5,000 Hz"


def test_rate_over_the_limit_is_refused_with_the_max(win):
    type_rate(win, "100000")                  # 2 channels -> max 50,000
    assert win.cfg.sample_rate == 2000
    assert "50,000" in win.statusBar().currentMessage()
    assert win.rate_box.currentText() == "2,000 Hz"   # box put back


def test_rate_garbage_is_refused(win):
    type_rate(win, "fast")
    assert win.cfg.sample_rate == 2000
    assert "isn't a sampling rate" in win.statusBar().currentMessage()


def test_rate_change_restarts_acquisition_at_new_rate(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    type_rate(win, "1000")
    assert win.btn_run.isChecked() and win.backend.running
    pump(app, 0.3)
    dt = win.buf_t[1] - win.buf_t[0]
    assert dt == pytest.approx(1 / 1000)
    win.btn_run.setChecked(False)


def test_rate_locked_while_recording(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    assert not win.rate_box.isEnabled() and not win.wiring_menu.isEnabled()
    assert win.apply_setup(sample_rate=1000) is False
    win.btn_rec.setChecked(False)
    assert win.rate_box.isEnabled() and win.wiring_menu.isEnabled()
    win.btn_run.setChecked(False)


def test_rate_change_after_recording_starts_new_session(app, win):
    record_block(app, win)
    old = win.session_dir
    win._confirm_new_session = lambda what: True
    assert win.apply_setup(sample_rate=1000)
    assert win.blocks == [] and win.session_dir is None
    assert session.list_blocks(old)              # old blocks untouched on disk


def test_rate_change_cancelled_keeps_session(app, win):
    record_block(app, win)
    win._confirm_new_session = lambda what: False
    assert not win.apply_setup(sample_rate=1000)
    assert win.cfg.sample_rate == 2000 and len(win.blocks) == 1


# ---------------------------------------------------------------- wiring
def test_wiring_switch_and_status_label(win):
    assert "RSE" in win.wiring_label.text() and win.act_wiring["RSE"].isChecked()
    assert win.apply_setup(terminal_config="DIFF")
    assert win.cfg.terminal_config == "DIFF"
    assert "DIFF" in win.wiring_label.text() and win.act_wiring["DIFF"].isChecked()
    assert [win.strips[0].input.itemText(i) for i in range(win.strips[0].input.count())] == \
        ["ai0", "ai1", "ai2", "ai3"]


def test_switch_to_diff_refused_when_a_channel_uses_ai4_to_ai7(win):
    edit(win, 1, "physical", "ai5")
    assert not win.apply_setup(terminal_config="DIFF")
    assert win.cfg.terminal_config == "RSE"
    assert win.act_wiring["RSE"].isChecked()          # menu tick put back
    assert "Ch2 (ai5)" in win.statusBar().currentMessage()


def test_switched_off_channel_on_ai6_also_blocks_diff(win):
    edit(win, 1, "physical", "ai6")
    edit(win, 1, "enabled", False)
    assert not win.apply_setup(terminal_config="DIFF")


def test_wiring_and_rate_are_saved_with_the_channel_setup(tmp_path, win):
    win.apply_setup(terminal_config="DIFF")
    win.apply_setup(sample_rate=1000)
    p = config.save(win.cfg, tmp_path / "setup.toml")
    back = config.load(p)
    assert back.terminal_config == "DIFF" and back.sample_rate == 1000 and back.chunk_size == 50


def test_review_shows_recorded_wiring(app, win):
    win.apply_setup(terminal_config="DIFF")
    record_block(app, win)
    win.open_session(win.session_dir)
    assert "recorded in DIFF" in win.wiring_label.text()
    win.back_to_live()
    assert "DIFF wiring" in win.wiring_label.text()


def test_max_rate_helper():
    assert config.max_rate(1) == 100_000 and config.max_rate(3) == 33_333
    assert config.auto_chunk_size(10) == 1 and config.auto_chunk_size(20000) == 1000
