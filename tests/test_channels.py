"""Changing channels from inside the app."""
import time

import numpy as np
import pytest

from neusci_daq import config, session
from neusci_daq.config import Channel


def make_cfg(tmp_path, n=2, rate=2000):
    chans = [Channel(f"ai{i}", f"Ch{i + 1}") for i in range(n)]
    return config.Config(backend="simulated", sample_rate=rate, chunk_size=100,
                         channels=chans, data_dir=str(tmp_path / "data"))


@pytest.fixture
def app():
    import pyqtgraph as pg
    return pg.mkQApp()


@pytest.fixture
def win(tmp_path, app):
    from pyqtgraph.Qt import QtCore
    from neusci_daq.gui import MainWindow
    st = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)
    w = MainWindow(make_cfg(tmp_path), settings=st, config_path=tmp_path / "config.toml")
    w._ask_unsaved = lambda: "discard"
    yield w
    w.close()


def pump(app, s):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def edit(w, i, what, value):
    rows = w._copy_rows()
    setattr(rows[i].ch, what, value)
    return w.apply_channels(rows)


# ---------------------------------------------------------------- config
def test_config_rejects_duplicate_inputs_and_names():
    c = config.Config(backend="simulated", channels=[Channel("ai0", "A"), Channel("ai0", "B")])
    with pytest.raises(ValueError, match="both use ai0"):
        config.validate(c)
    c = config.Config(backend="simulated", channels=[Channel("ai0", "A"), Channel("ai1", "A")])
    with pytest.raises(ValueError, match="named"):
        config.validate(c)
    c = config.Config(backend="simulated", spare_channels=[Channel("ai0", "A", enabled=False)])
    with pytest.raises(ValueError, match="At least one"):
        config.validate(c)


def test_setup_save_and_load_round_trip(tmp_path):
    cfg = config.Config(backend="simulated", sample_rate=20000, terminal_config="DIFF").with_channels([
        Channel("ai0", "Leg nerve", "µV", scale=10000),
        Channel("ai1", "Stim \"drive\"", "V", enabled=False),
    ])
    back = config.load(config.save(cfg, tmp_path / "setup.toml"))
    assert [(c.physical, c.name, c.units, c.scale) for c in back.channels] == [("ai0", "Leg nerve", "µV", 10000)]
    assert [(c.physical, c.name, c.enabled) for c in back.spare_channels] == [("ai1", 'Stim "drive"', False)]
    assert back.sample_rate == 20000 and back.terminal_config == "DIFF"


# ---------------------------------------------------------------- GUI
def test_each_plot_has_its_own_input(win, app):
    assert [s.input.currentText() for s in win.strips] == ["ai0", "ai1"]
    assert len(win.plots) == 2
    win.btn_run.setChecked(True)
    pump(app, 0.3)
    assert win.buf_y.shape[0] == 2

    assert edit(win, 1, "physical", "ai3")          # Ch2 now reads ai3
    assert win.btn_run.isChecked() and win.backend.running   # acquisition restarted
    assert [c.physical for c in win.cfg.channels] == ["ai0", "ai3"]
    assert win.backend.cfg is win.cfg and win.recorder.cfg is win.cfg
    pump(app, 0.3)
    assert win.buf_y.shape[0] == 2 and win.buf_t.size > 0


def test_switch_off_add_and_remove(win, app):
    assert edit(win, 0, "enabled", False)
    assert len(win.plots) == 1 and len(win.strips) == 2       # Ch1's controls stay
    assert [c.name for c in win.cfg.channels] == ["Ch2"]
    assert [c.name for c in win.cfg.spare_channels] == ["Ch1"]

    assert win.add_channel()                                   # picks the first free input
    assert [r.ch.physical for r in win.rows] == ["ai0", "ai1", "ai2"]
    assert len(win.plots) == 2

    win._channel_removed(win.strips[2])
    assert [r.ch.physical for r in win.rows] == ["ai0", "ai1"]

    assert not edit(win, 1, "enabled", False)                  # can't turn every channel off
    assert [c.name for c in win.cfg.channels] == ["Ch2"]


def test_bad_choices_are_refused_and_controls_restored(win):
    assert not edit(win, 1, "physical", "ai0")                 # already used by Ch1
    assert [c.physical for c in win.cfg.channels] == ["ai0", "ai1"]
    assert win.strips[1].input.currentText() == "ai1"
    assert "both use ai0" in win.statusBar().currentMessage()
    assert not edit(win, 1, "name", "Ch1")                     # duplicate name


def test_channels_locked_while_recording(win, app):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    pump(app, 0.2)
    # Everything is locked except the colour, which is display-only.
    assert not any(w.isEnabled() for s in win.strips for w in (s.on, s.name, s.input, s.remove))
    assert all(s.color_btn.isEnabled() for s in win.strips) and not win.btn_add_ch.isEnabled()
    assert not edit(win, 1, "physical", "ai2")
    assert [c.physical for c in win.cfg.channels] == ["ai0", "ai1"]
    win.btn_rec.setChecked(False)
    assert all(w.isEnabled() for s in win.strips for w in (s.on, s.name, s.input, s.remove))
    win.btn_run.setChecked(False)


def test_changing_channels_after_recording_starts_new_session(win, app):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    pump(app, 0.3)
    win.btn_rec.setChecked(False)
    first = win.session_dir
    assert first is not None and win.blocks

    win._confirm_new_session = lambda what: False      # Cancel
    assert not edit(win, 1, "physical", "ai2")
    assert win.session_dir == first

    win._confirm_new_session = lambda what: True      # OK
    assert edit(win, 1, "physical", "ai2")
    assert win.session_dir is None and not win.blocks          # fresh session
    assert win.btn_run.isChecked()                             # still acquiring
    win.btn_rec.setChecked(True)
    pump(app, 0.3)
    win.btn_rec.setChecked(False)
    info = session.block_info(win.blocks[-1])
    assert info["physical"] == ["ai0", "ai2"] and info["terminal_config"] == "RSE"
    assert session.list_blocks(first)                          # old blocks untouched
    win.btn_run.setChecked(False)


def test_rename_only_keeps_acquiring_without_new_session(win, app):
    win.btn_run.setChecked(True)
    pump(app, 0.3)
    assert edit(win, 0, "name", "Leg nerve")
    assert win.plots[0].getAxis("left").labelText == "Leg nerve"
    assert win.buf_t.size > 0                                  # display buffer kept
    win.btn_run.setChecked(False)


def test_review_mode_hides_channel_controls(win, app, tmp_path):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    pump(app, 0.3)
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    sess = win.save_as(tmp_path / "s")
    win.open_session(sess)
    assert win.strips == [] and win.channel_bar.isHidden()
    assert len(win.plots) == 2
    win.back_to_live()
    assert len(win.strips) == 2 and not win.channel_bar.isHidden()
