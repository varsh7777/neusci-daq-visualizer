"""Trace colours: choosing them, keeping them, and recolouring recorded blocks."""
import h5py
import numpy as np
import pytest

from neusci_daq import config, session
from neusci_daq.config import PALETTE, Channel

from test_channels import app, edit, make_cfg, pump, win  # noqa: F401  (fixtures)

P = [config.normalize_color(c) for c in PALETTE]
RED = "#ff0000"


def record_block(app, w, secs=0.3):
    if not w.btn_run.isChecked():
        w.btn_run.setChecked(True)
        pump(app, 0.2)
    w.btn_rec.setChecked(True)
    pump(app, secs)
    w.btn_rec.setChecked(False)


def pen_hex(curve):
    return curve.opts["pen"].color().name()


def stored_colors(path):
    with h5py.File(path, "r") as f:
        return [str(c) for c in f.attrs["colors"]]


# ---------------------------------------------------------------- config
def test_color_parsing():
    assert config.normalize_color("#E69F00") == "#e69f00"
    assert config.normalize_color("abc") == "#aabbcc"
    with pytest.raises(ValueError, match="isn't a colour"):
        config.normalize_color("blue-ish")


def test_new_channels_get_unused_palette_colors():
    chans = [Channel("ai0", "A", color=P[1]), Channel("ai1", "B"), Channel("ai2", "C")]
    config.fill_colors(chans)
    assert [c.color for c in chans] == [P[1], P[0], P[2]]


def test_color_saved_with_channel_setup(tmp_path):
    cfg = make_cfg(tmp_path)
    config.fill_colors(cfg.channels)
    cfg.channels[1].color = RED
    back = config.load(config.save(cfg, tmp_path / "c.toml"))
    assert [c.color for c in back.channels] == [P[0], RED]


# ---------------------------------------------------------------- live
def test_colors_stay_put_when_channels_are_added_or_removed(win):
    before = [r.ch.color for r in win.rows]
    assert win.add_channel() and win.add_channel()
    assert [r.ch.color for r in win.rows][:2] == before          # old channels unchanged
    assert len({r.ch.color for r in win.rows}) == 4              # new ones all different
    win.apply_channels(win._copy_rows()[1:])                     # remove Ch1
    assert [r.ch.color for r in win.rows][0] == before[1]


def test_plot_and_strip_use_the_channel_color(win):
    win.set_channel_color(1, RED)
    assert pen_hex(win.curves[1]) == RED
    assert win.strips[1].color_btn.color == RED
    assert win.cfg.channels[1].color == RED


def test_color_change_while_recording_keeps_acquiring_and_updates_block(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    pump(app, 0.2)
    n_before = win.buf_t.size
    win.set_channel_color(0, RED)
    assert win.recorder.recording and win.backend.running        # no restart
    pump(app, 0.2)
    assert win.buf_t.size >= n_before                             # buffer not cleared
    win.btn_rec.setChecked(False)
    assert stored_colors(win.blocks[-1])[0] == RED
    win.btn_run.setChecked(False)


def test_recolor_after_recording_asks_about_old_blocks(app, win):
    record_block(app, win)
    record_block(app, win)
    win.btn_run.setChecked(False)
    asked = []
    win._ask_recolor = lambda name, n: asked.append((name, n)) or (False, False)   # No
    win.set_channel_color(1, RED)
    assert asked == [("Ch2", 2)]
    assert all(stored_colors(b)[1] == P[1] for b in win.blocks)          # unchanged

    win._ask_recolor = lambda name, n: (True, False)                      # Yes
    win.set_channel_color(1, "#00ff00")
    assert all(stored_colors(b)[1] == "#00ff00" for b in win.blocks)
    assert all(stored_colors(b)[0] == P[0] for b in win.blocks)          # other channel untouched


def test_recolor_without_blocks_does_not_ask(win):
    win._ask_recolor = lambda *a: pytest.fail("shouldn't ask")
    win.set_channel_color(0, RED)


# ---------------------------------------------------------------- review and append
def test_review_shows_recorded_colors_and_recolors_whole_session(app, win):
    win.set_channel_color(1, RED)
    record_block(app, win)
    record_block(app, win)
    win.btn_run.setChecked(False)
    folder, blocks = win.session_dir, list(win.blocks)
    data_before = [session.read_block(b)["data"].copy() for b in blocks]

    win.open_session(folder)
    assert pen_hex(win.curves[1]) == RED and win.review_color_btns[1].color == RED

    win.set_review_color(0, "#123456")
    assert pen_hex(win.curves[0]) == "#123456"
    assert all(stored_colors(b) == ["#123456", RED] for b in blocks)
    for b, d in zip(blocks, data_before):                                 # data untouched
        np.testing.assert_array_equal(session.read_block(b)["data"], d)
    win.block_combo.setCurrentIndex(1)                                    # other block too
    assert pen_hex(win.curves[0]) == "#123456"


def test_blocks_without_saved_colors_fall_back_to_palette(app, win):
    record_block(app, win)
    win.btn_run.setChecked(False)
    with h5py.File(win.blocks[-1], "r+") as f:
        del f.attrs["colors"]                     # like a block from an older version
    assert session.read_block(win.blocks[-1])["colors"] == P[:2]
    assert not session.recolor_blocks(win.blocks, {"Ch1": RED})
    assert stored_colors(win.blocks[-1]) == [RED, P[1]]


def test_append_uses_the_sessions_colors(app, win, tmp_path):
    win.set_channel_color(0, RED)
    record_block(app, win)
    win.btn_run.setChecked(False)
    folder = tmp_path / "saved"
    win.save_as(folder)
    win.new_session()
    win.set_channel_color(0, P[3])                 # a different colour in the new session
    win.append_session(folder)
    assert win.rows[0].ch.color == RED and pen_hex(win.curves[0]) == RED


def test_remembered_recolor_choice_lasts_until_a_new_session(app, win):
    record_block(app, win)
    win.btn_run.setChecked(False)
    asked = []
    win._ask_recolor = lambda name, n: asked.append(name) or (True, True)   # Yes, remember
    win.set_channel_color(0, RED)
    win.set_channel_color(0, "#00ff00")                     # not asked again
    win.set_channel_color(1, "#0000ff")                     # any channel
    assert asked == ["Ch1"]
    assert stored_colors(win.blocks[0]) == ["#00ff00", "#0000ff"]

    win.new_session()                                       # new session: asks again
    record_block(app, win)
    win.btn_run.setChecked(False)
    win.set_channel_color(0, P[0])
    assert asked == ["Ch1", "Ch1"]


def test_remembered_never(app, win):
    record_block(app, win)
    win.btn_run.setChecked(False)
    win._ask_recolor = lambda name, n: (False, True)        # No, remember
    win.set_channel_color(0, RED)
    win._ask_recolor = lambda *a: pytest.fail("shouldn't ask again")
    win.set_channel_color(0, "#00ff00")
    assert stored_colors(win.blocks[0])[0] == P[0]


def test_append_notice_lists_rules_and_colour_changes(app, win, tmp_path):
    win.set_channel_color(0, RED)
    record_block(app, win)
    win.btn_run.setChecked(False)
    folder = tmp_path / "saved"
    win.save_as(folder)
    win.new_session()
    win.set_channel_color(0, P[3])
    notes = " ".join(win._append_notes(folder))
    assert "same channel names" in notes and "matched by name" in notes
    assert f"Ch1 → {RED}" in notes                          # custom colour: hex code
    assert config.color_label(P[0]) == "orange" and "Ch2" not in notes.split("switch colour:")[1]


def test_append_notice_can_be_turned_off(win, tmp_path):
    win.settings.setValue("skip_append_notice", True)
    assert win._confirm_append(tmp_path)                    # no dialog shown
