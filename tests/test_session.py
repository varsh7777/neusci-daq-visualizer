"""Sessions on disk, plus the File menu's New / Open / Append / Export / Recent."""
import time

import numpy as np
import pytest

from neusci_daq import config, session


def make_cfg(tmp_path, n=2, rate=2000, scale=1000.0):
    chans = [config.Channel(f"ai{i}", f"Ch{i}", "mV", scale=scale) for i in range(n)]
    return config.Config(backend="simulated", sample_rate=rate, chunk_size=100,
                         channels=chans, data_dir=str(tmp_path / "data"))


# ---------------------------------------------------------------- GUI helpers
@pytest.fixture
def app():
    import pyqtgraph as pg
    return pg.mkQApp()


@pytest.fixture
def settings(tmp_path):
    from pyqtgraph.Qt import QtCore
    return QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.Format.IniFormat)


def pump(app, s):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def record_blocks(app, w, n, secs=0.4, comment=None):
    w.btn_run.setChecked(True)
    pump(app, 0.2)
    for i in range(n):
        w.btn_rec.setChecked(True)
        pump(app, secs)
        if comment and i == 0:
            w.comment_edit.setText(comment)
            w._add_comment()
            pump(app, 0.1)
        w.btn_rec.setChecked(False)
        pump(app, 0.1)
    w.btn_run.setChecked(False)


# ---------------------------------------------------------------- pure helpers
def test_list_blocks_orders_by_number_and_ignores_others(tmp_path):
    for name in ["block_010.h5", "block_002.h5", "block_001.h5", "notes.txt", "block_x.h5", "s.mat"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in session.list_blocks(tmp_path)] == ["block_001.h5", "block_002.h5", "block_010.h5"]
    assert session.next_block_path(tmp_path).name == "block_011.h5"   # never reuses a gap
    assert session.next_block_path(tmp_path / "empty").name == "block_001.h5"
    assert not session.is_session(tmp_path / "empty")


def test_add_recent_dedupes_and_limits(tmp_path):
    r = []
    for i in range(10):
        r = session.add_recent(r, tmp_path / f"s{i}")
    r = session.add_recent(r, tmp_path / "s5")
    assert len(r) == 8 and r[0].endswith("s5") and len(set(r)) == 8


# ---------------------------------------------------------------- GUI flows
def test_record_save_open_append_export_recent(tmp_path, app, settings):
    import neuscitk
    from neusci_daq.gui import MainWindow, REVIEW, LIVE

    cfg = make_cfg(tmp_path)
    w = MainWindow(cfg, settings=settings)

    # Record without naming anything: blocks go straight to data/unsaved/
    record_blocks(app, w, 2, comment="TTX applied")
    unsaved = w.session_dir
    assert unsaved.parent.name == "unsaved" and not w.session_saved
    assert "Untitled session*" in w.windowTitle()
    assert len(session.list_blocks(unsaved)) == 2

    # First Save (as): moves the folder, writes the .mat, remembers it
    sess = w.save_as(tmp_path / "my_prep")
    assert not unsaved.exists() and w.session_saved
    assert [p.name for p in session.list_blocks(sess)] == ["block_001.h5", "block_002.h5"]
    assert len(neuscitk.LabChartDataset(str(sess / "my_prep.mat")).data["Channel 1"]) == 2
    assert w.recent_sessions()[0] == str(sess)
    assert "my_prep — Live" in w.windowTitle()

    # Keep recording into the saved session; Save updates the .mat
    record_blocks(app, w, 1)
    assert [p.name for p in session.list_blocks(sess)][-1] == "block_003.h5"
    assert w.save()
    assert len(neuscitk.LabChartDataset(str(sess / "my_prep.mat")).data["Channel 1"]) == 3

    # Save As refuses a non-empty folder; a copy keeps the original
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "x.txt").write_text("x")
    with pytest.raises(session.SessionError, match="isn't empty"):
        w.save_as(tmp_path / "taken")
    copy = w.save_as(tmp_path / "my_prep_copy")
    assert session.list_blocks(sess) and len(session.list_blocks(copy)) == 3
    assert (copy / "my_prep_copy.mat").exists() and not (copy / "my_prep.mat").exists()

    # Open: review mode shows block 1 with its comment
    w.open_session(sess)
    assert w.mode == REVIEW and w.block_combo.count() == 3
    blk = session.read_block(sess / "block_001.h5")
    x, y = w.curves[0].getOriginalDataset()
    assert len(x) == blk["data"].shape[1]
    np.testing.assert_allclose(y, blk["data"][0])
    assert len(w.markers) == 1 and blk["comments"][0][1] == "TTX applied"
    w.block_combo.setCurrentIndex(1)
    assert len(w.markers) == 0  # block 2 has no comments

    # scrollbar moves the view
    w.window_box.setValue(0.1)
    w.scroll.setValue(w.scroll.maximum())
    lo, hi = w.plots[0].viewRange()[0]
    assert hi == pytest.approx(blk_dur := session.block_info(sess / "block_002.h5")["duration_s"], abs=0.01)

    # Export from review
    out = w.export(tmp_path / "review.mat")
    assert len(neuscitk.LabChartDataset(str(out)).data["Channel 1"]) == 3

    # Append: back to live, new block continues numbering, old blocks untouched
    before = (sess / "block_001.h5").stat().st_mtime
    w.append_session(sess)
    assert w.mode == LIVE and w.session_saved
    record_blocks(app, w, 1)
    assert [p.name for p in session.list_blocks(sess)][-1] == "block_004.h5"
    assert (sess / "block_001.h5").stat().st_mtime == before
    assert w.close()  # saved session: no prompt, auto-exports
    ds = neuscitk.LabChartDataset(str(sess / "my_prep.mat"))
    assert len(ds.data["Channel 1"]) == 4

    # Recent sessions remembered across windows
    w2 = MainWindow(cfg, settings=settings)
    assert str(sess) in w2.recent_sessions()
    w2.close()


def test_append_refuses_mismatched_config(tmp_path, app, settings):
    from neusci_daq.gui import MainWindow

    w = MainWindow(make_cfg(tmp_path), settings=settings)
    record_blocks(app, w, 1)
    sess = w.save_as(tmp_path / "s")
    w.close()

    problems, _ = session.append_check(sess, make_cfg(tmp_path, n=3))
    assert any("channels differ" in p for p in problems)
    problems, _ = session.append_check(sess, make_cfg(tmp_path, rate=4000))
    assert any("Sample rate" in p for p in problems)
    problems, warnings = session.append_check(sess, make_cfg(tmp_path, scale=10000.0))
    assert not problems and any("gain" in x for x in warnings)

    w2 = MainWindow(make_cfg(tmp_path, scale=10000.0), settings=settings)
    with pytest.raises(session.SessionError):
        w2.append_session(sess)            # warning needs confirmation
    w2.append_session(sess, force=True)    # confirmed
    w2.close()


def test_open_recent_drops_missing(tmp_path, app, settings):
    from neusci_daq.gui import MainWindow
    from pyqtgraph.Qt import QtWidgets

    w = MainWindow(make_cfg(tmp_path), settings=settings)
    settings.setValue("recent_sessions", [str(tmp_path / "gone")])
    orig = QtWidgets.QMessageBox.warning
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    try:
        w.open_recent(str(tmp_path / "gone"))
    finally:
        QtWidgets.QMessageBox.warning = orig
    assert w.recent_sessions() == []
    w.close()


def test_review_handles_session_with_different_channels(tmp_path, app, settings):
    """Open works even when the session's channels differ from the current config."""
    from neusci_daq.gui import MainWindow

    w = MainWindow(make_cfg(tmp_path, n=3), settings=settings)
    record_blocks(app, w, 1)
    sess = w.save_as(tmp_path / "three")
    w.close()

    w2 = MainWindow(make_cfg(tmp_path, n=1), settings=settings)
    w2.open_session(sess)
    assert len(w2.plots) == 3
    w2.back_to_live()
    assert len(w2.plots) == 1
    w2.close()


def test_unsaved_session_prompts_before_leaving(tmp_path, app, settings):
    from neusci_daq.gui import MainWindow

    w = MainWindow(make_cfg(tmp_path), settings=settings)
    record_blocks(app, w, 1)
    unsaved = w.session_dir

    w._ask_unsaved = lambda: "cancel"
    assert not w.new_session() and w.session_dir == unsaved   # nothing changed
    assert w.close() is False                                  # window stays open

    w._ask_unsaved = lambda: "discard"
    assert w.new_session() and w.session_dir is None
    assert session.list_blocks(unsaved)                        # "Don't Save" deletes nothing
    assert w.close()


def test_save_without_recording_does_nothing(tmp_path, app, settings):
    from neusci_daq.gui import MainWindow

    w = MainWindow(make_cfg(tmp_path), settings=settings)
    assert w.save() is False
    with pytest.raises(session.SessionError, match="Nothing"):
        w.save_as(tmp_path / "x")
    w.close()


def test_restart_discards_leftover_chunks_from_previous_run(tmp_path, app, settings):
    """After Stop then Start, data left over from the old run must not be drawn.

    Each run's timestamps start again from 0, so an old chunk (with large
    timestamps) would be drawn on top of the new run, joined to it by a long line.
    """
    from neusci_daq.backends import Chunk
    from neusci_daq.gui import MainWindow

    w = MainWindow(make_cfg(tmp_path), settings=settings)
    w.btn_run.setChecked(True)
    pump(app, 0.3)
    w.btn_run.setChecked(False)
    # A chunk from the old run, still waiting when Start is pressed again.
    n = len(w.cfg.channels)
    w.display_q.put(Chunk(np.zeros((n, 100)), 100.0 + np.arange(100) / w.cfg.sample_rate, 200_000))
    w.btn_run.setChecked(True)
    pump(app, 0.3)
    w.btn_run.setChecked(False)
    assert w.buf_t.size, "nothing was acquired after restarting"
    assert np.all(np.diff(w.buf_t) > 0), "time went backwards in the display buffer"
    assert w.buf_t.max() < 100, "data from the previous run was drawn"
    w.close()