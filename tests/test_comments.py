"""Comments: the box, the panel, and editing comments in recorded blocks."""
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore

from neusci_daq import session, style

from test_channels import app, pump, win  # noqa: F401  (fixtures)


def start_recording(app, w):
    w.btn_run.setChecked(True)
    pump(app, 0.2)
    w.btn_rec.setChecked(True)
    pump(app, 0.2)


def comment(w, app, text):
    w.comment_edit.setText(text)
    w._add_comment()
    pump(app, 0.15)


def texts(path):
    return [c["text"] for c in session.read_comments(path)]


def panel_texts(w):
    tree = w.comments_panel.tree
    return [tree.topLevelItem(i).text(1) for i in range(tree.topLevelItemCount())]


# ---------------------------------------------------------------- the box
def test_comment_box_only_works_while_recording(app, win):
    assert not win.comment_edit.isEnabled()
    assert win.comment_edit.placeholderText() == "Press Record to add comments"
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    assert not win.comment_edit.isEnabled()              # running but not recording
    win.btn_rec.setChecked(True)
    assert win.comment_edit.isEnabled()
    assert win.comment_edit.placeholderText() == "Press Enter to add comment"
    win.btn_rec.setChecked(False)
    assert not win.comment_edit.isEnabled()
    win.btn_run.setChecked(False)


def test_comment_is_saved_as_soon_as_it_is_added(app, win):
    start_recording(app, win)
    comment(win, app, "stimulus on")
    block = win.recorder.path
    assert [c["text"] for c in win.recorder.get_comments()] == ["stimulus on"]   # in the open file
    assert panel_texts(win) == ["stimulus on"]
    assert win.btn_comments.text() == "Comments (1)"
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    assert texts(block) == ["stimulus on"]


def test_record_button_shows_elapsed_time(app, win):
    start_recording(app, win)
    pump(app, 0.3)
    assert win.btn_rec.text().startswith("Stop recording · 0:0")
    win.btn_rec.setChecked(False)
    assert win.btn_rec.text() == "Record"
    win.btn_run.setChecked(False)


def test_c_shortcut_jumps_to_the_box_only_when_usable(app, win):
    win._focus_comment()
    assert not win.comment_edit.hasFocus()
    start_recording(app, win)
    win.resize(1300, 600)        # narrow windows push the box into the toolbar's overflow menu
    win.show()
    win.activateWindow()
    pump(app, 0.05)
    win._focus_comment()
    pump(app, 0.05)
    assert win.focusWidget() is win.comment_edit
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


# ---------------------------------------------------------------- the panel
def test_panel_toggles_and_is_remembered(win, tmp_path):
    from neusci_daq.gui import MainWindow
    from test_channels import make_cfg
    win.show()
    assert win.comments_panel.isVisible()
    win.btn_comments.setChecked(False)
    assert not win.comments_panel.isVisible()
    win.btn_comments.setChecked(True)
    win.comments_panel.closed.emit()                     # its ✕
    assert not win.btn_comments.isChecked()
    w2 = MainWindow(make_cfg(tmp_path), settings=win.settings)
    assert not w2.btn_comments.isChecked()
    w2.close()


def test_edit_and_delete_while_recording(app, win):
    start_recording(app, win)
    comment(win, app, "a")
    comment(win, app, "b")
    comment(win, app, "c")
    win.edit_comment(1, "b edited")
    win.delete_comment(0)
    assert panel_texts(win) == ["b edited", "c"]
    assert len(win.markers) == 2                         # lines on the plots follow
    comment(win, app, "d")                               # adding still works after a rewrite
    block = win.recorder.path
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    assert texts(block) == ["b edited", "c", "d"]


def test_last_block_stays_editable_after_recording(app, win):
    start_recording(app, win)
    comment(win, app, "typo")
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    win.edit_comment(0, "fixed")
    assert texts(win.blocks[-1]) == ["fixed"]


def test_clearing_text_deletes(app, win):
    start_recording(app, win)
    comment(win, app, "x")
    item = win.comments_panel.tree.topLevelItem(0)
    item.setText(1, "   ")                               # what in-place editing does
    pump(app, 0.05)
    assert panel_texts(win) == []
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


# ---------------------------------------------------------------- review
def recorded_session(app, w, tmp_path, comments=("one", "two")):
    start_recording(app, w)
    for c in comments:
        comment(w, app, c)
    pump(app, 0.3)
    w.btn_rec.setChecked(False)
    w.btn_run.setChecked(False)
    folder = tmp_path / "saved"
    w.save_as(folder)
    return folder


def test_review_add_edit_delete_saves_to_the_block(app, win, tmp_path):
    folder = recorded_session(app, win, tmp_path)
    block = session.list_blocks(folder)[0]
    data_before = session.read_block(block)["data"].copy()
    win.open_session(folder)
    assert panel_texts(win) == ["one", "two"] and len(win.markers) == 2

    win.add_comment_at(0.05, "early")                    # e.g. a double-click on the trace
    assert panel_texts(win)[0] == "early"                # kept in time order
    win.edit_comment(1, "ONE")
    win.delete_comment(2)
    assert texts(block) == ["early", "ONE"]
    assert len(win.markers) == 2
    np.testing.assert_array_equal(session.read_block(block)["data"], data_before)
    c = session.read_comments(block)[0]
    assert c["sample_index"] == round(0.05 * win.review_block["sample_rate"])


def test_review_edits_update_the_sessions_mat(app, win, tmp_path):
    from scipy.io import loadmat
    folder = recorded_session(app, win, tmp_path)
    win.open_session(folder)
    win.edit_comment(0, "renamed")
    win.back_to_live()                                   # leaving review re-exports
    m = loadmat(folder / f"{folder.name}.mat")
    assert "renamed" in str(m["comtext"])


def test_jump_centres_the_plot(app, win, tmp_path):
    folder = recorded_session(app, win, tmp_path)
    win.open_session(folder)
    win.window_box.setValue(0.2)
    win.jump_to(0.3)
    lo, hi = win.plots[0].viewRange()[0]
    assert abs((lo + hi) / 2 - 0.3) < 1e-6


def test_double_click_opens_the_popup(app, win, tmp_path):
    folder = recorded_session(app, win, tmp_path)
    win.open_session(folder)
    win.show()
    pump(app, 0.1)
    vb = win.plots[0].vb
    centre = vb.sceneBoundingRect().center()

    class Ev:
        def double(self): return True
        def scenePos(self): return centre

    win._plot_clicked(win.plot_widgets[0], Ev())
    assert win._popup is not None and win._popup.isVisible()
    win._popup.edit.setText("from popup")
    win._popup._submit()
    assert "from popup" in panel_texts(win)


# ---------------------------------------------------------------- style
def test_stylesheet_applies_and_its_images_exist(app, win):
    for name in ("chevron-down.svg", "chevron-up.svg", "chevron-down-disabled.svg", "chevron-up-disabled.svg"):
        assert (style.ICON_DIR / name).exists()
    old = app.styleSheet()
    try:
        app.setStyleSheet(style.stylesheet())
        win.show()
        pump(app, 0.05)
        win.grab()                                       # renders without errors
    finally:
        app.setStyleSheet(old)


def test_nearby_comment_labels_are_staggered(app, win):
    start_recording(app, win)
    for t in ("a", "b", "c", "d"):
        comment(win, app, t)
    ys = [lines[0].label.orthoPos for _, lines in win.markers]
    assert len(set(ys[:3])) == 3 and ys[3] == ys[0]
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


# ---------------------------------------------------------------- live time axis
def axis_end(w):
    return w.plots[0].viewRange()[0][1]


def test_axis_restarts_at_zero_when_recording_starts(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.8)
    assert axis_end(win) > 0.7                          # counting from Start
    win.btn_rec.setChecked(True)
    pump(app, 0.3)
    secs = win.recorder.samples_written / win.backend.actual_rate
    assert abs(axis_end(win) - secs) < 0.1              # 0 = first recorded sample
    assert (win.curves[0].xData < 0).any()              # earlier data shown at negative times
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


def test_comment_lines_match_the_listed_times(app, win):
    start_recording(app, win)
    pump(app, 0.2)
    comment(win, app, "here")
    listed = session_t = win.recorder.get_comments()[0]["t"]
    assert abs(win.markers[0][0] - listed) < 1e-6
    assert win.comments_panel.tree.topLevelItem(0).text(0) == f"{session_t:.1f} s"
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


def test_axis_keeps_counting_after_stop_recording_and_resets_on_next_record(app, win):
    start_recording(app, win)
    comment(win, app, "first block")
    pump(app, 0.2)
    win.btn_rec.setChecked(False)
    end_of_block = axis_end(win)
    pump(app, 0.4)
    assert axis_end(win) > end_of_block + 0.25           # still counting, not reset
    marker_t = win.markers[0][0]
    win.btn_rec.setChecked(True)                         # new block: back to 0
    pump(app, 0.3)
    assert axis_end(win) < 0.5
    assert win.markers and win.markers[0][0] < marker_t - 0.5   # old line moved left with the data
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


def test_restarting_acquisition_clears_old_lines_and_axis(app, win):
    start_recording(app, win)
    comment(win, app, "x")
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    win.btn_run.setChecked(True)
    pump(app, 0.3)
    assert not win.markers and win._axis_zero == 0.0
    win.refresh_comments()                               # last block isn't on screen any more
    assert not win.markers
    win.btn_run.setChecked(False)
