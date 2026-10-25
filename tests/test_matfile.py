"""Opening LabChart-style .mat files (LabChart's own exports, and ours) for review."""
import hashlib
import json
import time

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from neusci_daq import matfile, session

from test_channels import app, make_cfg, pump, win  # noqa: F401  (fixtures)


def labchart_like(path, **override):
    """A file shaped like LabChart's 'Export as MATLAB': 2 channels, 3 blocks.
    Block 2: channel 2 switched off. Block 3: channel 2 recorded at half the rate."""
    a1 = np.arange(1000, dtype=np.float32) / 1000          # block 1, ch1: 1 s at 1 kHz
    a2 = -np.arange(1000, dtype=np.float32)                 # block 1, ch2
    b1 = np.full(500, 2.0, dtype=np.float32)                # block 2, ch1 only
    c1 = np.sin(np.arange(1000) / 1000 * 2 * np.pi).astype(np.float32)   # block 3, ch1 at 1 kHz
    c2 = np.linspace(0, 1, 500).astype(np.float32)                        # block 3, ch2 at 500 Hz
    data = np.concatenate([a1, a2, b1, c1, c2])
    starts = np.array([[1, 2001, 2501], [1001, -1, 3501]], float)
    ends = np.array([[1000, 2500, 3500], [2000, -1, 4000]], float)
    d = {
        "data": data.reshape(1, -1),
        "datastart": starts, "dataend": ends,
        "samplerate": np.array([[1000, 1000, 1000], [1000, 1000, 500]], float),
        "tickrate": np.array([[1000, 1000, 1000]], float),
        "titles": np.array(["Nerve  ", "Stim   "]),                     # char matrix, padded
        "unittext": np.array(["mV", "V "]),
        "unittextmap": np.array([[1, 1, 1], [2, 2, 2]], float),
        "blocktimes": np.array([[739000.5, 739000.6, 739000.7]]),       # MATLAB datenums
        "com": np.array([[-1, 1, 251, 1, 1], [2, 1, 501, 1, 2], [-1, 3, 1, 1, 3]], float),
        "comtext": np.array(["start      ", "on stim only", "block 3    "]),
    }
    d.update(override)
    savemat(path, d)
    return path


# ---------------------------------------------------------------- reading
def test_reads_a_labchart_export(tmp_path):
    blocks = matfile.read_labchart_mat(labchart_like(tmp_path / "lab.mat"))
    assert len(blocks) == 3
    b1, b2, b3 = blocks
    assert b1["names"] == ["Nerve", "Stim"] and b1["units"] == ["mV", "V"] and b1["rate"] == 1000
    assert b1["data"].shape == (2, 1000) and b1["data"][0, 500] == pytest.approx(0.5)
    assert b1["data"][1, 10] == -10
    assert b1["comments"] == [(0.25, "start"), (0.5, "on stim only [Stim]")]
    assert b1["start_time"].startswith("2023-")


def test_a_channel_switched_off_in_a_block_is_left_out(tmp_path):
    b2 = matfile.read_labchart_mat(labchart_like(tmp_path / "lab.mat"))[1]
    assert b2["names"] == ["Nerve"] and b2["data"].shape == (1, 500)


def test_slower_channels_are_resampled_to_the_fastest_and_noted(tmp_path):
    b3 = matfile.read_labchart_mat(labchart_like(tmp_path / "lab.mat"))[2]
    assert b3["rate"] == 1000 and b3["data"].shape == (2, 1000)
    assert b3["data"][1, 500] == pytest.approx(c2_at(0.5), abs=0.01)
    assert any("500 Hz" in n for n in b3["notes"])
    assert b3["comments"] == [(0.0, "block 3")]


def c2_at(t):
    return t / (499 / 500)        # linspace(0, 1, 500) at 500 Hz, at time t


def test_minimal_file_with_only_data_and_positions(tmp_path):
    p = tmp_path / "min.mat"
    savemat(p, {"data": np.arange(10.0), "datastart": np.array([[1.0]]), "dataend": np.array([[10.0]]),
                "samplerate": np.array([[100.0]])})
    (b,) = matfile.read_labchart_mat(p)
    assert b["names"] == ["Channel 1"] and b["units"] == [""] and b["comments"] == [] and b["start_time"] is None


def test_duplicate_channel_names_are_told_apart(tmp_path):
    p = labchart_like(tmp_path / "dup.mat", titles=np.array(["EMG", "EMG"]))
    assert matfile.read_labchart_mat(p)[0]["names"] == ["EMG", "EMG (2)"]


@pytest.mark.parametrize("what,msg", [
    ("not_labchart", "isn't a LabChart export"),
    ("v73", "MATLAB 7.3"),
    ("garbage", "couldn't be read"),
    ("past_end", "past the end"),
])
def test_problems_are_explained(tmp_path, what, msg):
    p = tmp_path / f"{what}.mat"
    if what == "not_labchart":
        savemat(p, {"x": np.arange(5)})
    elif what == "v73":
        with h5py.File(p, "w") as f:          # MATLAB 7.3 files are HDF5 with a 512-byte header
            f["data"] = np.arange(5.0)
        raw = p.read_bytes()
        p.write_bytes(b"MATLAB 7.3 MAT-file".ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM"
                      + b"\x00" * (512 - 128) + raw)
    elif what == "garbage":
        p.write_bytes(b"not a mat file at all")
    else:
        labchart_like(p, dataend=np.array([[1000, 2500, 9999], [2000, -1, 4000]], float))
    with pytest.raises(matfile.MatError, match=msg):
        matfile.read_labchart_mat(p)


# ---------------------------------------------------------------- importing
def test_import_makes_a_session_and_never_changes_the_file(tmp_path):
    src = labchart_like(tmp_path / "lab.mat")
    before = hashlib.md5(src.read_bytes()).hexdigest()
    folder, notes = matfile.import_mat(src, tmp_path / "imported")
    assert folder == tmp_path / "imported" / "lab"
    blocks = session.list_blocks(folder)
    assert [b.name for b in blocks] == ["block_001.h5", "block_002.h5", "block_003.h5"]
    blk = session.read_block(blocks[0])
    assert blk["channel_names"] == ["Nerve", "Stim"] and blk["units"] == ["mV", "V"]
    assert [c[1] for c in blk["comments"]] == ["start", "on stim only [Stim]"]
    info = session.block_info(blocks[0])
    assert info["backend"] == "imported" and len(set(info["colors"])) == 2
    assert session.block_info(blocks[1])["colors"][0] == info["colors"][0]   # same colour in every block
    assert notes and "Block 3" in notes[0]
    assert hashlib.md5(src.read_bytes()).hexdigest() == before


def test_importing_again_reuses_the_copy_unless_the_file_changed(tmp_path):
    src = labchart_like(tmp_path / "lab.mat")
    folder, _ = matfile.import_mat(src, tmp_path / "imp")
    session.write_comments(session.list_blocks(folder)[0], [{"t": 0.1, "sample_index": 100, "text": "mine"}])
    again, notes = matfile.import_mat(src, tmp_path / "imp")
    assert again == folder and notes                        # notes remembered too
    assert session.read_comments(session.list_blocks(again)[0])[0]["text"] == "mine"
    time.sleep(0.01)
    labchart_like(src, titles=np.array(["A", "B"]))         # a different file under the same name
    newer, _ = matfile.import_mat(src, tmp_path / "imp")
    assert newer == tmp_path / "imp" / "lab_2"
    assert session.read_block(session.list_blocks(newer)[0])["channel_names"] == ["A", "B"]
    assert json.loads((newer / matfile.MARKER).read_text())["path"] == str(src.resolve())


def test_a_failed_import_leaves_nothing_behind(tmp_path):
    p = tmp_path / "bad.mat"
    savemat(p, {"x": 1})
    with pytest.raises(matfile.MatError):
        matfile.import_mat(p, tmp_path / "imp")
    assert not (tmp_path / "imp" / "bad").exists()


# ---------------------------------------------------------------- in the window
def test_our_own_export_opens_back_up_with_the_same_data(app, win, tmp_path):
    win.btn_run.setChecked(True)
    pump(app, 0.3)
    win.btn_rec.setChecked(True)
    pump(app, 0.5)
    win.comment_edit.setText("hello")
    win._add_comment()
    pump(app, 0.3)
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    original = session.read_block(win.blocks[0])
    out = win.export(tmp_path / "ours.mat")
    win.open_mat(out)
    blk = win.review_block
    assert win.mode == "review" and "ours.mat (imported copy)" in win.windowTitle()
    assert blk["channel_names"] == original["channel_names"]
    assert np.allclose(blk["data"], original["data"], atol=1e-5)
    assert [c[1] for c in blk["comments"]] == ["hello"]
    assert blk["comments"][0][0] == pytest.approx(original["comments"][0][0], abs=1e-3)
    assert win.recent_sessions()[0] == str(out.resolve())


def test_comments_added_in_review_go_to_the_copy_not_the_mat(app, win, tmp_path):
    src = labchart_like(tmp_path / "lab.mat")
    before = hashlib.md5(src.read_bytes()).hexdigest()
    folder = win.open_mat(src)
    assert win.block_combo.count() == 3
    win.add_comment_at(0.75, "added in review")
    win.back_to_live()
    win.open_recent(str(src.resolve()))                       # reopen from File › Recent
    assert win.review_dir == folder
    assert "added in review" in [c[1] for c in win.review_block["comments"]]
    assert hashlib.md5(src.read_bytes()).hexdigest() == before


def test_a_bad_file_is_refused_with_a_message(win, tmp_path):
    from neusci_daq.session import SessionError
    p = tmp_path / "x.mat"
    savemat(p, {"x": 1})
    with pytest.raises(SessionError, match="isn't a LabChart export"):
        win.open_mat(p)
    assert win.mode == "live"


def test_units_that_already_have_a_prefix_are_not_prefixed_again(win, tmp_path):
    src = labchart_like(tmp_path / "lab.mat")
    win.open_mat(src)
    ax = win.plots[0].getAxis("left")                         # Nerve, in mV
    assert not ax.autoSIPrefix and "mV" in ax.labelText and "kmV" not in ax.labelString()
    assert win.plots[1].getAxis("left").labelUnits == "V"   # plain V still gets mV, µV, kV
