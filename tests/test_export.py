"""The .mat export must load unchanged in neuscitk's LabChartDataset."""
import time

import numpy as np
import pytest
from scipy.io import loadmat

from neusci_daq import config
from neusci_daq.backends import SimulatedBackend
from neusci_daq.export import to_labchart_mat
from neusci_daq.recorder import Recorder, load_recording

neuscitk = pytest.importorskip("neuscitk")


def record_blocks(tmp_path, n_channels, n_blocks, comment=None):
    chans = [config.Channel(f"ai{i}", f"Ch{i}", "mV", scale=1000.0) for i in range(n_channels)]
    cfg = config.Config(backend="simulated", sample_rate=2000, chunk_size=100, channels=chans)
    be = SimulatedBackend(cfg)
    rec = Recorder(be)
    be.start()
    paths = []
    for b in range(n_blocks):
        p = rec.start(tmp_path / f"block_{b + 1:03d}.h5")
        time.sleep(0.3 + 0.1 * b)  # blocks of different lengths
        if comment and b == 0:
            rec.add_comment(comment)
            time.sleep(0.1)
        rec.stop()
        paths.append(p)
        time.sleep(0.1)
    be.stop()
    return paths


@pytest.mark.parametrize("n_channels,n_blocks", [(4, 3), (1, 1), (2, 1), (1, 2)])
def test_neuscitk_loads_export(tmp_path, n_channels, n_blocks):
    paths = record_blocks(tmp_path, n_channels, n_blocks)
    out = to_labchart_mat(paths, tmp_path / "session.mat")
    ds = neuscitk.LabChartDataset(str(out))

    assert ds.n_channels == n_channels
    assert ds.fs == 2000
    for b, p in enumerate(paths):
        expected = load_recording(p)["data"]  # scaled, (n_ch, n)
        for c in range(n_channels):
            got = ds.data[f"Channel {c + 1}"][b]
            np.testing.assert_allclose(got, expected[c], rtol=1e-6)
    # the call students actually use
    blk = ds.get_block(0)
    assert blk.shape[-1] == load_recording(paths[0])["data"].shape[1]


def test_metadata_and_comments(tmp_path):
    paths = record_blocks(tmp_path, 2, 2, comment="TTX applied")
    out = to_labchart_mat(paths, tmp_path / "s.mat")
    m = loadmat(out)
    assert [t.strip() for t in m["titles"]] == ["Ch0", "Ch1"]
    assert [u.strip() for u in m["unittext"]] == ["mV"]
    assert m["com"].shape == (1, 5) and m["com"][0, 1] == 1  # block 1
    assert m["comtext"][0].strip() == "TTX applied"
    assert m["blocktimes"][0, 1] > m["blocktimes"][0, 0] > 700_000  # sane datenums


def test_rejects_mismatched_channels(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = record_blocks(tmp_path / "a", 2, 1)
    b = record_blocks(tmp_path / "b", 3, 1)
    with pytest.raises(ValueError, match="different channels"):
        to_labchart_mat(a + b, tmp_path / "x.mat")
