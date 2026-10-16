import time
import warnings

import numpy as np
import pytest

from neusci_daq import config
from neusci_daq.backends import SimulatedBackend
from neusci_daq.recorder import Recorder, load_recording


def make_cfg(**kw):
    c = config.Config(backend="simulated", sample_rate=1000, chunk_size=50,
                      channels=[config.Channel("ai0", "A", "mV", scale=2.0, offset=1.0),
                                config.Channel("ai1", "B")], **kw)
    config.validate(c)
    return c


def test_record_roundtrip(tmp_path):
    be = SimulatedBackend(make_cfg())
    rec = Recorder(be)
    be.start()
    time.sleep(0.2)
    rec.start(tmp_path / "r.h5")
    time.sleep(0.5)
    rec.add_comment("drug on")
    time.sleep(0.2)
    rec.stop()
    be.stop()
    assert be.error is None
    r = load_recording(tmp_path / "r.h5")
    assert r["data"].shape[0] == 2 and r["data"].shape[1] > 300
    np.testing.assert_allclose(r["data"][0], r["raw_volts"][0] * 2 + 1)
    assert r["comments"]["text"][0] in (b"drug on", "drug on")
    assert list(r["attrs"]["channel_names"]) == ["A", "B"]
    assert r["t"][0] == 0


def test_rejects_over_aggregate_rate():
    c = make_cfg()
    c.backend, c.sample_rate = "nidaqmx", 60_000
    with pytest.raises(ValueError, match="aggregate"):
        config.validate(c)


def test_diff_limits_channels():
    c = make_cfg()
    c.backend, c.terminal_config = "nidaqmx", "DIFF"
    c.channels[1].physical = "ai5"
    with pytest.raises(ValueError, match="ai0-ai3"):
        config.validate(c)


def test_warns_on_fake_gain():
    c = make_cfg()
    c.backend = "nidaqmx"
    c.channels[0].range = (-0.1, 0.1)
    with pytest.warns(UserWarning, match="does NOT add gain"):
        config.validate(c)


def test_example_config_loads():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = config.load("config.toml")
    assert cfg.backend == "nidaqmx" and len(cfg.channels) == 2
