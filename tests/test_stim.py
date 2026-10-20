"""The stimulator: shapes, safety, the NI output task, and the menu."""
import sys
import time
import types

import numpy as np
import pytest

from neusci_daq import config, session
from neusci_daq.stim import Stimulus, check_limit, load_wave_file

from test_channels import app, make_cfg, pump, win  # noqa: F401  (fixtures)

RATE = 5000.0


# ---------------------------------------------------------------- shapes
def test_single_pulse():
    w = Stimulus(shape="pulse", amplitude_v=2, width_ms=1, delay_ms=2).waveform(RATE)
    assert len(w) == 10 + 5 + 1                       # 2 ms delay, 1 ms pulse, final 0 V
    assert np.all(w[:10] == 0) and np.all(w[10:15] == 2) and w[-1] == 0


def test_train_and_biphasic():
    w = Stimulus(shape="pulse", amplitude_v=1, width_ms=1, count=3, interval_ms=4).waveform(RATE)
    starts = np.flatnonzero(np.diff(np.r_[0, w]) > 0)
    assert list(starts) == [0, 20, 40] and w.sum() == 15
    b = Stimulus(shape="pulse", amplitude_v=1.5, width_ms=1, biphasic=True).waveform(RATE)
    assert np.all(b[:5] == 1.5) and np.all(b[5:10] == -1.5) and b[-1] == 0


def test_step_ramp_sine():
    s = Stimulus(shape="step", amplitude_v=-0.5, duration_ms=10).waveform(RATE)
    assert np.all(s[:-1] == -0.5) and len(s) == 51
    r = Stimulus(shape="ramp", start_v=0, end_v=4, duration_ms=10).waveform(RATE)
    assert r[0] == 0 and r[-2] == pytest.approx(4) and np.all(np.diff(r[:-1]) > 0)
    n = Stimulus(shape="sine", amplitude_v=1, freq_hz=100, duration_ms=10, offset_v=0.5).waveform(RATE)
    assert n[:-1].max() == pytest.approx(1.5, abs=0.01) and n[:-1].min() == pytest.approx(-0.5, abs=0.01)


def test_custom_segments_make_any_shape():
    segs = [{"duration_ms": 2, "start_v": 0, "end_v": 2},    # ramp up
            {"duration_ms": 2, "start_v": 2, "end_v": 2},    # hold
            {"duration_ms": 1, "start_v": -1, "end_v": -1}]  # jump negative
    w = Stimulus(shape="custom", segments=segs).waveform(RATE)
    assert len(w) == 10 + 10 + 5 + 1
    assert w[0] == 0 and np.all(w[10:20] == 2) and np.all(w[20:25] == -1) and w[-1] == 0


def test_waveform_files(tmp_path):
    one = tmp_path / "one.csv"
    one.write_text("volts\n0\n1\n2\n1\n")
    w = Stimulus(shape="file", file=str(one), file_rate=1000).waveform(RATE)
    assert w.max() == pytest.approx(2) and len(w) == 3 * 5 + 1 + 1    # 3 ms resampled, + 0 V
    two = tmp_path / "two.txt"
    two.write_text("# time volts\n0 0\n0.002 3\n")
    w2 = load_wave_file(two, RATE, RATE)
    assert len(w2) == 11 and w2[-1] == pytest.approx(3) and w2[5] == pytest.approx(1.5)
    bad = tmp_path / "bad.csv"
    bad.write_text("0,1\n1\n")
    with pytest.raises(ValueError, match="same 1 or 2 columns"):
        load_wave_file(bad, RATE, RATE)


def test_problems_are_explained():
    with pytest.raises(ValueError, match="shorter than one output sample"):
        Stimulus(shape="pulse", width_ms=0.1).waveform(RATE)
    with pytest.raises(ValueError, match="Interval"):
        Stimulus(shape="pulse", width_ms=2, count=2, interval_ms=1).waveform(RATE)
    with pytest.raises(ValueError, match="half the output rate"):
        Stimulus(shape="sine", freq_hz=3000).waveform(RATE)
    with pytest.raises(ValueError, match="limit is 60"):
        Stimulus(shape="step", duration_ms=61000).waveform(RATE)
    with pytest.raises(ValueError, match="5 V limit"):
        check_limit(Stimulus(amplitude_v=6).waveform(RATE), 5.0)
    assert Stimulus(width_ms=0.3).rounding_notes(RATE) == \
        ["Width 0.3 ms rounds to 0.4 ms (output steps are 0.2 ms)."]
    assert len(Stimulus(width_ms=0.3).waveform(RATE)) == 2 + 1     # notes match what's sent
    assert len(Stimulus(width_ms=0.5).waveform(RATE)) == 3 + 1     # halves round up, not to even


def test_save_load_and_describe(tmp_path):
    s = Stimulus(shape="pulse", amplitude_v=2, width_ms=1, count=5, interval_ms=100)
    assert s.describe() == "Pulse 2 V, 1 ms ×5 every 100 ms"
    back = Stimulus.load(s.save(tmp_path / "s.json"))
    assert back == s
    (tmp_path / "junk.json").write_text("not json")
    with pytest.raises(ValueError, match="isn't a saved stimulus"):
        Stimulus.load(tmp_path / "junk.json")


def test_config_stimulator_section(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.stim_output, cfg.stim_max_v, cfg.stim_monitor = "ao1", 8.0, "ai3"
    back = config.load(config.save(cfg, tmp_path / "c.toml"))
    assert (back.stim_output, back.stim_rate, back.stim_max_v, back.stim_monitor) == ("ao1", 5000, 8.0, "ai3")
    cfg.stim_rate = 10000
    with pytest.raises(ValueError, match="5,000"):
        config.validate(cfg)


# ---------------------------------------------------------------- NI output task
class FakeTask:
    log = []

    def __init__(self):
        self.done_at = None
        FakeTask.log.append(("task",))
        self.ao_channels = types.SimpleNamespace(add_ao_voltage_chan=self._add)
        self.timing = types.SimpleNamespace(cfg_samp_clk_timing=self._timing)

    def _add(self, name, min_val, max_val):
        FakeTask.log.append(("chan", name, min_val, max_val))

    def _timing(self, rate, sample_mode, samps_per_chan):
        FakeTask.log.append(("timing", rate, sample_mode, samps_per_chan))
        self.duration = samps_per_chan / rate

    def write(self, data, auto_start):
        FakeTask.log.append(("write", np.atleast_1d(np.asarray(data, float)).copy(), auto_start))

    def start(self):
        self.done_at = time.monotonic() + self.duration
        FakeTask.log.append(("start",))

    def is_task_done(self):
        return time.monotonic() >= self.done_at

    def stop(self):
        FakeTask.log.append(("stop",))

    def close(self):
        FakeTask.log.append(("close",))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def fake_nidaqmx(monkeypatch):
    mod = types.ModuleType("nidaqmx")
    mod.Task = FakeTask
    consts = types.ModuleType("nidaqmx.constants")
    consts.AcquisitionType = types.SimpleNamespace(FINITE="FINITE", CONTINUOUS="CONTINUOUS")
    monkeypatch.setitem(sys.modules, "nidaqmx", mod)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", consts)
    FakeTask.log = []
    return FakeTask.log


def test_nidaqmx_output_sends_a_finite_hardware_timed_task(fake_nidaqmx, tmp_path):
    from neusci_daq.stimout import NidaqmxOutput
    cfg = make_cfg(tmp_path)
    cfg.device, cfg.stim_output = "Dev1", "ao1"
    out = NidaqmxOutput(cfg)
    wave = Stimulus(amplitude_v=2, width_ms=1).waveform(RATE)
    out.fire(wave, RATE)
    deadline = time.time() + 2
    while out.busy and time.time() < deadline:
        time.sleep(0.01)
    assert out.error is None
    kinds = [e[0] for e in fake_nidaqmx]
    assert kinds == ["task", "chan", "timing", "write", "start", "stop", "close"]
    assert fake_nidaqmx[1] == ("chan", "Dev1/ao1", -10.0, 10.0)
    assert fake_nidaqmx[2][1:] == (RATE, "FINITE", len(wave))
    np.testing.assert_array_equal(fake_nidaqmx[3][1], wave)
    assert fake_nidaqmx[3][2] is False


def test_nidaqmx_stop_cuts_it_short_and_sets_zero(fake_nidaqmx, tmp_path):
    from neusci_daq.stimout import NidaqmxOutput
    out = NidaqmxOutput(make_cfg(tmp_path))
    out.fire(Stimulus(shape="step", amplitude_v=1, duration_ms=5000).waveform(RATE), RATE)
    time.sleep(0.1)
    t0 = time.time()
    out.stop()
    assert time.time() - t0 < 1 and not out.busy
    writes = [e for e in fake_nidaqmx if e[0] == "write"]
    assert writes[-1][1].tolist() == [0.0] and writes[-1][2] is True     # left at 0 V


def test_nidaqmx_error_is_reported_and_output_zeroed(fake_nidaqmx, tmp_path, monkeypatch):
    from neusci_daq.stimout import NidaqmxOutput
    monkeypatch.setattr(FakeTask, "start", lambda self: (_ for _ in ()).throw(RuntimeError("DAQmx -200xxx")))
    out = NidaqmxOutput(make_cfg(tmp_path))
    out.fire(Stimulus().waveform(RATE), RATE)
    out._thread.join(2)
    assert "200xxx" in str(out.error)
    assert [e for e in fake_nidaqmx if e[0] == "write"][-1][1].tolist() == [0.0]


# ---------------------------------------------------------------- the menu, simulated
def test_fire_needs_acquisition_running(win):
    assert not win.fire_stimulus()
    assert "Press Start" in win.statusBar().currentMessage()


def test_record_stimulus_on_adds_and_removes_a_stim_channel(app, win):
    assert win.set_stim_monitor("ai3")
    assert [c.name for c in win.cfg.channels] == ["Ch1", "Ch2", "Stim"]
    assert win.cfg.channels[2].physical == "ai3" and win.cfg.stim_monitor == "ai3"
    assert not win.set_stim_monitor("ai0")                  # used by Ch1
    assert win.set_stim_monitor("ai4")                      # move it
    assert [c.physical for c in win.cfg.channels] == ["ai0", "ai1", "ai4"]
    assert win.set_stim_monitor("")
    assert [c.name for c in win.cfg.channels] == ["Ch1", "Ch2"] and win.cfg.stim_monitor == ""


def test_fired_stimulus_is_recorded_on_the_stim_channel_and_logged(app, win):
    win.set_stim_monitor("ai3")
    win.set_stimulus(Stimulus(shape="step", amplitude_v=3, duration_ms=100, delay_ms=20))
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    pump(app, 0.2)
    assert win.fire_stimulus()
    pump(app, 0.4)
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)
    blk = session.read_block(win.blocks[-1])
    stim = blk["data"][2]
    assert stim.max() == pytest.approx(3, abs=0.02)
    assert (stim > 1.5).sum() == pytest.approx(0.1 * win.cfg.sample_rate, abs=3)   # 100 ms at the AI rate
    assert any(text.startswith("Stim:") and "Step 3 V" in text for _, text in blk["comments"])
    assert blk["data"][0].max() > 0.5                       # the fake response on the other channels


def test_limit_and_busy_are_refused_with_a_message(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.set_stimulus(Stimulus(amplitude_v=9))
    assert not win.fire_stimulus() and "limit" in win.statusBar().currentMessage()
    win.set_stimulus(Stimulus(shape="step", amplitude_v=1, duration_ms=1000))
    assert win.fire_stimulus()
    assert not win.fire_stimulus() and "still running" in win.statusBar().currentMessage()
    win.stop_stimulus()
    assert not win.stim_out.busy
    win.btn_run.setChecked(False)


def test_repeat_fires_until_stopped(app, win):
    win.stim_repeat_s = 0.15
    win.set_stimulus(Stimulus(amplitude_v=1, width_ms=1))
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.act_repeat.setChecked(True)
    pump(app, 0.55)
    n = win.stim_out.fired
    assert n >= 3
    win.btn_run.setChecked(False)                          # Stop also stops repeating
    assert not win.act_repeat.isChecked()
    pump(app, 0.3)
    assert win.stim_out.fired == n


def test_output_choice_and_stimulus_are_remembered(win, tmp_path):
    from neusci_daq.gui import MainWindow
    win.set_stim_output("ao1")
    assert win.cfg.stim_output == "ao1" and win.act_stim_out["ao1"].isChecked()
    win.set_stimulus(Stimulus(shape="ramp", end_v=2))
    w2 = MainWindow(make_cfg(tmp_path), settings=win.settings)
    assert w2.stimulus.shape == "ramp" and w2.stimulus.end_v == 2
    w2.close()


# ---------------------------------------------------------------- the window
def test_window_fields_build_the_stimulus_and_preview(app, win):
    sw = win.open_stim_window()
    sw.shape.setCurrentIndex(sw.shape.findData("pulse"))
    sw.p_amp.setValue(2.5)
    sw.p_width.setValue(2)
    sw.p_count.setValue(3)
    sw.p_interval.setValue(10)
    s = win.stimulus
    assert (s.shape, s.amplitude_v, s.width_ms, s.count, s.interval_ms) == ("pulse", 2.5, 2, 3, 10)
    x, y = sw.curve.getData()
    assert y.max() == pytest.approx(2.5) and "Peak 2.5 V" in sw.message.text()


def test_window_custom_table_and_errors(app, win):
    sw = win.open_stim_window()
    sw.shape.setCurrentIndex(sw.shape.findData("custom"))
    sw._add_segment()
    assert win.stimulus.shape == "custom" and len(win.stimulus.segments) == 4
    sw.table.item(0, 0).setText("abc")
    assert "must be a number" in sw.message.text()
    sw.table.item(0, 0).setText("5")
    sw.shape.setCurrentIndex(sw.shape.findData("pulse"))
    sw.p_amp.setValue(8)                                   # over the 5 V limit
    assert "limit" in sw.message.text()


def test_loading_a_stimulus_updates_preview_and_what_fire_sends(app, win, tmp_path):
    sw = win.open_stim_window()
    custom = Stimulus(shape="custom", segments=[{"duration_ms": 10, "start_v": 0, "end_v": 3},
                                                {"duration_ms": 10, "start_v": -1, "end_v": -1}])
    sw.set_stimulus(custom)
    assert win.stimulus.shape == "custom" and win.stimulus.segments == custom.segments
    x, y = sw.curve.getData()
    assert y.min() == pytest.approx(-1) and y.max() < 3
