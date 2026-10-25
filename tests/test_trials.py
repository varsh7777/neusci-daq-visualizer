"""Trials: protocols that step one or two stimulus settings, lined up and measured."""
import csv
import json

import h5py
import numpy as np
import pytest

from neusci_daq import trials as T
from neusci_daq.stim import Stimulus

from test_channels import app, make_cfg, pump, win  # noqa: F401  (fixtures)


# ---------------------------------------------------------------- values
def test_range_never_goes_past_to():
    assert T.range_values(0.5, 3.0, 0.5) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert T.range_values(0.5, 3.0, 0.7) == [0.5, 1.2, 1.9, 2.6]          # stops short of 3.0


def test_range_has_no_rounding_drift_and_counts_down():
    v = T.range_values(0.1, 1.0, 0.1)
    assert len(v) == 10 and v[-1] == 1.0
    assert T.range_values(3, 0.5, 0.5) == [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    assert T.range_values(2, 2, 1) == [2.0]


@pytest.mark.parametrize("step", [0, -0.5])
def test_range_needs_a_positive_step(step):
    with pytest.raises(ValueError, match="more than 0"):
        T.range_values(0, 1, step)


def test_range_has_a_limit():
    with pytest.raises(ValueError, match="most is"):
        T.range_values(0, 1000, 1)


def test_list_takes_commas_or_spaces_in_the_order_typed():
    assert T.parse_list("2, 0.5  1;3") == [2.0, 0.5, 1.0, 3.0]


@pytest.mark.parametrize("text,msg", [("", "at least one"), ("1, x", "isn't a number"),
                                      ("1 2 1", "twice")])
def test_list_mistakes_are_explained(text, msg):
    with pytest.raises(ValueError, match=msg):
        T.parse_list(text)


# ---------------------------------------------------------------- order
def _p(order, repeats=2):
    return T.Protocol(a=T.Variable("amplitude_v"), b=T.Variable("width_ms"), repeats=repeats, order=order)


def test_blocked_runs_all_repeats_of_a_combination_together():
    seq = T.sequence(_p("blocked"), [1, 2], [5, 6])
    assert [(t.ia, t.ib) for t in seq] == [(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1), (1, 1), (1, 1)]
    assert [t.n for t in seq] == list(range(1, 9))


def test_interleaved_runs_every_combination_once_per_repeat():
    seq = T.sequence(_p("interleaved"), [1, 2], [5, 6])
    assert [(t.rep, t.ia, t.ib) for t in seq[:4]] == [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    assert [t.rep for t in seq[4:]] == [2] * 4


def test_random_shuffles_within_each_repeat():
    seq = T.sequence(_p("random", repeats=5), [1, 2, 3], [5, 6], np.random.default_rng(3))
    for r in range(1, 6):
        combos = sorted((t.ia, t.ib) for t in seq if t.rep == r)
        assert combos == sorted((i, j) for i in range(3) for j in range(2))
    assert [(t.ia, t.ib) for t in seq] != [(t.ia, t.ib) for t in T.sequence(_p("interleaved", 5), [1, 2, 3], [5, 6])]


def test_one_variable_has_no_b():
    p = T.Protocol(a=T.Variable("amplitude_v"), b=None, repeats=1)
    seq = T.sequence(p, [1, 2, 3], None)
    assert [t.b for t in seq] == [None] * 3


# ---------------------------------------------------------------- applying values
def test_values_change_the_waveform_for_every_shape():
    rate = 5000
    w, delay = T.make_wave(Stimulus(shape="pulse", delay_ms=4, width_ms=1), _p("blocked"), 2.0, 0.4, rate)
    assert w.max() == 2.0 and np.count_nonzero(w) == 2 and delay == 4
    p = T.Protocol(a=T.Variable("count"))
    w, _ = T.make_wave(Stimulus(shape="pulse", width_ms=1, interval_ms=10), p, 3, None, rate)
    assert np.count_nonzero(np.diff(np.concatenate([[0.0], w])) > 0) == 3   # 3 rising edges
    p = T.Protocol(a=T.Variable("freq_hz"))
    w, _ = T.make_wave(Stimulus(shape="sine", amplitude_v=1, duration_ms=100), p, 20, None, rate)
    assert abs(np.sum(np.diff(np.sign(w[:-1])) > 0) - 2) <= 1          # 2 cycles in 100 ms
    p = T.Protocol(a=T.Variable("scale"))
    base = Stimulus(shape="custom").waveform(rate)
    w, _ = T.make_wave(Stimulus(shape="custom"), p, 0.5, None, rate)
    assert np.allclose(w, base * 0.5)


def test_count_must_be_whole():
    with pytest.raises(ValueError, match="whole number"):
        T.apply(Stimulus(shape="pulse"), "count", 2.5)


# ---------------------------------------------------------------- checks
def test_check_counts_trials_time_and_peak():
    p = T.Protocol(a=T.Variable("amplitude_v", "range", 0.5, 2, 0.5), b=T.Variable("width_ms", "list", text="0.2 1"),
                   repeats=3, gap_s=2)
    c = T.check(p, Stimulus(shape="pulse", delay_ms=5), 5000, 5)
    assert c.ok and c.n_trials == 24 and c.duration_s == 48 and c.peak_v == 2


def test_check_refuses_values_over_the_safety_limit():
    p = T.Protocol(a=T.Variable("amplitude_v", "range", 1, 8, 1))
    c = T.check(p, Stimulus(shape="pulse"), 5000, 5)
    assert not c.ok and "above the 5 V limit" in c.errors[0]


def test_check_warns_when_values_round_to_the_same_output():
    p = T.Protocol(a=T.Variable("width_ms", "list", text="0.3 0.4 1"))
    c = T.check(p, Stimulus(shape="pulse"), 5000, 5)
    assert c.ok and "both round to 0.4 ms" in c.warnings[0]


def test_check_refuses_trials_that_would_overlap():
    p = T.Protocol(a=T.Variable("duration_ms", "list", text="100 500"), gap_s=0.3)
    c = T.check(p, Stimulus(shape="step"), 5000, 5)
    assert not c.ok and "would overlap" in c.errors[0]


def test_check_refuses_settings_that_dont_fit_the_shape_and_same_a_b():
    c = T.check(T.Protocol(a=T.Variable("freq_hz")), Stimulus(shape="pulse"), 5000, 5)
    assert "isn't a setting" in c.errors[0]
    c = T.check(T.Protocol(a=T.Variable("amplitude_v"), b=T.Variable("amplitude_v")), Stimulus(), 5000, 5)
    assert any("same setting" in e for e in c.errors)


def test_protocol_saves_and_loads():
    p = T.Protocol(name="x", a=T.Variable("width_ms", "list", text="1 2"), b=T.Variable("amplitude_v"),
                   repeats=4, order="random", gap_s=1.5, measure="latency")
    assert T.Protocol.loads(p.dumps()) == p
    assert T.Protocol.loads(T.Protocol(b=None).dumps()).b is None
    with pytest.raises(ValueError, match="isn't a saved protocol"):
        T.Protocol.loads("{nope")


# ---------------------------------------------------------------- finding and measuring
def test_finds_the_stimulus_exactly_whatever_its_size_and_offset():
    rng = np.random.default_rng(0)
    wave, _ = T.make_wave(Stimulus(shape="pulse", delay_ms=5, width_ms=0.4), T.Protocol(), 2.0, None, 5000)
    tpl, pad = T.template(wave, 5000, 10000)
    x = 0.003 * rng.standard_normal(20000) + 0.7          # an offset on the input
    body = tpl[pad:-pad]
    x[12345:12345 + len(body)] += 0.3 * body              # recorded at 30%
    k, match = T.find_stimulus(x, tpl)
    assert k + pad == 12345 and match > 0.95


def test_finds_a_stimulus_with_no_delay_from_its_edge():
    wave = Stimulus(shape="step", amplitude_v=1, duration_ms=20).waveform(5000)
    tpl, pad = T.template(wave, 5000, 10000)
    x = np.zeros(5000)
    x[1000:1200] = 1.0
    k, match = T.find_stimulus(x, tpl)
    assert k + pad == 1000 and match > 0.99


def test_no_stimulus_means_no_match():
    tpl, _ = T.template(Stimulus().waveform(5000), 5000, 10000)
    _, match = T.find_stimulus(np.random.default_rng(1).standard_normal(5000) * 0.01, tpl)
    assert match < T.MATCH_THRESHOLD


def test_cut_and_measure():
    rate = 1000.0
    t = np.arange(1000) / rate
    data = np.vstack([np.where((t > 0.502) & (t < 0.505), 2.0, 0.0), t])
    t_ms, w = T.cut(t, data, 0.5, 10, 20, rate)
    assert t_ms[0] == -10 and t_ms[-1] == 20 and w.shape == (2, 31)
    assert T.measure("p2p", t_ms, w[0], 0, 10) == 2.0
    assert T.measure("max", t_ms, w[0], 0, 10) == 2.0 and T.measure("min", t_ms, w[0], 0, 10) == 0.0
    assert T.measure("latency", t_ms, w[0], 0, 10) == 3.0
    assert np.isnan(T.measure("p2p", t_ms, w[0], 50, 60))
    assert T.cut(t, data, 0.005, 10, 20, rate) is None        # not enough data before


def test_stats_ignore_missing_values():
    assert T.stats([1, 3, float("nan")]) == (2, 2.0, pytest.approx(1.4142, abs=1e-4))
    assert T.stats([5]) == (1, 5.0, 0.0)
    assert T.stats([])[0] == 0


def test_capture_never_takes_another_trials_stimulus():
    """Two pulses that differ only in size match equally well, so each trial must
    only look between its own send and the next one's. (Looking a fixed 0.25 s
    back, trial 2 used to find trial 1's pulse.)"""
    rate, out = 10000.0, 5000.0
    p = T.Protocol(a=T.Variable("amplitude_v"), gap_s=0.2, before_ms=5, after_ms=20)
    s = Stimulus(shape="pulse", delay_ms=5, width_ms=1)
    cap = T.Capture(rate, 0, lambda raw: raw, keep_s=10, read_s=0.05)
    t = np.arange(int(2 * rate)) / rate
    x = np.zeros_like(t)
    sent_at = {1: 0.40, 2: 0.60}                   # newest recorded sample when each was sent
    starts = {1: 0.42, 2: 0.62}                    # when each really started
    amps = {1: 1.0, 2: 2.0}
    for n, t0 in starts.items():
        i = int(round(t0 * rate))
        x[i + 50:i + 60] = amps[n]                 # 5 ms delay, 1 ms pulse
    done = 0
    for n in (1, 2):
        upto = int(round(sent_at[n] * rate))
        cap.add(t[done:upto], x[None, done:upto])
        done = upto
        w, d = T.make_wave(s, p, amps[n], None, out)
        cap.sent(T.Trial(n, 1, n - 1, 0, amps[n], None), w, out, d)
    cap.add(t[done:], x[None, done:])
    got = {r.trial.n: r for r in cap.resolve(p, force=True)}
    for n in (1, 2):
        assert got[n].found
        assert got[n].t0 == pytest.approx(starts[n] + 0.005, abs=1e-4)
        assert got[n].data[0].max() == amps[n]     # its own pulse, not the other one


def test_capture_waits_for_the_recording_to_move_on_before_the_next_trial():
    cap = T.Capture(1000, 0, lambda r: r, keep_s=5, read_s=0.05)
    assert cap.can_send(0.5)
    cap.add(np.arange(100) / 1000, np.zeros((1, 100)))
    w = np.ones(10)
    cap.sent(T.Trial(1, 1, 0, 0, 1, None), w, 5000, 0)
    assert not cap.can_send(0.5)                     # the clock moved on, the recording didn't
    cap.add(np.arange(100, 600) / 1000, np.zeros((1, 500)))
    assert cap.can_send(0.5)


# ---------------------------------------------------------------- in the window
def _ready(app, win, protocol):
    win.set_stim_monitor("ai3")
    win.set_stimulus(Stimulus(shape="pulse", delay_ms=5, amplitude_v=1, width_ms=1))
    win.trials_page._load_protocol_into_widgets(protocol)
    win.btn_run.setChecked(True)
    pump(app, 0.3)
    return win.trials_page


def _run_to_end(app, tp, limit=20):
    import time
    t0 = time.time()
    while tp.running and time.time() - t0 < limit:
        pump(app, 0.03)


def test_trials_tab_replaces_the_placeholder(win):
    assert win.tabs.pages["trials"] is win.trials_page
    assert win.tabs.panes[0].bar.tabText(2) == "Trials"


def test_run_needs_start_and_a_recorded_stimulus(app, win):
    tp = win.trials_page
    tp._load_protocol_into_widgets(T.Protocol(a=T.Variable("amplitude_v", "range", 0.5, 1, 0.5), repeats=1))
    assert not tp.btn_run.isEnabled() and "Record the stimulus" in tp.summary.text()
    assert not tp.run_protocol()
    win.set_stim_monitor("ai3")
    assert tp.btn_run.isEnabled()
    assert not tp.run_protocol() and "Press Start" in tp.message.text()


def test_a_full_protocol_lines_up_every_trial_and_saves_it(app, win, tmp_path):
    p = T.Protocol(name="sd", a=T.Variable("amplitude_v", "range", 0.5, 1.5, 0.5),
                   b=T.Variable("width_ms", "list", text="0.4 1"), repeats=2, order="random",
                   gap_s=0.3, before_ms=5, after_ms=20)
    tp = _ready(app, win, p)
    win.btn_rec.setChecked(True)
    pump(app, 0.1)
    assert tp.run_protocol()
    pump(app, 0.4)
    assert tp.running and win._tb_chip.isVisible() and win.stim_chip_text.text().startswith("Trial ")
    assert not win.strips[0].name.isEnabled() or not win.btn_add_ch.isEnabled()   # channels locked
    _run_to_end(app, tp)
    assert not tp.running and len(tp.results) == 12
    assert all(r.found for r in tp.results.values()), tp.message.text()
    mon = tp._run["channels"].index("Stim")
    for r in tp.results.values():                 # each lined up on its own stimulus
        k0 = int(np.argmin(np.abs(r.t_ms)))
        assert abs(r.data[mon][k0:k0 + 30].max() - r.trial.a) < 0.05
        assert abs(r.data[mon][:k0 - 2]).max() < 0.05
    assert "12 of 12" in tp.message.text()
    path = win.recorder.path
    win.btn_rec.setChecked(False)
    with h5py.File(path) as f:
        saved = json.loads(f.attrs["trials"])
        texts = [c["text"].decode() if isinstance(c["text"], bytes) else c["text"] for c in f["comments"][:]]
    assert saved["protocol"]["name"] == "sd" and len(saved["trials"]) == 12
    assert all(t["found"] and t["t"] > 0 for t in saved["trials"])
    assert texts[0].startswith("Trial 1/12: a = ")
    out = tp.export_csv(tmp_path / "t.csv")
    rows = list(csv.reader(out.open()))
    assert rows[0][:4] == ["trial", "repeat", "a: Amplitude (V)", "b: Pulse width (ms)"] and len(rows) == 13
    win.btn_run.setChecked(False)


def test_leaving_a_trial_out_changes_the_average_and_curve(app, win):
    tp = _ready(app, win, T.Protocol(a=T.Variable("amplitude_v", "list", text="1 2"), repeats=2, gap_s=0.25,
                                     after_ms=20))
    assert tp.run_protocol()
    _run_to_end(app, tp)
    assert len(tp.results) == 4
    r = tp.results[1]
    item = tp.table.item(0, 0)
    item.setCheckState(item.checkState().__class__.Unchecked)
    assert not r.use
    tp.show_view("curve")
    assert tp.stack.currentIndex() == 2
    win.btn_run.setChecked(False)


def test_escape_and_the_chip_stop_a_protocol_and_keep_the_trials_so_far(app, win):
    tp = _ready(app, win, T.Protocol(a=T.Variable("amplitude_v", "range", 0.5, 2, 0.5), repeats=3, gap_s=0.3,
                                     after_ms=20))
    assert tp.run_protocol()
    pump(app, 1.0)
    win._escape()
    assert not tp.running and not win._tb_chip.isVisible()
    assert 1 <= len(tp.results) < 12 and "Protocol stopped" in tp.message.text()
    assert win.btn_add_ch.isEnabled()                 # channels unlocked again
    assert tp.run_protocol()
    pump(app, 0.5)
    win.btn_chip_stop.click()
    assert not tp.running
    win.btn_run.setChecked(False)


def test_stopping_acquisition_stops_the_protocol(app, win):
    tp = _ready(app, win, T.Protocol(a=T.Variable("amplitude_v", "range", 0.5, 2, 0.5), repeats=3, gap_s=0.3,
                                     after_ms=20))
    assert tp.run_protocol()
    pump(app, 0.5)
    win.btn_run.setChecked(False)
    pump(app, 0.1)
    assert not tp.running


def test_fire_repeat_and_channel_changes_are_refused_while_running(app, win):
    tp = _ready(app, win, T.Protocol(a=T.Variable("amplitude_v", "range", 0.5, 2, 0.5), repeats=3, gap_s=0.3,
                                     after_ms=20))
    assert tp.run_protocol()
    assert not win.fire_stimulus()
    win.set_repeat(True)
    assert not win.repeating
    assert not win.add_channel()
    assert len(win.cfg.channels) == 3
    win.stop_stimulus()
    win.btn_run.setChecked(False)


def test_settings_are_locked_while_running(app, win):
    tp = _ready(app, win, T.Protocol(a=T.Variable("amplitude_v", "range", 0.5, 2, 0.5), repeats=3, gap_s=0.3,
                                     after_ms=20))
    assert tp.run_protocol()
    assert not tp.row_a.setting.isEnabled() and not tp.repeats.isEnabled() and not tp.btn_run.isEnabled()
    assert tp.btn_pause.isEnabled() and tp.btn_stop.isEnabled()
    tp.btn_pause.click()
    assert tp.paused and win.stim_chip_text.text().startswith("Paused")
    n = tp.next_i
    pump(app, 0.8)
    assert tp.next_i == n                              # nothing sent while paused
    tp.btn_pause.click()
    pump(app, 0.5)
    assert tp.next_i > n
    win.stop_stimulus()
    assert tp.row_a.setting.isEnabled()
    win.btn_run.setChecked(False)


def test_shape_follows_the_stimulator_and_offers_its_settings(win):
    tp = win.trials_page
    win.set_stimulus(Stimulus(shape="sine"))
    assert tp.shape.currentData() == "sine"
    keys = [tp.row_a.setting.itemData(i) for i in range(tp.row_a.setting.count())]
    assert keys == [k for k, _, _ in T.SETTINGS["sine"]]
    tp.shape.setCurrentIndex(tp.shape.findData("step"))
    tp._shape_picked()
    assert win.stimulus.shape == "step" and win.stim_page.shape.currentData() == "step"


def test_range_or_list_for_each_variable_and_b_can_be_added_and_removed(win):
    tp = win.trials_page
    win.set_stimulus(Stimulus(shape="pulse"))
    tp.btn_add_b.click()
    assert tp.protocol.b is not None and tp.protocol.b.key != tp.protocol.a.key
    tp.row_b.mode.buttons["list"].click()
    tp.row_b.list_edit.setText("0.2, 1")
    tp.row_b.list_edit.editingFinished.emit()
    assert tp.protocol.b.values() == [0.2, 1.0] and "2 values" in tp.row_b.count.text()
    tp.row_a.mode.buttons["range"].click()
    tp.row_a.start.setValue(0.5); tp.row_a.stop.setValue(3); tp.row_a.step.setValue(0.7)
    assert "4 values, last 2.6 V" in tp.row_a.count.text()
    assert tp.grid.rowCount() == 2 and tp.grid.columnCount() == 4
    tp.row_b.btn_remove.click()
    assert tp.protocol.b is None and tp.btn_add_b.isVisibleTo(tp)


def test_protocol_is_remembered(win, tmp_path):
    tp = win.trials_page
    tp.repeats.setValue(7)
    from neusci_daq.trialspage import TrialsPage
    again = TrialsPage(win)
    assert again.protocol.repeats == 7
