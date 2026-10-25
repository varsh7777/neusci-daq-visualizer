"""Trial protocols: send a stimulus many times while stepping one or two of its
settings (a, and optionally b), then line up and measure the responses.

Everything here is plain Python and numpy, so it can be tested without a window.

How a protocol runs (see trialspage.py for the window):
  1. `values()` turns each variable's Range (from / to / step) or List into numbers.
  2. `sequence()` orders every a × b combination, `repeats` times: Blocked,
     Interleaved or Random.
  3. For each trial, `make_wave()` builds the waveform with a (and b) applied.
  4. The output and the recording are separate tasks, so the moment a stimulus
     is sent isn't known exactly. `find_stimulus()` finds it in the recording of
     the stimulus (Stimulator › Record on) by matching the waveform that was sent.
     Time 0 of each trial is where the stimulus starts (after its delay).
  5. `cut()` takes the window around time 0 from every channel, and `measure()`
     measures one channel within a window, e.g. peak-to-peak 2–15 ms.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import stim as stimmod
from .stim import Stimulus

# Settings that can be stepped, per shape: (key, label, unit). "scale" multiplies
# the whole waveform, for shapes whose settings don't step sensibly.
SETTINGS = {
    "pulse": [("amplitude_v", "Amplitude", "V"), ("width_ms", "Pulse width", "ms"),
              ("count", "Number of pulses", ""), ("interval_ms", "Interval", "ms"),
              ("delay_ms", "Delay", "ms")],
    "step": [("amplitude_v", "Level", "V"), ("duration_ms", "Duration", "ms"), ("delay_ms", "Delay", "ms")],
    "ramp": [("end_v", "End level", "V"), ("start_v", "Start level", "V"),
             ("duration_ms", "Duration", "ms"), ("delay_ms", "Delay", "ms")],
    "sine": [("amplitude_v", "Amplitude", "V"), ("freq_hz", "Frequency", "Hz"),
             ("offset_v", "Offset", "V"), ("duration_ms", "Duration", "ms"), ("delay_ms", "Delay", "ms")],
    "custom": [("scale", "Scale (× the whole shape)", "×"), ("delay_ms", "Delay", "ms")],
    "file": [("scale", "Scale (× the whole shape)", "×"), ("delay_ms", "Delay", "ms")],
}
TIME_SETTINGS = {"width_ms", "interval_ms", "delay_ms", "duration_ms"}   # rounded to output samples
# A starting range (from, to, step) when a setting is picked.
DEFAULT_RANGES = {
    "amplitude_v": (0.5, 3.0, 0.5), "start_v": (0.0, 2.0, 0.5), "end_v": (0.5, 3.0, 0.5),
    "offset_v": (0.0, 1.0, 0.25), "width_ms": (0.2, 1.0, 0.2), "interval_ms": (10, 100, 30),
    "delay_ms": (0, 20, 5), "duration_ms": (10, 50, 10), "count": (1, 5, 1),
    "freq_hz": (5, 20, 5), "scale": (0.5, 2.0, 0.5),
}
ORDERS = {"blocked": "Blocked", "interleaved": "Interleaved", "random": "Random"}
MEASURES = {"p2p": "Peak-to-peak", "max": "Maximum", "min": "Minimum", "latency": "Time of peak"}
MAX_VALUES = 200          # per variable
MAX_TRIALS = 2000
LONG_PROTOCOL_S = 600     # warn above 10 minutes
MATCH_THRESHOLD = 0.6     # how closely the recorded stimulus must match the one sent (0–1)


def setting_info(shape: str, key: str) -> tuple[str, str]:
    """(label, unit) of a setting."""
    for k, label, unit in SETTINGS.get(shape, []) + [("scale", "Scale", "×")]:
        if k == key:
            return label, unit
    return key, ""


def fmt(v: float, unit: str = "") -> str:
    """A value with its unit: '1.5 V', '0.2 ms', '×2'."""
    s = f"{v:.6g}"
    if unit == "×":
        return f"×{s}"
    return f"{s} {unit}".strip()


# ---------------------------------------------------------------------------- variables
_SPLIT = re.compile(r"[,;\s]+")


@dataclass
class Variable:
    key: str = "amplitude_v"
    mode: str = "range"           # "range" (from / to / step) or "list"
    start: float = 0.5
    stop: float = 3.0
    step: float = 0.5
    text: str = "0.5, 1, 2, 3"    # the list, as typed

    def values(self) -> list[float]:
        """The values, in order. Raises ValueError with a message for the window."""
        if self.mode == "list":
            return parse_list(self.text)
        return range_values(self.start, self.stop, self.step)


def range_values(start: float, stop: float, step: float) -> list[float]:
    """from, from ± step, ... never going past `stop`. Counts down if stop < start."""
    if not step > 0:
        raise ValueError("Step must be more than 0 (from and to set the direction).")
    span = abs(stop - start)
    n = int(math.floor(span / step + 1e-9)) + 1
    if n > MAX_VALUES:
        raise ValueError(f"That's {n} values; the most is {MAX_VALUES}. Use a bigger step.")
    sign = 1.0 if stop >= start else -1.0
    # whole steps from `start`, so 0.1-sized steps don't drift
    return [round(start + sign * k * step, 9) for k in range(n)]


def parse_list(text: str) -> list[float]:
    """'0.5, 1, 2 3' -> [0.5, 1, 2, 3], kept in the order typed."""
    cells = [c for c in _SPLIT.split(text.strip()) if c]
    if not cells:
        raise ValueError("Type at least one value.")
    out = []
    for c in cells:
        try:
            out.append(float(c))
        except ValueError:
            raise ValueError(f"{c!r} isn't a number. Separate values with commas or spaces.") from None
    dupes = sorted({v for v in out if out.count(v) > 1})
    if dupes:
        raise ValueError(f"{', '.join(f'{v:g}' for v in dupes)} is in the list twice.")
    if len(out) > MAX_VALUES:
        raise ValueError(f"That's {len(out)} values; the most is {MAX_VALUES}.")
    return out


# ---------------------------------------------------------------------------- protocol
@dataclass
class Protocol:
    name: str = "Protocol"
    a: Variable = field(default_factory=Variable)
    b: Variable | None = None
    repeats: int = 3
    order: str = "interleaved"
    gap_s: float = 2.0            # from the start of one trial to the start of the next
    before_ms: float = 5.0        # window shown around each stimulus
    after_ms: float = 30.0
    measure: str = "p2p"
    measure_from_ms: float = 2.0
    measure_to_ms: float = 15.0
    measure_channel: str = ""     # channel name; "" = first channel that isn't the stimulus

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Protocol":
        d = dict(d)
        a = Variable(**{k: v for k, v in (d.pop("a", None) or {}).items() if k in Variable.__dataclass_fields__})
        bd = d.pop("b", None)
        b = Variable(**{k: v for k, v in bd.items() if k in Variable.__dataclass_fields__}) if bd else None
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(a=a, b=b, **known)

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def loads(cls, text: str) -> "Protocol":
        try:
            return cls.from_dict(json.loads(text))
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            raise ValueError(f"That isn't a saved protocol: {e}") from e


@dataclass
class Trial:
    n: int            # 1-based, in the order they run
    rep: int          # 1-based
    ia: int           # index into a's values
    ib: int           # index into b's values (0 when there's no b)
    a: float
    b: float | None


def sequence(p: Protocol, a_vals: list[float], b_vals: list[float] | None,
             rng: np.random.Generator | None = None) -> list[Trial]:
    """Every a × b combination, `repeats` times, in the protocol's order."""
    nb = len(b_vals) if b_vals else 1
    combos = [(ia, ib) for ib in range(nb) for ia in range(len(a_vals))]
    order: list[tuple[int, int, int]] = []          # (rep, ia, ib)
    if p.order == "blocked":
        for ia, ib in combos:
            order += [(r, ia, ib) for r in range(1, p.repeats + 1)]
    else:
        rng = rng or np.random.default_rng()
        for r in range(1, p.repeats + 1):
            these = list(combos)
            if p.order == "random":
                rng.shuffle(these)
            order += [(r, ia, ib) for ia, ib in these]
    return [Trial(n=k + 1, rep=r, ia=ia, ib=ib, a=a_vals[ia], b=(b_vals[ib] if b_vals else None))
            for k, (r, ia, ib) in enumerate(order)]


def apply(s: Stimulus, key: str, value: float) -> tuple[Stimulus, float]:
    """The stimulus with `key` set to `value`, and the scale for the whole waveform."""
    if key == "scale":
        return s, float(value)
    if key == "count":
        if abs(value - round(value)) > 1e-9:
            raise ValueError(f"Number of pulses must be a whole number, not {value:g}.")
        return replace(s, count=int(round(value))), 1.0
    if key == "amplitude_v" and s.shape == "sine":
        value = abs(value)
    return replace(s, **{key: float(value)}), 1.0


def make_wave(s: Stimulus, p: Protocol, a: float, b: float | None, rate: float) -> tuple[np.ndarray, float]:
    """(waveform, delay in ms) for one trial. Raises ValueError."""
    s, scale_a = apply(s, p.a.key, a)
    scale_b = 1.0
    if p.b is not None and b is not None:
        s, scale_b = apply(s, p.b.key, b)
    wave = s.waveform(rate) * (scale_a * scale_b)
    return wave, stimmod.samples(s.delay_ms, rate) * 1000.0 / rate


def describe(p: Protocol, shape: str, a: float, b: float | None) -> str:
    """'a = 1.5 V, b = 0.4 ms' for comments and the chip."""
    la, ua = setting_info(shape, p.a.key)
    out = f"a = {fmt(a, ua)}"
    if p.b is not None and b is not None:
        out += f", b = {fmt(b, setting_info(shape, p.b.key)[1])}"
    return out


@dataclass
class Check:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    a_vals: list[float] = field(default_factory=list)
    b_vals: list[float] | None = None
    n_trials: int = 0
    duration_s: float = 0.0
    peak_v: float = 0.0
    longest_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors


def check(p: Protocol, s: Stimulus, rate: float, max_v: float) -> Check:
    """Everything that would stop a protocol, found before it starts."""
    c = Check()
    keys = {k for k, _, _ in SETTINGS.get(s.shape, [])}
    for name, var in (("a", p.a), ("b", p.b)):
        if var is None:
            continue
        if var.key not in keys:
            c.errors.append(f"{name}: {setting_info(s.shape, var.key)[0]} isn't a setting of a "
                            f"{stimmod.SHAPES.get(s.shape, s.shape).lower()} stimulus.")
            continue
        try:
            vals = var.values()
        except ValueError as e:
            c.errors.append(f"{name}: {e}")
            continue
        if name == "a":
            c.a_vals = vals
        else:
            c.b_vals = vals
        if var.key in TIME_SETTINGS:              # values the output can't tell apart
            got: dict[int, float] = {}
            for v in vals:
                k = stimmod.samples(v, rate)
                if k in got:
                    c.warnings.append(f"{name}: {got[k]:g} and {v:g} ms both round to "
                                      f"{k * 1000 / rate:g} ms at {rate:,.0f} samples/s.")
                got.setdefault(k, v)
    if p.b is not None and p.a.key == p.b.key:
        c.errors.append("a and b are the same setting; pick two different ones.")
    if p.repeats < 1:
        c.errors.append("Repeats must be at least 1.")
    if p.before_ms < 0 or p.after_ms <= 0:
        c.errors.append("The window needs 0 ms or more before and more than 0 ms after the stimulus.")
    if c.errors:
        return c
    nb = len(c.b_vals) if c.b_vals else 1
    c.n_trials = len(c.a_vals) * nb * p.repeats
    if c.n_trials > MAX_TRIALS:
        c.errors.append(f"{c.n_trials} trials is too many (the most is {MAX_TRIALS}).")
        return c
    bad = []
    for b in (c.b_vals or [None]):
        for a in c.a_vals:
            try:
                wave, _ = make_wave(s, p, a, b, rate)
                stimmod.check_limit(wave, max_v)
            except ValueError as e:
                bad.append(f"{describe(p, s.shape, a, b)}: {e}")
                continue
            c.peak_v = max(c.peak_v, float(np.max(np.abs(wave))) if wave.size else 0.0)
            c.longest_s = max(c.longest_s, len(wave) / rate)
    if bad:
        c.errors += bad[:3] + ([f"…and {len(bad) - 3} more."] if len(bad) > 3 else [])
        return c
    need = c.longest_s + p.after_ms / 1000.0 + 0.1
    if p.gap_s < need:
        c.errors.append(f"Trials {p.gap_s:g} s apart would overlap: the longest stimulus plus the "
                        f"{p.after_ms:g} ms after it needs at least {need:.2f} s.")
    c.duration_s = c.n_trials * p.gap_s
    if c.duration_s > LONG_PROTOCOL_S:
        c.warnings.append(f"This takes about {c.duration_s / 60:.0f} minutes. Long protocols can "
                          "outlast the preparation; consider fewer values or repeats.")
    return c


# ---------------------------------------------------------------------------- finding the stimulus
def template(wave: np.ndarray, out_rate: float, in_rate: float, pad_s: float = 0.02) -> tuple[np.ndarray, int]:
    """The sent waveform as the input would record it, with 0 V before and after it
    (so a stimulus that starts straight away still has an edge to find).
    Returns (template, samples of padding before the waveform)."""
    t_out = np.arange(len(wave)) / out_rate
    t_in = np.arange(0, len(wave) / out_rate, 1.0 / in_rate)
    # zero-order hold: what the output really does between its samples
    idx = np.minimum((t_in * out_rate + 1e-9).astype(int), len(wave) - 1)
    body = wave[idx] if len(wave) else np.zeros(0)
    pad = max(1, int(round(pad_s * in_rate)))
    return np.concatenate([np.zeros(pad), body, np.zeros(pad)]), pad


def find_stimulus(x: np.ndarray, tpl: np.ndarray) -> tuple[int, float]:
    """Where `tpl` best matches inside `x`: (index of the template's start, match 0–1).

    Normalised cross-correlation, so it doesn't matter how strongly the stimulus
    was recorded, or whether the input has an offset.
    """
    from scipy.signal import correlate
    x = np.asarray(x, float)
    m = len(tpl)
    if len(x) < m or m < 2:
        return 0, 0.0
    t = tpl - tpl.mean()
    tn = float(np.sqrt(np.sum(t * t)))
    if tn == 0:
        return 0, 0.0
    num = correlate(x, t, mode="valid", method="fft")
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    var = np.maximum(s2 - s1 * s1 / m, 0.0)
    den = np.sqrt(var) * tn
    ncc = np.where(den > 1e-12 * max(1.0, tn), num / np.where(den > 0, den, 1.0), 0.0)
    k = int(np.argmax(ncc))
    return k, float(ncc[k])


def cut(t: np.ndarray, data: np.ndarray, t0: float, before_ms: float, after_ms: float,
        rate: float) -> tuple[np.ndarray, np.ndarray] | None:
    """The window from `before_ms` before `t0` to `after_ms` after it, from every
    channel: (time in ms relative to t0, data). None if the data doesn't cover it."""
    i0 = int(round((t0 - t[0]) * rate))
    lo = i0 - int(round(before_ms / 1000.0 * rate))
    hi = i0 + int(round(after_ms / 1000.0 * rate)) + 1
    if lo < 0 or hi > len(t):
        return None
    return (np.arange(lo - i0, hi - i0) / rate * 1000.0), data[:, lo:hi].copy()


def measure(kind: str, t_ms: np.ndarray, y: np.ndarray, t_from: float, t_to: float) -> float:
    """One number from one trace between t_from and t_to (ms after the stimulus)."""
    sel = (t_ms >= t_from) & (t_ms <= t_to)
    if not np.any(sel):
        return float("nan")
    seg = y[sel]
    if kind == "max":
        return float(np.max(seg))
    if kind == "min":
        return float(np.min(seg))
    if kind == "latency":
        base_sel = t_ms < 0
        base = float(np.mean(y[base_sel])) if np.any(base_sel) else 0.0
        return float(t_ms[sel][int(np.argmax(np.abs(seg - base)))])
    return float(np.max(seg) - np.min(seg))


def stats(xs) -> tuple[int, float, float]:
    """(n, mean, SD) ignoring NaN; SD is 0 for one value."""
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    if xs.size == 0:
        return 0, float("nan"), float("nan")
    return int(xs.size), float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0


# ---------------------------------------------------------------------------- capturing trials
@dataclass
class Result:
    trial: Trial
    found: bool
    match: float = 0.0
    t0: float = float("nan")          # stream time (s) of the stimulus start
    t_ms: np.ndarray | None = None    # window time axis, ms after the stimulus
    data: np.ndarray | None = None    # (channels, samples), in each channel's units
    use: bool = True                  # left out of averages and the curve when False


@dataclass
class _Pending:
    trial: Trial
    t_fire: float          # newest recorded sample when it was sent (stream time, s)
    tpl: np.ndarray
    pad: int
    delay_s: float
    dur_s: float
    t_next: float | None = None   # the same, for the trial sent after it


class Capture:
    """Keeps the last few seconds of the recording and turns each sent stimulus
    into a Result once the data after it has arrived.

    Each stimulus is looked for between when it was sent and when the next one
    was: the next is only sent once this one has finished, so it must be in
    between, however slow the computer was. Looking any further could find the
    next trial's stimulus instead, which matches just as well when only its size
    differs.
    """

    SEARCH_AFTER_S = 1.0      # for the last trial, which has no next one

    def __init__(self, rate: float, monitor_row: int, to_units, keep_s: float, read_s: float = 0.05):
        self.rate = float(rate)
        # 'When it was sent' is the newest recorded sample at the time, which can
        # be up to one read away from the real moment, either way.
        self.slack = read_s + 0.03
        self.monitor_row = monitor_row
        self.to_units = to_units              # (channels, n) volts -> display units
        self.keep_s = keep_s
        self.t = np.zeros(0)
        self.data: np.ndarray | None = None
        self.pending: list[_Pending] = []
        self.last_fire: float | None = None    # stream time of the last stimulus sent
        self.last_dur = 0.0

    @property
    def latest(self) -> float:
        return float(self.t[-1]) if self.t.size else 0.0

    def add(self, t: np.ndarray, data: np.ndarray) -> None:
        if self.data is None or self.data.shape[0] != data.shape[0]:
            self.t, self.data = np.asarray(t, float), np.asarray(data, float)
        else:
            self.t = np.concatenate([self.t, t])
            self.data = np.concatenate([self.data, data], axis=1)
        keep = int(self.keep_s * self.rate)
        if self.t.size > keep:
            self.t, self.data = self.t[-keep:], self.data[:, -keep:]

    def sent(self, trial: Trial, wave: np.ndarray, out_rate: float, delay_ms: float) -> None:
        tpl, pad = template(wave, out_rate, self.rate)
        now = self.latest
        if self.pending and self.pending[-1].t_next is None:
            self.pending[-1].t_next = now
        self.pending.append(_Pending(trial, now, tpl, pad, delay_ms / 1000.0, len(wave) / out_rate))
        self.last_fire, self.last_dur = now, len(wave) / out_rate

    def can_send(self, gap_s: float) -> bool:
        """True once the recording itself has moved on far enough since the last
        stimulus. Going by the clock alone isn't enough: after the computer stalls,
        a late trial would be sent straight after the one before, and the two
        stimuli couldn't be told apart."""
        if self.last_fire is None:
            return True
        need = max(0.8 * gap_s, self.last_dur + 2 * self.slack + 0.05)
        return self.latest - self.last_fire >= need

    def _span(self, pd: _Pending) -> tuple[float, float]:
        """Where the stimulus can start (stream time, s)."""
        lo = pd.t_fire - self.slack
        hi = (pd.t_next - self.slack) if pd.t_next is not None else pd.t_fire + self.SEARCH_AFTER_S
        return lo, max(hi, lo + 0.01)

    def resolve(self, p: Protocol, force: bool = False) -> list[Result]:
        """Results for sent stimuli whose data has all arrived (or all of them, when `force`)."""
        done, keep = [], []
        for pd in self.pending:
            ready = self._span(pd)[1] + pd.dur_s + p.after_ms / 1000.0 + 0.05
            if self.latest < ready and not force:
                keep.append(pd)
                continue
            done.append(self._result(pd, p))
        self.pending = keep
        return done

    def _result(self, pd: _Pending, p: Protocol) -> Result:
        if self.data is None or self.t.size == 0:
            return Result(pd.trial, False)
        s_lo, s_hi = self._span(pd)
        lo = max(0, int((s_lo - self.t[0]) * self.rate) - pd.pad)
        hi = min(self.t.size, int((s_hi + pd.dur_s - self.t[0]) * self.rate) + pd.pad + 1)
        k, match = find_stimulus(self.data[self.monitor_row, lo:hi], pd.tpl)
        if match < MATCH_THRESHOLD:
            return Result(pd.trial, False, match)
        t0 = float(self.t[lo + k + pd.pad]) + pd.delay_s
        got = cut(self.t, self.data, t0, p.before_ms, p.after_ms, self.rate)
        if got is None:
            return Result(pd.trial, False, match, t0)
        t_ms, raw = got
        return Result(pd.trial, True, match, t0, t_ms, self.to_units(raw))
