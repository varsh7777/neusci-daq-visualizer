"""Stimulus shapes for the analog outputs, and how they become a waveform.

A Stimulus describes one stimulus (a pulse train, a ramp, a custom shape...).
`waveform(rate)` turns it into the voltages written to ao0/ao1, one per
output sample. Every waveform ends at 0 V, because the output keeps its last
value after a stimulus finishes.

Shapes:
  pulse   one pulse or a train (optionally biphasic: +A then -A)
  step    a constant level for a while
  ramp    a straight line from one level to another
  sine    a sine wave (amplitude, offset, frequency)
  custom  any shape, as straight-line segments: (duration, start V, end V)
  file    voltages from a CSV/text file: one column (at a given rate) or
          two columns (time in s, volts), resampled to the output rate
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

# USB-6003 analog output (NI spec): 2 channels, 16-bit, +/-10 V, 5 kS/s.
AO_MAX_RATE = 5000.0
AO_RANGE = (-10.0, 10.0)
AO_CHANNELS = ("ao0", "ao1")
MAX_DURATION_S = 60.0

SHAPES = {
    "pulse": "Pulse / train",
    "step": "Step",
    "ramp": "Ramp",
    "sine": "Sine",
    "custom": "Custom segments",
    "file": "From file",
}


def samples(ms: float, rate: float) -> int:
    """Number of output samples for `ms` milliseconds, rounding halves up (0.3 ms at 5 kS/s -> 2)."""
    return int(math.floor(ms * rate / 1000.0 + 0.5 + 1e-9))


def _default_segments():
    return [{"duration_ms": 10.0, "start_v": 0.0, "end_v": 1.0},
            {"duration_ms": 10.0, "start_v": 1.0, "end_v": 1.0},
            {"duration_ms": 10.0, "start_v": 1.0, "end_v": 0.0}]


@dataclass
class Stimulus:
    shape: str = "pulse"
    delay_ms: float = 0.0          # 0 V before the stimulus starts
    amplitude_v: float = 1.0       # pulse, step, sine
    width_ms: float = 1.0          # pulse
    count: int = 1                 # pulse: number of pulses
    interval_ms: float = 100.0     # pulse: start of one pulse to start of the next
    biphasic: bool = False         # pulse: +A for width, then -A for width
    duration_ms: float = 100.0     # step, ramp, sine
    start_v: float = 0.0           # ramp
    end_v: float = 1.0             # ramp
    offset_v: float = 0.0          # sine
    freq_hz: float = 10.0          # sine
    segments: list = field(default_factory=_default_segments)   # custom
    file: str = ""                 # file
    file_rate: float = 0.0         # file: rate of a one-column file (0 = output rate)

    # -- description -----------------------------------------------------
    def describe(self) -> str:
        """Short text for comments and the status bar, e.g. 'Pulse 2 V, 1 ms ×5 @ 100 ms'."""
        d = f"{self.delay_ms:g} ms delay, " if self.delay_ms else ""
        if self.shape == "pulse":
            s = f"{'Biphasic pulse' if self.biphasic else 'Pulse'} {self.amplitude_v:g} V, {self.width_ms:g} ms"
            if self.count > 1:
                s += f" ×{self.count} every {self.interval_ms:g} ms"
        elif self.shape == "step":
            s = f"Step {self.amplitude_v:g} V for {self.duration_ms:g} ms"
        elif self.shape == "ramp":
            s = f"Ramp {self.start_v:g}→{self.end_v:g} V over {self.duration_ms:g} ms"
        elif self.shape == "sine":
            s = f"Sine {self.amplitude_v:g} V, {self.freq_hz:g} Hz for {self.duration_ms:g} ms"
            if self.offset_v:
                s += f", offset {self.offset_v:g} V"
        elif self.shape == "custom":
            s = f"Custom ({len(self.segments)} segments)"
        else:
            s = f"File {Path(self.file).name or '(none)'}"
        return d + s

    # -- waveform ----------------------------------------------------------
    def waveform(self, rate: float) -> np.ndarray:
        """Voltages, one per output sample at `rate`, ending at 0 V. Raises ValueError."""
        if not 0 < rate <= AO_MAX_RATE:
            raise ValueError(f"Output rate must be between 0 and {AO_MAX_RATE:,.0f} S/s (USB-6003 limit).")
        if self.shape not in SHAPES:
            raise ValueError(f"Unknown stimulus shape {self.shape!r}.")
        n = lambda ms: samples(ms, rate)
        if self.delay_ms < 0:
            raise ValueError("Delay can't be negative.")
        parts = [np.zeros(n(self.delay_ms))]
        dt_ms = 1000.0 / rate

        if self.shape == "pulse":
            w = n(self.width_ms)
            if self.width_ms * rate / 1000.0 < 1 - 1e-9:      # never stretch a pulse that's too short
                raise ValueError(f"Pulse width {self.width_ms:g} ms is shorter than one output sample "
                                 f"({dt_ms:g} ms at {rate:,.0f} S/s). Let the stimulus isolator set "
                                 "short widths, or use a wider pulse.")
            if self.count < 1:
                raise ValueError("Number of pulses must be at least 1.")
            one = np.full(w, float(self.amplitude_v))
            if self.biphasic:
                one = np.concatenate([one, -one])
            gap = n(self.interval_ms) - len(one)
            if self.count > 1 and gap < 0:
                raise ValueError(f"Interval {self.interval_ms:g} ms is shorter than one pulse "
                                 f"({len(one) * dt_ms:g} ms). Use a longer interval.")
            for i in range(self.count):
                parts.append(one)
                if i < self.count - 1 and gap > 0:
                    parts.append(np.zeros(gap))
        elif self.shape == "step":
            k = self._need(self.duration_ms, "Duration", dt_ms, rate)
            parts.append(np.full(k, float(self.amplitude_v)))
        elif self.shape == "ramp":
            k = self._need(self.duration_ms, "Duration", dt_ms, rate)
            parts.append(np.linspace(self.start_v, self.end_v, k))
        elif self.shape == "sine":
            k = self._need(self.duration_ms, "Duration", dt_ms, rate)
            if self.freq_hz <= 0 or self.freq_hz > rate / 2:
                raise ValueError(f"Frequency must be between 0 and {rate / 2:,.0f} Hz "
                                 f"(half the output rate) to be drawn correctly.")
            t = np.arange(k) / rate
            parts.append(self.offset_v + self.amplitude_v * np.sin(2 * np.pi * self.freq_hz * t))
        elif self.shape == "custom":
            if not self.segments:
                raise ValueError("Add at least one segment.")
            for i, seg in enumerate(self.segments, 1):
                k = n(float(seg["duration_ms"]))
                if float(seg["duration_ms"]) * rate / 1000.0 < 1 - 1e-9:
                    raise ValueError(f"Segment {i} is shorter than one output sample ({dt_ms:g} ms).")
                parts.append(np.linspace(float(seg["start_v"]), float(seg["end_v"]), k, endpoint=False)
                             if seg["start_v"] != seg["end_v"] else np.full(k, float(seg["start_v"])))
        else:
            parts.append(load_wave_file(self.file, rate, self.file_rate or rate))

        wave = np.concatenate(parts + [np.zeros(1)])     # always end at 0 V
        if len(wave) / rate > MAX_DURATION_S:
            raise ValueError(f"Stimulus is {len(wave) / rate:.1f} s long; the limit is {MAX_DURATION_S:g} s.")
        return wave

    @staticmethod
    def _need(ms: float, what: str, dt_ms: float, rate: float) -> int:
        k = samples(ms, rate)
        if ms * rate / 1000.0 < 1 - 1e-9:
            raise ValueError(f"{what} is shorter than one output sample ({dt_ms:g} ms at {rate:,.0f} S/s).")
        return k

    def rounding_notes(self, rate: float) -> list[str]:
        """Times that don't fit the output's sample grid and will be rounded."""
        dt = 1000.0 / rate
        notes = []
        checks = [("Delay", self.delay_ms)]
        if self.shape == "pulse":
            checks += [("Width", self.width_ms)] + ([("Interval", self.interval_ms)] if self.count > 1 else [])
        elif self.shape in ("step", "ramp", "sine"):
            checks.append(("Duration", self.duration_ms))
        for name, ms in checks:
            got = samples(ms, rate) * dt
            if ms and abs(got - ms) > 1e-9:
                notes.append(f"{name} {ms:g} ms rounds to {got:g} ms (output steps are {dt:g} ms).")
        return notes

    # -- files ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Stimulus":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Stimulus":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"{Path(path).name} isn't a saved stimulus: {e}") from e


def check_limit(wave: np.ndarray, max_v: float) -> None:
    """Refuse waveforms beyond the configured safety limit."""
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > max_v + 1e-9:
        raise ValueError(f"Stimulus reaches {peak:g} V, above the {max_v:g} V limit set in the config "
                         "([stimulator] max_voltage). Lower the amplitude, or ask an instructor to raise the limit.")


_SPLIT = re.compile(r"[,;\t ]+")


def load_wave_file(path: str | Path, rate: float, file_rate: float) -> np.ndarray:
    """Read a waveform file and resample it to `rate`.

    One column: volts, one per sample at `file_rate`.
    Two columns: time (s) and volts; any spacing.
    Lines that aren't numbers (headers, comments) are skipped.
    """
    if not path:
        raise ValueError("Choose a waveform file.")
    p = Path(path)
    if not p.exists():
        raise ValueError(f"{p.name} not found.")
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [c for c in _SPLIT.split(line.strip()) if c]
        try:
            rows.append([float(c) for c in cells])
        except ValueError:
            continue
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError(f"{p.name} has no numbers in it.")
    width = len(rows[0])
    if any(len(r) != width for r in rows) or width > 2:
        raise ValueError(f"{p.name}: every line needs the same 1 or 2 columns (volts, or time and volts).")
    a = np.asarray(rows, dtype=float)
    if width == 1:
        v = a[:, 0]
        if file_rate <= 0:
            raise ValueError("Give the file's sample rate for a one-column file.")
        t = np.arange(len(v)) / file_rate
    else:
        t, v = a[:, 0] - a[0, 0], a[:, 1]
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{p.name}: times in the first column must increase.")
    if len(v) == 1:
        return v.copy()
    out_t = np.arange(0, t[-1] + 0.5 / rate, 1.0 / rate)
    return np.interp(out_t, t, v)
