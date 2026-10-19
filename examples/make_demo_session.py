"""Write a synthetic session you can open in review mode (File > Open...).

It mimics a cockroach tibial-spine recording through an amplifier at gain 100
into a USB-6003 at 20 kHz:

  Ch "Leg nerve" (ai0, µV at the electrode): extracellular spikes from two units
      (a large spine afferent and a smaller one), downward-going like real
      extracellular spikes, on ~10 µV RMS noise, quantized to the USB-6003's
      ~305 µV step at the DAQ (~3 µV at the electrode at gain 100).
  Ch "Stim drive" (ai1, V): the speaker drive signal, bursts of a sine wave.

  block_001: 20 Hz stimulus bursts  -> phasic-tonic firing that adapts
  block_002: 50 Hz stimulus bursts
  block_003: no stimulus (spontaneous firing only)

The data are synthetic. Don't analyse them as if they were real recordings.

Usage:  python examples/make_demo_session.py [output_folder]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

FS = 20_000.0
GAIN = 100.0
LSB_AT_DAQ = 20.0 / 2**16          # USB-6003: ±10 V over 16 bits ≈ 305 µV
NOISE_UV = 10.0
_COMMENT_DT = np.dtype([("t", "f8"), ("sample_index", "i8"), ("text", h5py.string_dtype())])


def spike_waveform(amp_uv: float) -> np.ndarray:
    """~1.5 ms biphasic extracellular spike: sharp negative peak, slower positive overshoot."""
    t = np.arange(int(0.0015 * FS)) / FS
    neg = -np.exp(-((t - 0.0003) / 0.00012) ** 2)
    pos = 0.35 * np.exp(-((t - 0.0007) / 0.00025) ** 2)
    return amp_uv * (neg + pos)


def firing_rate(t, stim_on, base_hz, peak_hz, tonic_hz, tau_s):
    """Spontaneous rate, plus a phasic-tonic response that adapts during each stimulus burst."""
    rate = np.full_like(t, base_hz)
    onsets = np.flatnonzero(np.diff(stim_on.astype(int)) == 1) + 1
    for on in onsets:
        off = on + np.argmin(stim_on[on:]) if not stim_on[on:].all() else len(t)
        dt = t[on:off] - t[on]
        rate[on:off] = tonic_hz + (peak_hz - tonic_hz) * np.exp(-dt / tau_s)
    return rate


def make_block(rng, dur_s, stim_hz, start_time):
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    # Stimulus: 2 s on, 3 s off, starting at 2 s
    if stim_hz:
        phase = (t - 2.0) % 5.0
        stim_on = (t >= 2.0) & (phase < 2.0)
        stim = np.where(stim_on, 1.0 * np.sin(2 * np.pi * stim_hz * t), 0.0)
    else:
        stim_on = np.zeros(n, bool)
        stim = np.zeros(n)

    nerve = rng.normal(0, NOISE_UV, n)
    units = [  # (amplitude µV, spontaneous Hz, peak Hz, tonic Hz, adaptation tau s)
        (160.0, 2.0, 160.0 if stim_hz <= 20 else 120.0, 25.0, 0.25),
        (60.0, 6.0, 40.0, 12.0, 0.6),
    ]
    for amp, base, peak, tonic, tau in units:
        rate = firing_rate(t, stim_on, base, peak, tonic, tau)
        spikes = np.flatnonzero(rng.random(n) < rate / FS)
        # 2 ms refractory period
        keep, last = [], -10**9
        for s in spikes:
            if s - last > 0.002 * FS:
                keep.append(s)
                last = s
        wf = spike_waveform(amp * rng.uniform(0.9, 1.1))
        for s in keep:
            e = min(n, s + wf.size)
            nerve[s:e] += wf[: e - s]
    # slow electrode drift, and a little 60 Hz the Humbug didn't remove
    nerve += 4.0 * np.sin(2 * np.pi * 0.05 * t) + 3.0 * np.sin(2 * np.pi * 60 * t)

    # What the DAQ saw: amplified, then quantized to the ADC step
    nerve_at_daq = np.round(nerve * 1e-6 * GAIN / LSB_AT_DAQ) * LSB_AT_DAQ
    stim_at_daq = np.round(stim / LSB_AT_DAQ) * LSB_AT_DAQ
    raw = np.vstack([nerve_at_daq, stim_at_daq]).astype("f4")
    return raw, start_time


def write_block(path, raw, start_time, comments):
    with h5py.File(path, "w") as f:
        f.create_dataset("raw_volts", data=raw, chunks=(raw.shape[0], 1024), maxshape=(raw.shape[0], None))
        c = f.create_dataset("comments", shape=(len(comments),), maxshape=(None,), dtype=_COMMENT_DT)
        for i, (t, text) in enumerate(comments):
            c[i] = (t, int(round(t * FS)), text)
        a = f.attrs
        a["sample_rate"] = FS
        a["hardware_timed"] = True
        a["device"] = "Dev1"
        a["backend"] = "synthetic-demo"
        a["start_time"] = start_time.isoformat()
        a["channel_names"] = ["Leg nerve", "Stim drive"]
        a["physical"] = ["ai0", "ai1"]
        a["units"] = ["µV", "V"]
        a["scale"] = [1e6 / GAIN, 1.0]   # µV at the electrode per V at the DAQ
        a["offset"] = [0.0, 0.0]


def main(out="demo_cockroach_leg"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(302)
    t0 = datetime(2026, 10, 18, 14, 0, 0)
    plan = [  # duration s, stim Hz, comments
        (15.0, 20, [(0.5, "probe touching spine"), (2.0, "stim 20 Hz, 2 s on / 3 s off")]),
        (20.0, 50, [(2.0, "stim 50 Hz")]),
        (15.0, 0, [(0.5, "stimulus off: spontaneous activity")]),
    ]
    start = t0
    for i, (dur, hz, comments) in enumerate(plan, 1):
        raw, st = make_block(rng, dur, hz, start)
        write_block(out / f"block_{i:03d}.h5", raw, st, comments)
        start = start + timedelta(seconds=dur + 30)
    print(f"Wrote {len(plan)} blocks to {out.resolve()}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
