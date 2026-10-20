"""Write acquired chunks to HDF5 on a background thread.

File layout:
  /raw_volts   (n_channels, N) float32   -- unscaled ADC volts
  /t           (N,) float64              -- only for software-timed backends
  /comments    compound(t, sample_index, text)
  attrs: sample_rate, hardware_timed, device, backend, start_time,
         channel_names, physical, units, scale, offset, colors
         (colors are display-only and may be changed later)
"""
from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .backends import Backend, Chunk

_COMMENT_DT = np.dtype([("t", "f8"), ("sample_index", "i8"),
                        ("text", h5py.string_dtype())])


class Recorder:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.cfg = backend.cfg
        self._q = backend.subscribe()
        self._lock = threading.Lock()
        self._file: h5py.File | None = None
        self.path: Path | None = None
        self.samples_written = 0
        self.last_t = 0.0
        self.last_index = 0
        self._thread = threading.Thread(target=self._consume, daemon=True, name="daq-rec")
        self._thread.start()

    @property
    def recording(self) -> bool:
        return self._file is not None

    def start(self, path: str | Path | None = None) -> Path:
        cfg = self.cfg
        if path is None:
            Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
            path = Path(cfg.data_dir) / datetime.now().strftime("rec_%Y%m%d_%H%M%S.h5")
        n = len(cfg.channels)
        f = h5py.File(path, "w")
        f.create_dataset("raw_volts", shape=(n, 0), maxshape=(n, None), dtype="f4",
                         chunks=(n, max(1024, cfg.chunk_size)))
        if not self.backend.hardware_timed:
            f.create_dataset("t", shape=(0,), maxshape=(None,), dtype="f8", chunks=(4096,))
        f.create_dataset("comments", shape=(0,), maxshape=(None,), dtype=_COMMENT_DT)
        a = f.attrs
        a["sample_rate"] = self.backend.actual_rate
        a["hardware_timed"] = self.backend.hardware_timed
        a["device"], a["backend"] = cfg.device, cfg.backend
        a["terminal_config"] = cfg.terminal_config
        a["start_time"] = datetime.now().isoformat()
        a["channel_names"] = [c.name for c in cfg.channels]
        a["physical"] = [c.physical for c in cfg.channels]
        a["units"] = [c.units for c in cfg.channels]
        a["scale"] = [c.scale for c in cfg.channels]
        a["offset"] = [c.offset for c in cfg.channels]
        a["colors"] = [c.color for c in cfg.channels]
        with self._lock:
            self.samples_written = 0
            self._t_offset = None
            self._file, self.path = f, Path(path)
        return self.path

    def close(self) -> None:
        """Stop recording and end the writer thread (call before the program exits)."""
        self.stop()
        self._q.put(None)
        self._thread.join(timeout=5)

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                if not self.backend.hardware_timed:
                    self._file.attrs["sample_rate"] = self.backend.actual_rate
                self._file.close()
                self._file = None

    def set_colors(self, colors: list[str]) -> None:
        """Update the colours of the block being recorded (display metadata only)."""
        with self._lock:
            if self._file is not None:
                self._file.attrs["colors"] = list(colors)
                self._file.flush()

    def add_comment(self, text: str) -> float:
        """Comment at the latest acquired sample. Returns its stream time (s)."""
        with self._lock:
            if self._file is not None:
                ds = self._file["comments"]
                ds.resize((ds.shape[0] + 1,))
                ds[-1] = (self.last_t - (self._t_offset or 0.0),
                          self.samples_written, text)
                self._file.flush()
        return self.last_t

    def _consume(self):
        while True:
            chunk: Chunk | None = self._q.get()
            if chunk is None:   # sent by close()
                return
            self.last_t, self.last_index = float(chunk.t[-1]), chunk.start_index
            with self._lock:
                if self._file is None:
                    continue
                if self._t_offset is None:
                    self._t_offset = float(chunk.t[0])
                ds = self._file["raw_volts"]
                n0, k = ds.shape[1], chunk.data.shape[1]
                ds.resize((ds.shape[0], n0 + k))
                ds[:, n0:] = chunk.data
                if "t" in self._file:
                    tds = self._file["t"]
                    tds.resize((n0 + k,))
                    tds[n0:] = chunk.t - self._t_offset
                self.samples_written = n0 + k


def load_recording(path: str | Path) -> dict:
    """Read a recording back with scaling applied. Returns dict of arrays + metadata."""
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        raw = f["raw_volts"][:].astype("f8")
        scale = np.asarray(a["scale"])[:, None]
        offset = np.asarray(a["offset"])[:, None]
        t = f["t"][:] if "t" in f else np.arange(raw.shape[1]) / a["sample_rate"]
        return {"t": t, "data": raw * scale + offset, "raw_volts": raw,
                "comments": f["comments"][:], "attrs": a}
