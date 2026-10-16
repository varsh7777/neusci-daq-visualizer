"""Acquisition backends. Each runs a thread that publishes Chunks to subscriber queues.

Data is always published as raw volts; per-channel scaling happens at display/export
time so the recorded file keeps what the ADC actually measured.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Chunk:
    data: np.ndarray      # (n_channels, n_samples), volts
    t: np.ndarray         # (n_samples,), seconds since start
    start_index: int      # index of first sample in the stream


class Backend:
    hardware_timed = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.actual_rate = cfg.sample_rate
        self.error: BaseException | None = None
        self._subs: list[queue.Queue] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public API -------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        self._subs.append(q)
        return q

    def start(self) -> None:
        """Configure hardware in the caller's thread so setup errors raise here."""
        self.error = None
        self._stop.clear()
        self._setup()
        self._thread = threading.Thread(target=self._run, daemon=True, name="daq-acq")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- internals --------------------------------------------------------
    def _run(self):
        try:
            self._loop()
        except BaseException as e:  # surfaced to the GUI via .error
            self.error = e
        finally:
            try:
                self._teardown()
            except Exception:
                pass

    def _publish(self, chunk: Chunk):
        for q in self._subs:
            q.put(chunk)

    def _setup(self): ...
    def _loop(self): raise NotImplementedError
    def _teardown(self): ...


class SimulatedBackend(Backend):
    """Fake signals for developing without hardware: sine + noise + occasional spikes."""

    def _setup(self):
        self._rng = np.random.default_rng(0)

    def _loop(self):
        fs, n, cs = self.cfg.sample_rate, len(self.cfg.channels), self.cfg.chunk_size
        idx, t0 = 0, time.perf_counter()
        while not self._stop.is_set():
            t = (idx + np.arange(cs)) / fs
            data = np.empty((n, cs))
            for i in range(n):
                data[i] = 0.5 * np.sin(2 * np.pi * (1 + i) * t) + 0.02 * self._rng.standard_normal(cs)
                spikes = self._rng.random(cs) < 2 / fs   # ~2 spikes/s
                data[i, spikes] += 2.0
            self._publish(Chunk(data, t, idx))
            idx += cs
            # pace to real time
            delay = t0 + idx / fs - time.perf_counter()
            if delay > 0:
                time.sleep(delay)


class NidaqmxBackend(Backend):
    """Hardware-timed continuous acquisition with NI's official `nidaqmx` package.
    All channels are in ONE task, so each scan samples every channel in order on the
    device clock (still multiplexed: ~10 us inter-channel skew, not simultaneous)."""

    def _setup(self):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        term = {"RSE": TerminalConfiguration.RSE,
                "DIFF": getattr(TerminalConfiguration, "DIFF", None)
                or getattr(TerminalConfiguration, "DIFFERENTIAL")}[self.cfg.terminal_config]
        self._task = nidaqmx.Task()
        try:
            for ch in self.cfg.channels:
                self._task.ai_channels.add_ai_voltage_chan(
                    f"{self.cfg.device}/{ch.physical}", name_to_assign_to_channel=ch.name,
                    terminal_config=term, min_val=ch.range[0], max_val=ch.range[1])
            # Host buffer: ~10 s, so a GUI hiccup doesn't overflow the 2047-sample FIFO.
            buf = int(max(10 * self.cfg.sample_rate, 10 * self.cfg.chunk_size))
            self._task.timing.cfg_samp_clk_timing(
                rate=self.cfg.sample_rate, sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=buf)
            self.actual_rate = self._task.timing.samp_clk_rate  # device may coerce
            self._reader = AnalogMultiChannelReader(self._task.in_stream)
            self._task.start()
        except Exception:
            self._task.close()
            raise

    def _loop(self):
        n, cs, idx = len(self.cfg.channels), self.cfg.chunk_size, 0
        while not self._stop.is_set():
            data = np.empty((n, cs))
            # Blocks until cs samples/channel are available. Overflow raises
            # DaqError -200279, which ends the run and is shown in the GUI.
            self._reader.read_many_sample(data, number_of_samples_per_channel=cs, timeout=10.0)
            t = (idx + np.arange(cs)) / self.actual_rate
            self._publish(Chunk(data, t, idx))
            idx += cs

    def _teardown(self):
        self._task.stop()
        self._task.close()


class MultiChannelIOBackend(Backend):
    """Adapter for sjfleming/MultiChannelIO. SOFTWARE-TIMED polling:
    every sample is a separate start/read/stop per channel, so the rate is whatever
    the USB round-trips allow, with jitter, and channels are not sampled together.
    Timestamps are recorded per scan; `actual_rate` reports what was achieved."""

    hardware_timed = False

    def _setup(self):
        try:
            from MultiChannelIO import MultiChannelIO
        except ImportError as e:
            raise ImportError("MultiChannelIO.py not importable. Clone sjfleming/MultiChannelIO "
                              "and add it to PYTHONPATH (see README).") from e
        dev = self.cfg.device
        self._names = [f"{dev}/{ch.physical}" for ch in self.cfg.channels]
        # Override the library defaults: scaling would otherwise be x100,
        # reset=True would zero your outputs, and the DO line defaults to Dev1.
        self._daq = MultiChannelIO(
            self._names, [], digitalChannels=[f"/{dev}/port0/line0"],
            scalings=[1.0] * len(self._names),
            limit=[tuple(ch.range) for ch in self.cfg.channels],
            reset=False)
        self._daq.configure()
        self.actual_rate = float("nan")

    def _loop(self):
        n, cs, fs = len(self._names), self.cfg.chunk_size, self.cfg.sample_rate
        period, idx = 1.0 / fs, 0
        t0 = time.perf_counter()
        next_t = t0
        data, ts = np.empty((n, cs)), np.empty(cs)
        k = 0
        while not self._stop.is_set():
            ts[k] = time.perf_counter() - t0
            for i, name in enumerate(self._names):
                data[i, k] = self._daq.read(name, 1)
            k += 1
            if k == cs:
                self.actual_rate = (cs - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else float("nan")
                self._publish(Chunk(data.copy(), ts.copy(), idx))
                idx += cs
                k = 0
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()  # behind: don't burst to catch up

    def _teardown(self):
        from PyDAQmx.DAQmxFunctions import DAQmxClearTask  # library never clears its tasks
        for handles in (self._daq.inputTaskHandles, self._daq.outputTaskHandles,
                        self._daq.digitalTaskHandles):
            for h in handles.values():
                DAQmxClearTask(h)


BACKENDS = {"nidaqmx": NidaqmxBackend, "multichannelio": MultiChannelIOBackend,
            "simulated": SimulatedBackend}


def make_backend(cfg: Config) -> Backend:
    return BACKENDS[cfg.backend](cfg)
