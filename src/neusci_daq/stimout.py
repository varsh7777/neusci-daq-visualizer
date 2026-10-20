"""Sending a stimulus waveform out of an analog output (ao0/ao1).

Each stimulus is one hardware-timed, finite output task, run on a worker
thread so the window stays responsive. Recording keeps running on its own
task at the same time.

Safety: the waveform always ends at 0 V (stim.py), and on Stop, on errors
and when the program closes the output is set to 0 V.

Timing: the output and the recording are separate tasks, so their start
times differ by a few milliseconds. For exact timing, wire the output to a
spare input and record it (Stimulator > Record stimulus on).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .config import Config


class StimOutput:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.error: BaseException | None = None
        self.fired = 0                          # stimuli started (for tests and the status bar)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def fire(self, wave: np.ndarray, rate: float) -> None:
        """Start sending `wave`. Raises RuntimeError if the previous stimulus is still going."""
        if self.busy:
            raise RuntimeError("The previous stimulus is still running.")
        self.error = None
        self._stop.clear()
        self.fired += 1
        self._thread = threading.Thread(target=self._run, args=(np.asarray(wave, float), float(rate)),
                                        daemon=True, name="daq-stim")
        self._thread.start()

    def stop(self) -> None:
        """Stop the current stimulus (if any) and set the output to 0 V."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        try:
            self.zero()
        except Exception as e:      # reported, but never blocks stopping
            self.error = e

    def _run(self, wave, rate):
        try:
            self._send(wave, rate)
        except BaseException as e:
            self.error = e
            try:
                self.zero()
            except Exception:
                pass

    def _send(self, wave, rate): raise NotImplementedError
    def zero(self) -> None: ...


class NidaqmxOutput(StimOutput):
    """Hardware-timed finite output with NI's `nidaqmx` package."""

    def _channel(self, task):
        task.ao_channels.add_ao_voltage_chan(f"{self.cfg.device}/{self.cfg.stim_output}",
                                             min_val=-10.0, max_val=10.0)

    def _send(self, wave, rate):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        task = nidaqmx.Task()
        try:
            self._channel(task)
            if len(wave) < 2:
                task.write(float(wave[0]) if len(wave) else 0.0, auto_start=True)
                return
            task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=len(wave))
            task.write(wave, auto_start=False)
            task.start()
            deadline = time.monotonic() + len(wave) / rate + 5.0
            while not task.is_task_done():
                if self._stop.is_set():
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("The stimulus didn't finish in time.")
                time.sleep(0.005)
        finally:
            stopped_early = self._stop.is_set()
            try:
                task.stop()
            finally:
                task.close()
            if stopped_early:
                self.zero()

    def zero(self) -> None:
        import nidaqmx
        with nidaqmx.Task() as task:
            self._channel(task)
            task.write(0.0, auto_start=True)


class SimulatedOutput(StimOutput):
    """For trying the stimulator without hardware: the waveform appears on the
    monitor input of the simulated recording, plus a small fake response on the
    other channels."""

    def __init__(self, cfg: Config, backend):
        super().__init__(cfg)
        self.backend = backend

    def _send(self, wave, rate):
        self.backend.inject(self.cfg.stim_monitor, wave, rate)
        end = time.monotonic() + len(wave) / rate
        while time.monotonic() < end:
            if self._stop.is_set():
                self.backend.inject(self.cfg.stim_monitor, np.zeros(1), rate)
                return
            time.sleep(0.005)

    def zero(self) -> None:
        pass


def make_output(cfg: Config, backend) -> StimOutput | None:
    """The output for this backend, or None if it can't send stimuli."""
    if cfg.backend == "nidaqmx":
        return NidaqmxOutput(cfg)
    if cfg.backend == "simulated":
        return SimulatedOutput(cfg, backend)
    return None
