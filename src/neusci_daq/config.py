"""Load and validate the acquisition config (TOML)."""
from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# USB-6003 limits (NI spec): one multiplexed ADC, 100 kS/s aggregate,
# 8 single-ended / 4 differential inputs, fixed +/-10 V range.
USB6003_MAX_AGGREGATE_RATE = 100_000
USB6003_RANGE = (-10.0, 10.0)


@dataclass
class Channel:
    physical: str                 # e.g. "ai0"
    name: str = ""
    units: str = "V"
    scale: float = 1.0            # displayed = volts * scale + offset
    offset: float = 0.0
    range: tuple[float, float] = USB6003_RANGE

    def to_units(self, volts):
        return volts * self.scale + self.offset


@dataclass
class Config:
    backend: str = "nidaqmx"      # "nidaqmx" | "multichannelio" | "simulated"
    device: str = "Dev1"
    sample_rate: float = 1000.0   # per channel
    chunk_size: int = 100         # samples per channel per read
    terminal_config: str = "RSE"  # "RSE" | "DIFF"
    channels: list[Channel] = field(default_factory=list)
    window_s: float = 10.0
    data_dir: str = "data"

    @property
    def aggregate_rate(self) -> float:
        return self.sample_rate * len(self.channels)


def load(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    dev, disp, rec = raw.get("device", {}), raw.get("display", {}), raw.get("recording", {})
    chans = []
    for c in raw.get("channels", []):
        ch = Channel(**{k: v for k, v in c.items() if k != "range"})
        if "range" in c:
            ch.range = tuple(c["range"])
        ch.name = ch.name or ch.physical
        chans.append(ch)
    cfg = Config(
        backend=dev.get("backend", "nidaqmx"),
        device=dev.get("device", "Dev1"),
        sample_rate=float(dev.get("sample_rate", 1000)),
        chunk_size=int(dev.get("chunk_size", 100)),
        terminal_config=dev.get("terminal_config", "RSE").upper(),
        channels=chans,
        window_s=float(disp.get("window_s", 10)),
        data_dir=rec.get("directory", "data"),
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not cfg.channels:
        raise ValueError("Config has no [[channels]].")
    if cfg.backend not in ("nidaqmx", "multichannelio", "simulated"):
        raise ValueError(f"Unknown backend {cfg.backend!r}")
    if cfg.terminal_config not in ("RSE", "DIFF"):
        raise ValueError("terminal_config must be RSE or DIFF (USB-600x has no NRSE).")
    if cfg.backend == "simulated":
        return
    if cfg.aggregate_rate > USB6003_MAX_AGGREGATE_RATE:
        raise ValueError(
            f"{len(cfg.channels)} ch x {cfg.sample_rate:g} Hz = {cfg.aggregate_rate:g} S/s "
            f"exceeds the USB-6003's {USB6003_MAX_AGGREGATE_RATE} S/s aggregate limit.")
    max_ai = 3 if cfg.terminal_config == "DIFF" else 7
    for ch in cfg.channels:
        if not ch.physical.startswith("ai") or int(ch.physical[2:]) > max_ai:
            raise ValueError(f"{ch.physical}: valid inputs are ai0-ai{max_ai} in {cfg.terminal_config} mode.")
        if tuple(ch.range) != USB6003_RANGE:
            warnings.warn(f"{ch.name}: USB-6003 only supports +/-10 V; the driver will coerce "
                          f"{ch.range} to that. It does NOT add gain.")
    if cfg.backend == "multichannelio" and cfg.terminal_config != "RSE":
        raise ValueError("MultiChannelIO hardcodes RSE wiring; use the nidaqmx backend for DIFF.")
    if cfg.chunk_size / cfg.sample_rate > 0.5:
        warnings.warn("chunk_size/sample_rate > 0.5 s: display will update sluggishly.")
