"""Load, validate and save the acquisition config (TOML)."""
from __future__ import annotations

import json
import tomllib
import warnings
from dataclasses import dataclass, field, replace
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
    enabled: bool = True

    def to_units(self, volts):
        return volts * self.scale + self.offset


@dataclass
class Config:
    backend: str = "nidaqmx"      # "nidaqmx" | "multichannelio" | "simulated"
    device: str = "Dev1"
    sample_rate: float = 1000.0   # per channel
    chunk_size: int = 100         # samples per channel per read
    terminal_config: str = "RSE"  # "RSE" | "DIFF"
    channels: list[Channel] = field(default_factory=list)        # switched on: acquired and shown
    spare_channels: list[Channel] = field(default_factory=list)  # switched off: kept so they can be switched back on
    window_s: float = 10.0
    data_dir: str = "data"

    @property
    def aggregate_rate(self) -> float:
        return self.sample_rate * len(self.channels)

    def with_channels(self, all_channels: list[Channel], **changes) -> "Config":
        """A copy with a new channel list (on and off), plus any other field changes."""
        return replace(self, channels=[c for c in all_channels if c.enabled],
                       spare_channels=[c for c in all_channels if not c.enabled], **changes)


def valid_inputs(cfg: Config) -> list[str]:
    """Inputs available in the current wiring mode: in DIFF, ai4-ai7 are the minus inputs of ai0-ai3."""
    return [f"ai{i}" for i in range(4 if cfg.terminal_config == "DIFF" else 8)]


def load(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
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
        window_s=float(disp.get("window_s", 10)),
        data_dir=rec.get("directory", "data"),
    ).with_channels(chans)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not cfg.channels:
        raise ValueError("At least one channel must be on." if cfg.spare_channels
                         else "Config has no [[channels]].")
    if cfg.backend not in ("nidaqmx", "multichannelio", "simulated"):
        raise ValueError(f"Unknown backend {cfg.backend!r}")
    if cfg.terminal_config not in ("RSE", "DIFF"):
        raise ValueError("terminal_config must be RSE or DIFF (USB-600x has no NRSE).")
    if cfg.sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if cfg.aggregate_rate > USB6003_MAX_AGGREGATE_RATE:
        n = len(cfg.channels)
        raise ValueError(
            f"{n} ch × {cfg.sample_rate:g} Hz = {cfg.aggregate_rate:g} S/s, over the USB-6003's "
            f"{USB6003_MAX_AGGREGATE_RATE:,} S/s limit. With {n} channel(s), use at most "
            f"{USB6003_MAX_AGGREGATE_RATE // n:,} Hz per channel.")
    allowed = valid_inputs(cfg)
    seen: dict[str, str] = {}
    names: set[str] = set()
    for ch in cfg.channels:
        if ch.physical not in allowed:
            raise ValueError(f"{ch.name}: {ch.physical} isn't available; use {allowed[0]}–{allowed[-1]} "
                             f"in {cfg.terminal_config} mode.")
        if ch.physical in seen:
            raise ValueError(f"{seen[ch.physical]} and {ch.name} both use {ch.physical}. "
                             "Each input can only be one channel.")
        if not ch.name.strip():
            raise ValueError("Every channel needs a name.")
        if ch.name in names:
            raise ValueError(f"Two channels are named {ch.name!r}; give each its own name.")
        seen[ch.physical] = ch.name
        names.add(ch.name)
    if cfg.backend == "simulated":
        return
    for ch in cfg.channels:
        if tuple(ch.range) != USB6003_RANGE:
            warnings.warn(f"{ch.name}: USB-6003 only supports +/-10 V; the driver will coerce "
                          f"{ch.range} to that. It does NOT add gain.")
    if cfg.backend == "multichannelio" and cfg.terminal_config != "RSE":
        raise ValueError("MultiChannelIO hardcodes RSE wiring; use the nidaqmx backend for DIFF.")
    if cfg.chunk_size / cfg.sample_rate > 0.5:
        warnings.warn("chunk_size/sample_rate > 0.5 s: display will update sluggishly.")


def _s(x: str) -> str:
    return json.dumps(x, ensure_ascii=False)  # JSON strings are valid TOML basic strings


def dumps(cfg: Config) -> str:
    """The config as TOML that `load` reads back, switched-off channels included."""
    lines = [
        "[device]",
        f"backend = {_s(cfg.backend)}",
        f"device = {_s(cfg.device)}",
        f"sample_rate = {cfg.sample_rate:g}",
        f"chunk_size = {cfg.chunk_size}",
        f"terminal_config = {_s(cfg.terminal_config)}",
        "",
        "[display]",
        f"window_s = {cfg.window_s:g}",
        "",
        "[recording]",
        f"directory = {_s(cfg.data_dir)}",
    ]
    for c in cfg.channels + cfg.spare_channels:
        lines += ["", "[[channels]]",
                  f"physical = {_s(c.physical)}",
                  f"name = {_s(c.name)}",
                  f"units = {_s(c.units)}",
                  f"scale = {c.scale:g}"]
        if c.offset:
            lines.append(f"offset = {c.offset:g}")
        if tuple(c.range) != USB6003_RANGE:
            lines.append(f"range = [{c.range[0]:g}, {c.range[1]:g}]")
        if not c.enabled:
            lines.append("enabled = false")
    return "\n".join(lines) + "\n"


def save(cfg: Config, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(dumps(cfg), encoding="utf-8")
    return path
