"""Sessions on disk.

A session is a folder of numbered blocks, `block_001.h5`, `block_002.h5`, ...,
one per Record press (plus the exported `.mat`, if any). These helpers find the
blocks, check whether a session can be appended to with the current config,
and read a block for review.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .config import Config

_BLOCK_RE = re.compile(r"block_(\d+)\.h5")
# Device-coerced rates differ slightly from the requested rate.
_RATE_TOLERANCE = 0.005


class SessionError(Exception):
    pass


def _block_number(p: Path) -> int | None:
    m = _BLOCK_RE.fullmatch(p.name)
    return int(m.group(1)) if m else None


def list_blocks(folder: str | Path) -> list[Path]:
    """Block files in a session folder, in block-number order."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    blocks = [p for p in folder.iterdir() if p.is_file() and _block_number(p) is not None]
    return sorted(blocks, key=_block_number)


def is_session(folder: str | Path) -> bool:
    return bool(list_blocks(folder))


def next_block_path(folder: str | Path) -> Path:
    """Path for the next block. Never reuses a number, even if blocks were deleted."""
    nums = [_block_number(p) for p in list_blocks(folder)]
    return Path(folder) / f"block_{(max(nums) if nums else 0) + 1:03d}.h5"


def block_info(path: str | Path) -> dict:
    """Metadata of one block, without loading its data."""
    with h5py.File(path, "r") as f:
        a = f.attrs
        n = f["raw_volts"].shape[1]
        fs = float(a["sample_rate"])
        return {
            "channel_names": [str(x) for x in a["channel_names"]],
            "physical": [str(x) for x in a["physical"]],
            "units": [str(x) for x in a["units"]],
            "scale": [float(x) for x in a["scale"]],
            "offset": [float(x) for x in a["offset"]],
            "sample_rate": fs,
            "n_samples": int(n),
            "duration_s": n / fs if fs > 0 else float("nan"),
            "n_comments": int(f["comments"].shape[0]),
            "start_time": str(a["start_time"]),
            "device": str(a["device"]),
            "backend": str(a["backend"]),
            "terminal_config": str(a["terminal_config"]) if "terminal_config" in a else None,
        }


def append_check(folder: str | Path, cfg: Config) -> tuple[list[str], list[str]]:
    """Can new blocks recorded with `cfg` be added to this session?

    Returns (problems, warnings). Problems block appending, because the session
    would no longer export as one consistent file. Warnings are allowed but
    worth confirming, e.g. a changed amplifier scale.
    """
    blocks = list_blocks(folder)
    if not blocks:
        return [f"{folder} has no block_NNN.h5 files, so it isn't a session."], []
    info = block_info(blocks[-1])
    problems, warnings = [], []

    have_phys = [c.physical for c in cfg.channels]
    if info["physical"] != have_phys:
        problems.append(f"Input channels differ: session uses {info['physical']}, "
                        f"config uses {have_phys}.")
    have_names = [c.name for c in cfg.channels]
    if info["channel_names"] != have_names:
        problems.append(f"Channel names differ: session has {info['channel_names']}, "
                        f"config has {have_names}.")
    have_units = [c.units for c in cfg.channels]
    if info["units"] != have_units:
        problems.append(f"Units differ: session has {info['units']}, config has {have_units}.")
    fs = info["sample_rate"]
    if not np.isfinite(fs) or abs(fs - cfg.sample_rate) > _RATE_TOLERANCE * cfg.sample_rate:
        problems.append(f"Sample rate differs: session recorded at {fs:g} Hz, "
                        f"config asks for {cfg.sample_rate:g} Hz.")
    if info["terminal_config"] and info["terminal_config"] != cfg.terminal_config:
        problems.append(f"Wiring differs: session recorded in {info['terminal_config']} mode, "
                        f"config uses {cfg.terminal_config}.")

    if not problems:
        for c, s, o in zip(cfg.channels, info["scale"], info["offset"]):
            if not np.isclose(c.scale, s) or not np.isclose(c.offset, o):
                warnings.append(f"{c.name}: scale/offset changed from {s:g}/{o:g} to "
                                f"{c.scale:g}/{c.offset:g}. Did the amplifier gain change?")
    return problems, warnings


def read_block(path: str | Path) -> dict:
    """Load one block for review: scaled data as float32 to keep memory down."""
    with h5py.File(path, "r") as f:
        a = f.attrs
        fs = float(a["sample_rate"])
        raw = f["raw_volts"][:].astype(np.float32, copy=False)
        scale = np.asarray(a["scale"], dtype=np.float32)[:, None]
        offset = np.asarray(a["offset"], dtype=np.float32)[:, None]
        data = raw * scale + offset
        if "t" in f:
            t = f["t"][:]
        else:
            t = np.arange(raw.shape[1], dtype=np.float64) / fs
        comments = [(float(c["t"]), c["text"].decode() if isinstance(c["text"], bytes) else str(c["text"]))
                    for c in f["comments"][:]]
        return {
            "t": t, "data": data, "comments": comments, "sample_rate": fs,
            "channel_names": [str(x) for x in a["channel_names"]],
            "units": [str(x) for x in a["units"]],
            "terminal_config": str(a["terminal_config"]) if "terminal_config" in a else None,
            "duration_s": float(t[-1]) if t.size else 0.0,
        }


def add_recent(recent: list[str], path: str | Path, limit: int = 8) -> list[str]:
    """Move `path` to the front of the recent list, without duplicates."""
    p = str(Path(path).resolve())
    return ([p] + [r for r in recent if r != p])[:limit]
