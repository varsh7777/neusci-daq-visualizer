"""Open LabChart-style .mat files for review: exports from LabChart itself
("Export as MATLAB"), and the .mat files this app's Export writes.

The .mat is read once and turned into a normal session folder of blocks
(<data_dir>/imported/<name>/block_001.h5, ...), which review mode opens like
any other session. The original .mat is never changed: comments you add or
colours you change go into the imported copy. Opening the same .mat again
reuses that copy (unless the .mat has changed since), so they're kept.

LabChart's layout (see export.py, which writes the same thing):
  data            all samples, block by block, channel by channel within each block
  datastart/end   (channels, blocks) 1-based positions in `data`; a channel that
                  was off in a block has no data there (start -1, or end < start)
  samplerate      (channels, blocks) Hz        tickrate  (1, blocks) Hz
  titles          channel names                unittext + unittextmap: units
  blocktimes      (1, blocks) MATLAB datenums  com + comtext: comments
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from .config import next_color, normalize_color
from .recorder import _COMMENT_DT

MARKER = "imported_from.json"


class MatError(ValueError):
    pass


def _strings(v) -> list[str]:
    """MATLAB char matrices and cell arrays as a list of trimmed strings."""
    if v is None:
        return []
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return [str(x).strip() for x in a.reshape(-1)]
    if a.dtype == object:
        out = []
        for x in a.reshape(-1):
            out += _strings(x) if not isinstance(x, str) else [x.strip()]
        return out
    return [str(x).strip() for x in a.reshape(-1)]


def _matrix(d: dict, key: str, n_ch: int, n_blk: int, fill=np.nan) -> np.ndarray:
    """A (channels, blocks) array, whatever shape scipy returned it in."""
    if key not in d:
        return np.full((n_ch, n_blk), fill, dtype=float)
    a = np.asarray(d[key], dtype=float)
    if a.size == n_ch * n_blk:
        return a.reshape(n_ch, n_blk) if a.shape != (n_blk, n_ch) or n_ch == n_blk else a.T
    if a.size == n_blk:                     # one value per block, the same for every channel
        return np.tile(a.reshape(1, n_blk), (n_ch, 1))
    if a.size == 1:
        return np.full((n_ch, n_blk), float(a.reshape(-1)[0]))
    raise MatError(f"'{key}' has {a.size} values; expected {n_ch} channels × {n_blk} blocks.")


def _datenum(v: float) -> str | None:
    """MATLAB datenum -> ISO time, or None."""
    try:
        if not np.isfinite(v) or v <= 0:
            return None
        whole = int(v)
        return (datetime.fromordinal(whole - 366) + timedelta(days=v - whole)).isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        return None


def read_labchart_mat(path: str | Path) -> list[dict]:
    """Blocks of a LabChart-style .mat: [{names, units, rate, data (ch, n), comments [(t, text)],
    start_time, notes}]. Raises MatError with a message for the window."""
    from scipy.io import loadmat
    path = Path(path)
    try:
        d = loadmat(path, squeeze_me=False, chars_as_strings=True)
    except NotImplementedError:
        raise MatError(f"{path.name} is a MATLAB 7.3 file, which can't be read here yet. In MATLAB: "
                       "load it and save it again with  save('copy.mat', '-v7')  — or export it from "
                       "LabChart again.") from None
    except Exception as e:
        raise MatError(f"{path.name} couldn't be read as a MATLAB file: {e}") from None
    missing = [k for k in ("data", "datastart", "dataend") if k not in d]
    if missing:
        have = ", ".join(k for k in d if not k.startswith("__")) or "nothing"
        raise MatError(f"{path.name} isn't a LabChart export: it has no {', '.join(missing)} "
                       f"(it has {have}). Export from LabChart with File › Export › MATLAB.")
    data = np.asarray(d["data"]).reshape(-1)
    starts = np.atleast_2d(np.asarray(d["datastart"], dtype=float))
    n_ch, n_blk = starts.shape
    if n_ch == 0 or n_blk == 0:
        raise MatError(f"{path.name} has no blocks.")
    ends = _matrix(d, "dataend", n_ch, n_blk)
    titles = _strings(d.get("titles"))
    titles = (titles + [f"Channel {i + 1}" for i in range(len(titles), n_ch)])[:n_ch]
    titles = [t or f"Channel {i + 1}" for i, t in enumerate(titles)]
    seen: dict[str, int] = {}
    for i, t in enumerate(titles):                 # names must be unique to tell channels apart
        if t in seen:
            seen[t] += 1
            titles[i] = f"{t} ({seen[t]})"
        else:
            seen[t] = 1
    rates = _matrix(d, "samplerate", n_ch, n_blk)
    tick = _matrix(d, "tickrate", 1, n_blk)[0] if "tickrate" in d else np.nanmax(rates, axis=0)
    unittext = _strings(d.get("unittext"))
    umap = _matrix(d, "unittextmap", n_ch, n_blk, fill=0)
    btimes = _matrix(d, "blocktimes", 1, n_blk)[0] if "blocktimes" in d else np.full(n_blk, np.nan)
    com = np.atleast_2d(np.asarray(d.get("com", np.zeros((0, 5))), dtype=float))
    comtext = _strings(d.get("comtext"))

    blocks = []
    for b in range(n_blk):
        chans, notes = [], []
        for c in range(n_ch):
            s, e = starts[c, b], ends[c, b]
            if not (np.isfinite(s) and np.isfinite(e)) or s < 1 or e < s:
                continue                            # channel off in this block
            if e > data.size:
                raise MatError(f"Block {b + 1}, {titles[c]}: dataend {int(e)} is past the end of the data.")
            fs = rates[c, b] if np.isfinite(rates[c, b]) and rates[c, b] > 0 else tick[b]
            if not (np.isfinite(fs) and fs > 0):
                raise MatError(f"Block {b + 1}, {titles[c]}: no sample rate in the file.")
            chans.append((c, np.asarray(data[int(s) - 1:int(e)], dtype=np.float32), float(fs)))
        if not chans:
            continue
        fast = max(fs for _, _, fs in chans)
        n = max(int(round(len(y) / fs * fast)) for _, y, fs in chans)
        rows = []
        for c, y, fs in chans:
            if fs != fast or len(y) != n:
                if fs != fast:
                    notes.append(f"{titles[c]} was recorded at {fs:g} Hz and is shown resampled to {fast:g} Hz.")
                t_src = np.arange(len(y)) / fs
                t_dst = np.arange(n) / fast
                y = np.interp(t_dst, t_src, y, right=np.nan).astype(np.float32)
            rows.append(y)
        comments = []
        tr = tick[b] if np.isfinite(tick[b]) and tick[b] > 0 else fast
        for row in com:
            if row.size < 3 or int(round(row[1])) != b + 1:
                continue
            idx = int(round(row[4])) - 1 if row.size >= 5 else len(comments)
            text = comtext[idx] if 0 <= idx < len(comtext) else ""
            ch = int(round(row[0]))
            if ch > 0 and ch <= n_ch and titles[ch - 1] not in [titles[c] for c, _, _ in chans]:
                continue
            if ch > 0 and ch <= n_ch:
                text = f"{text} [{titles[ch - 1]}]" if len(chans) > 1 else text
            comments.append(((row[2] - 1) / tr, text))
        blocks.append({
            "names": [titles[c] for c, _, _ in chans],
            "units": [unittext[int(umap[c, b]) - 1] if 1 <= umap[c, b] <= len(unittext) else ""
                      for c, _, _ in chans],
            "rate": fast,
            "data": np.vstack(rows),
            "comments": sorted(comments),
            "start_time": _datenum(btimes[b]),
            "notes": notes,
            "source_block": b + 1,
        })
    if not blocks:
        raise MatError(f"{path.name} has no data in any block.")
    return blocks


def _stamp(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime": st.st_mtime}


def import_mat(path: str | Path, dest_root: str | Path) -> tuple[Path, list[str]]:
    """Turn a .mat into a session folder under `dest_root` (or reuse one made
    earlier from the same, unchanged file). Returns (folder, notes)."""
    path = Path(path)
    if not path.is_file():
        raise MatError(f"{path} not found.")
    dest_root = Path(dest_root)
    stamp = _stamp(path)
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in path.stem) or "imported"
    k = 1
    while True:
        folder = dest_root / (stem if k == 1 else f"{stem}_{k}")
        marker = folder / MARKER
        if not folder.exists():
            break
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if {k: saved.get(k) for k in stamp} == stamp:
                return folder, list(saved.get("notes", []))   # same file as before: keep its comments
        except (OSError, ValueError, AttributeError):
            pass
        k += 1
    blocks = read_labchart_mat(path)
    folder.mkdir(parents=True)
    colors: dict[str, str] = {}
    notes: list[str] = []
    try:
        for i, b in enumerate(blocks, 1):
            for nm in b["names"]:
                if nm not in colors:
                    colors[nm] = next_color(colors.values())
            _write_block(folder / f"block_{i:03d}.h5", b, [normalize_color(colors[nm]) for nm in b["names"]], path)
            notes += [f"Block {i}: {n}" for n in b["notes"]]
    except Exception:
        for p in folder.iterdir():
            p.unlink()
        folder.rmdir()
        raise
    marker.write_text(json.dumps({**stamp, "notes": notes}), encoding="utf-8")
    return folder, notes


def _write_block(path: Path, b: dict, colors: list[str], source: Path) -> None:
    """One block in the same layout the recorder writes, with the data already in its units."""
    n = len(b["names"])
    with h5py.File(path, "w") as f:
        f.create_dataset("raw_volts", data=b["data"].astype(np.float32), maxshape=(n, None),
                         chunks=(n, max(1, min(4096, b["data"].shape[1]))))
        rows = np.zeros(len(b["comments"]), dtype=_COMMENT_DT)
        for k, (t, text) in enumerate(b["comments"]):
            rows[k] = (float(t), int(round(t * b["rate"])), text)
        f.create_dataset("comments", data=rows, maxshape=(None,))
        a = f.attrs
        a["sample_rate"] = b["rate"]
        a["hardware_timed"] = True
        a["device"], a["backend"] = "", "imported"
        a["start_time"] = b["start_time"] or datetime.fromtimestamp(source.stat().st_mtime).isoformat(timespec="seconds")
        a["channel_names"] = b["names"]
        a["physical"] = [""] * n
        a["units"] = b["units"]
        a["scale"] = [1.0] * n                 # the .mat holds values already in their units
        a["offset"] = [0.0] * n
        a["colors"] = colors
        a["imported_from"] = f"{source.name}, block {b['source_block']}"
