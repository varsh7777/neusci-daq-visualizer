"""Export recordings to a LabChart-style MATLAB .mat file.

Each HDF5 recording (one Record press) becomes one LabChart "block". The layout
matches what LabChart's "Export as MATLAB" writes and what neuscitk's
`LabChartDataset` reads:

  data        (1, N) float32   all samples, concatenated block by block,
                               channel by channel within each block,
                               in display units (volts * scale + offset)
  datastart   (n_ch, n_blocks) 1-based index of each channel/block in `data`
  dataend     (n_ch, n_blocks) 1-based inclusive end index
  samplerate  (n_ch, n_blocks) Hz
  titles      channel names
  unittext    unique unit strings; unittextmap (n_ch, n_blocks) 1-based into it
  blocktimes  (1, n_blocks) MATLAB datenum of each block's start
  firstsampleoffset (n_ch, n_blocks) zeros
  com         (n_comments, 5): [channel(-1 = all), block, sample, type, comtext index]
  comtext     comment strings
  tickrate    (1, n_blocks) Hz

The `com` column layout follows LabChart's export as I understand it; neuscitk
doesn't read comments, so check it against a real LabChart export if your MATLAB
scripts use them.

Command line:  python -m neusci_daq.export session.mat block_001.h5 block_002.h5 ...
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .recorder import load_recording

# MATLAB v5 .mat files can't hold a variable over 2 GiB; scipy can't write v7.3.
_MAX_BYTES = 2**31 - 1


def _datenum(iso: str) -> float:
    dt = datetime.fromisoformat(iso)
    return dt.toordinal() + 366 + (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400


def to_labchart_mat(h5_paths: list[str | Path], out_path: str | Path) -> Path:
    if not h5_paths:
        raise ValueError("No recordings to export.")
    recs = [load_recording(p) for p in h5_paths]

    names = list(recs[0]["attrs"]["channel_names"])
    units = list(recs[0]["attrs"]["units"])
    for p, r in zip(h5_paths, recs):
        if list(r["attrs"]["channel_names"]) != names:
            raise ValueError(f"{p} has different channels {list(r['attrs']['channel_names'])} "
                             f"from the first recording {names}; export them separately.")

    n_ch, n_blk = len(names), len(recs)
    total = sum(r["data"].size for r in recs)
    if total * 4 > _MAX_BYTES:
        raise ValueError(f"{total * 4 / 1e9:.2f} GB of data exceeds the 2 GB .mat limit. "
                         "Export fewer blocks per file.")

    data = np.empty(total, dtype=np.float32)
    datastart = np.zeros((n_ch, n_blk))
    dataend = np.zeros((n_ch, n_blk))
    samplerate = np.zeros((n_ch, n_blk))
    unittext = sorted(set(units))
    unittextmap = np.zeros((n_ch, n_blk))
    blocktimes = np.zeros((1, n_blk))
    tickrate = np.zeros((1, n_blk))
    com_rows, comtext = [], []

    pos = 0  # 0-based write position in `data`
    for b, r in enumerate(recs):
        fs = float(r["attrs"]["sample_rate"])
        if not np.isfinite(fs):
            raise ValueError(f"{h5_paths[b]} has no valid sample rate.")
        blocktimes[0, b] = _datenum(str(r["attrs"]["start_time"]))
        tickrate[0, b] = fs
        for c in range(n_ch):
            x = r["data"][c]
            data[pos:pos + x.size] = x
            datastart[c, b] = pos + 1           # MATLAB 1-based
            dataend[c, b] = pos + x.size        # inclusive end
            samplerate[c, b] = fs
            unittextmap[c, b] = unittext.index(units[c]) + 1
            pos += x.size
        for cm in r["comments"]:
            text = cm["text"].decode() if isinstance(cm["text"], bytes) else str(cm["text"])
            comtext.append(text)
            com_rows.append([-1, b + 1, int(cm["sample_index"]) + 1, 1, len(comtext)])

    out = {
        "data": data.reshape(1, -1),
        "datastart": datastart, "dataend": dataend, "samplerate": samplerate,
        "titles": np.array(names),
        "unittext": np.array(unittext),
        "unittextmap": unittextmap,
        "blocktimes": blocktimes,
        "firstsampleoffset": np.zeros((n_ch, n_blk)),
        "com": np.array(com_rows, dtype=float).reshape(-1, 5),
        "comtext": np.array(comtext if comtext else [""]),
        "tickrate": tickrate,
    }
    out_path = Path(out_path)
    savemat(out_path, out, do_compression=False)
    return out_path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print("usage: python -m neusci_daq.export OUT.mat REC1.h5 [REC2.h5 ...]")
        return 2
    path = to_labchart_mat(argv[1:], argv[0])
    print(f"Wrote {path} ({len(argv) - 1} block(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
