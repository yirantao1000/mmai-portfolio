#!/usr/bin/env python3
"""Download a Dropbox shared-folder zip to disk, then extract it.

The on-the-fly zips Dropbox generates can't be processed by stream-unzip
(NotStreamUnzippable on the first big entry — the zip uses a feature that
requires the trailing central directory to decode). So we do a two-step:

  step 1: download zip to data_incoming/dataset.zip
  step 2: extract via stdlib zipfile (which reads the central directory at
          the end of the file)

Resumes the download if a partial file exists and the server reports the
total size.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests


DEFAULT_URL = (
    "https://www.dropbox.com/scl/fo/neiz1q4c2izx9rbbgmaa0/"
    "AHjE2Cc7BoJAnwu_KQ96IvA?rlkey=olssevu8t51h7dxa9n0rddqlj&dl=1"
)


def get_total_size(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=30)
    cl = r.headers.get("Content-Length")
    if cl:
        return int(cl)
    return None


def download_to(url: str, dst: Path, chunk_size: int = 1 << 20) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = get_total_size(url)
    if total:
        print(f"server reports total: {total/1024/1024:.1f} MB", flush=True)

    # Dropbox zips don't support Range, so we always start fresh
    if dst.exists():
        existing = dst.stat().st_size
        if total and existing == total:
            print(f"existing complete file at {dst} ({existing/1024/1024:.1f} MB); reusing", flush=True)
            return dst
        print(f"existing partial file at {dst} ({existing/1024/1024:.1f} MB); restarting (Dropbox doesn't support Range)", flush=True)
        dst.unlink()

    print(f"downloading -> {dst}", flush=True)
    t0 = time.time()
    written = 0
    last_log = t0
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_log >= 5.0:
                    last_log = now
                    elapsed = now - t0
                    mbps = (written / 1024 / 1024) / max(elapsed, 1e-6)
                    if total:
                        pct = written / total * 100.0
                        eta = (total - written) / max(written / elapsed, 1.0)
                        print(f"  {written/1024/1024:7.1f} / {total/1024/1024:.1f} MB "
                              f"({pct:5.1f}%) @ {mbps:5.1f} MB/s  ETA {eta/60:.1f} min",
                              flush=True)
                    else:
                        print(f"  {written/1024/1024:7.1f} MB @ {mbps:5.1f} MB/s", flush=True)
    elapsed = time.time() - t0
    print(f"download done. {written/1024/1024:.1f} MB in {elapsed/60:.1f} min "
          f"({(written/1024/1024)/elapsed:.1f} MB/s avg)", flush=True)
    return dst


def extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nextracting {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        n = len(infos)
        print(f"  archive has {n} entries", flush=True)
        t0 = time.time()
        last = t0
        bytes_total = sum(info.file_size for info in infos if not info.is_dir())
        bytes_done = 0
        for i, info in enumerate(infos, 1):
            if info.is_dir():
                continue
            z.extract(info, out_dir)
            bytes_done += info.file_size
            now = time.time()
            if now - last >= 5.0 or i == n:
                last = now
                pct = bytes_done / bytes_total * 100 if bytes_total else 0
                print(f"  [{i}/{n}] {info.filename}  ({bytes_done/1024/1024:.0f} / "
                      f"{bytes_total/1024/1024:.0f} MB, {pct:.1f}%)", flush=True)
    print(f"extracted in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default="data_incoming", help="Output staging dir.")
    ap.add_argument("--zip-name", default="dataset.zip")
    ap.add_argument("--no-extract", action="store_true",
                    help="Stop after download; do not extract.")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip download; extract an existing zip.")
    args = ap.parse_args()

    out_root = Path(args.out)
    zip_path = out_root / args.zip_name

    if not args.no_download:
        download_to(args.url, zip_path)

    if not args.no_extract:
        extract(zip_path, out_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
