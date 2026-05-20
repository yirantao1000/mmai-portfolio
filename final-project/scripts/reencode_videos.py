#!/usr/bin/env python3
"""Re-encode OpenCV-produced MP4s (mp4v / MPEG-4 Part 2) to H.264 + faststart.

OpenCV's default `cv2.VideoWriter_fourcc(*'mp4v')` writes MPEG-4 Part 2
streams that most modern players (Windows Media Player, Cursor preview,
browsers) refuse to play — they look "corrupted" but are actually fine,
just the wrong codec.

This script uses the ffmpeg binary bundled with imageio-ffmpeg to losslessly
re-mux into H.264 inside an MP4 container, with the moov atom moved to the
front for streaming (`+faststart`).

Usage:
    python scripts/reencode_videos.py renders/v1_heuristic_vs_vlm
    python scripts/reencode_videos.py renders/v1_heuristic_vs_vlm renders/v1_eval_*
    python scripts/reencode_videos.py --crf 28 renders/v1_heuristic_vs_vlm   # smaller files

By default the script overwrites the originals (after a successful encode);
pass `--keep-original` to leave them in place and write *.h264.mp4 alongside.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def get_ffmpeg() -> str:
    """Return path to ffmpeg, preferring the bundled imageio-ffmpeg copy."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        raise RuntimeError(
            "ffmpeg not found. Install imageio-ffmpeg (`pip install imageio-ffmpeg`) "
            "or put a system ffmpeg on PATH."
        )


def reencode(
    ffmpeg: str,
    src: Path,
    dst: Path,
    crf: int = 23,
    preset: str = "veryfast",
) -> tuple[bool, str]:
    """Re-encode `src` -> `dst` (H.264 + faststart). Returns (ok, message)."""
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"
    if result.returncode != 0:
        return False, result.stderr.strip().splitlines()[-1] if result.stderr else "ffmpeg failed"
    if not dst.exists() or dst.stat().st_size == 0:
        return False, "output missing or empty"
    return True, "ok"


def looks_already_h264(path: Path) -> bool:
    """Cheap check: read the first ~256 bytes and look for 'avc1'/'h264'/'H264'."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return False
    return b"avc1" in head or b"h264" in head or b"H264" in head


def process_dir(
    ffmpeg: str,
    directory: Path,
    crf: int,
    preset: str,
    keep_original: bool,
    skip_existing_h264: bool,
) -> tuple[int, int, int]:
    """Process all .mp4 files in `directory` (non-recursive)."""
    if not directory.exists():
        print(f"  ! does not exist: {directory}")
        return 0, 0, 0
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".mp4")
    converted = skipped = failed = 0
    total_in = total_out = 0
    for i, src in enumerate(files, 1):
        if skip_existing_h264 and looks_already_h264(src):
            print(f"  [{i:3d}/{len(files)}] {src.name}: already h264, skipping")
            skipped += 1
            continue
        in_size = src.stat().st_size
        if keep_original:
            dst_final = src.with_suffix(".h264.mp4")
        else:
            dst_final = src
        tmp = dst_final.with_name(dst_final.stem + ".__tmp__.mp4")
        t0 = time.time()
        ok, msg = reencode(ffmpeg, src, tmp, crf=crf, preset=preset)
        if not ok:
            print(f"  [{i:3d}/{len(files)}] {src.name}: FAILED ({msg})")
            tmp.unlink(missing_ok=True)
            failed += 1
            continue
        if not keep_original:
            os.replace(tmp, src)
            out_size = src.stat().st_size
        else:
            os.replace(tmp, dst_final)
            out_size = dst_final.stat().st_size
        total_in += in_size
        total_out += out_size
        elapsed = time.time() - t0
        ratio = (out_size / in_size * 100) if in_size > 0 else 0.0
        print(
            f"  [{i:3d}/{len(files)}] {src.name}: "
            f"{in_size/1024:6.0f} KB -> {out_size/1024:6.0f} KB ({ratio:5.1f}%) "
            f"in {elapsed:.1f}s"
        )
        converted += 1
    if total_in > 0:
        print(
            f"  totals: {converted} converted, {skipped} skipped, {failed} failed; "
            f"size {total_in/1024/1024:.1f} MB -> {total_out/1024/1024:.1f} MB "
            f"({total_out/total_in*100:.1f}%)"
        )
    else:
        print(f"  totals: {converted} converted, {skipped} skipped, {failed} failed")
    return converted, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-encode OpenCV mp4v files to H.264.")
    parser.add_argument("paths", nargs="+", help="One or more directories (non-recursive).")
    parser.add_argument("--crf", type=int, default=23, help="x264 CRF (lower=better quality, 18-28 typical).")
    parser.add_argument("--preset", type=str, default="veryfast",
                        help="x264 preset (ultrafast..veryslow). veryfast is a good speed/size balance.")
    parser.add_argument("--keep-original", action="store_true",
                        help="Write *.h264.mp4 alongside originals instead of overwriting.")
    parser.add_argument("--skip-existing-h264", action="store_true",
                        help="Skip files whose header already advertises H.264.")
    args = parser.parse_args()

    ffmpeg = get_ffmpeg()
    print(f"using ffmpeg: {ffmpeg}\n")

    grand_conv = grand_skip = grand_fail = 0
    for p in args.paths:
        d = Path(p)
        print(f"[dir] {d}")
        c, s, f = process_dir(
            ffmpeg, d,
            crf=args.crf, preset=args.preset,
            keep_original=args.keep_original,
            skip_existing_h264=args.skip_existing_h264,
        )
        grand_conv += c
        grand_skip += s
        grand_fail += f
        print()

    print(f"DONE. converted={grand_conv}, skipped={grand_skip}, failed={grand_fail}")
    return 0 if grand_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
