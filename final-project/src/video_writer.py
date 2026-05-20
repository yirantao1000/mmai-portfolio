"""Cross-platform H.264 MP4 writer that produces files modern players accept.

OpenCV's `cv2.VideoWriter` on Windows defaults to MPEG-4 Part 2 (`mp4v`
fourcc), which is technically a valid MP4 but is rejected by most modern
players (Cursor preview, Windows Media Player, browsers). We instead pipe
raw BGR frames to a bundled ffmpeg binary (via imageio-ffmpeg) and ask for
H.264 + faststart, which is universally supported.

Usage:

    from src.video_writer import H264Writer

    with H264Writer(out_path, fps=15.0) as w:
        for frame_bgr in frames:
            w.write(frame_bgr)
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def _resolve_ffmpeg() -> str:
    """Return path to a working ffmpeg binary."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        raise RuntimeError(
            "ffmpeg not found. Install imageio-ffmpeg "
            "(`pip install imageio-ffmpeg`) or put a system ffmpeg on PATH."
        )


class H264Writer:
    """Stream BGR frames to an ffmpeg subprocess that encodes H.264 + faststart.

    Parameters
    ----------
    path:
        Output .mp4 path.
    fps:
        Output frame rate (must match the rate at which `write()` is called).
    crf:
        x264 CRF (lower=better quality, 18-28 typical).
    preset:
        x264 preset. `veryfast` is a good speed/size tradeoff for live encode.
    pix_fmt:
        Output pixel format. `yuv420p` is required for compatibility with
        Apple/Windows/web players.

    The ffmpeg subprocess is started lazily on the first `write()` so that
    frame dimensions are taken from the first frame.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        crf: int = 23,
        preset: str = "veryfast",
        pix_fmt: str = "yuv420p",
    ):
        self.path = Path(path)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = preset
        self.pix_fmt = pix_fmt

        self._proc: Optional[subprocess.Popen] = None
        self._size: Optional[tuple[int, int]] = None  # (w, h)
        self._ffmpeg = _resolve_ffmpeg()
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def _start(self, w: int, h: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{self.fps}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self.pix_fmt,
            "-movflags", "+faststart",
            "-an",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._size = (w, h)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR frame, got shape {frame_bgr.shape}")
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        h, w, _ = frame_bgr.shape
        if self._proc is None:
            self._start(w, h)
        elif (w, h) != self._size:
            raise ValueError(
                f"frame size changed mid-stream: {self._size} -> {(w, h)}"
            )
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg pipe broke; stderr:\n{stderr}")
        self._n_written += 1

    def release(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        rc = self._proc.wait(timeout=120)
        if rc != 0:
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg exited with code {rc}; stderr:\n{stderr}")
        self._proc = None

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None and self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
            return
        self.release()
