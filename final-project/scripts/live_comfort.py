#!/usr/bin/env python3
"""Live comfort scoring from a RealSense camera — deployable reference.

Designed to be dropped into a robot-arm control stack. Two usage modes:

1. As a script (demo / verification):
       python scripts/live_comfort.py
       python scripts/live_comfort.py --headless --verbose
       python scripts/live_comfort.py --phase execution --config config/deploy.yaml

2. As a library (robot integration):

       from scripts.live_comfort import LiveComfortMonitor

       monitor = LiveComfortMonitor(config_path="config/deploy.yaml")
       monitor.start()                        # opens RealSense + loads models
       try:
           while robot_running():
               monitor.set_phase("intent")    # drive from your state machine
               snapshot = monitor.step()      # pulls one frame, runs pipeline
               if snapshot is None:
                   continue                   # stream timeout — retry
               if snapshot.abort:
                   robot.abort_handover()
                   break
               robot.tick(snapshot.comfort)
       finally:
           monitor.stop()

`step()` returns a ComfortSnapshot — a flat, JSON-serializable view of the
pipeline's FrameResult with the abort decision already evaluated against the
calibrated τ* from the config. No pyrealsense2 or src.types symbols cross the
API boundary, so the monitor can be used across processes without plumbing.

The underlying pipeline is phase-aware (approach / intent / execution). Drive
`set_phase()` from the controller — the pipeline applies the phase-specific
fusion weights loaded from deploy.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import IntegratedPipeline
from src.types import FrameResult


Phase = Literal["approach", "intent", "execution"]


@dataclass
class ComfortSnapshot:
    """One frame's decision-relevant output, flattened for easy IPC.

    Everything here is JSON-serializable via `to_dict()`. Anything that would
    require an import of src.types on the consumer side stays out.
    """
    timestamp_ms: float
    phase: Phase

    comfort: float            # integrated 0-100, EMA-smoothed
    emotion_comfort: float    # 0-100
    posture_comfort: float    # 0-100

    abort: bool               # comfort <= abort_threshold (τ*)
    abort_threshold: float    # τ* from the loaded config

    face_detected: bool
    pose_detected: bool

    dominant_emotion: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    gaze_yaw_deg: Optional[float] = None
    gaze_pitch_deg: Optional[float] = None
    looking_at_camera: Optional[bool] = None

    interaction_z_m: Optional[float] = None
    is_covering_mouth: Optional[bool] = None
    is_withdrawing: Optional[bool] = None

    @classmethod
    def from_result(cls, result: FrameResult, phase: Phase, abort_threshold: float) -> "ComfortSnapshot":
        emo = result.emotion
        gaze = result.gaze
        pose = result.pose
        return cls(
            timestamp_ms=result.timestamp_ms,
            phase=phase,
            comfort=float(result.integrated_comfort_score),
            emotion_comfort=float(result.emotion_comfort_score),
            posture_comfort=float(result.posture_comfort_score),
            abort=result.integrated_comfort_score <= abort_threshold,
            abort_threshold=float(abort_threshold),
            face_detected=result.face_detected,
            pose_detected=bool(pose and pose.has_pose),
            dominant_emotion=emo.dominant_emotion if emo else None,
            valence=emo.valence if emo else None,
            arousal=emo.arousal if emo else None,
            gaze_yaw_deg=gaze.yaw if gaze else None,
            gaze_pitch_deg=gaze.pitch if gaze else None,
            looking_at_camera=gaze.is_looking_at_camera if gaze else None,
            interaction_z_m=pose.interaction_z_meters if pose else None,
            is_covering_mouth=pose.is_covering_mouth if pose else None,
            is_withdrawing=pose.is_withdrawing if pose else None,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class LiveComfortMonitor:
    """Live RealSense → pipeline → ComfortSnapshot.

    Lifecycle: __init__ → start() → step() (repeat) → stop().
    Models load lazily inside start(), so construction is cheap. Not thread-safe;
    if the controller runs it on a separate thread, own the lock externally.

    After each step(), the raw objects used to render it are available as:
        self.last_color_frame : np.ndarray (H, W, 3) BGR
        self.last_result      : src.types.FrameResult
    Integrations that need to draw their own HUD can consume these directly.
    """

    def __init__(
        self,
        config_path: str | Path = PROJECT_ROOT / "config" / "deploy.yaml",
        initial_phase: Phase = "intent",
        color_size: tuple[int, int] = (640, 480),
        depth_size: tuple[int, int] = (640, 480),
        fps: int = 30,
        device_serial: Optional[str] = None,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.abort_threshold = float(
            self.config.get("comfort", {}).get("abort_threshold", 80.0)
        )
        self._initial_phase: Phase = initial_phase
        self._current_phase: Phase = initial_phase

        self._color_size = color_size
        self._depth_size = depth_size
        self._fps = fps
        self._device_serial = device_serial

        self._pipeline: Optional[IntegratedPipeline] = None
        self._rs_pipeline = None
        self._align = None
        self._t0_ms: Optional[float] = None
        self._running = False

        self.last_color_frame: Optional[np.ndarray] = None
        self.last_result: Optional[FrameResult] = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Load models + open the RealSense stream. Idempotent."""
        if self._running:
            return

        self._pipeline = IntegratedPipeline(self.config)
        self._pipeline.load_models()
        self._pipeline.reset_state()
        self._pipeline.set_phase(self._initial_phase)

        import pyrealsense2 as rs

        cfg = rs.config()
        if self._device_serial:
            cfg.enable_device(self._device_serial)
        cfg.enable_stream(rs.stream.color, *self._color_size, rs.format.bgr8, self._fps)
        cfg.enable_stream(rs.stream.depth, *self._depth_size, rs.format.z16, self._fps)

        self._rs_pipeline = rs.pipeline()
        self._rs_pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        self._t0_ms = None
        self._running = True

    def stop(self) -> None:
        """Tear down the RealSense stream. Idempotent."""
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            finally:
                self._rs_pipeline = None
                self._align = None
        self._running = False

    def __enter__(self) -> "LiveComfortMonitor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- control ----------------------------------------------------------

    def set_phase(self, phase: Phase) -> None:
        """Drive phase from the controller's state machine.

        `approach`   — robot approaching user (gaze-leaning fusion weights)
        `intent`     — intent signaled, waiting for acceptance
        `execution`  — physical handover underway
        """
        self._current_phase = phase
        if self._pipeline is not None:
            self._pipeline.set_phase(phase)

    # -- step -------------------------------------------------------------

    def step(self, timeout_ms: int = 5000) -> Optional[ComfortSnapshot]:
        """Pull one aligned color+depth frame and score it.

        Returns None on stream timeout. Callers should treat repeated Nones as
        a camera disconnect. Otherwise returns a ComfortSnapshot; the raw
        frame and FrameResult are also cached on the instance for HUD rendering.
        """
        if not self._running or self._pipeline is None or self._rs_pipeline is None:
            raise RuntimeError("LiveComfortMonitor.step() called before start()")

        try:
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None

        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or color_frame.get_data_size() == 0:
            return None

        ts_ms = color_frame.get_timestamp()
        if self._t0_ms is None:
            self._t0_ms = ts_ms
        rel_ms = ts_ms - self._t0_ms

        color = np.asanyarray(color_frame.get_data()).copy()
        depth = depth_frame if depth_frame else None

        result = self._pipeline.process_frame(color, rel_ms, depth_frame=depth)

        self.last_color_frame = color
        self.last_result = result
        return ComfortSnapshot.from_result(result, self._current_phase, self.abort_threshold)

    # -- bulk driver ------------------------------------------------------

    def run(
        self,
        on_snapshot: Callable[[ComfortSnapshot], None],
        stop_condition: Optional[Callable[[ComfortSnapshot], bool]] = None,
    ) -> None:
        """Drive the camera loop, invoking `on_snapshot` each frame.

        The script mode uses this; most robot integrations will prefer to call
        step() from their own control loop for tighter timing/ownership.
        """
        if not self._running:
            self.start()
        try:
            while self._running:
                snap = self.step()
                if snap is None:
                    continue
                on_snapshot(snap)
                if stop_condition is not None and stop_condition(snap):
                    return
        except KeyboardInterrupt:
            return


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _format_snapshot(s: ComfortSnapshot, verbose: bool) -> str:
    tag = "ABORT" if s.abort else "ok   "
    line = (
        f"t={s.timestamp_ms/1000:6.2f}s  phase={s.phase:<9}  "
        f"comfort={s.comfort:5.1f}  E={s.emotion_comfort:5.1f}  P={s.posture_comfort:5.1f}  "
        f"[{tag}]"
    )
    if verbose:
        extra = []
        if s.dominant_emotion is not None:
            extra.append(f"emo={s.dominant_emotion}")
        if s.valence is not None:
            extra.append(f"V={s.valence:+.2f}")
        if s.gaze_yaw_deg is not None:
            extra.append(f"gaze=({s.gaze_yaw_deg:+.0f},{s.gaze_pitch_deg:+.0f})")
        if s.interaction_z_m is not None:
            extra.append(f"z={s.interaction_z_m:.2f}m")
        if extra:
            line += "  " + " ".join(extra)
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "config" / "deploy.yaml"),
                        help="Path to config YAML (default: config/deploy.yaml).")
    parser.add_argument("--phase", choices=["approach", "intent", "execution"],
                        default="intent",
                        help="Initial phase (controller overrides at runtime).")
    parser.add_argument("--headless", action="store_true",
                        help="Disable HUD window — prints scores to stdout only.")
    parser.add_argument("--device-serial", type=str, default=None,
                        help="Specific RealSense serial (optional).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print emotion/gaze/z detail each frame.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: until Ctrl-C / 'q').")
    args = parser.parse_args()

    cv2 = None
    viz = None
    if not args.headless:
        try:
            import cv2  # noqa: F401
            from src.visualization import Visualizer
        except ImportError:
            args.headless = True

    monitor = LiveComfortMonitor(
        config_path=args.config,
        initial_phase=args.phase,
        device_serial=args.device_serial,
    )
    monitor.start()

    if not args.headless:
        viz = Visualizer(monitor.config)

    t_start = time.time()
    try:
        while True:
            if args.duration is not None and (time.time() - t_start) >= args.duration:
                break

            snap = monitor.step()
            if snap is None:
                continue

            print(_format_snapshot(snap, args.verbose), flush=True)

            if not args.headless and viz is not None and monitor.last_color_frame is not None:
                display = viz.draw(monitor.last_color_frame, monitor.last_result)
                cv2.imshow("Live Comfort", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
