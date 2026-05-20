import time

import cv2
import numpy as np

from .types import FrameResult


# Color anchors for comfort bar (BGR). Designed around the calibrated τ*=80:
#   <65   firm abort / walk-by baseline   red
#    80   τ* marginal                     yellow-green
#    85   clear comfort                   teal-green
# Piecewise-linear lerp between anchors prevents HUD flicker near band edges.
_COMFORT_ANCHORS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0,   (60,  60,  230)),   # red
    (65.0,  (0,   140, 255)),   # amber
    (80.0,  (120, 220, 180)),   # yellow-green (marginal, at τ*)
    (85.0,  (180, 200, 0)),     # teal-green (clear comfort)
    (100.0, (180, 200, 0)),     # hold teal-green above 85
)


def _lerp_comfort_color(score: float) -> tuple[int, int, int]:
    s = max(0.0, min(100.0, float(score)))
    for (x0, c0), (x1, c1) in zip(_COMFORT_ANCHORS, _COMFORT_ANCHORS[1:]):
        if s <= x1:
            t = (s - x0) / (x1 - x0) if x1 > x0 else 0.0
            return (
                int(round(c0[0] + t * (c1[0] - c0[0]))),
                int(round(c0[1] + t * (c1[1] - c0[1]))),
                int(round(c0[2] + t * (c1[2] - c0[2]))),
            )
    return _COMFORT_ANCHORS[-1][1]


class Visualizer:
    """Draws combined emotion + posture overlays on frames."""

    def __init__(self, config: dict):
        viz_cfg = config.get("visualization", config)
        self.show_bbox = viz_cfg.get("show_bbox", True)
        self.show_comfort_bar = viz_cfg.get("show_comfort_bar", True)
        self.show_emotion_text = viz_cfg.get("show_emotion_text", True)
        self.show_gaze_text = viz_cfg.get("show_gaze_text", True)
        self.show_posture_text = viz_cfg.get("show_posture_text", True)
        self.show_depth_marker = viz_cfg.get("show_depth_marker", True)
        self.show_state_warnings = viz_cfg.get("show_state_warnings", True)
        self.show_fps = viz_cfg.get("show_fps", True)
        self.bbox_thickness = viz_cfg.get("bbox_thickness", 2)
        self.abort_threshold = float(
            config.get("comfort", {}).get("abort_threshold", 80.0)
        )
        self._prev_time = time.time()
        self._fps = 0.0

    def _update_fps(self) -> None:
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            self._fps = 0.3 * (1.0 / dt) + 0.7 * self._fps
        self._prev_time = now

    def _comfort_color(self, score: float) -> tuple[int, int, int]:
        return _lerp_comfort_color(score)

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        self._update_fps()
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        score = result.integrated_comfort_score
        color = self._comfort_color(score)

        # --- Top: Integrated comfort bar ---
        if self.show_comfort_bar:
            bar_x, bar_y = 20, 30
            bar_w, bar_h = 200, 25
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * max(0.0, min(100.0, score)) / 100.0)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
            tau_x = bar_x + int(bar_w * self.abort_threshold / 100.0)
            cv2.line(overlay, (tau_x, bar_y - 3), (tau_x, bar_y + bar_h + 3),
                     (255, 255, 255), 1)
            cv2.putText(overlay, f"Comfort: {score:.0f}/100  (abort<={self.abort_threshold:.0f})",
                        (bar_x, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Face bounding box ---
        if self.show_bbox and result.face_bbox is not None:
            bbox = result.face_bbox
            cv2.rectangle(overlay,
                          (int(bbox.x1), int(bbox.y1)),
                          (int(bbox.x2), int(bbox.y2)),
                          color, self.bbox_thickness)

        # --- Left column: Emotion + Gaze ---
        if self.show_emotion_text and result.emotion is not None:
            em = result.emotion
            cv2.putText(overlay,
                        f"{em.dominant_emotion} (V:{em.valence:+.2f} A:{em.arousal:+.2f})",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(overlay,
                        f"Emotion Score: {result.emotion_comfort_score:.0f}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if self.show_gaze_text and result.gaze is not None:
            gz = result.gaze
            status = "Looking at camera" if gz.is_looking_at_camera else "Looking away"
            gaze_color = (0, 255, 0) if gz.is_looking_at_camera else (0, 0, 255)
            cv2.putText(overlay,
                        f"Gaze: {status} (Y:{gz.yaw:.1f} P:{gz.pitch:.1f})",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 2)

        # --- Right column: Posture ---
        if self.show_posture_text and result.pose is not None:
            pose = result.pose
            posture_color = (0, int(255 * pose.open_posture_score),
                             int(255 * (1 - pose.open_posture_score)))
            cv2.putText(overlay,
                        f"Posture: {pose.open_posture_score:.2f}",
                        (w - 260, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 2)

            if pose.interaction_z_meters is not None:
                cv2.putText(overlay,
                            f"Depth Z: {pose.interaction_z_meters:.2f}m",
                            (w - 260, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.putText(overlay,
                        f"Posture Score: {result.posture_comfort_score:.0f}",
                        (w - 260, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # --- Depth marker on closest wrist ---
        if self.show_depth_marker and result.pose is not None and result.pose.closest_wrist_px is not None:
            cv2.circle(overlay, result.pose.closest_wrist_px, 8, (255, 255, 0), -1)

        # --- Bottom center: State warnings ---
        if self.show_state_warnings and result.pose is not None:
            warning = None
            if result.pose.is_covering_mouth:
                warning = "STATE: SCARED (mouth/face covered)"
            elif result.pose.is_withdrawing:
                warning = "STATE: SUDDEN WITHDRAWAL"

            if warning:
                text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(overlay, warning, (text_x, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- FPS ---
        if self.show_fps:
            cv2.putText(overlay, f"FPS: {self._fps:.1f}", (w - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return overlay
