"""Phase-aware integrated comfort scorer.

Scoring is decomposed into three phases — approach, intent, execution — each with its
own `(emotion_weight, posture_weight, gaze_weight)`. Phase drives what the score should
emphasize: gaze/emotion during intent signaling, posture/depth during execution.

Phase selection is external. At runtime, the pipeline is told the current phase via
`set_phase()` by either the calibration harness (reading JSON sidecar keypoints) or
the robot controller (driven by its state machine).
"""
from __future__ import annotations

import math

from .phases import PHASES, Phase
from .types import FrameResult

_DEFAULT_PHASE: Phase = "intent"


def _phase_config(raw: dict, phase: Phase, fallback: dict) -> dict:
    """Look up a phase's weights, falling back to sensible defaults if missing."""
    return raw.get(phase, fallback)


class IntegratedComfortScorer:
    """Computes emotion + posture + integrated comfort with phase-aware weighting.

    Per-frame control flow:
      1. `update(result, timestamp_s)` computes instant emotion/posture scores.
      2. Each is EMA-smoothed with a time constant (in seconds) so the same config
         behaves identically at 13 FPS and 60 FPS.
      3. Integration weights `(we, wp)` depend on current phase, set via `set_phase()`.
      4. After a phase switch, the integrated output is frozen for `phase_warmup_s`
         seconds — this suppresses the EMA transient that would otherwise leak into
         the per-phase mean that the calibration objective reads.
    """

    def __init__(self, config: dict):
        pw = config.get("phase_weights", {})

        # Sensible defaults if a phase block is missing
        default_intent = {"emotion_weight": 0.6, "posture_weight": 0.4, "gaze_weight": 0.6}
        default_exec = {"emotion_weight": 0.35, "posture_weight": 0.65, "gaze_weight": 0.1}
        default_approach = {"emotion_weight": 0.5, "posture_weight": 0.5, "gaze_weight": 0.3}

        self._phase_weights: dict[Phase, dict] = {
            "approach": _phase_config(pw, "approach", default_approach),
            "intent": _phase_config(pw, "intent", default_intent),
            "execution": _phase_config(pw, "execution", default_exec),
        }

        # Emotion-instant coefficients (shared across phases; gaze contribution is phase-dep.)
        self.gamma = config.get("gamma", 0.5)
        self.delta = config.get("delta", 0.3)

        # Posture-instant penalties
        self.mouth_cover_penalty = config.get("mouth_cover_penalty", 30.0)
        self.withdrawal_penalty = config.get("withdrawal_penalty", 25.0)

        # EMA: time-constant based, so smoothing is FPS-independent
        self.ema_time_constant_s = float(config.get("ema_time_constant_s", 0.5))

        # Phase-switch warmup: freeze integrated output for N seconds after a switch
        self.phase_warmup_s = float(config.get("phase_warmup_s", 0.3))

        # No-detection decay (per-second rate — converted to per-frame via dt)
        self.no_face_decay_rate = float(config.get("no_face_decay_rate", 0.15))
        self.no_face_decay_target = float(config.get("no_face_decay_target", 35.0))
        self.no_pose_decay_rate = float(config.get("no_pose_decay_rate", 0.30))
        self.no_pose_decay_target = float(config.get("no_pose_decay_target", 50.0))

        self._phase: Phase = _DEFAULT_PHASE
        self._phase_changed_at_s: float | None = None
        self._smoothed_emotion = 50.0
        self._smoothed_posture = 50.0
        self._last_integrated = 50.0
        self._last_timestamp_s: float | None = None

    # -- Phase control -------------------------------------------------------

    def set_phase(self, phase: Phase) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        if phase != self._phase:
            self._phase = phase
            self._phase_changed_at_s = self._last_timestamp_s

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def current_posture_weight(self) -> float:
        """Final fusion posture weight for the active phase."""
        return float(self._phase_weights[self._phase].get("posture_weight", 0.0))

    # -- Core update ---------------------------------------------------------

    def _ema_alpha(self, dt_s: float) -> float:
        """Convert time constant to per-frame blend factor: α = 1 - exp(-dt/τ)."""
        if self.ema_time_constant_s <= 0 or dt_s <= 0:
            return 1.0
        return 1.0 - math.exp(-dt_s / self.ema_time_constant_s)

    def _compute_emotion_instant(self, result: FrameResult) -> float | None:
        if not result.face_detected or result.emotion is None:
            return None
        weights = self._phase_weights[self._phase]
        wg = float(weights.get("gaze_weight", 0.0))

        gaze_signal = 0.0
        if result.gaze is not None and result.gaze.is_looking_at_camera:
            gaze_signal = 1.0

        raw = self.gamma * result.emotion.valence - self.delta * result.emotion.arousal + wg * gaze_signal
        max_mag = self.gamma + self.delta + wg
        if max_mag <= 0:
            return 50.0
        instant = ((raw + max_mag) / (2 * max_mag)) * 100.0
        return max(0.0, min(100.0, instant))

    def _compute_posture_instant(self, result: FrameResult) -> float | None:
        if result.pose is None or not result.pose.has_pose:
            return None
        instant = result.pose.open_posture_score * 100.0
        if result.pose.is_covering_mouth:
            instant -= self.mouth_cover_penalty
        if result.pose.is_withdrawing:
            instant -= self.withdrawal_penalty
        return max(0.0, min(100.0, instant))

    def update(self, result: FrameResult, timestamp_s: float) -> tuple[float, float, float]:
        """Update smoothed scores and return (emotion, posture, integrated)."""
        if self._last_timestamp_s is None:
            dt_s = 1.0 / 30.0
        else:
            dt_s = max(1e-3, timestamp_s - self._last_timestamp_s)
        self._last_timestamp_s = timestamp_s

        alpha = self._ema_alpha(dt_s)

        # Emotion branch
        e_instant = self._compute_emotion_instant(result)
        if e_instant is None:
            # exponential decay toward the "no-face" floor (also time-constant based)
            decay_alpha = 1.0 - math.exp(-dt_s * self.no_face_decay_rate)
            self._smoothed_emotion += (self.no_face_decay_target - self._smoothed_emotion) * decay_alpha
        else:
            self._smoothed_emotion = alpha * e_instant + (1 - alpha) * self._smoothed_emotion

        # Posture branch
        p_instant = self._compute_posture_instant(result)
        if p_instant is None:
            decay_alpha = 1.0 - math.exp(-dt_s * self.no_pose_decay_rate)
            self._smoothed_posture += (self.no_pose_decay_target - self._smoothed_posture) * decay_alpha
        else:
            self._smoothed_posture = alpha * p_instant + (1 - alpha) * self._smoothed_posture

        # Phase-weighted integration
        w = self._phase_weights[self._phase]
        we = float(w.get("emotion_weight", 0.5))
        wp = float(w.get("posture_weight", 0.5))
        total = we + wp
        if total > 0:
            we, wp = we / total, wp / total
        integrated = we * self._smoothed_emotion + wp * self._smoothed_posture

        # Warmup: freeze integrated output for phase_warmup_s after a phase switch
        if (
            self._phase_changed_at_s is not None
            and (timestamp_s - self._phase_changed_at_s) < self.phase_warmup_s
        ):
            integrated = self._last_integrated
        else:
            self._last_integrated = integrated

        return self._smoothed_emotion, self._smoothed_posture, integrated

    def reset(self) -> None:
        self._phase = _DEFAULT_PHASE
        self._phase_changed_at_s = None
        self._smoothed_emotion = 50.0
        self._smoothed_posture = 50.0
        self._last_integrated = 50.0
        self._last_timestamp_s = None
