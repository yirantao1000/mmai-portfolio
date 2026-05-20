#!/usr/bin/env python3
"""Comfort-aware RBY1 interaction demo.

Behavior:
  1. When a person is detected, run the short POS1/POS2/POS3 sequence.
  2. When comfort is high, move from POS1 to POS4.
  3. Whenever comfort is low, cancel current control and return to POS1.

This script intentionally keeps the robot action policy small and explicit. The
vision loop runs in a background thread so low comfort can preempt long motions.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rby1_sdk as rby

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_comfort import ComfortSnapshot, LiveComfortMonitor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DEFAULT_ADDRESS = "192.168.30.1:50051"
DEFAULT_MODEL = "a"
DEFAULT_POWER = ".*"
DEFAULT_SERVO = "torso_.*|right_arm_.*|left_arm_.*"


POS1 = np.array(
    [
        0.31264445933094226,
        0.2670085308913444,
        0.0,
        0.7853981633974483,
        -1.5707963267948966,
        0.7853981633974483,
        0.0,
        0.0,
        -0.04196044972010946,
        -0.4211688538461661,
        0.009670913531546357,
        -2.2310801314259594,
        0.018196847096293416,
        -0.05137301658629012,
        1.2120058904123845,
        4.9360767927013206e-05,
        0.08566751122840555,
        1.8984910741158927e-05,
        -0.6628467675811193,
        2.6844663787998718e-05,
        1.2217083188957611,
        0.0,
        0.0,
        0.0030679609375,
    ],
    dtype=np.float64,
)

POS2 = np.array(
    [
        0.31104656267689473,
        0.2732083699090489,
        0.0,
        0.7853981633974483,
        -1.5707963267948966,
        0.7853981633974483,
        0.0,
        0.0,
        -0.04196044972010946,
        -0.4211574628997214,
        0.009670913531546357,
        -2.231042161604477,
        0.018193012144323704,
        -0.05136534668235069,
        1.9101052071712608,
        4.9360767927013206e-05,
        0.08566371424625731,
        1.8984910741158927e-05,
        -0.662842970598971,
        2.3009711818284616e-05,
        1.2217044839437914,
        0.0,
        -0.021475726562500002,
        0.03221358984375,
    ],
    dtype=np.float64,
)

POS3 = np.array(
    [
        0.31104656267689473,
        0.2732083699090489,
        0.0,
        0.7853981633974483,
        -1.5707963267948966,
        0.7853981633974483,
        0.0,
        0.0,
        -0.04196044972010946,
        -0.4211574628997214,
        0.009670913531546357,
        -2.231045958586625,
        0.018193012144323704,
        -0.05136918163432041,
        0.513799194998356,
        4.9360767927013206e-05,
        0.08566371424625731,
        1.8984910741158927e-05,
        -0.662842970598971,
        2.6844663787998718e-05,
        1.2217044839437914,
        0.0,
        -0.021475726562500002,
        0.030679609375,
    ],
    dtype=np.float64,
)

POS4 = np.array(
    [
        0.31072698334608523,
        0.2669446150251825,
        3.7969821482317853e-06,
        0.7853943664153001,
        -1.5708001237770448,
        0.7853943664153001,
        -7.593964296463571e-06,
        0.0,
        -1.251591631557347,
        -0.38176757009396484,
        0.015457514325451598,
        -0.5181399809298577,
        0.018196847096293416,
        0.12625045379495797,
        1.2120135603163238,
        4.9360767927013206e-05,
        0.0856143534783303,
        1.8984910741158927e-05,
        -0.6585599747357656,
        3.067961575771282e-05,
        1.2217083188957611,
        0.0,
        0.0,
        0.004601941406250001,
    ],
    dtype=np.float64,
)


@dataclass
class SharedPerception:
    snapshot: ComfortSnapshot | None = None
    updated_at: float = 0.0


def initialize_robot(address: str, model_name: str, power: str, servo: str):
    robot = rby.create_robot(address, model_name)
    if not robot.connect():
        raise RuntimeError(f"Failed to connect robot at {address}")

    if not robot.is_power_on(power) and not robot.power_on(power):
        raise RuntimeError(f"Failed to turn power on: {power}")

    if not robot.is_servo_on(servo) and not robot.servo_on(servo):
        raise RuntimeError(f"Failed to turn servo on: {servo}")

    state = robot.get_control_manager_state().state
    if state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        if not robot.reset_fault_control_manager():
            raise RuntimeError("Failed to reset control manager fault")

    if not robot.enable_control_manager():
        raise RuntimeError("Failed to enable control manager")

    return robot


def split_body_position(model, position: np.ndarray):
    body_dof = len(model.torso_idx) + len(model.right_arm_idx) + len(model.left_arm_idx)

    if len(position) == len(model.robot_joint_names):
        torso = position[model.torso_idx]
        right_arm = position[model.right_arm_idx]
        left_arm = position[model.left_arm_idx]
        return torso, right_arm, left_arm

    if len(position) == body_dof:
        torso_end = len(model.torso_idx)
        right_end = torso_end + len(model.right_arm_idx)
        torso = position[:torso_end]
        right_arm = position[torso_end:right_end]
        left_arm = position[right_end:]
        return torso, right_arm, left_arm

    raise ValueError(
        f"Position length {len(position)} does not match full DoF "
        f"({len(model.robot_joint_names)}) or body DoF ({body_dof})."
    )


def build_body_position_command(model, position: np.ndarray, minimum_time: float):
    torso, right_arm, left_arm = split_body_position(model, position)
    body = rby.BodyComponentBasedCommandBuilder()
    body.set_torso_command(
        rby.JointPositionCommandBuilder()
        .set_minimum_time(minimum_time)
        .set_position(torso)
    )
    body.set_right_arm_command(
        rby.JointPositionCommandBuilder()
        .set_minimum_time(minimum_time)
        .set_position(right_arm)
    )
    body.set_left_arm_command(
        rby.JointPositionCommandBuilder()
        .set_minimum_time(minimum_time)
        .set_position(left_arm)
    )
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body)
    )


class RobotMotion:
    def __init__(
        self,
        robot,
        model,
        stop_event: threading.Event,
        first_motion_emergency_delay_s: float,
    ):
        self.robot = robot
        self.model = model
        self.stop_event = stop_event
        self.first_motion_emergency_delay_s = first_motion_emergency_delay_s
        self.emergency_event = threading.Event()
        self._emergency_lock = threading.Lock()
        self._last_emergency_s = 0.0
        self._first_motion_started_at_s: float | None = None
        self._last_delay_log_s = 0.0

    def move_to(
        self,
        name: str,
        position: np.ndarray,
        minimum_time: float,
        priority: int = 1,
        raise_on_failure: bool = False,
        ignore_stop: bool = False,
    ) -> bool:
        if self.stop_event.is_set() and not ignore_stop:
            return False

        if self._first_motion_started_at_s is None and not name.startswith("pos1-emergency"):
            self._first_motion_started_at_s = time.monotonic()
            logging.info("First robot motion started; low-comfort home is armed after %.1fs.",
                         self.first_motion_emergency_delay_s)

        logging.info("Moving to %s (%.2fs, priority=%d)", name, minimum_time, priority)
        command = build_body_position_command(self.model, position, minimum_time)
        feedback = self.robot.send_command(command, priority=priority).get()

        if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            logging.warning("Move to %s finished with %s", name, feedback.finish_code)
            if raise_on_failure:
                raise RuntimeError(f"Failed to move robot to {name}: {feedback.finish_code}")
            return False
        return True

    def return_home_for_shutdown(self, minimum_time: float, priority: int) -> None:
        logging.info("Returning to POS1 before shutdown.")
        try:
            self.robot.cancel_control()
        except Exception as exc:
            logging.debug("cancel_control before shutdown failed or unavailable: %s", exc)

        try:
            self.move_to(
                "pos1-shutdown",
                POS1,
                minimum_time=minimum_time,
                priority=priority,
                raise_on_failure=False,
                ignore_stop=True,
            )
        except Exception:
            logging.exception("Shutdown POS1 command failed")

    def request_emergency_home(self, reason: str, minimum_time: float, priority: int) -> None:
        now = time.monotonic()
        with self._emergency_lock:
            if self._first_motion_started_at_s is None:
                if now - self._last_delay_log_s > 1.0:
                    logging.info(
                        "Low comfort detected before first motion; waiting until the first "
                        "motion has run for %.1fs before returning to POS1 (%s).",
                        self.first_motion_emergency_delay_s,
                        reason,
                    )
                    self._last_delay_log_s = now
                return

            elapsed = now - self._first_motion_started_at_s
            if elapsed < self.first_motion_emergency_delay_s:
                if now - self._last_delay_log_s > 0.5:
                    logging.info(
                        "Low comfort detected %.2fs after first motion start; delaying "
                        "POS1 return for another %.2fs (%s).",
                        elapsed,
                        self.first_motion_emergency_delay_s - elapsed,
                        reason,
                    )
                    self._last_delay_log_s = now
                return

            if now - self._last_emergency_s < 0.75:
                return
            self._last_emergency_s = now
            self.emergency_event.set()

        logging.warning("LOW COMFORT: returning to POS1 immediately (%s)", reason)
        try:
            self.robot.cancel_control()
        except Exception as exc:
            logging.debug("cancel_control failed or unavailable: %s", exc)

        try:
            self.move_to(
                "pos1-emergency",
                POS1,
                minimum_time=minimum_time,
                priority=priority,
                raise_on_failure=False,
            )
        except Exception:
            logging.exception("Emergency POS1 command failed")

    def clear_emergency_if_recovered(self, comfort: float, recovery_threshold: float) -> None:
        if self.emergency_event.is_set() and comfort >= recovery_threshold:
            logging.info(
                "Comfort recovered to %.1f >= %.1f; normal actions may resume",
                comfort,
                recovery_threshold,
            )
            self.emergency_event.clear()


def run_perception_loop(
    monitor: LiveComfortMonitor,
    shared: SharedPerception,
    shared_lock: threading.Lock,
    motion: RobotMotion,
    stop_event: threading.Event,
    low_threshold: float,
    recovery_threshold: float,
    emergency_time: float,
    emergency_priority: int,
    show_hud: bool,
    window_name: str,
) -> None:
    monitor.start()
    cv2 = None
    visualizer = None
    if show_hud:
        import cv2 as cv2_module

        from src.visualization import Visualizer

        cv2 = cv2_module
        visualizer = Visualizer(monitor.config)

    try:
        while not stop_event.is_set():
            snapshot = monitor.step()
            if snapshot is None:
                continue

            with shared_lock:
                shared.snapshot = snapshot
                shared.updated_at = time.monotonic()

            logging.info(
                "vision: comfort=%.1f phase=%s face=%s pose=%s abort=%s emotion=%s",
                snapshot.comfort,
                snapshot.phase,
                snapshot.face_detected,
                snapshot.pose_detected,
                snapshot.abort,
                snapshot.dominant_emotion,
            )

            if snapshot.comfort <= low_threshold or snapshot.abort:
                motion.request_emergency_home(
                    reason=f"comfort={snapshot.comfort:.1f}",
                    minimum_time=emergency_time,
                    priority=emergency_priority,
                )
            else:
                motion.clear_emergency_if_recovered(snapshot.comfort, recovery_threshold)

            if (
                show_hud
                and cv2 is not None
                and visualizer is not None
                and monitor.last_color_frame is not None
                and monitor.last_result is not None
            ):
                display = visualizer.draw(monitor.last_color_frame, monitor.last_result)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("HUD window requested stop via 'q'.")
                    stop_event.set()
                    break
    except Exception:
        stop_event.set()
        logging.exception("Perception loop failed")
    finally:
        monitor.stop()
        if cv2 is not None:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass


def latest_snapshot(
    shared: SharedPerception,
    shared_lock: threading.Lock,
    max_age_s: float,
) -> ComfortSnapshot | None:
    with shared_lock:
        if shared.snapshot is None:
            return None
        if time.monotonic() - shared.updated_at > max_age_s:
            return None
        return shared.snapshot


def person_detected(snapshot: ComfortSnapshot | None) -> bool:
    return bool(snapshot and (snapshot.face_detected or snapshot.pose_detected))


def comfort_is_high(snapshot: ComfortSnapshot | None, threshold: float) -> bool:
    return bool(snapshot and snapshot.comfort >= threshold)


def run_interruptible_sequence(
    motion: RobotMotion,
    sequence: list[tuple[str, np.ndarray, float]],
) -> bool:
    for name, position, minimum_time in sequence:
        if motion.stop_event.is_set() or motion.emergency_event.is_set():
            return False
        ok = motion.move_to(name, position, minimum_time, priority=1)
        if not ok or motion.emergency_event.is_set():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--address", type=str, default=DEFAULT_ADDRESS, help="Robot address")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Robot model name")
    parser.add_argument("--power", type=str, default=DEFAULT_POWER, help="Power regex")
    parser.add_argument("--servo", type=str, default=DEFAULT_SERVO, help="Servo regex")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "deploy.yaml"),
        help="Comfort detector config YAML",
    )
    parser.add_argument(
        "--initial-phase",
        choices=["approach", "intent", "execution"],
        default="intent",
        help="Initial perception phase",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=80.0,
        help="Return to POS1 when comfort is at or below this value.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=85.0,
        help="Run POS1->POS4 when comfort is at or above this value.",
    )
    parser.add_argument(
        "--recovery-threshold",
        type=float,
        default=83.0,
        help="Clear low-comfort emergency after comfort reaches this value.",
    )
    parser.add_argument(
        "--initial-time",
        type=float,
        default=0.8,
        help="Motion duration for moving to POS1 at the start of a sequence.",
    )
    parser.add_argument(
        "--step-time",
        type=float,
        default=0.5,
        help="Motion duration for each POS2/POS3/POS1 step in the short sequence.",
    )
    parser.add_argument("--pos4-time", type=float, default=1.5)
    parser.add_argument("--emergency-time", type=float, default=0.8)
    parser.add_argument("--emergency-priority", type=int, default=10)
    parser.add_argument(
        "--shutdown-home-time",
        type=float,
        default=1.0,
        help="Motion duration for returning to POS1 after Ctrl+C or shutdown.",
    )
    parser.add_argument(
        "--first-motion-emergency-delay",
        type=float,
        default=1.0,
        help=(
            "Minimum time after the first robot motion starts before a low-comfort "
            "event is allowed to command POS1."
        ),
    )
    parser.add_argument(
        "--snapshot-max-age",
        type=float,
        default=1.0,
        help="Ignore perception snapshots older than this many seconds.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the camera HUD window and print logs only.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Comfort-Aware Robot Interaction",
        help="OpenCV HUD window title.",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Stop after person-detected sequence and first high-comfort POS4 action.",
    )
    args = parser.parse_args()

    stop_event = threading.Event()
    shared = SharedPerception()
    shared_lock = threading.Lock()

    def handle_signal(_signum, _frame):
        logging.info("Stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logging.info("Connecting to robot at %s", args.address)
    robot = initialize_robot(args.address, args.model, args.power, args.servo)
    model = robot.model()
    motion = RobotMotion(
        robot,
        model,
        stop_event,
        first_motion_emergency_delay_s=args.first_motion_emergency_delay,
    )

    monitor = LiveComfortMonitor(
        config_path=args.config,
        initial_phase=args.initial_phase,
    )
    perception_thread = threading.Thread(
        target=run_perception_loop,
        name="comfort-perception",
        args=(
            monitor,
            shared,
            shared_lock,
            motion,
            stop_event,
            args.low_threshold,
            args.recovery_threshold,
            args.emergency_time,
            args.emergency_priority,
            not args.headless,
            args.window_name,
        ),
        daemon=True,
    )
    perception_thread.start()

    detected_sequence_done = False
    high_action_done = False

    try:
        while not stop_event.is_set():
            snapshot = latest_snapshot(shared, shared_lock, args.snapshot_max_age)

            if motion.emergency_event.is_set():
                high_action_done = False
                time.sleep(0.1)
                continue

            if not detected_sequence_done and person_detected(snapshot):
                logging.info("Person detected; running short POS sequence.")
                sequence = [
                    ("pos1", POS1, args.initial_time),
                    ("pos2", POS2, args.step_time),
                    ("pos3", POS3, args.step_time),
                    ("pos2", POS2, args.step_time),
                    ("pos3", POS3, args.step_time),
                    ("pos1", POS1, args.step_time),
                ]
                detected_sequence_done = run_interruptible_sequence(motion, sequence)
                continue

            if detected_sequence_done and not high_action_done and comfort_is_high(
                snapshot, args.high_threshold
            ):
                logging.info(
                    "Comfort %.1f >= %.1f; running POS1 -> POS4.",
                    snapshot.comfort,
                    args.high_threshold,
                )
                high_action_done = run_interruptible_sequence(
                    motion,
                    [
                        ("pos1", POS1, args.initial_time),
                        ("pos4", POS4, args.pos4_time),
                    ],
                )
                if args.one_shot and high_action_done:
                    logging.info("One-shot behavior complete.")
                    stop_event.set()
                continue

            time.sleep(0.1)
    finally:
        stop_event.set()
        perception_thread.join(timeout=2.0)
        motion.return_home_for_shutdown(
            minimum_time=args.shutdown_home_time,
            priority=args.emergency_priority,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
