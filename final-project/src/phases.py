"""Interaction-phase definitions and sidecar-JSON parsing.

Each .bag recording has a sidecar JSON (see data_explanation.txt) with up to four
temporal keypoints that segment the interaction into three phases:

  [start_time] ──approach── [signal_time] ──intent── [handover|abort_time] ──execution── [end_time]

sc01 (walk-by) has no handover — `abort_time` closes the intent window and
`execution` is undefined.

Phase drives scoring weights at runtime. Calibration reads JSON keypoints; the
robot deployment in §7.3 drives phase directly from the controller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Phase = Literal["approach", "intent", "execution"]
PHASES: tuple[Phase, ...] = ("approach", "intent", "execution")


@dataclass(frozen=True)
class PhaseWindows:
    """Time intervals (seconds from recording start) for each phase.

    `execution` is None for sc01 (no handover occurred).
    """
    approach: tuple[float, float] | None
    intent: tuple[float, float] | None
    execution: tuple[float, float] | None


def find_sidecar(bag_path: str | Path) -> Path | None:
    """Locate the sidecar JSON for a `.bag`.

    Tries `<bag>.json` next to the bag first; otherwise looks one directory up
    (the layout used in `data/<scenario>/RawData_unlabelled_bagfiles/*.bag` with
    sidecars at `data/<scenario>/*.json`).
    """
    bag = Path(bag_path)
    stem = bag.stem
    candidates = [bag.with_suffix(".json"), bag.parent.parent / f"{stem}.json"]
    for c in candidates:
        if c.exists():
            return c
    return None


def windows_from_sidecar(json_path: str | Path) -> PhaseWindows:
    """Parse a sidecar JSON into phase time windows.

    Accepts either `handover_time` (sc02–sc05) or `abort_time` (sc01) as the
    intent→execution boundary.
    """
    with open(json_path) as f:
        data = json.load(f)
    labels = data.get("labels", {})

    start = labels.get("start_time")
    signal = labels.get("signal_time")
    handover = labels.get("handover_time")
    abort = labels.get("abort_time")
    end = labels.get("end_time")

    approach = (start, signal) if start is not None and signal is not None else None

    if handover is not None:
        intent = (signal, handover) if signal is not None else None
        execution = (handover, end) if end is not None else None
    elif abort is not None:
        intent = (signal, abort) if signal is not None else None
        execution = None
    else:
        intent = None
        execution = None

    return PhaseWindows(approach=approach, intent=intent, execution=execution)


def phase_at(windows: PhaseWindows, t_seconds: float) -> Phase:
    """Return the current phase at time `t_seconds`.

    Boundary rule: a timestamp exactly on the boundary belongs to the *later* phase.
    After the last-defined window closes we stick to the last phase rather than
    wrapping back to approach.
    """
    if windows.execution is not None and t_seconds >= windows.execution[0]:
        return "execution"
    if windows.intent is not None and t_seconds >= windows.intent[0]:
        return "intent"
    return "approach"
