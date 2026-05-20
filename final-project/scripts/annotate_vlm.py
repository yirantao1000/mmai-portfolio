#!/usr/bin/env python3
"""VLM annotator — uses OpenAI Responses API (gpt-5.5) to label each .bag with
a per-timestamp comfort score (1-5 ordinal).

The VLM only ever sees raw frames + their timestamps. It is NEVER told the
scenario name, the interaction phase, sidecar events, or any other human-
labelled metadata that would not be available at test time on the robot.

Design tricks (each maps to a paper-worthy methods choice):

  1. Sliding-window calls with overlap. Each call sees K frames at stride S < K
     so adjacent calls share (K - S) frames. The model returns one score per
     frame in the window; we average the predictions across overlapping windows.
     This stabilizes neighbor-frame scores and gives a "consensus across
     context" signal.

  2. Ordinal soft-distribution. We ask the model to output, for each frame, a
     soft distribution over 1..5 (probabilities sum to 1) — not a single
     integer. This unlocks ordinal-regression / KL-distillation training
     downstream.

  3. Rationale capture. We ask for a 1-sentence rationale per call (not per
     frame -> too expensive) which is logged for inspection / future rationale
     distillation.

  4. Temporal smoothing (post-process). After overlap-averaging we apply a
     causal EMA so the resulting label looks like what a real-time system
     would produce.

Output: annotations/<scenario>/<bag_stem>__vlm.parquet (one row per sampled
timestamp; default sample rate = 2 Hz). The `scenario` and `phase` columns in
the output parquet are filled in from directory / sidecar metadata after the
VLM has already run — they are NOT fed to the model.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/annotate_vlm.py --only sc02_comfortable
    python scripts/annotate_vlm.py --window 6 --stride 3 --fps 2
    python scripts/annotate_vlm.py --dry-run    # exercise the pipeline without spending tokens
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import (
    annotation_path,
    find_bags,
    iter_bag_frames,
    score5_to_100,
    scenario_of,
    to_typed,
    write_annotations,
)
from src.phases import find_sidecar, phase_at, windows_from_sidecar


# Default model — change via --model. As of 2026: gpt-5.5 (best), gpt-5.4-mini
# (cheaper), gpt-4.1-mini (very cheap, still vision-capable).
DEFAULT_MODEL = "gpt-5.5"


SYSTEM_PROMPT = """You are an expert annotator labeling human BEHAVIORAL comfort during a human-robot handover. A robot is offering a yellow cup; your job is to score whether the person is on track to accept it smoothly.

You are given a sequence of video frames sampled at low frame-rate; each frame is captioned with its timestamp only. You must judge the interaction from what you see — you are NOT told the scenario, the expected outcome, or what phase of the handover is occurring.

============================================================
CRITICAL CALIBRATION RULES — read carefully, models tend to get this wrong:

(R1) Score BEHAVIOR, not facial affect. A neutral expression is the DEFAULT
     state for a comfortable adult during a normal interaction. Do NOT
     downgrade a frame just because the person is not smiling. A smile is a
     bonus signal; its absence is NOT evidence of discomfort. Conversely, a
     polite smile while the body is withdrawing is still uncomfortable.

(R2) Cue priority — when cues conflict, the higher one wins:

       1. Acceptance action      — is the person reaching for / orienting
                                   toward the cup, or pulling back / turning
                                   away / stepping back?
       2. Body posture           — open, forward-leaning, hands ready
                                   vs. closed, tense, hands guarding the face
                                   or chest, arms folded.
       3. Gaze + head pose       — looking at cup/robot/hand vs. averted,
                                   looking down at phone, looking past the
                                   robot.
       4. Facial micro-affect    — stress (grimace, lip-press, jaw clench,
                                   widened eyes) vs. relief (smile, soft eyes).
                                   USE ONLY AS A TIE-BREAKER. Facial affect
                                   alone may shift the score by AT MOST 1 bin.

(R3) Default for a healthy mid-handover frame is 4, not 3. Only drop to 3 if
     you can name at least one mild concerning behavioral signal (brief
     pause with closed posture, brief gaze aversion, hesitant half-reach
     that pulls back). "I cannot read the expression clearly" is NOT a
     reason to drop to 3 — if the body is engaged and oriented, score 4.

(R4) "Absent" (no person visible, or only motion blur) is unambiguous: score
     1 with HIGH confidence. It is not "unreadable" — it is clearly not
     engaging.

============================================================
ORDINAL SCALE (1-5):

  5 = engaged AND actively accepting: clearly oriented toward the cup,
      reaching with hand(s), or hands already on / very near the cup. No
      withdrawal signals. Expression can be neutral.

  4 = engaged and on-track: oriented toward the robot/cup, stable posture,
      no withdrawal, action is in progress but not yet completed (e.g. still
      approaching, pausing briefly before reaching, mid-reach). This is the
      DEFAULT for a healthy frame.

  3 = ambiguous / mildly concerning: at least one behavioral red flag —
      brief gaze aversion, a hand pulling back slightly, posture closing —
      but not yet a clear refusal or withdrawal. Use sparingly.

  2 = uncomfortable: clear behavioral discomfort — withdrawing motion (hand
      pulled back from the cup), body turning away, stepping back, hand
      shielding the face/chest, head shake. The person is still in frame
      but is moving away from acceptance.

  1 = refusing or absent: the person is walking away / no longer in frame /
      distracted to phone / actively blocking the interaction with hands or
      body. This includes "no person visible".

============================================================
TIE-BREAKING DEFAULTS (use when truly ambiguous):
  - between 4 and 5 on a frame with stable engagement but no clear reach yet → 4
  - between 3 and 4 on a stable, oriented, non-withdrawing frame             → 4
  - between 2 and 3 on a frame with mild closed posture but no clear motion  → 3
  - between 1 and 2 on a partially-out-of-frame person still oriented        → 2

============================================================
OUTPUT FORMAT — return ONLY a JSON object:

{
  "frames": [
    {"t": <timestamp_seconds_as_float>,
     "p": [p1, p2, p3, p4, p5],   // soft distribution, must sum to 1.0
     "score": <integer 1..5>,     // argmax of p (your point estimate)
     "tag": "engaged|hesitant|withdraw|distract|absent|neutral"}
    ...one entry per input frame, in the same order...
  ],
  "rationale": "<one short sentence summarizing the BEHAVIORAL trajectory across this window — refer to actions, not feelings>"
}

Be calibrated and use the full 1-5 range. Do NOT assume the abort/continue label of the recording — judge each frame on what you actually see, following the rules above."""


# ---------- helpers ------------------------------------------------------------

def encode_jpeg(img_bgr: np.ndarray, quality: int = 70) -> str:
    """JPEG-encode a BGR image and return a base64 data URL."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_response(text: str) -> dict | None:
    """Extract the JSON object from a model response, robust to ``` fences."""
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Strip code fences and retry
        cleaned = re.sub(r"^```(?:json)?|```$", "", m.group(0).strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


@dataclass
class WindowResult:
    timestamps: list[float]            # one per frame in window (seconds)
    soft: np.ndarray                   # shape (K, 5), each row sums ~ 1.0
    rationale: str


def call_vlm(
    client,
    model: str,
    frames: list[tuple[float, np.ndarray, int]],   # [(t_s, bgr, frame_idx), ...]
    max_retries: int = 3,
) -> WindowResult | None:
    """One Responses-API call covering K frames. Returns soft (K,5) + rationale.

    `frames` is a list of (timestamp_s, bgr_image, frame_idx). The VLM is only
    shown timestamps and pixels — no scenario name, no phase, no sidecar info.
    """
    header = (
        f"This window contains {len(frames)} consecutive frames sampled "
        f"chronologically from a single recording.\n"
        f"Each frame is captioned with its timestamp in seconds."
    )

    content: list[dict] = [{"type": "input_text", "text": header}]
    for t_s, bgr, _fi in frames:
        content.append({
            "type": "input_text",
            "text": f"[t={t_s:.2f}s]",
        })
        content.append({
            "type": "input_image",
            "image_url": encode_jpeg(bgr),
        })
    content.append({"type": "input_text", "text": "Now produce the JSON described in the system prompt."})

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",   "content": content},
                ],
            )
            text = getattr(resp, "output_text", None) or ""
            data = parse_response(text)
            if not data or "frames" not in data:
                raise ValueError(f"could not parse JSON: {text[:300]}")

            entries = data["frames"]
            if len(entries) != len(frames):
                # Models occasionally drop or add a frame — align by index, pad
                # missing rows with uniform.
                aligned = [None] * len(frames)
                for i, e in enumerate(entries[:len(frames)]):
                    aligned[i] = e
                entries = aligned

            soft = np.full((len(frames), 5), 0.2, dtype=np.float32)
            for i, e in enumerate(entries):
                if e is None:
                    continue
                p = e.get("p")
                if isinstance(p, list) and len(p) == 5:
                    arr = np.asarray(p, dtype=np.float32)
                    arr = np.clip(arr, 0.0, None)
                    s = arr.sum()
                    if s > 0:
                        arr /= s
                    soft[i] = arr
                else:
                    score = int(e.get("score", 3))
                    score = max(1, min(5, score))
                    soft[i, score - 1] = 1.0

            rationale = str(data.get("rationale", "")).strip()
            return WindowResult(
                timestamps=[t for t, _, _ in frames],
                soft=soft,
                rationale=rationale,
            )
        except Exception as e:
            last_err = e
            wait_s = 2 ** attempt
            print(f"      ! VLM call failed (attempt {attempt+1}/{max_retries}): {e}; sleeping {wait_s}s",
                  flush=True)
            time.sleep(wait_s)

    print(f"      ! VLM call failed permanently: {last_err}", flush=True)
    return None


# ---------- per-bag pipeline ---------------------------------------------------

def sample_frames(bag_path: Path, fps: float, resize_max: int,
                  ) -> list[tuple[float, np.ndarray, int]]:
    """Sample frames from the bag. Returns (timestamp_s, bgr, frame_idx) tuples.

    Intentionally returns no phase information — the VLM should not be given
    any human-labelled metadata. Phase, if needed for the output parquet, is
    looked up separately in `annotate_bag`.
    """
    out: list[tuple[float, np.ndarray, int]] = []
    for s in iter_bag_frames(
        bag_path,
        target_fps=fps,
        resize_max_side=resize_max,
    ):
        out.append((float(s.timestamp_s), s.bgr, int(s.frame_idx)))
    return out


def overlap_average(
    n_frames: int,
    window_outputs: list[tuple[list[int], np.ndarray]],   # [(indices, soft (K,5))]
) -> tuple[np.ndarray, np.ndarray]:
    """Average soft predictions across overlapping windows."""
    soft_sum = np.zeros((n_frames, 5), dtype=np.float64)
    counts = np.zeros(n_frames, dtype=np.float64)
    for idxs, soft in window_outputs:
        for j, fi in enumerate(idxs):
            soft_sum[fi] += soft[j]
            counts[fi] += 1.0
    counts_safe = np.where(counts > 0, counts, 1.0)
    avg = soft_sum / counts_safe[:, None]
    return avg.astype(np.float32), counts.astype(np.float32)


def causal_ema(scores: np.ndarray, timestamps: np.ndarray, tau_s: float) -> np.ndarray:
    """Causal EMA, time-constant based (FPS-independent)."""
    if tau_s <= 0:
        return scores.astype(np.float32)
    out = np.empty_like(scores, dtype=np.float32)
    prev = float(scores[0])
    out[0] = prev
    for i in range(1, len(scores)):
        dt = max(1e-3, float(timestamps[i] - timestamps[i - 1]))
        alpha = 1.0 - float(np.exp(-dt / tau_s))
        prev = alpha * float(scores[i]) + (1 - alpha) * prev
        out[i] = prev
    return out


def annotate_bag(
    client,
    model: str,
    bag_path: Path,
    data_root: Path,
    fps: float,
    window_size: int,
    stride: int,
    resize_max: int,
    smoothing_tau_s: float,
    dry_run: bool = False,
    sleep_between: float = 0.0,
) -> pd.DataFrame | None:
    samples = sample_frames(bag_path, fps=fps, resize_max=resize_max)
    if not samples:
        print(f"      ! no frames sampled from {bag_path}")
        return None

    n = len(samples)
    timestamps = np.array([s[0] for s in samples], dtype=np.float32)
    scen = scenario_of(bag_path, data_root)
    print(f"      sampled {n} frames @ {fps} Hz")

    rationales: list[str] = []
    window_outputs: list[tuple[list[int], np.ndarray]] = []
    if dry_run:
        print(f"      [dry-run] would issue ~{max(1, (n - window_size) // max(stride, 1) + 1)} VLM calls")
        rng = np.random.default_rng(0)
        for start in range(0, max(1, n - window_size + 1), max(stride, 1)):
            idxs = list(range(start, min(start + window_size, n)))
            fake = rng.dirichlet(alpha=[1.0, 1.0, 2.0, 2.0, 1.0], size=len(idxs)).astype(np.float32)
            window_outputs.append((idxs, fake))
            rationales.append("[dry-run synthetic]")
    else:
        for start in range(0, max(1, n - window_size + 1), max(stride, 1)):
            idxs = list(range(start, min(start + window_size, n)))
            window_frames = [samples[i] for i in idxs]
            res = call_vlm(client, model, window_frames)
            if res is None:
                continue
            window_outputs.append((idxs, res.soft))
            if res.rationale:
                rationales.append(res.rationale)
            if sleep_between > 0:
                time.sleep(sleep_between)

    if not window_outputs:
        print(f"      ! all VLM calls failed for {bag_path.name}")
        return None

    soft_avg, counts = overlap_average(n, window_outputs)

    grid = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    expected_score_5 = (soft_avg * grid).sum(axis=1)
    score_100 = score5_to_100(expected_score_5)
    score_100_smoothed = causal_ema(score_100, timestamps, smoothing_tau_s)

    confidence = soft_avg.max(axis=1)

    # Phase is recovered ONLY for the output parquet (analysis metadata).
    # It is never shown to the VLM during inference.
    sidecar_path = find_sidecar(bag_path)
    phase_windows = windows_from_sidecar(sidecar_path) if sidecar_path is not None else None

    rows: list[dict] = []
    for i in range(n):
        t_s, _bgr, frame_idx = samples[i]
        rationale_idx = min(i // max(stride, 1), len(rationales) - 1) if rationales else -1
        phase_str = phase_at(phase_windows, float(t_s)) if phase_windows is not None else ""
        rows.append({
            "timestamp_s": float(t_s),
            "frame_idx": int(frame_idx),
            "comfort_score": float(score_100_smoothed[i]),
            "comfort_score_5": float(expected_score_5[i]),
            "confidence": float(confidence[i]),
            "rationale": rationales[rationale_idx] if rationale_idx >= 0 else "",
            "source": "vlm",
            "scenario": scen,
            "bag_stem": bag_path.stem,
            "phase": phase_str,
        })
    return to_typed(pd.DataFrame(rows))


# ---------- main ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="VLM annotator (OpenAI Responses API).")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--annotations-root", type=str, default=str(PROJECT_ROOT / "annotations"))
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional list of scenarios to process.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frame sampling rate fed to the VLM.")
    parser.add_argument("--window", type=int, default=6,
                        help="Number of frames per VLM call.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Stride between consecutive VLM calls (overlap = window - stride).")
    parser.add_argument("--resize-max", type=int, default=512,
                        help="Resize so max(h, w) == this before encoding (saves tokens).")
    parser.add_argument("--smooth-tau-s", type=float, default=0.6,
                        help="Causal EMA time constant (seconds).")
    parser.add_argument("--sleep-between", type=float, default=0.0,
                        help="Seconds to sleep between API calls (rate-limit guard).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the API; emit synthetic soft labels to exercise the pipeline.")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Optional split JSON; combined with --split-set to filter bags.")
    parser.add_argument("--split-set", type=str, default="test", choices=["train", "test"])
    args = parser.parse_args()

    if args.stride < 1 or args.window < 1 or args.stride > args.window:
        print("error: require 1 <= stride <= window")
        return 2

    data_root = Path(args.data_root)
    out_root = Path(args.annotations_root)
    bags = find_bags(data_root, args.only)

    if args.split_file:
        with open(args.split_file) as f:
            split_data = json.load(f)
        key = "all_test_stems" if args.split_set == "test" else "all_train_stems"
        keep_keys = {(s, t) for s, t in split_data.get(key, [])}
        bags = [b for b in bags if (scenario_of(b, data_root), b.stem) in keep_keys]
        print(f"  filtered to {len(bags)} bags via --split-file (set={args.split_set})")

    if not bags:
        print(f"No .bag files under {data_root}")
        return 1

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            print("error: `openai` package not installed. Run `pip install openai`.")
            return 1

        # Resolve API key: project-root `.openai_key` file takes precedence
        # over the OPENAI_API_KEY env var (so you can set-and-forget on
        # Windows without messing with shell env). The file is gitignored.
        api_key = os.environ.get("OPENAI_API_KEY")
        key_file = PROJECT_ROOT / ".openai_key"
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip() or api_key
        if not api_key:
            print(f"error: no OpenAI API key found.\n"
                  f"  put your key in {key_file}\n"
                  f"  or set the OPENAI_API_KEY env var.")
            return 1
        client = OpenAI(api_key=api_key)

    t0 = time.time()
    written = skipped = failed = 0
    for i, bag in enumerate(bags, 1):
        scen = scenario_of(bag, data_root)
        out_path = annotation_path(out_root, scen, bag.stem, "vlm")
        if out_path.exists() and not args.force:
            print(f"  [{i}/{len(bags)}] {scen}/{bag.name} — already annotated, skipping")
            skipped += 1
            continue
        print(f"  [{i}/{len(bags)}] annotating {scen}/{bag.name}", flush=True)
        df = annotate_bag(
            client, args.model, bag, data_root,
            fps=args.fps, window_size=args.window, stride=args.stride,
            resize_max=args.resize_max, smoothing_tau_s=args.smooth_tau_s,
            dry_run=args.dry_run, sleep_between=args.sleep_between,
        )
        if df is None:
            failed += 1
            continue
        write_annotations(df, out_path)
        try:
            disp = out_path.relative_to(PROJECT_ROOT)
        except ValueError:
            disp = out_path
        print(f"      wrote {len(df)} rows -> {disp}")
        written += 1

    print(f"\nDone in {time.time() - t0:.1f}s. wrote={written}, skipped={skipped}, failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
