#!/usr/bin/env python3
"""Interactive human annotator (OpenCV-based, single-file).

Default workflow ("dense" mode):
  - Frames are pre-extracted at --fps (default 2 Hz) so the cursor steps
    through one labeling decision every 0.5 s of video.
  - Press 1-5 to set the comfort score for the current frame; the cursor
    auto-advances one step.
  - Press 0 to clear the current frame's label.
  - Use arrows to step manually, Space to play/pause through your labels.
  - Press 's' to save and quit, 'q' to quit without saving.

Controls:
  1..5    set comfort score (1=clearly refusing, 5=clearly comfortable)
  0       clear current frame's score
  ←/→     step backward/forward one frame
  ↓/↑     step backward/forward 5 frames
  Space   toggle auto-play (uses your label rate as playback fps)
  e       mark/clear an event flag at current frame (typed into rationale)
  n       add a free-text note (terminal stdin) on the current frame
  s       save and quit
  q       quit without saving
  ?/h     toggle help overlay

Output: annotations/<scenario>/<bag_stem>__human.parquet, same schema as
the VLM/heuristic annotators (one row per sampled frame; unlabeled rows
have comfort_score=NaN).

Usage:
    python scripts/annotate_human.py path/to/data/sc02_comfortable/Bright/foo.bag
    python scripts/annotate_human.py --interactive    # pick from list
    python scripts/annotate_human.py path/to/file.bag --fps 4 --rater alice
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


COLOR_BG = (32, 32, 32)
COLOR_TEXT = (240, 240, 240)
COLOR_DIM = (160, 160, 160)
COLOR_CURSOR = (0, 200, 255)
SCORE_COLORS = {
    1: (60, 60, 230),     # red
    2: (50, 130, 230),    # orange
    3: (60, 200, 230),    # yellow
    4: (140, 220, 100),   # light green
    5: (60, 200, 60),     # green
}
COLOR_NONE = (90, 90, 90)


def _scenario_from_path(bag_path: Path) -> str:
    parts = bag_path.parts
    for p in parts:
        if p.startswith("sc") and "_" in p:
            return p
    return bag_path.parent.parent.name


def select_bag_interactively(data_root: Path) -> Path | None:
    bags = find_bags(data_root)
    if not bags:
        print(f"No .bag files under {data_root}")
        return None
    print(f"\nFound {len(bags)} bags:")
    for i, b in enumerate(bags, 1):
        try:
            rel = b.relative_to(data_root)
        except ValueError:
            rel = b
        print(f"  {i:>2}) {rel}")
    while True:
        try:
            choice = input("Select index: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        try:
            i = int(choice)
            if 1 <= i <= len(bags):
                return bags[i - 1]
        except ValueError:
            pass
        print(f"Enter 1..{len(bags)}.")


def precache_frames(bag_path: Path, fps: float, resize_max: int) -> list[tuple[float, int, np.ndarray]]:
    """Pre-decode frames at sampling fps. Returns (timestamp_s, frame_idx, bgr)."""
    out: list[tuple[float, int, np.ndarray]] = []
    print(f"Caching frames @ {fps} Hz from {bag_path.name} ...", flush=True)
    t0 = time.time()
    for s in iter_bag_frames(bag_path, target_fps=fps, resize_max_side=resize_max):
        out.append((s.timestamp_s, s.frame_idx, s.bgr))
    print(f"  cached {len(out)} frames in {time.time() - t0:.1f}s")
    return out


def render_ui(
    canvas: np.ndarray,
    frame: np.ndarray,
    info: dict,
    scores: list[int | None],
    cursor: int,
    show_help: bool,
) -> None:
    """Composite the UI onto `canvas` (in-place)."""
    canvas[:] = COLOR_BG

    h_main = canvas.shape[0] - 90
    fh, fw = frame.shape[:2]
    scale = min(canvas.shape[1] / fw, h_main / fh)
    sw, sh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (sw, sh))
    x0 = (canvas.shape[1] - sw) // 2
    canvas[0:sh, x0:x0 + sw] = resized

    title = (f"{info['scenario']} / {info['stem']}   "
             f"frame {cursor+1}/{len(scores)}   "
             f"t={info['t']:.2f}s   phase={info['phase']}   rater={info['rater']}")
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    cur_score = scores[cursor]
    cur_text = "—" if cur_score is None else str(cur_score)
    cur_color = COLOR_NONE if cur_score is None else SCORE_COLORS.get(cur_score, COLOR_TEXT)
    cv2.putText(canvas, f"score: {cur_text}", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cur_color, 2, cv2.LINE_AA)

    # Timeline
    bar_y0 = canvas.shape[0] - 64
    bar_y1 = canvas.shape[0] - 20
    bar_x0 = 12
    bar_x1 = canvas.shape[1] - 12
    n = len(scores)
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), (60, 60, 60), -1)
    if n > 0:
        cell_w = max(1.0, (bar_x1 - bar_x0) / n)
        for i, sc in enumerate(scores):
            color = SCORE_COLORS[sc] if sc is not None else COLOR_NONE
            x = int(round(bar_x0 + i * cell_w))
            x_next = int(round(bar_x0 + (i + 1) * cell_w))
            cv2.rectangle(canvas, (x, bar_y0), (max(x + 1, x_next), bar_y1), color, -1)
        # cursor marker
        cx = int(round(bar_x0 + cursor * cell_w))
        cv2.line(canvas, (cx, bar_y0 - 3), (cx, bar_y1 + 3), COLOR_CURSOR, 2)

    legend = "1-5 score | 0 clear | <-/-> step | Up/Dn ±5 | space play | e event | n note | s save | q quit | ? help"
    cv2.putText(canvas, legend, (12, canvas.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DIM, 1, cv2.LINE_AA)

    if show_help:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (40, 60), (canvas.shape[1] - 40, canvas.shape[0] - 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        lines = [
            "Comfort Likert scale:",
            "   5  clearly comfortable: engaged, relaxed, reaching",
            "   4  mostly comfortable: minor hesitation",
            "   3  ambiguous / neutral",
            "   2  uncomfortable: hesitation, withdrawing, mouth covered",
            "   1  clearly refusing or absent: turned away, distracted, leaving",
            "",
            "Keys:",
            "   1..5      set score and auto-advance one step",
            "   0         clear current frame's score",
            "   <- / ->   step ±1 frame",
            "   Up / Dn   step ±5 frames",
            "   Space     toggle play/pause (advances at sample rate)",
            "   e         toggle 'event' flag on this frame",
            "   n         type a note in the terminal for this frame",
            "   s         save and quit",
            "   q         quit without saving",
            "   ? / h     toggle this help",
        ]
        y = 95
        for line in lines:
            cv2.putText(canvas, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
            y += 22


def annotate_bag_interactively(
    bag_path: Path,
    out_path: Path,
    rater: str,
    fps: float,
    resize_max: int,
    auto_play_fps: float,
    data_root: Path,
) -> bool:
    samples = precache_frames(bag_path, fps=fps, resize_max=resize_max)
    if not samples:
        print(f"  ! no frames cached, aborting")
        return False

    sidecar = find_sidecar(bag_path)
    windows = windows_from_sidecar(sidecar) if sidecar is not None else None
    scen = scenario_of(bag_path, data_root) if data_root in bag_path.parents else _scenario_from_path(bag_path)

    n = len(samples)
    scores: list[int | None] = [None] * n
    events: list[bool] = [False] * n
    notes: list[str] = [""] * n

    if out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            for _, row in prev.iterrows():
                idx = int(row["frame_idx"])
                if 0 <= idx < n and not pd.isna(row.get("comfort_score_5", np.nan)):
                    val = int(round(float(row["comfort_score_5"])))
                    if 1 <= val <= 5:
                        scores[idx] = val
                if 0 <= idx < n and isinstance(row.get("rationale", ""), str):
                    notes[idx] = str(row["rationale"])
                    events[idx] = "[event]" in notes[idx]
            print(f"  loaded {sum(s is not None for s in scores)} prior labels from {out_path}")
        except Exception as e:
            print(f"  ! could not load prior labels ({e}); starting fresh")

    cursor = 0
    show_help = False
    playing = False
    last_play_t = time.time()
    play_period = 1.0 / max(auto_play_fps, 0.5)

    win = "human-annotator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    canvas = np.zeros((720, 960, 3), dtype=np.uint8)

    while True:
        t_s, _, frame = samples[cursor]
        phase = phase_at(windows, t_s) if windows is not None else "intent"
        info = {
            "scenario": scen,
            "stem": bag_path.stem,
            "t": float(t_s),
            "phase": phase,
            "rater": rater,
        }
        render_ui(canvas, frame, info, scores, cursor, show_help)
        cv2.imshow(win, canvas)
        delay_ms = 30
        key = cv2.waitKey(delay_ms) & 0xFFFF

        if playing and (time.time() - last_play_t) >= play_period:
            cursor = min(n - 1, cursor + 1)
            last_play_t = time.time()

        if key == 0xFFFF:
            continue
        c = key & 0xFF

        if c in (ord("q"), 27):  # q or ESC
            confirm = input("  Quit WITHOUT saving? [y/N]: ").strip().lower()
            if confirm == "y":
                cv2.destroyWindow(win)
                return False
        elif c == ord("s"):
            cv2.destroyWindow(win)
            break
        elif c == ord("?") or c == ord("h"):
            show_help = not show_help
        elif c in (ord("0"),):
            scores[cursor] = None
        elif ord("1") <= c <= ord("5"):
            scores[cursor] = int(chr(c))
            cursor = min(n - 1, cursor + 1)
        elif c == ord(" "):
            playing = not playing
            last_play_t = time.time()
        elif c == ord("e"):
            events[cursor] = not events[cursor]
            tag = "[event]"
            if events[cursor]:
                if tag not in notes[cursor]:
                    notes[cursor] = (tag + " " + notes[cursor]).strip()
            else:
                notes[cursor] = notes[cursor].replace(tag, "").strip()
        elif c == ord("n"):
            try:
                msg = input(f"  note for frame {cursor} (t={t_s:.2f}s): ").strip()
            except (EOFError, KeyboardInterrupt):
                msg = ""
            if msg:
                notes[cursor] = msg
        elif c in (81, ord(",")):  # left arrow / ,
            cursor = max(0, cursor - 1)
        elif c in (83, ord(".")):  # right arrow / .
            cursor = min(n - 1, cursor + 1)
        elif c in (82,):  # up arrow
            cursor = min(n - 1, cursor + 5)
        elif c in (84,):  # down arrow
            cursor = max(0, cursor - 5)

    rows: list[dict] = []
    for i, (t_s, frame_idx, _) in enumerate(samples):
        sc = scores[i]
        row = {
            "timestamp_s": float(t_s),
            "frame_idx": int(frame_idx),
            "comfort_score": float(score5_to_100(sc)) if sc is not None else float("nan"),
            "comfort_score_5": float(sc) if sc is not None else float("nan"),
            "confidence": float("nan"),
            "rationale": notes[i] if notes[i] else "",
            "source": "human",
            "scenario": scen,
            "bag_stem": bag_path.stem,
            "phase": phase_at(windows, float(t_s)) if windows is not None else "intent",
        }
        rows.append(row)
    df = to_typed(pd.DataFrame(rows))
    df.attrs["rater"] = rater
    write_annotations(df, out_path)
    print(f"  saved {len(df)} rows ({sum(s is not None for s in scores)} labeled) -> {out_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive human annotator.")
    parser.add_argument("bag", nargs="?", default=None, help="Path to a .bag file.")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--annotations-root", type=str, default=str(PROJECT_ROOT / "annotations"))
    parser.add_argument("--rater", type=str, default="anon", help="Rater id (string).")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Sampling rate (one labeling step every 1/fps seconds).")
    parser.add_argument("--resize-max", type=int, default=720,
                        help="Resize so max(h, w) == this (display only).")
    parser.add_argument("--auto-play-fps", type=float, default=2.0,
                        help="Playback rate when SPACE is pressed.")
    parser.add_argument("--interactive", action="store_true",
                        help="Pick a file from a list under --data-root.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if args.bag:
        bag_path = Path(args.bag).resolve()
    elif args.interactive:
        bag_path = select_bag_interactively(data_root)
    else:
        print("error: provide a .bag path or use --interactive")
        return 2
    if bag_path is None or not bag_path.exists():
        print(f"error: {bag_path} not found")
        return 2

    try:
        scen = scenario_of(bag_path, data_root)
    except ValueError:
        scen = _scenario_from_path(bag_path)

    out_path = annotation_path(Path(args.annotations_root), scen, bag_path.stem,
                               f"human-{args.rater}")
    ok = annotate_bag_interactively(
        bag_path=bag_path,
        out_path=out_path,
        rater=args.rater,
        fps=args.fps,
        resize_max=args.resize_max,
        auto_play_fps=args.auto_play_fps,
        data_root=data_root,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
