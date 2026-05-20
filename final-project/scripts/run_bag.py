#!/usr/bin/env python3
"""Run integrated emotion + posture detection on Intel RealSense .bag files."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import IntegratedPipeline
from src.bag_source import BagSource
from src.phases import find_sidecar, phase_at, windows_from_sidecar
from src.visualization import Visualizer


def _parse_selection(choice: str, bags: list[Path]) -> list[Path] | None:
    """Parse user input like '0', '3', '1,4,7', or '2-5' into a list of bag paths."""
    choice = choice.strip()
    if not choice:
        return None

    indices = set()
    for part in choice.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            try:
                start, end = part.split("-", 1)
                start, end = int(start), int(end)
                if start < 0 or end > len(bags):
                    return None
                indices.update(range(start, end + 1))
            except ValueError:
                return None
        else:
            try:
                indices.add(int(part))
            except ValueError:
                return None

    if not indices:
        return None
    if 0 in indices:
        return bags

    if any(i < 1 or i > len(bags) for i in indices):
        return None

    return [bags[i - 1] for i in sorted(indices)]


def find_bag_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".bag":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.bag"))
    return []


def load_sidecar(bag_path: Path) -> dict | None:
    json_path = find_sidecar(bag_path)
    if json_path is not None:
        with open(json_path) as f:
            return json.load(f)
    return None


def process_bag(bag_path: Path, pipeline: IntegratedPipeline, viz: Visualizer,
                headless: bool, save: bool = False, save_dir: Path | None = None) -> None:
    """Process a single .bag file through the integrated pipeline."""
    metadata = load_sidecar(bag_path)
    label = bag_path.parent.parent.name
    lighting = bag_path.parent.name

    header = f"[{label}/{lighting}] {bag_path.name}"
    if metadata:
        header += f" - {metadata.get('scenario_name', '')} ({metadata.get('duration_seconds', '?')}s)"
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")

    pipeline.reset_state()

    sidecar_path = find_sidecar(bag_path)
    windows = windows_from_sidecar(sidecar_path) if sidecar_path is not None else None
    if windows is None:
        pipeline.set_phase("intent")  # sensible default when no sidecar exists

    interaction_start_s = windows.approach[0] if windows and windows.approach else None
    if windows and windows.execution:
        interaction_end_s = windows.execution[1]
    elif windows and windows.intent:
        interaction_end_s = windows.intent[1]
    else:
        interaction_end_s = None
    if interaction_start_s is not None and interaction_end_s is not None:
        print(f"  interaction window: [{interaction_start_s:.2f}s, {interaction_end_s:.2f}s]")

    source = BagSource(str(bag_path), real_time=(not headless and not save))
    if not source.open():
        print(f"  Skipping: could not open {bag_path}")
        return

    writer = None
    out_path = None
    if save:
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{label}_{lighting}_{bag_path.stem}.mp4"
        else:
            out_path = bag_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (640, 480))
        print(f"  Saving to: {out_path}")

    frame_count = 0

    try:
        while True:
            ret, frame, depth_frame, timestamp_ms = source.read()
            if not ret or frame is None:
                break

            t_s = timestamp_ms / 1000.0
            if interaction_start_s is not None and t_s < interaction_start_s:
                continue
            if interaction_end_s is not None and t_s > interaction_end_s:
                break

            if windows is not None:
                pipeline.set_phase(phase_at(windows, t_s))

            result = pipeline.process_frame(frame, timestamp_ms, depth_frame=depth_frame)
            frame_count += 1

            show = not headless
            if show or save:
                display = viz.draw(frame, result)
                cv2.putText(display, f"{label}/{lighting}", (20, display.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                phase_str = f"  phase={pipeline.phase}" if windows is not None else ""
                cv2.putText(display, f"t={t_s:.1f}s  frame={frame_count}{phase_str}",
                            (20, display.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if writer is not None:
                    writer.write(display)

                if show:
                    cv2.imshow("Integrated Detection - Bag Playback", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("  User quit.")
                        return
                    elif key == ord("n"):
                        print("  Skipping to next file.")
                        break

            if (headless or save) and frame_count % 100 == 0:
                print(f"  Frame {frame_count}, t={timestamp_ms/1000:.1f}s, "
                      f"comfort={result.integrated_comfort_score:.0f} "
                      f"(E:{result.emotion_comfort_score:.0f} P:{result.posture_comfort_score:.0f})",
                      flush=True)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
        raise
    finally:
        source.release()
        if writer is not None:
            writer.release()

    print(f"  Processed {frame_count} frames.")
    if out_path:
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run integrated emotion + posture detection on RealSense .bag files")
    parser.add_argument(
        "input", type=str, nargs="?",
        default=str(PROJECT_ROOT / "data"),
        help="Path to a .bag file or directory (default: data/)")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "config" / "deploy.yaml"),
        help="Path to config YAML file")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI display")
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated video as .mp4")
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save annotated videos")
    args = parser.parse_args()

    input_path = Path(args.input)
    bags = find_bag_files(input_path)
    if not bags:
        print(f"Error: No .bag files found at {input_path}")
        sys.exit(1)

    print(f"\nFound {len(bags)} .bag file(s):\n")
    for i, bag_path in enumerate(bags):
        try:
            rel = bag_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = bag_path
        metadata = load_sidecar(bag_path)
        desc = ""
        if metadata:
            desc = f" - {metadata.get('scenario_name', '')} ({metadata.get('duration_seconds', '?')}s)"
        print(f"  {i+1:>2}) {rel}{desc}")

    print(f"\n   0) Run ALL files\n")

    # Non-interactive: single .bag file given → run it without prompting.
    # Also skip prompt when headless (e.g. overnight/automated runs).
    if input_path.is_file() and input_path.suffix == ".bag":
        selected = bags
    elif args.headless or args.save:
        selected = bags
    else:
        while True:
            try:
                choice = input("Select file(s) (e.g. 3 or 1,4,7 or 2-5 or 0 for all): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                sys.exit(0)

            selected = _parse_selection(choice, bags)
            if selected is not None:
                break
            print(f"Invalid input. Enter 0-{len(bags)}, a comma-separated list, or a range.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = IntegratedPipeline(config)
    pipeline.load_models()

    viz = Visualizer(config)

    try:
        for i, bag_path in enumerate(selected):
            print(f"\n[{i+1}/{len(selected)}]", end="")
            process_bag(bag_path, pipeline, viz, args.headless,
                        save=args.save,
                        save_dir=Path(args.save_dir) if args.save_dir else None)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cv2.destroyAllWindows()

    print("\nDone.")


if __name__ == "__main__":
    main()
