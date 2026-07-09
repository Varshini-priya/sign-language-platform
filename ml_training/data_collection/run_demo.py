"""
run_demo.py
───────────
Week 2 deliverable — polished real-time hand landmark demo.

This script is your proof-of-concept for the entire project.
Run it to verify that MediaPipe + OpenCV are working correctly
on your machine before moving to ML training (Week 3).

What it shows:
  - Live webcam feed at 30fps
  - Hand skeleton with color-coded joints
  - Bounding box with Left/Right label and detection confidence
  - FPS counter
  - Landmark coordinate table (press T to toggle)
  - Two-hand simultaneous detection
  - CSV logging of landmark data (press L to toggle)

Keyboard controls:
  Q  — quit
  T  — toggle landmark coordinate table
  D  — toggle debug joint index labels
  L  — toggle CSV landmark logging
  S  — save current frame as screenshot
  F  — toggle fullscreen

Usage:
  python run_demo.py
  python run_demo.py --camera 1        (use external webcam)
  python run_demo.py --no-flip         (don't mirror image)
  python run_demo.py --width 1280 --height 720
"""

import cv2
import numpy as np
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from landmark_detector import HandLandmarkDetector, DetectionResult
from visualizer import (
    draw_prediction_overlay,
    draw_hud,
    draw_controls_legend,
    draw_hand_skeleton,
    COLORS,
)
from landmark_utils import LANDMARK_NAMES


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SignAI — Week 2 Hand Landmark Demo")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-flip", action="store_true", help="Disable horizontal flip")
    p.add_argument("--hands", type=int, default=2, choices=[1, 2],
                   help="Max number of hands to detect")
    return p.parse_args()


# ── Landmark table overlay ─────────────────────────────────────────────────────
def draw_landmark_table(
    frame: np.ndarray,
    feature_vector: np.ndarray,
    hand_label: str = "Right",
) -> np.ndarray:
    """
    Draw a small table of the first 8 landmark (x, y) values.
    Useful for verifying normalization is working correctly.
    """
    h, w = frame.shape[:2]
    table_x = w - 230
    table_y = 80

    # Background
    cv2.rectangle(frame, (table_x - 8, table_y - 18),
                  (w - 8, table_y + 8 * 18 + 6), (20, 20, 20), -1)

    header = f"Landmarks ({hand_label}) — normalized"
    cv2.putText(frame, header, (table_x, table_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Show first 8 landmarks
    for i in range(min(8, 21)):
        x_val = feature_vector[i * 3]
        y_val = feature_vector[i * 3 + 1]
        z_val = feature_vector[i * 3 + 2]
        name  = LANDMARK_NAMES[i][:12]
        row   = f"{i:2d} {name:<12} {x_val:+.3f} {y_val:+.3f}"
        color = (0, 255, 200) if i == 0 else (200, 200, 200)
        cv2.putText(frame, row, (table_x, table_y + (i + 1) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        _ = z_val  # available but not shown for space

    return frame


# ── CSV logger ─────────────────────────────────────────────────────────────────
class CSVLogger:
    """Logs landmark data to CSV for manual inspection."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._writer = None

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.path, "w", newline="")
        headers      = ["timestamp_ms", "hand"] + [
            f"{name}_{axis}"
            for name in LANDMARK_NAMES
            for axis in ("x", "y", "z")
        ]
        self._writer = csv.writer(self._file)
        self._writer.writerow(headers)
        print(f"  📝 CSV logging → {self.path}")

    def log(self, result: DetectionResult):
        if not self._writer:
            return
        for fv, label in zip(result.feature_vectors, result.handedness):
            row = [f"{result.timestamp_ms:.0f}", label] + fv.tolist()
            self._writer.writerow(row)

    def close(self):
        if self._file:
            self._file.close()
            print(f"  ✅ CSV saved: {self.path}")


# ── Screenshot helper ──────────────────────────────────────────────────────────
def save_screenshot(frame: np.ndarray, out_dir: Path = Path("screenshots")) -> str:
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"signai_{ts}.png"
    cv2.imwrite(str(path), frame)
    return str(path)


# ── Info panel ─────────────────────────────────────────────────────────────────
def draw_info_panel(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Show a banner at top when no hands are detected."""
    if result.num_hands == 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.putText(
            frame, "No hand detected — show your hand to the camera",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (100, 100, 100), 1, cv2.LINE_AA
        )
    return frame


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 55)
    print("  🤟  SignAI — Week 2 Hand Landmark Demo")
    print("=" * 55)
    print(f"  Camera  : {args.camera}")
    print(f"  Size    : {args.width}×{args.height}")
    print(f"  Max hands: {args.hands}")
    print()
    print("  Controls:")
    print("    Q  — quit")
    print("    T  — toggle landmark table")
    print("    D  — toggle debug joint labels")
    print("    L  — toggle CSV logging")
    print("    S  — save screenshot")
    print("    F  — toggle fullscreen")
    print()

    WINDOW = "SignAI — Hand Landmark Demo  (Q to quit)"

    # State flags
    show_table     = False
    debug_joints   = False
    csv_logging    = False
    fullscreen     = False

    # CSV logger (opens on demand)
    log_path = Path("logs") / f"landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger   = CSVLogger(log_path)

    # FPS smoothing
    fps_history: list[float] = []
    SMOOTH_N = 10

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with HandLandmarkDetector(
        camera_index=args.camera,
        max_num_hands=args.hands,
        frame_width=args.width,
        frame_height=args.height,
        flip_horizontal=not args.no_flip,
    ) as detector:

        print("  ✅ Started. Press Q in the window to quit.\n")

        while True:
            result = detector.process_frame(draw_skeleton=True, draw_hud=False)
            if result is None:
                print("  ⚠️  Frame read failed. Retrying...")
                time.sleep(0.05)
                continue

            frame = result.frame

            # ── Smooth FPS ─────────────────────────────────────────────────────
            fps_history.append(result.fps)
            if len(fps_history) > SMOOTH_N:
                fps_history.pop(0)
            smooth_fps = sum(fps_history) / len(fps_history)

            # ── Overlays ───────────────────────────────────────────────────────
            draw_info_panel(frame, result)

            draw_hud(
                frame,
                fps=smooth_fps,
                num_hands=result.num_hands,
            )

            # Landmark table (first hand only)
            if show_table and result.feature_vectors:
                draw_landmark_table(
                    frame, result.feature_vectors[0],
                    result.handedness[0] if result.handedness else "?"
                )

            # Legend
            draw_controls_legend(frame)

            # CSV logging indicator
            if csv_logging:
                h_frame = frame.shape[0]
                cv2.putText(
                    frame, "⬤ CSV LOGGING",
                    (frame.shape[1] - 160, h_frame - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 220), 1, cv2.LINE_AA
                )
                logger.log(result)

            cv2.imshow(WINDOW, frame)

            # ── Key handling ───────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n  👋 Quitting...")
                break

            elif key == ord('t'):
                show_table = not show_table
                print(f"  Landmark table: {'ON' if show_table else 'OFF'}")

            elif key == ord('d'):
                debug_joints = not debug_joints
                print(f"  Debug joints: {'ON' if debug_joints else 'OFF'}")

            elif key == ord('l'):
                csv_logging = not csv_logging
                if csv_logging:
                    logger.open()
                else:
                    logger.close()

            elif key == ord('s'):
                path = save_screenshot(frame)
                print(f"  📸 Screenshot saved: {path}")

            elif key == ord('f'):
                fullscreen = not fullscreen
                flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, flag)

    # Cleanup
    if csv_logging:
        logger.close()

    print("\n  Session summary:")
    print(f"    Frames processed : {detector.frame_count}")
    print(f"    Avg FPS          : {sum(fps_history)/max(len(fps_history),1):.1f}")
    print("\n  ✅ Week 2 complete. Next: collect dataset (data_collector.py)")


if __name__ == "__main__":
    main()
