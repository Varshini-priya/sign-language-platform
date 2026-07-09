"""
data_collector.py — UPDATED for variable-length dynamic gestures
─────────────────────────────────────────────────────────────────
Previously: every recording was forced to exactly 30 frames (~1 second),
which can't represent longer signs like compound or directional ASL signs.

Now: press SPACE to START recording, then either:
  - press SPACE again to STOP manually (for shorter signs), or
  - recording auto-stops at MAX_SEQUENCE_LENGTH (4 seconds) if you don't
    stop it yourself.
Each saved sequence keeps its REAL, variable length — padding to a fixed
size happens later in build_dataset.py, not here. This keeps the raw
recorded data honest and reusable even if you change the max length later.

How it works:
  1. Run the script
  2. Type a sign label when prompted (e.g. "hello")
  3. Press SPACE — countdown — recording starts
  4. Perform the sign at natural speed (can take 0.5s to 4s)
  5. Press SPACE again to stop, OR let it auto-stop at 4 seconds
  6. Sequence is saved to dataset/landmarks/<label>/seq_NNN.npy (variable shape!)
  7. Repeat 30-50 times per sign, varying pace slightly for robustness
  8. Press Q to quit

Output:
  dataset/landmarks/<sign>/seq_000.npy   ← shape (N, 63), N varies per file!
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from landmark_detector import HandLandmarkDetector
from visualizer import draw_hud, draw_controls_legend

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 120   # 4 seconds at 30fps — hard auto-stop cap
MIN_SEQUENCE_LENGTH = 12    # 0.4 seconds — sequences shorter than this are rejected
DATASET_DIR    = Path("../../dataset/landmarks")
LABEL_MAP_PATH = Path("../../dataset/label_map.json")
COUNTDOWN_SECS = 2


def load_or_create_label_map(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_label_map(label_map: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  📝 Label map saved → {path}  ({len(label_map)} classes)")


def get_next_seq_index(sign_dir: Path) -> int:
    return len(list(sign_dir.glob("seq_*.npy")))


def prompt_sign_label() -> str:
    print("\n" + "─" * 50)
    label = input("🤟 Enter sign label to record (e.g. hello, thank_you): ").strip().lower()
    return (label or "unknown").replace(" ", "_")


def run_countdown(window_name: str, detector: HandLandmarkDetector, sign: str, secs: int = 2) -> bool:
    start = time.time()
    while time.time() - start < secs:
        result = detector.process_frame(draw_skeleton=True, draw_hud=False)
        if result is None:
            continue
        frame = result.frame
        remaining = secs - (time.time() - start)
        cv2.putText(frame, str(int(remaining) + 1),
                    (frame.shape[1] // 2 - 30, frame.shape[0] // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, f"Get ready: {sign.upper()}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def collect_sequences(
    detector: HandLandmarkDetector,
    sign_label: str,
    label_map: dict,
    window_name: str = "SignAI — Data Collector",
) -> int:
    """Collect variable-length sequences for one sign label."""
    sign_dir = DATASET_DIR / sign_label
    sign_dir.mkdir(parents=True, exist_ok=True)

    if sign_label not in label_map.values():
        new_idx = len(label_map)
        label_map[str(new_idx)] = sign_label
        save_label_map(label_map, LABEL_MAP_PATH)
        print(f"  ✅ New class registered: '{sign_label}' → index {new_idx}")

    seq_idx = get_next_seq_index(sign_dir)
    sequences_saved = 0
    recording = False
    current_sequence: list[np.ndarray] = []
    waiting_space = True

    print(f"  📂 Existing sequences: {seq_idx}")
    print(f"  👆 SPACE to start, SPACE again to stop early (or auto-stops at 4s)")

    while True:
        result = detector.process_frame(draw_skeleton=True, draw_hud=False)
        if result is None:
            continue

        frame = result.frame
        h, w = frame.shape[:2]

        if waiting_space:
            cv2.putText(frame, f"Sign: {sign_label.upper()}   Saved: {sequences_saved}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "SPACE to record  |  R to change sign  |  Q to quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
            draw_hud(frame, fps=result.fps, num_hands=result.num_hands)

        if recording:
            if result.has_hands:
                current_sequence.append(result.primary_hand_vector)
            else:
                current_sequence.append(np.zeros(63, dtype=np.float32))

            elapsed_secs = len(current_sequence) / 30
            draw_hud(
                frame, fps=result.fps, num_hands=result.num_hands,
                recording=True, recording_sign=sign_label,
                frame_count=len(current_sequence), target_frames=MAX_SEQUENCE_LENGTH,
            )
            cv2.putText(frame, f"{elapsed_secs:.1f}s  (SPACE to stop)",
                        (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

            auto_stop = len(current_sequence) >= MAX_SEQUENCE_LENGTH
            if auto_stop:
                sequences_saved += _finalize_recording(current_sequence, sign_dir, seq_idx)
                seq_idx += 1 if len(current_sequence) >= MIN_SEQUENCE_LENGTH else 0
                current_sequence, recording, waiting_space = [], False, True

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return sequences_saved

        if key == ord(' '):
            if waiting_space and not recording:
                ok = run_countdown(window_name, detector, sign_label, COUNTDOWN_SECS)
                if ok:
                    current_sequence, recording, waiting_space = [], True, False
                    print(f"  🔴 Recording sequence {seq_idx}... (press SPACE to stop)")
            elif recording:
                # Manual stop
                saved = _finalize_recording(current_sequence, sign_dir, seq_idx)
                sequences_saved += saved
                seq_idx += 1 if saved else 0
                current_sequence, recording, waiting_space = [], False, True

        if key == ord('r'):
            return sequences_saved


def _finalize_recording(frames: list[np.ndarray], sign_dir: Path, seq_idx: int) -> int:
    """Save the sequence if it meets the minimum length. Returns 1 if saved, 0 if discarded."""
    if len(frames) < MIN_SEQUENCE_LENGTH:
        print(f"  ⚠️  Too short ({len(frames)} frames, need ≥{MIN_SEQUENCE_LENGTH}) — discarded")
        return 0

    sequence_array = np.array(frames, dtype=np.float32)  # variable shape (N, 63) — NOT padded here
    save_path = sign_dir / f"seq_{seq_idx:03d}.npy"
    np.save(save_path, sequence_array)
    print(f"  💾 Saved: {save_path}  (shape {sequence_array.shape}, {len(frames)/30:.1f}s)")
    return 1


def main():
    print("=" * 55)
    print("  🤟  SignAI — Variable-Length Data Collector  (Week 2, updated)")
    print("=" * 55)
    print(f"  Dataset dir : {DATASET_DIR.resolve()}")
    print(f"  Length range: {MIN_SEQUENCE_LENGTH/30:.1f}s – {MAX_SEQUENCE_LENGTH/30:.1f}s (variable per recording)")
    print()

    label_map = load_or_create_label_map(LABEL_MAP_PATH)
    print(f"  📚 Existing classes ({len(label_map)}): {list(label_map.values())}")

    window_name = "SignAI — Data Collector"

    with HandLandmarkDetector(max_num_hands=2, flip_horizontal=True) as detector:
        while True:
            sign_label = prompt_sign_label()
            print(f"\n  Recording: '{sign_label}'")
            saved = collect_sequences(detector, sign_label, label_map, window_name)
            print(f"\n  ✅ {saved} sequences saved for '{sign_label}'")

            print("\n  Continue with another sign? (Y / N)")
            if input("  > ").strip().lower() != 'y':
                break

    print("\n  📊 Final label map:")
    for idx, name in sorted(label_map.items(), key=lambda x: int(x[0])):
        sign_dir = DATASET_DIR / name
        count = len(list(sign_dir.glob("seq_*.npy"))) if sign_dir.exists() else 0
        print(f"    [{idx:>3}] {name:<20} — {count} sequences")

    print("\n  Done! Run build_dataset.py next (Week 3) to pad and split the data.\n")


if __name__ == "__main__":
    main()
