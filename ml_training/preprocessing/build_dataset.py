"""
build_dataset.py — UPDATED for variable-length gesture sequences
──────────────────────────────────────────────────────────────────
Previously: all sequences were already exactly 30 frames, so no padding
was needed here.

Now: each seq_NNN.npy file is a different shape (N, 63) where N varies
per recording. This script:
  1. Loads every .npy file (variable shape)
  2. Pads each one with zero-rows up to MAX_SEQUENCE_LENGTH (120 frames)
  3. Stacks them into uniform train/val/test NumPy arrays
  4. The Masking layer in the LSTM model will ignore the zero-padded rows

Run:
    python build_dataset.py
Output:
    dataset/train/X.npy  shape (N_train, 120, 63)
    dataset/train/y.npy  shape (N_train,)
    dataset/val/X.npy    shape (N_val, 120, 63)
    dataset/test/X.npy   shape (N_test, 120, 63)
    dataset/label_map.json (already created by data_collector.py)
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

LANDMARKS_DIR     = Path("../../dataset/landmarks")
LABEL_MAP_PATH    = Path("../../dataset/label_map.json")
OUT_DIR           = Path("../../dataset")
MAX_SEQUENCE_LEN  = 120    # must match config.py MAX_SEQUENCE_LENGTH
NUM_FEATURES      = 63
TRAIN_SPLIT       = 0.70
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15


def load_label_map() -> dict:
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)  # {"0": "hello", "1": "thank_you", ...}


def pad_sequence(seq: np.ndarray, max_len: int = MAX_SEQUENCE_LEN) -> np.ndarray:
    """
    Pad (or truncate) a variable-length (N, 63) array to (max_len, 63).
    Padding value is 0.0 — the Masking layer ignores these during training.
    If the sequence is longer than max_len (e.g. a very slow signer),
    we take the middle portion to preserve the most expressive frames.
    """
    n = len(seq)
    if n == max_len:
        return seq
    if n > max_len:
        # Take the middle section — most signs have a preparation phase at
        # the start and a hold at the end; the middle is most informative.
        start = (n - max_len) // 2
        return seq[start: start + max_len]
    # Pad with zeros at the end
    pad = np.zeros((max_len - n, NUM_FEATURES), dtype=np.float32)
    return np.vstack([seq.astype(np.float32), pad])


def load_all_sequences(label_map: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Load every .npy file, pad to MAX_SEQUENCE_LEN, return X and y arrays
    plus a list of the raw (unpadded) frame counts for reporting.
    """
    name_to_idx = {v: int(k) for k, v in label_map.items()}
    X, y, raw_lengths = [], [], []

    for sign_dir in sorted(LANDMARKS_DIR.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign_name = sign_dir.name
        if sign_name not in name_to_idx:
            print(f"  ⚠️  '{sign_name}' not in label_map — skipping")
            continue

        class_idx = name_to_idx[sign_name]
        seq_files = sorted(sign_dir.glob("seq_*.npy"))

        for seq_file in seq_files:
            seq = np.load(seq_file)

            # Validate shape — must be (N, 63)
            if seq.ndim != 2 or seq.shape[1] != NUM_FEATURES:
                print(f"  ⚠️  Bad shape {seq.shape} in {seq_file} — skipping")
                continue

            raw_lengths.append(len(seq))
            X.append(pad_sequence(seq))
            y.append(class_idx)

        avg_len = np.mean([len(np.load(f)) for f in seq_files]) if seq_files else 0
        print(f"  📂 {sign_name:<20} {len(seq_files)} sequences  "
              f"(avg {avg_len/30:.1f}s, range varies)")

    return (
        np.array(X, dtype=np.float32),  # (N, MAX_SEQUENCE_LEN, 63)
        np.array(y, dtype=np.int64),    # (N,)
        raw_lengths,
    )


def main():
    print("=" * 55)
    print("  📊 Building Dataset Splits — Week 3 (variable-length)")
    print("=" * 55)
    print(f"  Padding all sequences to {MAX_SEQUENCE_LEN} frames "
          f"({MAX_SEQUENCE_LEN/30:.1f}s max)\n")

    label_map = load_label_map()
    print(f"  Classes ({len(label_map)}): {list(label_map.values())}\n")

    X, y, raw_lengths = load_all_sequences(label_map)

    if len(X) < 10:
        print("\n  ❌ Not enough data. Record more sequences with data_collector.py first.")
        return

    print(f"\n  Total sequences : {len(X)}")
    print(f"  X shape         : {X.shape}   (padded to {MAX_SEQUENCE_LEN} frames)")
    print(f"  y shape         : {y.shape}")
    print(f"  Gesture lengths : min={min(raw_lengths)/30:.1f}s  "
          f"max={max(raw_lengths)/30:.1f}s  "
          f"avg={np.mean(raw_lengths)/30:.1f}s")

    # Split: train vs (val + test)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=y, random_state=42
    )
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=(1 - val_ratio), stratify=y_rest, random_state=42
    )

    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for split_name, (Xs, ys) in splits.items():
        out_path = OUT_DIR / split_name
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "X.npy", Xs)
        np.save(out_path / "y.npy", ys)
        print(f"  💾 {split_name:<6} → X{Xs.shape}  y{ys.shape}  → {out_path}")

    print("\n  ✅ Dataset build complete.")
    print(f"  Next: python train.py (Week 5-6)\n")


if __name__ == "__main__":
    main()
