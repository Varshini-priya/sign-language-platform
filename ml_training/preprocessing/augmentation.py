"""
augmentation.py — UPDATED for padded variable-length sequences
──────────────────────────────────────────────────────────────
Previously: augmentations were applied uniformly across all 30 frames,
because every frame was real.

Now: sequences are padded with zeros to MAX_SEQUENCE_LENGTH. We must
only augment the REAL (non-zero) frames so the Masking layer can still
detect the padding boundary. Augmenting the zero rows would corrupt the
mask and confuse the LSTM into thinking padding is real data.
"""
import numpy as np


def _real_frame_mask(sequence: np.ndarray) -> np.ndarray:
    """Returns a boolean mask: True for real frames, False for zero-padding."""
    return (sequence != 0).any(axis=1)  # (T,) bool


def random_scale(sequence: np.ndarray, scale_range=(0.85, 1.15)) -> np.ndarray:
    """Simulate hand at different distances from camera."""
    mask = _real_frame_mask(sequence)
    seq = sequence.copy()
    scale = np.random.uniform(*scale_range)
    seq[mask] *= scale
    return seq


def random_rotation_2d(sequence: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """Rotate (x, y) of every real landmark by a small random angle — simulates wrist tilt."""
    mask = _real_frame_mask(sequence)
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    seq = sequence.copy()
    for t in np.where(mask)[0]:
        xy = seq[t].reshape(21, 3)[:, :2]  # (21, 2)
        seq[t].reshape(21, 3)[:, :2] = (rot @ xy.T).T
    return seq


def random_noise(sequence: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to real frames only."""
    mask = _real_frame_mask(sequence)
    seq = sequence.copy()
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    noise[~mask] = 0.0  # never add noise to padding
    return seq + noise


def random_time_shift(sequence: np.ndarray, max_shift: int = 5) -> np.ndarray:
    """
    Shift the real frames earlier or later in time within the padded window.
    This simulates signing at slightly different tempos without changing
    the gesture's duration relative to the padding boundary.
    """
    mask = _real_frame_mask(sequence)
    real_frames = sequence[mask]
    n_real = len(real_frames)
    max_len = len(sequence)

    shift = np.random.randint(-max_shift, max_shift + 1)
    new_start = max(0, min(max_len - n_real, shift))

    seq = np.zeros_like(sequence)
    seq[new_start: new_start + n_real] = real_frames
    return seq


def random_speed(sequence: np.ndarray, speed_range=(0.8, 1.25)) -> np.ndarray:
    """
    Speed-warp the gesture — compresses or stretches the real frames in time.
    Simulates signing the same sign faster or slower. Works by interpolating
    the real frames to a new length, then re-placing them in the padded window.

    This is only possible with variable-length sequences (new pipeline) —
    with fixed 30-frame windows there was no room to stretch into.
    """
    mask = _real_frame_mask(sequence)
    real_frames = sequence[mask]
    n_real = len(real_frames)
    max_len = len(sequence)

    speed = np.random.uniform(*speed_range)
    new_len = min(int(n_real / speed), max_len)
    new_len = max(new_len, 4)  # never collapse to < 4 frames

    # Linear interpolation along time axis
    old_idx = np.linspace(0, n_real - 1, new_len)
    new_frames = np.array([
        (1 - (i - int(i))) * real_frames[int(i)] +
        (i - int(i)) * real_frames[min(int(i) + 1, n_real - 1)]
        for i in old_idx
    ], dtype=np.float32)

    seq = np.zeros_like(sequence)
    seq[:new_len] = new_frames
    return seq


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """Apply a random combination of augmentations to one padded sequence."""
    seq = sequence.copy()
    if np.random.rand() < 0.70:
        seq = random_scale(seq)
    if np.random.rand() < 0.50:
        seq = random_rotation_2d(seq)
    if np.random.rand() < 0.60:
        seq = random_noise(seq)
    if np.random.rand() < 0.40:
        seq = random_time_shift(seq)
    if np.random.rand() < 0.50:
        seq = random_speed(seq)  # new — only useful with variable-length
    return seq


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the dataset by generating `multiplier` augmented copies of each sequence.
    Original data is always preserved; augmented copies are appended.

    Args:
        X:          (N, MAX_SEQ_LEN, 63) padded sequences
        y:          (N,) integer class labels
        multiplier: number of augmented copies per original sample

    Returns:
        (X_augmented, y_augmented) of size N × (1 + multiplier)
    """
    X_list, y_list = [X], [y]
    for i in range(multiplier):
        X_aug = np.array([augment_sequence(seq) for seq in X], dtype=np.float32)
        X_list.append(X_aug)
        y_list.append(y.copy())
        print(f"  Augmentation pass {i+1}/{multiplier} done")
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
