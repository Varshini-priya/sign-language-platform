"""
landmark_utils.py
─────────────────
Utility functions for extracting, normalizing, and processing
MediaPipe hand landmarks.

Key concepts:
  - Raw landmarks: 21 points with (x, y, z) in [0, 1] image-relative coords
  - Normalized: wrist-relative, scale-invariant coordinates
  - Flattened: 63-element 1D array ready for the ML model

Usage:
    from landmark_utils import extract_landmarks, normalize_landmarks, flatten_landmarks
"""

import numpy as np
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_LANDMARKS = 21
NUM_COORDS = 3        # x, y, z per landmark
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63 total features

# MediaPipe hand connection pairs (for drawing skeleton)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17),                 # Palm base
]

# Landmark names for interpretability
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Convert a MediaPipe hand landmark result into a (21, 3) numpy array.

    Args:
        hand_landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList

    Returns:
        np.ndarray of shape (21, 3) — raw [x, y, z] values in [0, 1] range
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    assert coords.shape == (21, 3), f"Expected (21, 3), got {coords.shape}"
    return coords


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make landmarks scale- and position-invariant by normalizing
    relative to the wrist joint (landmark 0).

    Steps:
      1. Subtract wrist position → all points relative to wrist origin
      2. Divide by max absolute value → scale into [-1, 1]

    This ensures the model sees the same gesture regardless of:
      - Where on screen the hand is
      - How far the hand is from the camera

    Args:
        landmarks: np.ndarray of shape (21, 3) — raw landmarks

    Returns:
        np.ndarray of shape (21, 3) — normalized landmarks
    """
    normalized = landmarks.copy()

    # Step 1: translate so wrist is at origin
    wrist = normalized[0].copy()
    normalized -= wrist

    # Step 2: scale so the largest coordinate magnitude is 1.0
    max_val = np.max(np.abs(normalized))
    if max_val > 1e-6:  # avoid division by zero
        normalized /= max_val

    return normalized


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Flatten (21, 3) array to a (63,) 1D feature vector for the ML model.

    Args:
        landmarks: np.ndarray of shape (21, 3)

    Returns:
        np.ndarray of shape (63,)
    """
    return landmarks.flatten()


def landmarks_to_feature_vector(hand_landmarks) -> np.ndarray:
    """
    Full pipeline: MediaPipe result → normalized 63-element feature vector.
    This is the main function called every frame during inference.

    Args:
        hand_landmarks: MediaPipe NormalizedLandmarkList

    Returns:
        np.ndarray of shape (63,)
    """
    raw = extract_landmarks(hand_landmarks)
    normalized = normalize_landmarks(raw)
    return flatten_landmarks(normalized)


# ── Multi-hand support ─────────────────────────────────────────────────────────

def extract_both_hands(
    multi_hand_landmarks,
    multi_handedness
) -> dict[str, Optional[np.ndarray]]:
    """
    Extract normalized feature vectors for both hands (left and right).
    Returns zeros for any hand not detected.

    Args:
        multi_hand_landmarks: list of NormalizedLandmarkList (one per hand)
        multi_handedness:     list of Classification (contains 'Left'/'Right' label)

    Returns:
        dict with keys 'left' and 'right', each a (63,) array or None
    """
    result = {"left": None, "right": None}

    if not multi_hand_landmarks:
        return result

    for hand_lm, handedness in zip(multi_hand_landmarks, multi_handedness):
        # MediaPipe labels from the camera's perspective (mirrored)
        # So "Right" in MediaPipe = person's right hand
        label = handedness.classification[0].label.lower()
        feature_vector = landmarks_to_feature_vector(hand_lm)
        result[label] = feature_vector

    return result


def build_two_hand_vector(
    multi_hand_landmarks,
    multi_handedness
) -> np.ndarray:
    """
    Build a 126-element feature vector combining both hands.
    Zeros for any absent hand — preserves consistent input shape.

    Returns:
        np.ndarray of shape (126,) = left (63) + right (63)
    """
    hands = extract_both_hands(multi_hand_landmarks, multi_handedness)
    left  = hands["left"]  if hands["left"]  is not None else np.zeros(63, dtype=np.float32)
    right = hands["right"] if hands["right"] is not None else np.zeros(63, dtype=np.float32)
    return np.concatenate([left, right])


# ── Sequence utilities ─────────────────────────────────────────────────────────

def pad_or_truncate(
    sequence: list[np.ndarray],
    target_length: int = 30
) -> np.ndarray:
    """
    Ensure a landmark sequence is exactly target_length frames.
    - If longer:  take the last target_length frames (most recent)
    - If shorter: pad with zero vectors at the end

    Args:
        sequence:      list of (63,) feature vectors
        target_length: desired sequence length (default 30)

    Returns:
        np.ndarray of shape (target_length, 63)
    """
    arr = np.array(sequence, dtype=np.float32)

    if len(arr) >= target_length:
        return arr[-target_length:]

    pad_count = target_length - len(arr)
    padding = np.zeros((pad_count, NUM_FEATURES), dtype=np.float32)
    return np.vstack([arr, padding])


# ── Pixel coordinate helpers ───────────────────────────────────────────────────

def landmark_to_pixel(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    """
    Convert a normalized landmark (x, y in [0,1]) to pixel coordinates.

    Args:
        landmark:     single MediaPipe NormalizedLandmark
        frame_width:  image width in pixels
        frame_height: image height in pixels

    Returns:
        (px, py) integer pixel coordinates
    """
    px = int(landmark.x * frame_width)
    py = int(landmark.y * frame_height)
    return px, py


def all_landmarks_to_pixels(
    hand_landmarks,
    frame_width: int,
    frame_height: int
) -> list[tuple[int, int]]:
    """
    Convert all 21 landmarks to pixel coordinates.

    Returns:
        list of 21 (px, py) tuples
    """
    return [
        landmark_to_pixel(lm, frame_width, frame_height)
        for lm in hand_landmarks.landmark
    ]


# ── Motion energy (for dynamic gesture segmentation) ───────────────────────────

def compute_motion_energy(prev_vector: np.ndarray, curr_vector: np.ndarray) -> float:
    """
    Measures how much the hand moved between two consecutive frames.
    Used to automatically detect when a gesture STARTS (motion rises above
    a threshold) and ENDS (motion settles back down) — enabling variable-
    length gesture capture (0.4s to 4s+) instead of a fixed window.

    Args:
        prev_vector: (63,) normalized feature vector from the previous frame
        curr_vector: (63,) normalized feature vector from the current frame

    Returns:
        Mean per-landmark Euclidean displacement (typically 0.0–0.3 range
        in normalized coordinate space; tune MOTION_THRESHOLD against this).
    """
    if prev_vector is None:
        return 0.0
    prev_points = prev_vector.reshape(21, 3)
    curr_points = curr_vector.reshape(21, 3)
    displacements = np.linalg.norm(curr_points - prev_points, axis=1)  # (21,)
    return float(np.mean(displacements))
