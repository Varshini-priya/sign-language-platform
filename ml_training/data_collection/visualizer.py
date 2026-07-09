"""
visualizer.py
─────────────
Custom hand landmark visualization using OpenCV.
Draws a color-coded skeleton overlay on the webcam frame —
joint type determines color, confidence score shown as text.

Color scheme:
  - Fingertips       : bright cyan
  - Middle joints    : green
  - Base joints      : orange
  - Wrist            : white
  - Connections      : gray
  - Bounding box     : brand indigo (when enabled)
"""

import cv2
import numpy as np
from landmark_utils import (
    all_landmarks_to_pixels,
    HAND_CONNECTIONS,
    LANDMARK_NAMES,
)


# ── Color palette (BGR format for OpenCV) ──────────────────────────────────────
COLORS = {
    "wrist":       (255, 255, 255),   # White
    "fingertip":   (0, 255, 220),     # Cyan
    "middle":      (0, 210, 90),      # Green
    "base":        (30, 160, 255),    # Orange
    "connection":  (160, 160, 160),   # Gray
    "bbox":        (180, 100, 255),   # Indigo
    "text_bg":     (30,  30,  30),    # Dark background
    "text_fg":     (255, 255, 255),   # White text
    "confidence":  (0, 200, 100),     # Green for confidence bar
    "left_hand":   (255, 150,  50),   # Orange tint for left hand label
    "right_hand":  (50,  200, 255),   # Blue tint for right hand label
}

# Landmark indices grouped by type
WRIST_IDX      = {0}
FINGERTIP_IDX  = {4, 8, 12, 16, 20}
MIDDLE_IDX     = {3, 7, 11, 15, 19, 6, 10, 14, 18}
BASE_IDX       = {1, 2, 5, 9, 13, 17}


def _get_joint_color(landmark_idx: int) -> tuple[int, int, int]:
    """Return the BGR color for a given landmark index."""
    if landmark_idx in WRIST_IDX:
        return COLORS["wrist"]
    if landmark_idx in FINGERTIP_IDX:
        return COLORS["fingertip"]
    if landmark_idx in MIDDLE_IDX:
        return COLORS["middle"]
    return COLORS["base"]


def draw_hand_skeleton(
    frame: np.ndarray,
    hand_landmarks,
    handedness_label: str = "",
    confidence: float = 0.0,
    draw_bbox: bool = True,
    draw_joint_labels: bool = False,
) -> np.ndarray:
    """
    Draw a full hand skeleton overlay on a frame.

    Args:
        frame:             BGR image from OpenCV
        hand_landmarks:    MediaPipe NormalizedLandmarkList
        handedness_label:  "Left" or "Right" (from MediaPipe handedness)
        confidence:        detection confidence [0, 1]
        draw_bbox:         whether to draw bounding box around hand
        draw_joint_labels: whether to label each joint (slow, use for debug)

    Returns:
        Modified frame (in-place + returned)
    """
    h, w = frame.shape[:2]
    pixel_coords = all_landmarks_to_pixels(hand_landmarks, w, h)

    # ── 1. Draw connections ────────────────────────────────────────────────────
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = pixel_coords[start_idx]
        pt2 = pixel_coords[end_idx]
        cv2.line(frame, pt1, pt2, COLORS["connection"], thickness=2, lineType=cv2.LINE_AA)

    # ── 2. Draw joints (circles) ───────────────────────────────────────────────
    for idx, (px, py) in enumerate(pixel_coords):
        color = _get_joint_color(idx)
        # Fingertips slightly larger
        radius = 8 if idx in FINGERTIP_IDX else 6
        cv2.circle(frame, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
        # White border for visibility
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # ── 3. Bounding box ────────────────────────────────────────────────────────
    if draw_bbox and pixel_coords:
        xs = [p[0] for p in pixel_coords]
        ys = [p[1] for p in pixel_coords]
        x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
        bbox_color = COLORS["left_hand"] if "left" in handedness_label.lower() else COLORS["right_hand"]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 2, lineType=cv2.LINE_AA)

        # Hand label inside bbox
        if handedness_label:
            label = f"{handedness_label} ({confidence:.0%})"
            _draw_label(frame, label, x_min, y_min - 10, bbox_color)

    # ── 4. Joint labels (debug mode) ──────────────────────────────────────────
    if draw_joint_labels:
        for idx, (px, py) in enumerate(pixel_coords):
            cv2.putText(
                frame, str(idx),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                COLORS["text_fg"], 1, cv2.LINE_AA
            )

    return frame


def draw_prediction_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    x: int = 20,
    y: int = 60,
) -> np.ndarray:
    """
    Draw the current prediction label + confidence bar in the top-left corner.

    Args:
        frame:      BGR image
        label:      predicted sign label (e.g. "hello")
        confidence: model confidence [0, 1]
        x, y:       top-left position of the overlay

    Returns:
        Modified frame
    """
    if not label:
        return frame

    # ── Sign label ─────────────────────────────────────────────────────────────
    sign_text = label.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(sign_text, font, 1.2, 2)

    # Background pill
    pad = 8
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        COLORS["text_bg"], -1
    )
    cv2.putText(frame, sign_text, (x, y), font, 1.2, COLORS["text_fg"], 2, cv2.LINE_AA)

    # ── Confidence bar ─────────────────────────────────────────────────────────
    bar_y = y + baseline + pad + 10
    bar_w = 160
    bar_h = 8

    # Background track
    cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    # Filled portion
    filled_w = int(bar_w * confidence)
    bar_color = (
        COLORS["confidence"] if confidence >= 0.7
        else (0, 165, 255)  # Orange for uncertain
    )
    if filled_w > 0:
        cv2.rectangle(frame, (x, bar_y), (x + filled_w, bar_y + bar_h), bar_color, -1)

    # Percentage label
    conf_text = f"{confidence:.0%}"
    cv2.putText(
        frame, conf_text,
        (x + bar_w + 8, bar_y + bar_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        COLORS["text_fg"], 1, cv2.LINE_AA
    )

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    num_hands: int,
    recording: bool = False,
    recording_sign: str = "",
    frame_count: int = 0,
    target_frames: int = 30,
) -> np.ndarray:
    """
    Draw heads-up display info: FPS, hand count, recording indicator.

    Args:
        frame:          BGR image
        fps:            current frames per second
        num_hands:      number of detected hands (0, 1, or 2)
        recording:      whether data collection is active
        recording_sign: sign label being recorded
        frame_count:    frames collected so far
        target_frames:  target frames for current recording

    Returns:
        Modified frame
    """
    h, w = frame.shape[:2]

    # ── FPS counter ────────────────────────────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Hand count ────────────────────────────────────────────────────────────
    hand_color = (0, 200, 100) if num_hands > 0 else (100, 100, 100)
    hand_text  = f"Hands: {num_hands}"
    cv2.putText(frame, hand_text, (w - 110, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1, cv2.LINE_AA)

    # ── Recording indicator ────────────────────────────────────────────────────
    if recording:
        # Pulsing red circle
        cv2.circle(frame, (20, 20), 8, (0, 0, 220), -1)
        rec_text = f"REC [{recording_sign}] {frame_count}/{target_frames}"
        cv2.putText(frame, rec_text, (36, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 220), 1, cv2.LINE_AA)

        # Progress bar at bottom
        bar_filled = int((frame_count / max(target_frames, 1)) * w)
        cv2.rectangle(frame, (0, h - 5), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 5), (bar_filled, h), (0, 0, 200), -1)

    return frame


def draw_controls_legend(frame: np.ndarray) -> np.ndarray:
    """
    Draw keyboard controls in the bottom-left corner.
    Only shown when not recording.
    """
    h = frame.shape[0]
    controls = [
        "Q — quit",
        "SPACE — start recording",
        "D — toggle debug joints",
    ]
    for i, text in enumerate(controls):
        y = h - 20 - (i * 22)
        cv2.putText(frame, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ── Internal helper ────────────────────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a small filled-background text label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
