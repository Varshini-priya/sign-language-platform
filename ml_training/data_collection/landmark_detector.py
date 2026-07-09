"""
landmark_detector.py
────────────────────
Core computer vision class for the SignAI platform.

⚠️ UPDATED: Google removed the old `mp.solutions.hands` legacy API from
recent MediaPipe pip wheels (0.10.30+). This file now uses the new
MediaPipe Tasks API (`HandLandmarker`) internally — still 100% free,
just a different interface. Requires a one-time free model download:
run `download_model.py` once before using this file.

The public interface of this class (DetectionResult, process_frame, etc.)
is UNCHANGED — landmark_utils.py, visualizer.py, data_collector.py, and
run_demo.py all work exactly as before with zero modifications.

Usage:
    detector = HandLandmarkDetector()
    with detector:
        while True:
            result = detector.process_frame()
            # result.feature_vectors — list of (63,) arrays per hand
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from landmark_utils import (
    extract_landmarks,
    normalize_landmarks,
    flatten_landmarks,
)

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


# ── Adapter: makes the new Tasks API output look like the old API ─────────────
class _LandmarkListAdapter:
    """
    The legacy `mp.solutions.hands` API returned objects with a `.landmark`
    attribute (list of 21 points). The new Tasks API returns the list of 21
    points directly, with no wrapper. This adapter re-wraps it so
    landmark_utils.py and visualizer.py — both written against the old
    shape — work completely unchanged.
    """
    def __init__(self, landmark_list):
        self.landmark = landmark_list


# ── Result dataclass — UNCHANGED from before ───────────────────────────────────

@dataclass
class DetectionResult:
    """
    Structured result returned by HandLandmarkDetector.process_frame().

    Attributes:
        frame:            The processed BGR frame (with or without overlay drawn)
        raw_landmarks:    list of (21, 3) raw landmark arrays, one per detected hand
        feature_vectors:  list of (63,) normalized+flattened vectors, one per hand
        handedness:       list of 'Left' or 'Right' strings, one per hand
        confidences:      list of detection confidence floats [0, 1], one per hand
        num_hands:        number of detected hands (0, 1, or 2)
        fps:              current frames-per-second estimate
        timestamp_ms:     Unix timestamp in milliseconds
    """
    frame: np.ndarray
    raw_landmarks:   list[np.ndarray] = field(default_factory=list)
    feature_vectors: list[np.ndarray] = field(default_factory=list)
    handedness:      list[str]        = field(default_factory=list)
    confidences:     list[float]      = field(default_factory=list)
    num_hands:       int              = 0
    fps:             float            = 0.0
    timestamp_ms:    float            = 0.0

    @property
    def has_hands(self) -> bool:
        return self.num_hands > 0

    @property
    def primary_hand_vector(self) -> Optional[np.ndarray]:
        """Return the first detected hand's feature vector, or None."""
        return self.feature_vectors[0] if self.feature_vectors else None


# ── Main detector class ────────────────────────────────────────────────────────

class HandLandmarkDetector:
    """
    Webcam-based hand landmark detector using MediaPipe's Tasks API
    (HandLandmarker) — the current, actively-maintained replacement for
    the removed `mp.solutions.hands` legacy API.

    Public interface is identical to before: open(), close(), read_frame(),
    process_frame(), context manager support, .fps, .frame_count, etc.

    Args:
        camera_index:          Webcam index (0 = built-in, 1 = first external)
        max_num_hands:         Maximum hands to detect (1 or 2)
        min_detection_conf:    Minimum detection confidence threshold [0, 1]
        min_tracking_conf:     Minimum tracking confidence threshold [0, 1]
        frame_width:           Requested camera resolution width
        frame_height:          Requested camera resolution height
        flip_horizontal:       Mirror the frame (natural webcam feel)
        model_path:            Path to hand_landmarker.task (auto-detected by default)
    """

    def __init__(
        self,
        camera_index: int = 0,
        max_num_hands: int = 2,
        min_detection_conf: float = 0.70,
        min_tracking_conf: float = 0.60,
        frame_width: int = 640,
        frame_height: int = 480,
        flip_horizontal: bool = True,
        model_path: Optional[Path] = None,
        swap_handedness: bool = True,
    ):
        self.camera_index      = camera_index
        self.max_num_hands     = max_num_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf  = min_tracking_conf
        self.frame_width       = frame_width
        self.frame_height      = frame_height
        self.flip_horizontal   = flip_horizontal
        self.model_path         = Path(model_path) if model_path else MODEL_PATH

        # MediaPipe's handedness assumes a mirrored ("selfie") input image.
        # In practice, many webcams (especially built-in laptop cameras on
        # Windows) still report it backwards even after we flip the frame
        # for display. This flag inverts the label as a correction.
        # If your hand ever shows correctly WITHOUT this flag, set it False.
        self.swap_handedness = swap_handedness

        self._cap: Optional[cv2.VideoCapture] = None
        self._hands = None  # HandLandmarker instance (Tasks API)

        # FPS tracking
        self._prev_time = time.time()
        self._fps = 0.0

        # Frame counter (also used as the monotonic VIDEO-mode timestamp)
        self._frame_count = 0
        self._start_time_ms = 0.0

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open webcam and initialize the MediaPipe HandLandmarker."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"❌ Model file not found at {self.model_path}\n"
                f"   Run 'python download_model.py' once to download it (free, ~10MB)."
            )

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"❌ Cannot open camera at index {self.camera_index}. "
                "Check that your webcam is connected and not used by another app."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        # ── Initialize the new Tasks API HandLandmarker ────────────────────────
        base_options = mp_tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf,
            min_hand_presence_confidence=self.min_detection_conf,
            running_mode=mp_vision.RunningMode.VIDEO,  # enables temporal tracking
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time_ms = time.time() * 1000

        print(f"✅ Camera opened ({self.frame_width}×{self.frame_height})")
        print(f"   HandLandmarker (Tasks API): max_hands={self.max_num_hands}, "
              f"det_conf={self.min_detection_conf}, track_conf={self.min_tracking_conf}")

    def close(self) -> None:
        """Release webcam and MediaPipe resources."""
        if self._hands:
            self._hands.close()
        if self._cap and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        print("👋 Camera released.")

    # ── Frame processing ───────────────────────────────────────────────────────

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single raw frame from the webcam."""
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")

        ret, frame = self._cap.read()
        if not ret:
            print("⚠️  Failed to read frame from camera.")
            return None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return frame

    def process_frame(
        self,
        draw_skeleton: bool = True,
        draw_hud: bool = True,
    ) -> Optional[DetectionResult]:
        """
        Read and process one frame: detect hands, extract landmarks.

        Args:
            draw_skeleton: draw joint/bone overlay on frame
            draw_hud:      draw FPS and hand count HUD

        Returns:
            DetectionResult or None if frame read failed.
        """
        frame = self.read_frame()
        if frame is None:
            return None

        self._frame_count += 1

        # ── MediaPipe Tasks API processing ─────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Monotonically increasing timestamp required by VIDEO running mode
        timestamp_ms = int(time.time() * 1000 - self._start_time_ms)
        mp_result = self._hands.detect_for_video(mp_image, timestamp_ms)

        # ── Extract landmarks ──────────────────────────────────────────────────
        raw_landmarks   = []
        feature_vectors = []
        handedness_list = []
        confidences     = []

        if mp_result.hand_landmarks:
            for raw_points, handedness_categories in zip(
                mp_result.hand_landmarks,
                mp_result.handedness,
            ):
                # Wrap to match the old API's `.landmark` shape expected
                # by landmark_utils.py and visualizer.py
                hand_lm = _LandmarkListAdapter(raw_points)

                raw = extract_landmarks(hand_lm)
                raw_landmarks.append(raw)

                fv = flatten_landmarks(normalize_landmarks(raw))
                feature_vectors.append(fv)

                category = handedness_categories[0]
                label = category.category_name
                if self.swap_handedness:
                    label = "Right" if label == "Left" else "Left"
                handedness_list.append(label)
                confidences.append(category.score)

                if draw_skeleton:
                    from visualizer import draw_hand_skeleton
                    draw_hand_skeleton(
                        frame, hand_lm,
                        handedness_label=label,
                        confidence=category.score,
                        draw_bbox=True,
                    )

        # ── FPS ───────────────────────────────────────────────────────────────
        self._fps = self._compute_fps()

        # ── HUD ───────────────────────────────────────────────────────────────
        if draw_hud:
            from visualizer import draw_hud as _draw_hud
            _draw_hud(
                frame,
                fps=self._fps,
                num_hands=len(raw_landmarks),
            )

        return DetectionResult(
            frame=frame,
            raw_landmarks=raw_landmarks,
            feature_vectors=feature_vectors,
            handedness=handedness_list,
            confidences=confidences,
            num_hands=len(raw_landmarks),
            fps=self._fps,
            timestamp_ms=time.time() * 1000,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _compute_fps(self) -> float:
        """Calculate FPS using time delta between frames."""
        now = time.time()
        fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now
        return fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_frame_size(self) -> tuple[int, int]:
        """Return actual (width, height) from the camera (may differ from requested)."""
        if not self._cap:
            return (self.frame_width, self.frame_height)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
