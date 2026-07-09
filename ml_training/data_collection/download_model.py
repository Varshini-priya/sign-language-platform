"""
download_model.py
──────────────────
One-time setup script — downloads the free HandLandmarker model file
from Google's public MediaPipe model CDN. No API key, no account,
no cost. Required because of MediaPipe's 2023 migration away from the
old `mp.solutions` API to the new Tasks API.

Run once:
    python download_model.py
"""
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
OUT_PATH = Path(__file__).parent / "hand_landmarker.task"


def main():
    print("=" * 55)
    print("  📥 Downloading MediaPipe HandLandmarker model")
    print("=" * 55)
    print(f"  Source: {MODEL_URL}")
    print(f"  Saving to: {OUT_PATH}\n")

    if OUT_PATH.exists():
        print("  ℹ️  Model file already exists. Delete it first to re-download.")
        return

    try:
        urllib.request.urlretrieve(MODEL_URL, OUT_PATH)
        size_kb = OUT_PATH.stat().st_size / 1024
        print(f"  ✅ Downloaded successfully ({size_kb:.0f} KB)")
        print(f"  Model ready at: {OUT_PATH}\n")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print("  Try manually downloading from the URL above in your browser")
        print(f"  and save it as: {OUT_PATH}")


if __name__ == "__main__":
    main()
