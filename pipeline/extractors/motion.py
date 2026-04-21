import cv2
import numpy as np


def extract_motion_features(
    video_path: str,
    sample_fps: float = 2.0,
) -> dict[str, list[float]]:
    """Compute optical flow magnitude between sampled frame pairs."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / sample_fps))

    timestamps = []
    magnitudes = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            t = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_mag = float(np.mean(mag))
                magnitudes.append(avg_mag)
                timestamps.append(t)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    return {
        "motion_magnitudes": magnitudes,
        "motion_timestamps": timestamps,
    }
