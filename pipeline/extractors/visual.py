import cv2
import numpy as np


def extract_visual_features(
    video_path: str,
    sample_fps: float = 2.0,
    shot_threshold: float = 0.5,
) -> dict:
    """Extract visual features from a video file.

    Samples frames at sample_fps rate, computes HSV histograms,
    frame-to-frame similarity, and detects shot boundaries.

    Returns dict with keys: frame_timestamps, histograms,
    frame_similarities, shot_boundaries, static_scores, edge_densities.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / sample_fps))

    timestamps = []
    histograms = []
    similarities = []
    shot_boundaries = []
    edge_densities = []

    frame_idx = 0
    prev_hist = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            t = frame_idx / fps
            timestamps.append(t)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

            if prev_hist is not None:
                sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                similarities.append(sim)
                if sim < shot_threshold:
                    shot_boundaries.append(t)
            else:
                similarities.append(1.0)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size
            edge_densities.append(edge_density)

            prev_hist = hist

        frame_idx += 1

    cap.release()

    static_scores = []
    for i in range(len(histograms)):
        if i + 5 < len(histograms):
            score = cv2.compareHist(histograms[i], histograms[i + 5], cv2.HISTCMP_CORREL)
        else:
            score = 1.0
        static_scores.append(score)

    return {
        "frame_timestamps": timestamps,
        "histograms": [h.tolist() for h in histograms],
        "frame_similarities": similarities,
        "shot_boundaries": shot_boundaries,
        "static_scores": static_scores,
        "edge_densities": edge_densities,
    }
