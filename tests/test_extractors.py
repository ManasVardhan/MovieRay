import numpy as np
import pytest
from pipeline.extractors.audio import extract_audio_features


def test_extract_audio_features_returns_expected_keys():
    import soundfile as sf
    import tempfile, os

    sr = 22050
    duration = 3.0
    samples = np.zeros(int(sr * duration))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, samples, sr)
    tmp.close()

    try:
        features = extract_audio_features(tmp.name)
        assert "rms_energy" in features
        assert "spectral_centroid" in features
        assert "timestamps" in features
        assert len(features["rms_energy"]) == len(features["timestamps"])
        assert len(features["spectral_centroid"]) == len(features["timestamps"])
    finally:
        os.unlink(tmp.name)


def test_silent_audio_has_low_rms():
    import soundfile as sf
    import tempfile, os

    sr = 22050
    samples = np.zeros(int(sr * 3.0))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, samples, sr)
    tmp.close()

    try:
        features = extract_audio_features(tmp.name)
        assert all(v < 0.01 for v in features["rms_energy"])
    finally:
        os.unlink(tmp.name)


def test_loud_audio_has_high_rms():
    import soundfile as sf
    import tempfile, os

    sr = 22050
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    samples = 0.8 * np.sin(2 * np.pi * 440 * t)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, samples, sr)
    tmp.close()

    try:
        features = extract_audio_features(tmp.name)
        assert all(v > 0.1 for v in features["rms_energy"])
    finally:
        os.unlink(tmp.name)


from pipeline.extractors.visual import extract_visual_features
import cv2
import tempfile, os


def _create_test_video(path: str, num_frames: int = 60, fps: int = 30):
    """Create a simple test video with solid color frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(num_frames):
        color = (0, 0, 255) if i < 30 else (255, 0, 0)
        frame = np.full((240, 320, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_visual_features_returns_expected_keys():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=30)

    try:
        features = extract_visual_features(tmp.name, sample_fps=2)
        assert "frame_timestamps" in features
        assert "histograms" in features
        assert "frame_similarities" in features
        assert "shot_boundaries" in features
        assert len(features["histograms"]) == len(features["frame_timestamps"])
    finally:
        os.unlink(tmp.name)


def test_shot_boundary_detected_on_color_change():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=30)

    try:
        features = extract_visual_features(tmp.name, sample_fps=2)
        assert len(features["shot_boundaries"]) >= 1
    finally:
        os.unlink(tmp.name)


from pipeline.extractors.motion import extract_motion_features


def test_motion_features_returns_expected_keys():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=30)

    try:
        features = extract_motion_features(tmp.name, sample_fps=2)
        assert "motion_magnitudes" in features
        assert "motion_timestamps" in features
        assert len(features["motion_magnitudes"]) == len(features["motion_timestamps"])
    finally:
        os.unlink(tmp.name)


def test_static_video_has_low_motion():
    """A video with identical frames should have near-zero motion."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 30, (320, 240))
    frame = np.full((240, 320, 3), (128, 128, 128), dtype=np.uint8)
    for _ in range(90):
        writer.write(frame)
    writer.release()

    try:
        features = extract_motion_features(tmp.name, sample_fps=2)
        assert all(m < 1.0 for m in features["motion_magnitudes"])
    finally:
        os.unlink(tmp.name)
