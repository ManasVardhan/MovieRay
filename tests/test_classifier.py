import pytest
from pipeline.classifier import classify_heuristic


def test_dead_air_detection():
    segment = {
        "start": 0.0, "end": 10.0,
        "avg_rms": 0.001, "avg_motion": 0.05, "static_score": 0.99,
        "has_speech": False, "avg_spectral_centroid": 100.0, "edge_density": 0.01,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "dead_air"
    assert result["confidence"] > 0.8


def test_non_dead_air_returns_none():
    """Anything that isn't dead air should be deferred to vision model."""
    segment = {
        "start": 100.0, "end": 200.0,
        "avg_rms": 0.4, "avg_motion": 2.0, "static_score": 0.3,
        "has_speech": True, "avg_spectral_centroid": 3000.0, "edge_density": 0.05,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result is None


def test_music_segment_deferred_to_vision():
    """Music segments should NOT be classified by heuristics anymore."""
    segment = {
        "start": 100.0, "end": 105.0,
        "avg_rms": 0.3, "avg_motion": 1.5, "static_score": 0.4,
        "has_speech": False, "avg_spectral_centroid": 3000.0, "edge_density": 0.02,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result is None  # Deferred to vision
