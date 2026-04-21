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


def test_intro_detection():
    segment = {
        "start": 0.0, "end": 12.0,
        "avg_rms": 0.3, "avg_motion": 0.2, "static_score": 0.95,
        "has_speech": False, "avg_spectral_centroid": 2000.0, "edge_density": 0.15,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "intro"


def test_outro_detection():
    segment = {
        "start": 580.0, "end": 600.0,
        "avg_rms": 0.2, "avg_motion": 0.1, "static_score": 0.95,
        "has_speech": False, "avg_spectral_centroid": 1500.0, "edge_density": 0.12,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "outro"


def test_unclassified_returns_none():
    segment = {
        "start": 100.0, "end": 200.0,
        "avg_rms": 0.4, "avg_motion": 2.0, "static_score": 0.3,
        "has_speech": True, "avg_spectral_centroid": 3000.0, "edge_density": 0.05,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result is None


def test_transition_detection():
    segment = {
        "start": 100.0, "end": 105.0,
        "avg_rms": 0.3, "avg_motion": 1.5, "static_score": 0.4,
        "has_speech": False, "avg_spectral_centroid": 3000.0, "edge_density": 0.02,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "transition"


def test_intro_music_no_speech():
    """Animated intro with music but no speech near start."""
    segment = {
        "start": 0.0, "end": 20.0,
        "avg_rms": 0.4, "avg_motion": 3.0, "static_score": 0.3,
        "has_speech": False, "avg_spectral_centroid": 2500.0, "edge_density": 0.03,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result is not None
    assert result["label"] == "intro"


def test_outro_music_near_end():
    """Music without speech near the end."""
    segment = {
        "start": 560.0, "end": 600.0,
        "avg_rms": 0.3, "avg_motion": 1.0, "static_score": 0.5,
        "has_speech": False, "avg_spectral_centroid": 2000.0, "edge_density": 0.02,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result is not None
    assert result["label"] == "outro"


def test_sponsorship_keyword_detection():
    """Transcript with sponsorship keywords."""
    segment = {
        "start": 200.0, "end": 260.0,
        "avg_rms": 0.4, "avg_motion": 1.0, "static_score": 0.5,
        "has_speech": True, "avg_spectral_centroid": 2000.0, "edge_density": 0.05,
    }
    transcript = "This video is brought to you by Squarespace. Use code TECH for 10% off."
    result = classify_heuristic(segment, video_duration=600.0, transcript=transcript)
    assert result is not None
    assert result["label"] == "sponsorship"


def test_self_promo_keyword_detection():
    """Transcript with self-promotion keywords."""
    segment = {
        "start": 550.0, "end": 590.0,
        "avg_rms": 0.4, "avg_motion": 1.0, "static_score": 0.5,
        "has_speech": True, "avg_spectral_centroid": 2000.0, "edge_density": 0.05,
    }
    transcript = "Don't forget to like and subscribe, and follow me on Instagram."
    result = classify_heuristic(segment, video_duration=600.0, transcript=transcript)
    assert result is not None
    assert result["label"] == "self_promotion"
