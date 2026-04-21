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
