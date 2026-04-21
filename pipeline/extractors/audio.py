import librosa
import numpy as np


def extract_audio_features(
    audio_path: str, hop_seconds: float = 1.0
) -> dict[str, list[float]]:
    """Extract RMS energy and spectral centroid from an audio file.

    Returns dict with keys: rms_energy, spectral_centroid, timestamps.
    Each value is a list of floats, one per hop window.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = int(sr * hop_seconds)

    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    min_len = min(len(rms), len(centroid))
    rms = rms[:min_len]
    centroid = centroid[:min_len]

    timestamps = [i * hop_seconds for i in range(min_len)]

    return {
        "rms_energy": rms.tolist(),
        "spectral_centroid": centroid.tolist(),
        "timestamps": timestamps,
    }
