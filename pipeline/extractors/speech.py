import mlx_whisper


def extract_speech_features(
    audio_path: str, model_name: str = "mlx-community/whisper-small-mlx"
) -> dict:
    """Transcribe audio using mlx-whisper and return timestamped segments.

    Returns dict with key 'segments', a list of dicts with
    keys: start, end, text.
    """
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_name, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
        })

    return {"segments": segments}
