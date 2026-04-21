import whisper


def extract_speech_features(
    audio_path: str, model_name: str = "tiny"
) -> dict:
    """Transcribe audio using Whisper and return timestamped segments.

    Returns dict with key 'segments', a list of dicts with
    keys: start, end, text.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
        })

    return {"segments": segments}
