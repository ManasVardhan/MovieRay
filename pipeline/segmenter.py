def build_segments(
    shot_boundaries: list[float], duration: float
) -> list[dict]:
    """Create segment dicts from shot boundary timestamps."""
    boundaries = sorted(set(shot_boundaries))
    points = [0.0] + boundaries + [duration]
    points = sorted(set(points))

    segments = []
    for i in range(len(points) - 1):
        segments.append({"start": points[i], "end": points[i + 1]})

    return segments


def merge_adjacent_same_label(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that share the same label."""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["label"] == merged[-1]["label"]:
            merged[-1]["end"] = seg["end"]
            if seg.get("confidence", 0) > merged[-1].get("confidence", 0):
                merged[-1]["confidence"] = seg["confidence"]
                merged[-1]["reason"] = seg.get("reason", "")
        else:
            merged.append(seg.copy())

    return merged


def merge_short_segments(
    segments: list[dict], min_duration: float = 3.0
) -> list[dict]:
    """Absorb segments shorter than min_duration into their neighbors."""
    if not segments:
        return []

    result = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < min_duration and result:
            result[-1]["end"] = seg["end"]
        else:
            result.append(seg.copy())

    if len(result) > 1 and (result[0]["end"] - result[0]["start"]) < min_duration:
        result[1]["start"] = result[0]["start"]
        result = result[1:]

    return result
