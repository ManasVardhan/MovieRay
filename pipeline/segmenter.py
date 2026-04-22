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


def merge_rapid_cuts(
    segments: list[dict], min_segment_duration: float = 5.0
) -> list[dict]:
    """Pre-merge clusters of very short segments (rapid cuts) into single segments.

    Ads and intros typically have many quick cuts (1-2s each) that create
    lots of micro-segments. These should be merged before classification
    so the vision model can see the full context of the rapid-cut sequence.
    """
    if not segments:
        return []

    result = [segments[0].copy()]
    for seg in segments[1:]:
        prev = result[-1]
        prev_duration = prev["end"] - prev["start"]
        cur_duration = seg["end"] - seg["start"]

        # If both the previous and current segment are short, merge them
        if prev_duration < min_segment_duration and cur_duration < min_segment_duration:
            result[-1]["end"] = seg["end"]
        # If only the current is short and the previous was also recently extended
        # from merging short segments, keep extending
        elif cur_duration < min_segment_duration and prev_duration < min_segment_duration * 3:
            result[-1]["end"] = seg["end"]
        else:
            result.append(seg.copy())

    return result


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


def merge_sandwiched_segments(
    segments: list[dict], max_sandwich_duration: float = 20.0
) -> list[dict]:
    """Absorb short differently-labeled segments sandwiched between two same-labeled segments.

    E.g., [sponsorship, transition(15s), sponsorship] -> [sponsorship]
    This prevents short misclassifications from breaking up contiguous regions.
    """
    if len(segments) < 3:
        return segments

    result = [segments[0].copy()]
    i = 1
    while i < len(segments) - 1:
        cur = segments[i]
        nxt = segments[i + 1]
        prev = result[-1]
        cur_duration = cur["end"] - cur["start"]

        # If prev and next have same label, and current is short, absorb it
        if prev["label"] == nxt["label"] and cur_duration <= max_sandwich_duration:
            # Extend prev to cover current and next
            result[-1]["end"] = nxt["end"]
            i += 2  # skip both current and next
        else:
            result.append(cur.copy())
            i += 1

    # Don't forget the last segment if we didn't skip it
    if i == len(segments) - 1:
        result.append(segments[-1].copy())

    return result
