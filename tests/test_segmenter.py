import pytest
from pipeline.segmenter import build_segments, merge_short_segments, merge_adjacent_same_label


def test_build_segments_from_shot_boundaries():
    shot_boundaries = [10.0, 25.0, 50.0]
    duration = 60.0

    segments = build_segments(shot_boundaries, duration)

    assert len(segments) == 4
    assert segments[0] == {"start": 0.0, "end": 10.0}
    assert segments[1] == {"start": 10.0, "end": 25.0}
    assert segments[2] == {"start": 25.0, "end": 50.0}
    assert segments[3] == {"start": 50.0, "end": 60.0}


def test_build_segments_no_boundaries():
    segments = build_segments([], 60.0)
    assert len(segments) == 1
    assert segments[0] == {"start": 0.0, "end": 60.0}


def test_merge_short_segments():
    segments = [
        {"start": 0.0, "end": 10.0, "label": "intro"},
        {"start": 10.0, "end": 11.5, "label": "core_content"},
        {"start": 11.5, "end": 40.0, "label": "core_content"},
        {"start": 40.0, "end": 41.0, "label": "transition"},
        {"start": 41.0, "end": 60.0, "label": "outro"},
    ]

    merged = merge_short_segments(segments, min_duration=3.0)
    assert all((s["end"] - s["start"]) >= 3.0 for s in merged)


def test_merge_adjacent_same_label():
    segments = [
        {"start": 0.0, "end": 10.0, "label": "core_content", "confidence": 0.9, "reason": "a"},
        {"start": 10.0, "end": 20.0, "label": "core_content", "confidence": 0.85, "reason": "b"},
        {"start": 20.0, "end": 30.0, "label": "sponsorship", "confidence": 0.8, "reason": "c"},
    ]

    merged = merge_adjacent_same_label(segments)

    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 20.0
    assert merged[0]["label"] == "core_content"
    assert merged[1]["label"] == "sponsorship"
