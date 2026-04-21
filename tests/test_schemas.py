import pytest
from pipeline.schemas import Segment, AnalysisResult


def test_segment_creation():
    seg = Segment(
        start=0.0,
        end=15.2,
        label="intro",
        type="non-content",
        confidence=0.92,
        reason="Static title card with music, no speech",
    )
    assert seg.start == 0.0
    assert seg.end == 15.2
    assert seg.label == "intro"
    assert seg.type == "non-content"
    assert seg.confidence == 0.92


def test_segment_label_validation():
    with pytest.raises(ValueError):
        Segment(
            start=0.0,
            end=5.0,
            label="invalid_label",
            type="non-content",
            confidence=0.5,
            reason="test",
        )


def test_segment_type_derived_from_label():
    content = Segment(
        start=0.0, end=10.0, label="core_content", confidence=0.9, reason="main"
    )
    assert content.type == "content"

    non = Segment(
        start=0.0, end=10.0, label="sponsorship", confidence=0.8, reason="ad"
    )
    assert non.type == "non-content"


def test_analysis_result_creation():
    result = AnalysisResult(
        video="test.mp4",
        duration=100.0,
        segments=[
            Segment(
                start=0.0,
                end=100.0,
                label="core_content",
                confidence=0.95,
                reason="main content",
            )
        ],
    )
    assert result.video == "test.mp4"
    assert len(result.segments) == 1


def test_analysis_result_to_json():
    result = AnalysisResult(
        video="test.mp4",
        duration=100.0,
        segments=[
            Segment(
                start=0.0,
                end=100.0,
                label="core_content",
                confidence=0.95,
                reason="main content",
            )
        ],
    )
    data = result.model_dump()
    assert "analyzed_at" in data
    assert data["video"] == "test.mp4"
