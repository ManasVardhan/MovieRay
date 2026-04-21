# MovieRay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multimodal video segmentation system that classifies long-form video segments as content or non-content, with an interactive React video player for navigating and skipping segments.

**Architecture:** Two independent components connected by a JSON contract. A Python CLI pipeline extracts audio, visual, motion, and speech features from video, classifies temporal segments using heuristics + LLM, and outputs `segments.json`. A React + FastAPI player loads the video and metadata, rendering an interactive timeline with skip/play controls.

**Tech Stack:** Python (librosa, OpenCV, Whisper, Anthropic SDK, click, pydantic), React + Vite + TypeScript + Tailwind CSS, FastAPI

**Spec:** `docs/superpowers/specs/2026-04-20-movieray-design.md`

---

## File Map

### Pipeline (`pipeline/`)

| File | Responsibility |
|---|---|
| `pipeline/__init__.py` | Package init |
| `pipeline/analyze.py` | CLI entry point — orchestrates extraction, segmentation, classification |
| `pipeline/schemas.py` | Pydantic models for segment data and JSON output |
| `pipeline/extractors/__init__.py` | Package init |
| `pipeline/extractors/audio.py` | Audio feature extraction (RMS energy, spectral centroid) via librosa |
| `pipeline/extractors/visual.py` | Frame sampling, HSV histograms, shot boundaries, static/text detection via OpenCV |
| `pipeline/extractors/motion.py` | Optical flow magnitude via OpenCV Farneback |
| `pipeline/extractors/speech.py` | Whisper transcription with timestamps |
| `pipeline/segmenter.py` | Shot boundary detection, window aggregation, short segment merging |
| `pipeline/classifier.py` | Heuristic rules + LLM batch classification + confidence scoring |
| `pipeline/requirements.txt` | Python dependencies |
| `tests/test_schemas.py` | Schema validation tests |
| `tests/test_segmenter.py` | Segmentation logic tests |
| `tests/test_classifier.py` | Classification logic tests |
| `tests/test_extractors.py` | Extractor unit tests |

### Player Backend (`player/server/`)

| File | Responsibility |
|---|---|
| `player/server/main.py` | FastAPI app — video listing, metadata serving, video streaming with range requests |
| `player/server/requirements.txt` | Python dependencies |

### Player Frontend (`player/frontend/`)

| File | Responsibility |
|---|---|
| `player/frontend/src/types.ts` | TypeScript types matching pipeline JSON schema |
| `player/frontend/src/App.tsx` | Root layout — three-panel structure, state management |
| `player/frontend/src/components/VideoPlayer.tsx` | HTML5 video element with custom controls |
| `player/frontend/src/components/SegmentTimeline.tsx` | Color-coded horizontal timeline bar with seek + hover |
| `player/frontend/src/components/SegmentPanel.tsx` | Right panel — scrollable segment list with Play/Skip per segment |
| `player/frontend/src/components/Controls.tsx` | Global controls — Play Content Only, Skip Non-Content toggle, category filters |

---

## Task 1: Project Scaffolding & Schemas

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/extractors/__init__.py`
- Create: `pipeline/schemas.py`
- Create: `pipeline/requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Initialize git repo and create directory structure**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git init
mkdir -p pipeline/extractors tests data
touch pipeline/__init__.py pipeline/extractors/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write pipeline/requirements.txt**

```
librosa>=0.10.0
opencv-python>=4.8.0
openai-whisper>=20231117
anthropic>=0.40.0
yt-dlp>=2024.0
click>=8.1.0
pydantic>=2.0.0
numpy>=1.24.0
ffmpeg-python>=0.2.0
pytest>=8.0.0
```

- [ ] **Step 3: Write the failing test for schemas**

```python
# tests/test_schemas.py
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.schemas'`

- [ ] **Step 5: Implement schemas**

```python
# pipeline/schemas.py
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

VALID_LABELS = [
    "core_content",
    "intro",
    "outro",
    "sponsorship",
    "self_promotion",
    "recap",
    "transition",
    "dead_air",
    "filler",
]

NON_CONTENT_LABELS = [l for l in VALID_LABELS if l != "core_content"]


class Segment(BaseModel):
    start: float
    end: float
    label: str
    type: Literal["content", "non-content"] = "content"
    confidence: float
    reason: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"Invalid label '{v}'. Must be one of {VALID_LABELS}")
        return v

    @model_validator(mode="after")
    def set_type_from_label(self):
        self.type = "content" if self.label == "core_content" else "non-content"
        return self


class AnalysisResult(BaseModel):
    video: str
    duration: float
    analyzed_at: str = ""
    segments: list[Segment]

    def model_post_init(self, __context):
        if not self.analyzed_at:
            self.analyzed_at = datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_schemas.py -v`
Expected: All 5 tests PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add -A
git commit -m "feat: project scaffolding and Pydantic schemas for segments"
```

---

## Task 2: Audio Feature Extraction

**Files:**
- Create: `pipeline/extractors/audio.py`
- Create: `tests/test_extractors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extractors.py
import numpy as np
import pytest
from pipeline.extractors.audio import extract_audio_features


def test_extract_audio_features_returns_expected_keys():
    # Create a 3-second silent WAV file for testing
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
    # Generate a loud sine wave
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement audio extractor**

```python
# pipeline/extractors/audio.py
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

    # RMS energy per window
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]

    # Spectral centroid per window
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    # Align lengths (centroid may differ by 1)
    min_len = min(len(rms), len(centroid))
    rms = rms[:min_len]
    centroid = centroid[:min_len]

    timestamps = [i * hop_seconds for i in range(min_len)]

    return {
        "rms_energy": rms.tolist(),
        "spectral_centroid": centroid.tolist(),
        "timestamps": timestamps,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/extractors/audio.py tests/test_extractors.py
git commit -m "feat: audio feature extraction with librosa (RMS, spectral centroid)"
```

---

## Task 3: Visual Feature Extraction

**Files:**
- Create: `pipeline/extractors/visual.py`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extractors.py`:

```python
from pipeline.extractors.visual import extract_visual_features
import cv2
import tempfile, os


def _create_test_video(path: str, num_frames: int = 60, fps: int = 30):
    """Create a simple test video with solid color frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(num_frames):
        # Alternate between red and blue frames every 30 frames (1 sec)
        color = (0, 0, 255) if i < 30 else (255, 0, 0)
        frame = np.full((240, 320, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_visual_features_returns_expected_keys():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=30)

    try:
        features = extract_visual_features(tmp.name, sample_fps=2)
        assert "frame_timestamps" in features
        assert "histograms" in features
        assert "frame_similarities" in features
        assert "shot_boundaries" in features
        assert len(features["histograms"]) == len(features["frame_timestamps"])
    finally:
        os.unlink(tmp.name)


def test_shot_boundary_detected_on_color_change():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=30)

    try:
        features = extract_visual_features(tmp.name, sample_fps=2)
        # Should detect at least one shot boundary at the color change
        assert len(features["shot_boundaries"]) >= 1
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py::test_visual_features_returns_expected_keys -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement visual extractor**

```python
# pipeline/extractors/visual.py
import cv2
import numpy as np


def extract_visual_features(
    video_path: str,
    sample_fps: float = 2.0,
    shot_threshold: float = 0.5,
) -> dict:
    """Extract visual features from a video file.

    Samples frames at sample_fps rate, computes HSV histograms,
    frame-to-frame similarity, and detects shot boundaries.

    Returns dict with keys: frame_timestamps, histograms,
    frame_similarities, shot_boundaries, static_scores, edge_densities.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / sample_fps))

    timestamps = []
    histograms = []
    similarities = []
    shot_boundaries = []
    static_scores = []
    edge_densities = []

    frame_idx = 0
    prev_hist = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            t = frame_idx / fps
            timestamps.append(t)

            # HSV histogram (8 bins per channel = 512 total)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

            # Frame similarity via histogram correlation
            if prev_hist is not None:
                sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                similarities.append(sim)

                # Shot boundary = low similarity
                if sim < shot_threshold:
                    shot_boundaries.append(t)
            else:
                similarities.append(1.0)

            # Edge density (proxy for text-heavy frames)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size
            edge_densities.append(edge_density)

            prev_hist = hist

        frame_idx += 1

    cap.release()

    # Static scores: compare each frame to the one 5 samples later
    static_scores = []
    for i in range(len(histograms)):
        if i + 5 < len(histograms):
            score = cv2.compareHist(histograms[i], histograms[i + 5], cv2.HISTCMP_CORREL)
        else:
            score = 1.0
        static_scores.append(score)

    return {
        "frame_timestamps": timestamps,
        "histograms": [h.tolist() for h in histograms],
        "frame_similarities": similarities,
        "shot_boundaries": shot_boundaries,
        "static_scores": static_scores,
        "edge_densities": edge_densities,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/extractors/visual.py tests/test_extractors.py
git commit -m "feat: visual feature extraction (histograms, shot boundaries, edge density)"
```

---

## Task 4: Motion Feature Extraction

**Files:**
- Create: `pipeline/extractors/motion.py`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extractors.py`:

```python
from pipeline.extractors.motion import extract_motion_features


def test_motion_features_returns_expected_keys():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _create_test_video(tmp.name, num_frames=60, fps=30)

    try:
        features = extract_motion_features(tmp.name, sample_fps=2)
        assert "motion_magnitudes" in features
        assert "motion_timestamps" in features
        assert len(features["motion_magnitudes"]) == len(features["motion_timestamps"])
    finally:
        os.unlink(tmp.name)


def test_static_video_has_low_motion():
    """A video with identical frames should have near-zero motion."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    # Create video with all same-color frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 30, (320, 240))
    frame = np.full((240, 320, 3), (128, 128, 128), dtype=np.uint8)
    for _ in range(90):
        writer.write(frame)
    writer.release()

    try:
        features = extract_motion_features(tmp.name, sample_fps=2)
        assert all(m < 1.0 for m in features["motion_magnitudes"])
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py::test_motion_features_returns_expected_keys -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement motion extractor**

```python
# pipeline/extractors/motion.py
import cv2
import numpy as np


def extract_motion_features(
    video_path: str,
    sample_fps: float = 2.0,
) -> dict[str, list[float]]:
    """Compute optical flow magnitude between sampled frame pairs.

    Returns dict with keys: motion_magnitudes, motion_timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / sample_fps))

    timestamps = []
    magnitudes = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            t = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_mag = float(np.mean(mag))
                magnitudes.append(avg_mag)
                timestamps.append(t)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    return {
        "motion_magnitudes": magnitudes,
        "motion_timestamps": timestamps,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/extractors/motion.py tests/test_extractors.py
git commit -m "feat: motion feature extraction via Farneback optical flow"
```

---

## Task 5: Speech Transcription Extractor

**Files:**
- Create: `pipeline/extractors/speech.py`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extractors.py`:

```python
from pipeline.extractors.speech import extract_speech_features


def test_speech_features_returns_expected_keys():
    import soundfile as sf

    # Create a short silent audio file
    sr = 16000
    samples = np.zeros(int(sr * 2.0))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, samples, sr)
    tmp.close()

    try:
        features = extract_speech_features(tmp.name)
        assert "segments" in features
        assert isinstance(features["segments"], list)
        # Silent audio may return empty segments
        for seg in features["segments"]:
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py::test_speech_features_returns_expected_keys -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement speech extractor**

```python
# pipeline/extractors/speech.py
import whisper


def extract_speech_features(
    audio_path: str, model_name: str = "small"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_extractors.py::test_speech_features_returns_expected_keys -v`
Expected: PASS (Whisper will load the model on first run — may take a minute to download)

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/extractors/speech.py tests/test_extractors.py
git commit -m "feat: speech transcription extractor using Whisper"
```

---

## Task 6: Segmenter (Shot Boundaries + Window Merging)

**Files:**
- Create: `pipeline/segmenter.py`
- Create: `tests/test_segmenter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segmenter.py
import pytest
from pipeline.segmenter import build_segments, merge_short_segments


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
        {"start": 10.0, "end": 11.5, "label": "core_content"},  # too short (1.5s)
        {"start": 11.5, "end": 40.0, "label": "core_content"},
        {"start": 40.0, "end": 41.0, "label": "transition"},  # too short (1s)
        {"start": 41.0, "end": 60.0, "label": "outro"},
    ]

    merged = merge_short_segments(segments, min_duration=3.0)

    # Short segments should be absorbed into neighbors
    assert all((s["end"] - s["start"]) >= 3.0 for s in merged)


def test_merge_adjacent_same_label():
    from pipeline.segmenter import merge_adjacent_same_label

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_segmenter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement segmenter**

```python
# pipeline/segmenter.py


def build_segments(
    shot_boundaries: list[float], duration: float
) -> list[dict]:
    """Create segment dicts from shot boundary timestamps.

    Returns list of {"start": float, "end": float} dicts covering
    the full video duration.
    """
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
            # Keep the higher confidence
            if seg.get("confidence", 0) > merged[-1].get("confidence", 0):
                merged[-1]["confidence"] = seg["confidence"]
                merged[-1]["reason"] = seg.get("reason", "")
        else:
            merged.append(seg.copy())

    return merged


def merge_short_segments(
    segments: list[dict], min_duration: float = 3.0
) -> list[dict]:
    """Absorb segments shorter than min_duration into their neighbors.

    Short segments take the label of their longer neighbor
    (preferring the previous segment).
    """
    if not segments:
        return []

    result = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < min_duration and result:
            # Absorb into previous segment
            result[-1]["end"] = seg["end"]
        else:
            result.append(seg.copy())

    # Final pass: if first segment is too short, merge into next
    if len(result) > 1 and (result[0]["end"] - result[0]["start"]) < min_duration:
        result[1]["start"] = result[0]["start"]
        result = result[1:]

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_segmenter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/segmenter.py tests/test_segmenter.py
git commit -m "feat: segmenter with shot boundary splitting and segment merging"
```

---

## Task 7: Classifier (Heuristics + LLM)

**Files:**
- Create: `pipeline/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier.py
import pytest
from pipeline.classifier import classify_heuristic, classify_with_llm_batch


def test_dead_air_detection():
    segment = {
        "start": 0.0,
        "end": 10.0,
        "avg_rms": 0.001,
        "avg_motion": 0.05,
        "static_score": 0.99,
        "has_speech": False,
        "avg_spectral_centroid": 100.0,
        "edge_density": 0.01,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "dead_air"
    assert result["confidence"] > 0.8


def test_intro_detection():
    segment = {
        "start": 0.0,
        "end": 12.0,
        "avg_rms": 0.3,
        "avg_motion": 0.2,
        "static_score": 0.95,
        "has_speech": False,
        "avg_spectral_centroid": 2000.0,
        "edge_density": 0.15,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "intro"


def test_outro_detection():
    segment = {
        "start": 580.0,
        "end": 600.0,
        "avg_rms": 0.2,
        "avg_motion": 0.1,
        "static_score": 0.95,
        "has_speech": False,
        "avg_spectral_centroid": 1500.0,
        "edge_density": 0.12,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "outro"


def test_unclassified_returns_none():
    segment = {
        "start": 100.0,
        "end": 200.0,
        "avg_rms": 0.4,
        "avg_motion": 2.0,
        "static_score": 0.3,
        "has_speech": True,
        "avg_spectral_centroid": 3000.0,
        "edge_density": 0.05,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result is None  # Needs LLM classification


def test_transition_detection():
    segment = {
        "start": 100.0,
        "end": 105.0,
        "avg_rms": 0.3,
        "avg_motion": 1.5,
        "static_score": 0.4,
        "has_speech": False,
        "avg_spectral_centroid": 3000.0,
        "edge_density": 0.02,
    }
    result = classify_heuristic(segment, video_duration=600.0)
    assert result["label"] == "transition"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement classifier**

```python
# pipeline/classifier.py
import json
import anthropic


def classify_heuristic(
    segment: dict, video_duration: float
) -> dict | None:
    """Apply rule-based heuristics to classify a segment.

    Returns {"label": str, "confidence": float, "reason": str} or None
    if the segment can't be classified by heuristics alone.
    """
    start = segment["start"]
    end = segment["end"]
    rms = segment["avg_rms"]
    motion = segment["avg_motion"]
    static = segment["static_score"]
    has_speech = segment["has_speech"]
    centroid = segment["avg_spectral_centroid"]
    edge = segment["edge_density"]

    is_silent = rms < 0.01
    is_static = static > 0.9
    is_low_motion = motion < 0.3
    has_music = centroid > 1000 and not has_speech and rms > 0.05
    has_text_overlay = edge > 0.1
    is_near_start = start < 60.0
    is_near_end = end > video_duration - 60.0

    # Dead air: silence + no motion + static
    if is_silent and is_low_motion and is_static:
        return {
            "label": "dead_air",
            "confidence": 0.95,
            "reason": "Silence with no motion and static frame",
        }

    # Intro: static/text near start, no speech
    if is_near_start and not has_speech and is_static and has_text_overlay:
        return {
            "label": "intro",
            "confidence": 0.85,
            "reason": "Static frame with text overlay near video start",
        }

    # Outro: static/text near end, no speech
    if is_near_end and not has_speech and is_static and has_text_overlay:
        return {
            "label": "outro",
            "confidence": 0.85,
            "reason": "Static frame with text overlay near video end",
        }

    # Transition: music + no speech + not static
    if has_music and not has_speech and not is_static:
        return {
            "label": "transition",
            "confidence": 0.75,
            "reason": "Music without speech, non-static visuals",
        }

    return None


def classify_with_llm_batch(
    segments: list[dict],
    transcript_map: dict[int, str],
) -> list[dict]:
    """Classify segments using Claude API based on transcript content.

    Args:
        segments: list of segment dicts (must have "start", "end" keys)
        transcript_map: dict mapping segment index to transcript text

    Returns:
        list of {"index": int, "label": str, "confidence": float, "reason": str}
    """
    if not transcript_map:
        return []

    client = anthropic.Anthropic()
    results = []

    # Batch segments in groups of 10
    items = list(transcript_map.items())
    for batch_start in range(0, len(items), 10):
        batch = items[batch_start : batch_start + 10]

        segments_text = ""
        for idx, text in batch:
            seg = segments[idx]
            segments_text += (
                f"Segment {idx} [{seg['start']:.1f}s - {seg['end']:.1f}s]:\n"
                f'"{text}"\n\n'
            )

        prompt = f"""Classify each video segment below as one of:
- core_content: the main material the viewer wants to watch
- sponsorship: paid promotion, discount codes, "brought to you by"
- self_promotion: subscribe reminders, merch plugs, social media callouts
- recap: summary of previously covered material, repeated boilerplate
- filler: unrelated tangents or low-information content

For each segment, respond with a JSON array of objects:
{{"index": <segment_number>, "label": "<label>", "confidence": <0.0-1.0>, "reason": "<one line>"}}

Segments:
{segments_text}

Respond with ONLY the JSON array, no other text."""

        message = client.messages.create(
            model="claude-sonnet-4-6-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            response_text = message.content[0].text
            batch_results = json.loads(response_text)
            results.extend(batch_results)
        except (json.JSONDecodeError, IndexError, KeyError):
            # If parsing fails, default all to core_content
            for idx, _ in batch:
                results.append({
                    "index": idx,
                    "label": "core_content",
                    "confidence": 0.5,
                    "reason": "LLM classification failed, defaulting to content",
                })

    return results
```

- [ ] **Step 4: Run heuristic tests to verify they pass**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pytest tests/test_classifier.py -v`
Expected: All 5 tests PASS (LLM tests are not run here — they require API key)

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/classifier.py tests/test_classifier.py
git commit -m "feat: classifier with heuristic rules and LLM batch classification"
```

---

## Task 8: CLI Entry Point (analyze.py)

**Files:**
- Create: `pipeline/analyze.py`

- [ ] **Step 1: Implement the CLI orchestrator**

```python
# pipeline/analyze.py
import json
import os
import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np

from pipeline.classifier import classify_heuristic, classify_with_llm_batch
from pipeline.extractors.audio import extract_audio_features
from pipeline.extractors.motion import extract_motion_features
from pipeline.extractors.speech import extract_speech_features
from pipeline.extractors.visual import extract_visual_features
from pipeline.schemas import AnalysisResult, Segment
from pipeline.segmenter import (
    build_segments,
    merge_adjacent_same_label,
    merge_short_segments,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration", "-of", "csv=p=0", video_path,
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _extract_audio_track(video_path: str, output_path: str) -> None:
    """Extract audio from video to WAV using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "22050", "-ac", "1", "-y", output_path,
        ],
        capture_output=True,
    )


def _aggregate_features_per_segment(
    segments: list[dict],
    audio_feats: dict,
    visual_feats: dict,
    motion_feats: dict,
    speech_feats: dict,
) -> tuple[list[dict], dict[int, str]]:
    """Aggregate extracted features for each segment.

    Returns (enriched_segments, transcript_map).
    """
    transcript_map = {}

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]

        # Audio: average RMS and spectral centroid in this window
        audio_ts = audio_feats["timestamps"]
        rms_vals = [
            audio_feats["rms_energy"][j]
            for j, t in enumerate(audio_ts)
            if start <= t < end
        ]
        centroid_vals = [
            audio_feats["spectral_centroid"][j]
            for j, t in enumerate(audio_ts)
            if start <= t < end
        ]
        seg["avg_rms"] = float(np.mean(rms_vals)) if rms_vals else 0.0
        seg["avg_spectral_centroid"] = float(np.mean(centroid_vals)) if centroid_vals else 0.0

        # Visual: average static score and edge density
        vis_ts = visual_feats["frame_timestamps"]
        static_vals = [
            visual_feats["static_scores"][j]
            for j, t in enumerate(vis_ts)
            if start <= t < end
        ]
        edge_vals = [
            visual_feats["edge_densities"][j]
            for j, t in enumerate(vis_ts)
            if start <= t < end
        ]
        seg["static_score"] = float(np.mean(static_vals)) if static_vals else 0.0
        seg["edge_density"] = float(np.mean(edge_vals)) if edge_vals else 0.0

        # Motion: average magnitude
        mot_ts = motion_feats["motion_timestamps"]
        mot_vals = [
            motion_feats["motion_magnitudes"][j]
            for j, t in enumerate(mot_ts)
            if start <= t < end
        ]
        seg["avg_motion"] = float(np.mean(mot_vals)) if mot_vals else 0.0

        # Speech: concatenate transcript text in this window
        speech_text = " ".join(
            s["text"]
            for s in speech_feats["segments"]
            if s["start"] < end and s["end"] > start
        ).strip()
        seg["has_speech"] = len(speech_text) > 0
        if speech_text:
            transcript_map[i] = speech_text

    return segments, transcript_map


@click.command()
@click.argument("video_path", required=False)
@click.option("--youtube", "-y", help="YouTube URL to download and analyze")
@click.option("--output", "-o", help="Output JSON path (default: data/<video_name>.json)")
def main(video_path: str | None, youtube: str | None, output: str | None):
    """Analyze a video and generate segment metadata."""
    if not video_path and not youtube:
        raise click.UsageError("Provide a video file path or --youtube URL")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download YouTube video if needed
    if youtube:
        click.echo(f"Downloading video from YouTube...")
        dl_path = str(DATA_DIR / "%(title)s.%(ext)s")
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", dl_path, youtube],
            check=True,
        )
        # Find the downloaded file (most recent in data/)
        files = sorted(DATA_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if not files:
            raise click.ClickException("YouTube download failed")
        video_path = str(files[0])
        click.echo(f"Downloaded: {video_path}")

    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).stem

    if not os.path.exists(video_path):
        raise click.ClickException(f"Video not found: {video_path}")

    click.echo(f"Analyzing: {video_path}")

    # Step 1: Get duration
    duration = _get_video_duration(video_path)
    click.echo(f"Duration: {duration:.1f}s")

    # Step 2: Extract audio track
    click.echo("Extracting audio track...")
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_tmp.close()
    _extract_audio_track(video_path, audio_tmp.name)

    # Step 3: Feature extraction
    click.echo("Extracting audio features...")
    audio_feats = extract_audio_features(audio_tmp.name)

    click.echo("Extracting visual features...")
    visual_feats = extract_visual_features(video_path)

    click.echo("Extracting motion features...")
    motion_feats = extract_motion_features(video_path)

    click.echo("Transcribing speech...")
    speech_feats = extract_speech_features(audio_tmp.name)

    os.unlink(audio_tmp.name)

    # Step 4: Build segments from shot boundaries
    click.echo("Building segments from shot boundaries...")
    raw_segments = build_segments(visual_feats["shot_boundaries"], duration)

    # Step 5: Aggregate features per segment
    enriched_segments, transcript_map = _aggregate_features_per_segment(
        raw_segments, audio_feats, visual_feats, motion_feats, speech_feats
    )

    # Step 6: Heuristic classification
    click.echo("Classifying segments (heuristics)...")
    unclassified_indices = []
    for i, seg in enumerate(enriched_segments):
        result = classify_heuristic(seg, duration)
        if result:
            seg["label"] = result["label"]
            seg["confidence"] = result["confidence"]
            seg["reason"] = result["reason"]
        else:
            unclassified_indices.append(i)

    # Step 7: LLM classification for remaining segments
    llm_transcript_map = {
        i: transcript_map[i]
        for i in unclassified_indices
        if i in transcript_map
    }

    if llm_transcript_map:
        click.echo(f"Classifying {len(llm_transcript_map)} segments via LLM...")
        llm_results = classify_with_llm_batch(enriched_segments, llm_transcript_map)
        for r in llm_results:
            idx = r["index"]
            enriched_segments[idx]["label"] = r["label"]
            enriched_segments[idx]["confidence"] = r["confidence"]
            enriched_segments[idx]["reason"] = r["reason"]

    # Default any still-unclassified segments to core_content
    for seg in enriched_segments:
        if "label" not in seg:
            seg["label"] = "core_content"
            seg["confidence"] = 0.5
            seg["reason"] = "Default classification (no strong signals)"

    # Step 8: Merge and smooth
    click.echo("Merging and smoothing segments...")
    merged = merge_adjacent_same_label(enriched_segments)
    merged = merge_short_segments(merged, min_duration=5.0)

    # Step 9: Build output
    final_segments = []
    for seg in merged:
        final_segments.append(
            Segment(
                start=seg["start"],
                end=seg["end"],
                label=seg["label"],
                confidence=seg.get("confidence", 0.5),
                reason=seg.get("reason", ""),
            )
        )

    result = AnalysisResult(
        video=os.path.basename(video_path),
        duration=duration,
        segments=final_segments,
    )

    # Save output
    if not output:
        output = str(DATA_DIR / f"{video_name}.json")

    with open(output, "w") as f:
        f.write(result.model_dump_json(indent=2))

    click.echo(f"Saved analysis to: {output}")
    click.echo(f"Found {len(final_segments)} segments:")
    for seg in final_segments:
        click.echo(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.label} ({seg.confidence:.0%})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the CLI help output**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -m pipeline.analyze --help`
Expected: Shows usage with `VIDEO_PATH` argument and `--youtube` option

- [ ] **Step 3: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add pipeline/analyze.py
git commit -m "feat: CLI entry point orchestrating full analysis pipeline"
```

---

## Task 9: FastAPI Backend

**Files:**
- Create: `player/server/main.py`
- Create: `player/server/requirements.txt`

- [ ] **Step 1: Create player directory structure**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
mkdir -p player/server
```

- [ ] **Step 2: Write player/server/requirements.txt**

```
fastapi>=0.110.0
uvicorn>=0.27.0
```

- [ ] **Step 3: Implement FastAPI server**

```python
# player/server/main.py
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

DATA_DIR = Path(__file__).parent.parent.parent / "data"

app = FastAPI(title="MovieRay Player")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_video_file(video_name: str) -> Path | None:
    """Find the video file matching a metadata entry."""
    for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        path = DATA_DIR / f"{video_name}{ext}"
        if path.exists():
            return path
    # Also check if video_name already has extension
    path = DATA_DIR / video_name
    if path.exists():
        return path
    return None


@app.get("/api/videos")
def list_videos():
    """List all analyzed videos with their metadata."""
    videos = []
    if not DATA_DIR.exists():
        return {"videos": []}

    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        video_name = json_file.stem
        video_file = _find_video_file(data.get("video", video_name))
        videos.append({
            "id": video_name,
            "name": data.get("video", video_name),
            "duration": data.get("duration", 0),
            "segment_count": len(data.get("segments", [])),
            "has_video_file": video_file is not None,
        })

    return {"videos": videos}


@app.get("/api/videos/{video_id}/metadata")
def get_metadata(video_id: str):
    """Return segment metadata for a video."""
    json_path = DATA_DIR / f"{video_id}.json"
    if not json_path.exists():
        raise HTTPException(404, f"No metadata found for '{video_id}'")

    with open(json_path) as f:
        return json.load(f)


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str):
    """Stream the video file with range request support."""
    json_path = DATA_DIR / f"{video_id}.json"
    if not json_path.exists():
        raise HTTPException(404, f"No metadata found for '{video_id}'")

    with open(json_path) as f:
        data = json.load(f)

    video_file = _find_video_file(data.get("video", video_id))
    if not video_file:
        raise HTTPException(404, f"Video file not found for '{video_id}'")

    return FileResponse(
        str(video_file),
        media_type="video/mp4",
        filename=video_file.name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Test server starts**

Run: `cd /Users/manasvardhan/Desktop/MovieRay && python -c "from player.server.main import app; print('Server module loads OK')"`
Expected: `Server module loads OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/server/main.py player/server/requirements.txt
git commit -m "feat: FastAPI backend for video listing, metadata, and streaming"
```

---

## Task 10: React Frontend Scaffolding & Types

**Files:**
- Create: `player/frontend/` (via Vite scaffold)
- Create: `player/frontend/src/types.ts`

- [ ] **Step 1: Scaffold React + Vite + TypeScript project**

```bash
cd /Users/manasvardhan/Desktop/MovieRay/player
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Configure Tailwind in vite.config.ts**

Replace `player/frontend/vite.config.ts` with:

```typescript
// player/frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

- [ ] **Step 3: Add Tailwind import to CSS**

Replace `player/frontend/src/index.css` with:

```css
@import "tailwindcss";
```

- [ ] **Step 4: Write TypeScript types matching pipeline JSON schema**

```typescript
// player/frontend/src/types.ts
export interface Segment {
  start: number;
  end: number;
  label: SegmentLabel;
  type: "content" | "non-content";
  confidence: number;
  reason: string;
}

export type SegmentLabel =
  | "core_content"
  | "intro"
  | "outro"
  | "sponsorship"
  | "self_promotion"
  | "recap"
  | "transition"
  | "dead_air"
  | "filler";

export interface VideoMetadata {
  video: string;
  duration: number;
  analyzed_at: string;
  segments: Segment[];
}

export interface VideoListItem {
  id: string;
  name: string;
  duration: number;
  segment_count: number;
  has_video_file: boolean;
}

export const SEGMENT_COLORS: Record<SegmentLabel, string> = {
  core_content: "#22c55e",   // green
  intro: "#3b82f6",          // blue
  outro: "#6366f1",          // indigo
  sponsorship: "#ef4444",    // red
  self_promotion: "#f97316", // orange
  recap: "#eab308",          // yellow
  transition: "#8b5cf6",     // violet
  dead_air: "#6b7280",       // gray
  filler: "#a855f7",         // purple
};

export const SEGMENT_DISPLAY_NAMES: Record<SegmentLabel, string> = {
  core_content: "Content",
  intro: "Intro",
  outro: "Outro",
  sponsorship: "Sponsorship",
  self_promotion: "Self Promo",
  recap: "Recap",
  transition: "Transition",
  dead_air: "Dead Air",
  filler: "Filler",
};
```

- [ ] **Step 5: Verify build works**

Run: `cd /Users/manasvardhan/Desktop/MovieRay/player/frontend && npm run build`
Expected: Build succeeds with no errors

- [ ] **Step 6: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/frontend/
git commit -m "feat: React frontend scaffolding with TypeScript types and Tailwind"
```

---

## Task 11: VideoPlayer Component

**Files:**
- Create: `player/frontend/src/components/VideoPlayer.tsx`

- [ ] **Step 1: Implement VideoPlayer component**

```tsx
// player/frontend/src/components/VideoPlayer.tsx
import { useRef, useEffect, forwardRef, useImperativeHandle } from "react";

interface VideoPlayerProps {
  videoUrl: string;
  onTimeUpdate: (currentTime: number) => void;
  onDurationChange: (duration: number) => void;
}

export interface VideoPlayerHandle {
  seek: (time: number) => void;
  play: () => void;
  pause: () => void;
  getCurrentTime: () => number;
  isPaused: () => boolean;
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(
  ({ videoUrl, onTimeUpdate, onDurationChange }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useImperativeHandle(ref, () => ({
      seek: (time: number) => {
        if (videoRef.current) {
          videoRef.current.currentTime = time;
        }
      },
      play: () => videoRef.current?.play(),
      pause: () => videoRef.current?.pause(),
      getCurrentTime: () => videoRef.current?.currentTime ?? 0,
      isPaused: () => videoRef.current?.paused ?? true,
    }));

    useEffect(() => {
      const video = videoRef.current;
      if (!video) return;

      const handleTimeUpdate = () => onTimeUpdate(video.currentTime);
      const handleDuration = () => onDurationChange(video.duration);

      video.addEventListener("timeupdate", handleTimeUpdate);
      video.addEventListener("loadedmetadata", handleDuration);

      return () => {
        video.removeEventListener("timeupdate", handleTimeUpdate);
        video.removeEventListener("loadedmetadata", handleDuration);
      };
    }, [onTimeUpdate, onDurationChange]);

    return (
      <div className="bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full aspect-video"
          controls
        />
      </div>
    );
  }
);

VideoPlayer.displayName = "VideoPlayer";
```

- [ ] **Step 2: Verify build**

Run: `cd /Users/manasvardhan/Desktop/MovieRay/player/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/frontend/src/components/VideoPlayer.tsx
git commit -m "feat: VideoPlayer component with imperative handle for seek/play/pause"
```

---

## Task 12: SegmentTimeline Component

**Files:**
- Create: `player/frontend/src/components/SegmentTimeline.tsx`

- [ ] **Step 1: Implement SegmentTimeline component**

```tsx
// player/frontend/src/components/SegmentTimeline.tsx
import { useState } from "react";
import {
  Segment,
  SEGMENT_COLORS,
  SEGMENT_DISPLAY_NAMES,
} from "../types";

interface SegmentTimelineProps {
  segments: Segment[];
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

export function SegmentTimeline({
  segments,
  duration,
  currentTime,
  onSeek,
}: SegmentTimelineProps) {
  const [hoveredSegment, setHoveredSegment] = useState<Segment | null>(null);
  const [tooltipX, setTooltipX] = useState(0);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    onSeek(ratio * duration);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    const time = ratio * duration;
    setTooltipX(x);

    const seg = segments.find((s) => time >= s.start && time < s.end);
    setHoveredSegment(seg ?? null);
  };

  const playheadPosition = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="relative">
      {/* Tooltip */}
      {hoveredSegment && (
        <div
          className="absolute bottom-full mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap pointer-events-none z-10"
          style={{ left: tooltipX, transform: "translateX(-50%)" }}
        >
          <span className="font-medium">
            {SEGMENT_DISPLAY_NAMES[hoveredSegment.label]}
          </span>
          <span className="text-gray-400 ml-2">
            {formatTime(hoveredSegment.start)} - {formatTime(hoveredSegment.end)}
          </span>
        </div>
      )}

      {/* Timeline bar */}
      <div
        className="relative h-8 rounded-md overflow-hidden cursor-pointer flex"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredSegment(null)}
      >
        {segments.map((seg, i) => {
          const widthPct = ((seg.end - seg.start) / duration) * 100;
          const isActive =
            currentTime >= seg.start && currentTime < seg.end;

          return (
            <div
              key={i}
              className={`h-full transition-opacity ${
                isActive ? "ring-2 ring-white ring-inset" : ""
              }`}
              style={{
                width: `${widthPct}%`,
                backgroundColor: SEGMENT_COLORS[seg.label],
                opacity: hoveredSegment === seg ? 1 : 0.8,
              }}
            />
          );
        })}

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
          style={{ left: `${playheadPosition}%` }}
        />
      </div>

      {/* Segment labels below timeline */}
      <div className="flex mt-1 text-xs text-gray-500">
        {segments.map((seg, i) => {
          const widthPct = ((seg.end - seg.start) / duration) * 100;
          if (widthPct < 5) return <div key={i} style={{ width: `${widthPct}%` }} />;
          return (
            <div
              key={i}
              className="truncate text-center"
              style={{ width: `${widthPct}%` }}
            >
              {SEGMENT_DISPLAY_NAMES[seg.label]}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
```

- [ ] **Step 2: Verify build**

Run: `cd /Users/manasvardhan/Desktop/MovieRay/player/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/frontend/src/components/SegmentTimeline.tsx
git commit -m "feat: SegmentTimeline with color-coded segments, playhead, hover tooltips"
```

---

## Task 13: SegmentPanel Component

**Files:**
- Create: `player/frontend/src/components/SegmentPanel.tsx`

- [ ] **Step 1: Implement SegmentPanel component**

```tsx
// player/frontend/src/components/SegmentPanel.tsx
import { useEffect, useRef } from "react";
import {
  Segment,
  SEGMENT_COLORS,
  SEGMENT_DISPLAY_NAMES,
} from "../types";

interface SegmentPanelProps {
  segments: Segment[];
  currentTime: number;
  onPlay: (segment: Segment) => void;
  onSkip: (segment: Segment) => void;
}

export function SegmentPanel({
  segments,
  currentTime,
  onPlay,
  onSkip,
}: SegmentPanelProps) {
  const activeRef = useRef<HTMLDivElement>(null);

  const activeIndex = segments.findIndex(
    (s) => currentTime >= s.start && currentTime < s.end
  );

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeIndex]);

  return (
    <div className="bg-gray-900 rounded-lg p-4 h-full overflow-y-auto">
      <h3 className="text-white font-semibold mb-3 text-sm uppercase tracking-wide">
        Segment Overview
      </h3>

      <div className="space-y-2">
        {segments.map((seg, i) => {
          const isActive = i === activeIndex;
          const isLowConfidence = seg.confidence < 0.7;

          return (
            <div
              key={i}
              ref={isActive ? activeRef : null}
              className={`rounded-lg p-3 transition-all ${
                isActive
                  ? "bg-gray-700 ring-1 ring-white/30"
                  : "bg-gray-800 hover:bg-gray-750"
              } ${isLowConfidence ? "border border-dashed border-yellow-500/50" : ""}`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: SEGMENT_COLORS[seg.label] }}
                  />
                  <span className="text-white text-sm font-medium">
                    {SEGMENT_DISPLAY_NAMES[seg.label]}
                  </span>
                  {isLowConfidence && (
                    <span className="text-yellow-500 text-xs" title="Low confidence">
                      ?
                    </span>
                  )}
                </div>
                <span className="text-gray-400 text-xs">
                  {formatTime(seg.start)} - {formatTime(seg.end)}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    seg.type === "content"
                      ? "bg-green-900/50 text-green-400"
                      : "bg-red-900/50 text-red-400"
                  }`}
                >
                  {seg.type === "content" ? "Content" : "Non-Content"}
                </span>

                <div className="flex gap-1">
                  <button
                    onClick={() => onPlay(seg)}
                    className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                  >
                    Play
                  </button>
                  <button
                    onClick={() => onSkip(seg)}
                    className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                  >
                    Skip
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
```

- [ ] **Step 2: Verify build**

Run: `cd /Users/manasvardhan/Desktop/MovieRay/player/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/frontend/src/components/SegmentPanel.tsx
git commit -m "feat: SegmentPanel with scrollable segment list, Play/Skip buttons"
```

---

## Task 14: Controls Component

**Files:**
- Create: `player/frontend/src/components/Controls.tsx`

- [ ] **Step 1: Implement Controls component**

```tsx
// player/frontend/src/components/Controls.tsx
import { SegmentLabel, SEGMENT_DISPLAY_NAMES, SEGMENT_COLORS } from "../types";

interface ControlsProps {
  skipNonContent: boolean;
  onToggleSkipNonContent: () => void;
  skippedCategories: Set<SegmentLabel>;
  onToggleCategory: (label: SegmentLabel) => void;
  onPlayContentOnly: () => void;
}

const NON_CONTENT_LABELS: SegmentLabel[] = [
  "intro",
  "outro",
  "sponsorship",
  "self_promotion",
  "recap",
  "transition",
  "dead_air",
  "filler",
];

export function Controls({
  skipNonContent,
  onToggleSkipNonContent,
  skippedCategories,
  onToggleCategory,
  onPlayContentOnly,
}: ControlsProps) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-wrap items-center gap-4">
      <button
        onClick={onPlayContentOnly}
        className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
      >
        Play Content Only
      </button>

      <button
        onClick={onToggleSkipNonContent}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          skipNonContent
            ? "bg-red-600 hover:bg-red-500 text-white"
            : "bg-gray-700 hover:bg-gray-600 text-gray-300"
        }`}
      >
        {skipNonContent ? "Skip Non-Content: ON" : "Skip Non-Content: OFF"}
      </button>

      <div className="h-6 w-px bg-gray-700" />

      <div className="flex flex-wrap gap-2">
        {NON_CONTENT_LABELS.map((label) => (
          <label
            key={label}
            className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={skippedCategories.has(label)}
              onChange={() => onToggleCategory(label)}
              className="rounded border-gray-600"
            />
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: SEGMENT_COLORS[label] }}
            />
            {SEGMENT_DISPLAY_NAMES[label]}
          </label>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

Run: `cd /Users/manasvardhan/Desktop/MovieRay/player/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/frontend/src/components/Controls.tsx
git commit -m "feat: Controls with Play Content Only, Skip Non-Content toggle, category filters"
```

---

## Task 15: App.tsx — Wire Everything Together

**Files:**
- Modify: `player/frontend/src/App.tsx`

- [ ] **Step 1: Implement App.tsx**

Replace `player/frontend/src/App.tsx` with:

```tsx
// player/frontend/src/App.tsx
import { useState, useRef, useCallback, useEffect } from "react";
import { VideoPlayer, VideoPlayerHandle } from "./components/VideoPlayer";
import { SegmentTimeline } from "./components/SegmentTimeline";
import { SegmentPanel } from "./components/SegmentPanel";
import { Controls } from "./components/Controls";
import {
  Segment,
  SegmentLabel,
  VideoMetadata,
  VideoListItem,
} from "./types";

function App() {
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [skipNonContent, setSkipNonContent] = useState(false);
  const [skippedCategories, setSkippedCategories] = useState<Set<SegmentLabel>>(
    new Set(["sponsorship", "dead_air"])
  );

  const playerRef = useRef<VideoPlayerHandle>(null);

  // Fetch video list on mount
  useEffect(() => {
    fetch("/api/videos")
      .then((r) => r.json())
      .then((data) => setVideos(data.videos))
      .catch(console.error);
  }, []);

  // Fetch metadata when video is selected
  useEffect(() => {
    if (!selectedVideoId) return;
    fetch(`/api/videos/${selectedVideoId}/metadata`)
      .then((r) => r.json())
      .then((data: VideoMetadata) => setMetadata(data))
      .catch(console.error);
  }, [selectedVideoId]);

  // Auto-skip logic
  const handleTimeUpdate = useCallback(
    (time: number) => {
      setCurrentTime(time);

      if (!skipNonContent || !metadata) return;

      const currentSegment = metadata.segments.find(
        (s) => time >= s.start && time < s.end
      );

      if (
        currentSegment &&
        currentSegment.type === "non-content" &&
        skippedCategories.has(currentSegment.label)
      ) {
        playerRef.current?.seek(currentSegment.end);
      }
    },
    [skipNonContent, metadata, skippedCategories]
  );

  const handleSeek = (time: number) => {
    playerRef.current?.seek(time);
  };

  const handlePlaySegment = (seg: Segment) => {
    playerRef.current?.seek(seg.start);
    playerRef.current?.play();
  };

  const handleSkipSegment = (seg: Segment) => {
    playerRef.current?.seek(seg.end);
  };

  const handlePlayContentOnly = () => {
    setSkipNonContent(true);
    setSkippedCategories(
      new Set([
        "intro", "outro", "sponsorship", "self_promotion",
        "recap", "transition", "dead_air", "filler",
      ])
    );
    playerRef.current?.seek(0);
    playerRef.current?.play();
  };

  const handleToggleCategory = (label: SegmentLabel) => {
    setSkippedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  // Video selection screen
  if (!selectedVideoId) {
    return (
      <div className="min-h-screen bg-gray-950 text-white p-8">
        <h1 className="text-3xl font-bold mb-2">MovieRay</h1>
        <p className="text-gray-400 mb-8">
          Multimodal Video Segmentation Player
        </p>

        {videos.length === 0 ? (
          <div className="text-gray-500">
            <p>No analyzed videos found.</p>
            <p className="text-sm mt-2">
              Run{" "}
              <code className="bg-gray-800 px-2 py-1 rounded">
                python pipeline/analyze.py video.mp4
              </code>{" "}
              to analyze a video first.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {videos.map((v) => (
              <button
                key={v.id}
                onClick={() => setSelectedVideoId(v.id)}
                className="text-left bg-gray-900 rounded-lg p-4 hover:bg-gray-800 transition-colors"
              >
                <h3 className="font-medium mb-1">{v.name}</h3>
                <p className="text-gray-400 text-sm">
                  {Math.floor(v.duration / 60)}m {Math.floor(v.duration % 60)}s
                  &middot; {v.segment_count} segments
                </p>
                {!v.has_video_file && (
                  <p className="text-yellow-500 text-xs mt-1">
                    Video file missing
                  </p>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Player screen
  const videoUrl = `/api/videos/${selectedVideoId}/stream`;
  const segments = metadata?.segments ?? [];

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => {
              setSelectedVideoId(null);
              setMetadata(null);
            }}
            className="text-gray-400 hover:text-white transition-colors"
          >
            &larr; Back
          </button>
          <h1 className="text-xl font-bold">
            {metadata?.video ?? selectedVideoId}
          </h1>
        </div>
        <span className="text-gray-500 text-sm">
          Processed by MovieRay
        </span>
      </div>

      {/* Main layout: video + panel */}
      <div className="flex gap-4">
        {/* Left: video + timeline + controls */}
        <div className="flex-1 space-y-3">
          <VideoPlayer
            ref={playerRef}
            videoUrl={videoUrl}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={setDuration}
          />

          <SegmentTimeline
            segments={segments}
            duration={duration}
            currentTime={currentTime}
            onSeek={handleSeek}
          />

          <Controls
            skipNonContent={skipNonContent}
            onToggleSkipNonContent={() => setSkipNonContent((p) => !p)}
            skippedCategories={skippedCategories}
            onToggleCategory={handleToggleCategory}
            onPlayContentOnly={handlePlayContentOnly}
          />
        </div>

        {/* Right: segment panel */}
        <div className="w-80 flex-shrink-0">
          <SegmentPanel
            segments={segments}
            currentTime={currentTime}
            onPlay={handlePlaySegment}
            onSkip={handleSkipSegment}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
```

- [ ] **Step 2: Clean up default Vite files**

Delete `player/frontend/src/App.css` and remove its import if present. Update `player/frontend/src/main.tsx` to just:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 3: Verify build**

Run: `cd /Users/manasvardhan/Desktop/MovieRay/player/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add player/frontend/src/
git commit -m "feat: wire up App.tsx — video selection, player layout, auto-skip logic"
```

---

## Task 16: End-to-End Integration Test

**Files:** No new files — this is a manual integration test.

- [ ] **Step 1: Install pipeline dependencies**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
pip install -r pipeline/requirements.txt
```

- [ ] **Step 2: Run pipeline on a test video**

Find or create a short test video (~2-3 minutes) and run:

```bash
cd /Users/manasvardhan/Desktop/MovieRay
python -m pipeline.analyze /path/to/test/video.mp4
```

Expected: Outputs `data/<video_name>.json` with classified segments. Copy the video file into `data/` alongside the JSON.

- [ ] **Step 3: Verify JSON output is valid**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
python -c "
import json
with open('data/<video_name>.json') as f:
    data = json.load(f)
print(f'Video: {data[\"video\"]}')
print(f'Duration: {data[\"duration\"]}s')
print(f'Segments: {len(data[\"segments\"])}')
for s in data['segments']:
    print(f'  [{s[\"start\"]:.1f}-{s[\"end\"]:.1f}] {s[\"label\"]} ({s[\"confidence\"]:.0%})')
"
```

- [ ] **Step 4: Start the backend**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
pip install -r player/server/requirements.txt
python -m player.server.main
```

Expected: Server starts on port 8000.

- [ ] **Step 5: Start the frontend dev server**

In a new terminal:

```bash
cd /Users/manasvardhan/Desktop/MovieRay/player/frontend
npm run dev
```

Expected: Vite dev server starts on port 5173.

- [ ] **Step 6: Open browser and verify**

Open `http://localhost:5173`. Verify:
1. Video selection screen shows the analyzed video
2. Clicking the video opens the player
3. Video plays with synchronized audio
4. Segment timeline shows color-coded segments
5. Segment panel lists all segments with Play/Skip buttons
6. Clicking a segment in the timeline seeks the video
7. "Skip Non-Content" toggle auto-skips marked categories
8. "Play Content Only" button works
9. Category filter checkboxes toggle correctly

- [ ] **Step 7: Commit final state**

```bash
cd /Users/manasvardhan/Desktop/MovieRay
git add -A
git commit -m "feat: end-to-end integration verified — MovieRay v1.0"
```
