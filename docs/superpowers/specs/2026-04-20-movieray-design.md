# MovieRay — Multimodal Video Segmentation System

**Project:** CSCI 576 Multimedia Final Project
**Author:** Manas Vardhan (solo)
**Demo date:** May 6-8, 2026
**Date:** 2026-04-20

## Overview

MovieRay is a multimodal system that automatically segments long-form videos into core content and non-content, paired with a custom video player that lets users navigate and skip segments using a visual "content map."

Two independent components connected by a JSON contract:
1. **Analysis Pipeline** (Python CLI) — ingests video, extracts multimodal features, classifies temporal segments, outputs `segments.json`
2. **Video Player** (React + FastAPI) — loads video + JSON metadata, renders interactive segmented timeline with navigation controls

## Non-Content Taxonomy

Operational definitions applied consistently across all video types:

| Label | Definition |
|---|---|
| `intro` | Opening sequence: title cards, animated logos, theme music before main content begins |
| `outro` | Closing sequence: end cards, credits, sign-off after main content ends |
| `sponsorship` | Paid promotion: "brought to you by", product mentions with discount codes, mid-roll ads |
| `self_promotion` | Channel/creator promotion: subscribe reminders, merch plugs, social media callouts |
| `recap` | Repeated boilerplate or summary of previously covered material |
| `transition` | Interstitial screens, bumpers, music-only bridges between content sections |
| `dead_air` | Silence with no motion — holding screens, "starting soon" countdowns, inactivity |
| `filler` | Unrelated tangents or low-information segments not central to the video's purpose |
| `core_content` | The main material the viewer wants to watch |

## Analysis Pipeline

### Input

- Local video file path (mp4, mkv, webm, etc.)
- YouTube URL (downloaded via yt-dlp to local file)

### Feature Extraction (4 parallel passes)

**Pass 1: Audio Analysis (librosa)**
- Extract audio track via ffmpeg to WAV
- Compute RMS energy over 1-second hop windows to detect silence/dead air
- Compute spectral centroid to distinguish music vs speech vs noise
- Detect audio scene changes via sudden shifts in energy/spectrum

**Pass 2: Visual Analysis (OpenCV)**
- Sample frames at ~2 FPS (every 15 frames at 30fps)
- Compute HSV color histograms per frame
- Detect shot boundaries via histogram difference thresholding
- Detect static/near-static frames by comparing consecutive frame similarity
- Detect text-heavy frames via edge density patterns

**Pass 3: Motion Analysis (OpenCV)**
- Compute Farneback optical flow between sampled frame pairs
- Average flow magnitude per frame
- Low motion + silence = dead air / holding screen
- Low motion + speech = talking head (likely content)

**Pass 4: Speech Transcription (Whisper)**
- Whisper small model (given 8GB RAM constraint)
- Output: timestamped transcript segments with word-level timing
- Richest signal for sponsorship/self-promotion detection

**Output:** Unified feature timeline — for every 1-second window: RMS energy, spectral centroid, frame similarity score, motion magnitude, transcript text (if any).

### Segmentation

1. **Shot Boundary Detection** — use visual histogram differences + audio scene changes to find natural cut points. Never split a segment mid-shot.
2. **Window Aggregation** — for each inter-shot segment, aggregate: average motion, average energy, silence ratio, transcript text, frame similarity stats.
3. **Short Segment Merging** — segments under 3 seconds get merged with neighbors.

### Classification

**Step 1: Heuristic Classification (handles obvious cases)**

| Signal Combination | Classification |
|---|---|
| Silence + no motion + static frame | `dead_air` |
| Static frame + text overlay + first 60s of video | `intro` |
| Static frame + text overlay + last 60s of video | `outro` |
| Music + no speech + uniform motion | `transition` |
| High frame similarity across distant segments | `recap` (repeated boilerplate) |

**Step 2: LLM Classification (transcript-based segments)**
- Segments with speech that weren't classified by heuristics get sent to an LLM in batches
- Prompt includes surrounding context (previous/next segment transcript)
- LLM classifies as: `core_content`, `sponsorship`, `self_promotion`, `recap`, or `filler`
- Batched 10 segments per request to minimize API calls

**Step 3: Segment Merging & Smoothing**
- Merge adjacent segments with the same label
- Remove spurious single-window classifications (isolated misclassifications get absorbed by neighbors)
- Enforce minimum segment duration of 5 seconds

**Confidence Scoring:** Each segment gets a confidence value (0-1). Heuristic matches with strong signal = high confidence. LLM classifications carry model-stated confidence. Low-confidence segments flagged in player UI.

### Output Schema

```json
{
  "video": "podcast_ep12.mp4",
  "duration": 2730.0,
  "analyzed_at": "2026-04-20T18:00:00Z",
  "segments": [
    {
      "start": 0.0,
      "end": 15.2,
      "label": "intro",
      "type": "non-content",
      "confidence": 0.92,
      "reason": "Static title card with music, no speech"
    }
  ]
}
```

## Video Player

### Architecture

- **Backend:** FastAPI serving video files (streamed) and JSON metadata via REST endpoints
- **Frontend:** React + Vite + Tailwind CSS SPA

### API Endpoints

- `GET /api/videos` — list all analyzed videos
- `GET /api/videos/<id>/metadata` — return segments.json for a video
- `GET /api/videos/<id>/stream` — stream video file with range request support

### UI Layout (three-panel)

**Main Area: Video Player**
- HTML5 `<video>` element with custom controls
- Play / Pause / Stop, volume, current time / duration, fullscreen toggle

**Bottom: Interactive Segment Timeline**
- Horizontal bar spanning full video duration
- Each segment rendered as a colored block (green = content, distinct colors per non-content subtype)
- Playhead indicator showing current position
- Click to seek, hover for tooltip (label + time range)
- Current segment highlighted

**Right: Segment Overview Panel**
- Scrollable list of all segments
- Each entry: label, time range, content/non-content badge, Play/Skip buttons
- Active segment auto-highlighted during playback
- Low-confidence segments show visual indicator (dashed border / "?" icon)

### Global Controls

- **"Play Content Only"** — plays video, auto-skips all non-content segments
- **"Skip Non-Content" toggle** — when enabled, playhead jumps past non-content during playback
- **Category filter checkboxes** — toggle which non-content types to skip (e.g., skip sponsorships but watch intros)

### Sync Behavior

The player listens to the `<video>` element's `timeupdate` event. On each tick, checks if current time has entered a non-content segment marked for skipping. If so, seeks to that segment's end time. Segment panel auto-scrolls to keep active segment visible.

## Tech Stack

### Pipeline
- `ffmpeg` — audio extraction, video probing
- `librosa` — audio features (RMS, spectral centroid)
- `opencv-python` — frame sampling, histograms, optical flow, shot detection
- `openai-whisper` (small model) — speech transcription
- `anthropic` — Claude API (Sonnet 4.6) for transcript classification
- `yt-dlp` — YouTube download
- `click` — CLI interface
- `pydantic` — output schema validation

### Player
- `React` + `Vite` + `TypeScript` — frontend
- `FastAPI` + `uvicorn` — backend
- `Tailwind CSS` — styling

## Project Structure

```
MovieRay/
├── pipeline/
│   ├── analyze.py          # CLI entry point
│   ├── extractors/
│   │   ├── audio.py        # librosa audio features
│   │   ├── visual.py       # OpenCV frame/histogram/text detection
│   │   ├── motion.py       # optical flow
│   │   └── speech.py       # Whisper transcription
│   ├── segmenter.py        # shot boundary + window merging
│   ├── classifier.py       # heuristics + LLM classification
│   ├── schemas.py          # JSON output schema (Pydantic)
│   └── requirements.txt
├── player/
│   ├── server/
│   │   └── main.py         # FastAPI backend
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── components/
│   │   │   │   ├── VideoPlayer.tsx
│   │   │   │   ├── SegmentTimeline.tsx
│   │   │   │   ├── SegmentPanel.tsx
│   │   │   │   └── Controls.tsx
│   │   │   └── types.ts
│   │   ├── package.json
│   │   └── vite.config.ts
│   └── requirements.txt
├── data/                   # analyzed videos + JSON metadata
└── README.md
```

## CLI Usage

```bash
# Analyze a local video
python pipeline/analyze.py /path/to/video.mp4

# Analyze a YouTube video
python pipeline/analyze.py --youtube "https://youtube.com/watch?v=..."

# Start the player
cd player && python server/main.py
# Frontend served at http://localhost:5173 (Vite dev) or built and served by FastAPI
```
