# pipeline/analyze.py
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np

from pipeline.classifier import classify_heuristic, classify_with_vision
from pipeline.extractors.audio import extract_audio_features
from pipeline.extractors.motion import extract_motion_features
from pipeline.extractors.speech import extract_speech_features
from pipeline.extractors.visual import extract_visual_features
from pipeline.schemas import AnalysisResult, Segment
from pipeline.segmenter import (
    build_segments,
    merge_adjacent_same_label,
    merge_sandwiched_segments,
    merge_short_segments,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def _get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _extract_audio_track(video_path: str, output_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", "-y", output_path],
        capture_output=True,
    )


def _aggregate_features_per_segment(
    segments: list[dict],
    audio_feats: dict,
    visual_feats: dict,
    motion_feats: dict,
    speech_feats: dict,
) -> tuple[list[dict], dict[int, str]]:
    transcript_map = {}

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]

        audio_ts = audio_feats["timestamps"]
        rms_vals = [audio_feats["rms_energy"][j] for j, t in enumerate(audio_ts) if start <= t < end]
        centroid_vals = [audio_feats["spectral_centroid"][j] for j, t in enumerate(audio_ts) if start <= t < end]
        seg["avg_rms"] = float(np.mean(rms_vals)) if rms_vals else 0.0
        seg["avg_spectral_centroid"] = float(np.mean(centroid_vals)) if centroid_vals else 0.0

        vis_ts = visual_feats["frame_timestamps"]
        static_vals = [visual_feats["static_scores"][j] for j, t in enumerate(vis_ts) if start <= t < end]
        edge_vals = [visual_feats["edge_densities"][j] for j, t in enumerate(vis_ts) if start <= t < end]
        seg["static_score"] = float(np.mean(static_vals)) if static_vals else 0.0
        seg["edge_density"] = float(np.mean(edge_vals)) if edge_vals else 0.0

        mot_ts = motion_feats["motion_timestamps"]
        mot_vals = [motion_feats["motion_magnitudes"][j] for j, t in enumerate(mot_ts) if start <= t < end]
        seg["avg_motion"] = float(np.mean(mot_vals)) if mot_vals else 0.0

        speech_text = " ".join(
            s["text"] for s in speech_feats["segments"] if s["start"] < end and s["end"] > start
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

    if youtube:
        click.echo("Downloading video from YouTube...")
        dl_path = str(DATA_DIR / "%(title)s.%(ext)s")
        subprocess.run(["yt-dlp", "-f", "mp4", "-o", dl_path, youtube], check=True)
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

    duration = _get_video_duration(video_path)
    click.echo(f"Duration: {duration:.1f}s")

    click.echo("Extracting audio track...")
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_tmp.close()
    _extract_audio_track(video_path, audio_tmp.name)

    click.echo("Extracting audio features...")
    audio_feats = extract_audio_features(audio_tmp.name)

    click.echo("Extracting visual features...")
    visual_feats = extract_visual_features(video_path)

    click.echo("Extracting motion features...")
    motion_feats = extract_motion_features(video_path)

    click.echo("Transcribing speech...")
    speech_feats = extract_speech_features(audio_tmp.name)

    os.unlink(audio_tmp.name)

    click.echo("Building segments from shot boundaries...")
    raw_segments = build_segments(visual_feats["shot_boundaries"], duration)

    enriched_segments, transcript_map = _aggregate_features_per_segment(
        raw_segments, audio_feats, visual_feats, motion_feats, speech_feats
    )

    click.echo("Classifying segments (heuristics)...")
    unclassified_indices = []
    for i, seg in enumerate(enriched_segments):
        transcript = transcript_map.get(i, "")
        result = classify_heuristic(seg, duration, transcript=transcript)
        if result:
            seg["label"] = result["label"]
            seg["confidence"] = result["confidence"]
            seg["reason"] = result["reason"]
        else:
            unclassified_indices.append(i)

    if unclassified_indices:
        click.echo(f"Classifying {len(unclassified_indices)} segments via vision LLM...")
        vision_results = classify_with_vision(
            enriched_segments, unclassified_indices, transcript_map, video_path
        )
        for r in vision_results:
            idx = r["index"]
            enriched_segments[idx]["label"] = r["label"]
            enriched_segments[idx]["confidence"] = r["confidence"]
            enriched_segments[idx]["reason"] = r["reason"]

    for seg in enriched_segments:
        if "label" not in seg:
            seg["label"] = "core_content"
            seg["confidence"] = 0.5
            seg["reason"] = "Default classification (no strong signals)"

    click.echo("Merging and smoothing segments...")
    merged = merge_adjacent_same_label(enriched_segments)
    merged = merge_sandwiched_segments(merged)
    merged = merge_short_segments(merged, min_duration=5.0)

    final_segments = []
    for seg in merged:
        final_segments.append(Segment(
            start=seg["start"], end=seg["end"], label=seg["label"],
            confidence=seg.get("confidence", 0.5), reason=seg.get("reason", ""),
        ))

    result = AnalysisResult(video=os.path.basename(video_path), duration=duration, segments=final_segments)

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
