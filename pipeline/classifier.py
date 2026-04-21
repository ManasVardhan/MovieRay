from __future__ import annotations

import json
import anthropic


def classify_heuristic(segment: dict, video_duration: float) -> dict | None:
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

    if is_silent and is_low_motion and is_static:
        return {"label": "dead_air", "confidence": 0.95, "reason": "Silence with no motion and static frame"}

    if is_near_start and not has_speech and is_static and has_text_overlay:
        return {"label": "intro", "confidence": 0.85, "reason": "Static frame with text overlay near video start"}

    if is_near_end and not has_speech and is_static and has_text_overlay:
        return {"label": "outro", "confidence": 0.85, "reason": "Static frame with text overlay near video end"}

    if has_music and not has_speech and not is_static:
        return {"label": "transition", "confidence": 0.75, "reason": "Music without speech, non-static visuals"}

    return None


def classify_with_llm_batch(segments: list[dict], transcript_map: dict[int, str]) -> list[dict]:
    if not transcript_map:
        return []

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        import click
        click.echo("  No ANTHROPIC_API_KEY set — skipping LLM classification, defaulting to core_content")
        return [
            {"index": idx, "label": "core_content", "confidence": 0.5, "reason": "No API key, defaulted to content"}
            for idx in transcript_map
        ]

    client = anthropic.Anthropic()
    results = []

    items = list(transcript_map.items())
    for batch_start in range(0, len(items), 10):
        batch = items[batch_start : batch_start + 10]

        segments_text = ""
        for idx, text in batch:
            seg = segments[idx]
            segments_text += f'Segment {idx} [{seg["start"]:.1f}s - {seg["end"]:.1f}s]:\n"{text}"\n\n'

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
            model="claude-sonnet-4-5-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            response_text = message.content[0].text
            batch_results = json.loads(response_text)
            results.extend(batch_results)
        except (json.JSONDecodeError, IndexError, KeyError):
            for idx, _ in batch:
                results.append({
                    "index": idx, "label": "core_content",
                    "confidence": 0.5, "reason": "LLM classification failed, defaulting to content",
                })

    return results
