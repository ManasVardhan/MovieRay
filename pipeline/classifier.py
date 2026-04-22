from __future__ import annotations

import json
import os
import re

import requests

# Keyword patterns for transcript-based heuristic classification
SPONSOR_KEYWORDS = [
    r"\bsponsored?\b", r"\bbought to you by\b", r"\bbrought to you by\b",
    r"\buse code\b", r"\bdiscount code\b", r"\bpromo code\b",
    r"\bcheck out the link\b", r"\blink in the description\b",
    r"\blink below\b", r"\blink in the bio\b",
    r"\bsign up at\b", r"\bgo to\b.*\.com", r"\bhead over to\b",
    r"\bfree trial\b", r"\bspecial offer\b", r"\bexclusive deal\b",
    r"\buse my link\b", r"\baffiliate\b", r"\bthanks to\b.*\bfor sponsoring\b",
    r"\btoday'?s sponsor\b", r"\bword from our sponsor\b",
]

SELF_PROMO_KEYWORDS = [
    r"\bsubscribe\b", r"\bhit the bell\b", r"\bnotification bell\b",
    r"\blike and subscribe\b", r"\bleave a comment\b", r"\bsmash that\b",
    r"\bfollow me on\b", r"\bfollow us on\b",
    r"\bcheck out my\b", r"\bcheck out our\b",
    r"\bmerch\b", r"\bmerchandise\b", r"\bpatreon\b",
    r"\bjoin the\b.*\b(discord|community|membership)\b",
    r"\bsocial media\b", r"\binstagram\b.*\bfollow\b",
    r"\btwitter\b.*\bfollow\b",
]

RECAP_KEYWORDS = [
    r"\blast time\b", r"\blast episode\b", r"\bpreviously\b",
    r"\bquick recap\b", r"\bto recap\b", r"\bas we discussed\b",
    r"\bif you missed\b", r"\bearlier we\b",
]


def _match_keywords(text: str, patterns: list[str]) -> int:
    """Count how many keyword patterns match in the text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def classify_heuristic(
    segment: dict, video_duration: float, transcript: str = ""
) -> dict | None:
    """Apply rule-based heuristics to classify a segment.

    Returns {"label": str, "confidence": float, "reason": str} or None.
    """
    start = segment["start"]
    end = segment["end"]
    duration = end - start
    rms = segment["avg_rms"]
    motion = segment["avg_motion"]
    static = segment["static_score"]
    has_speech = segment["has_speech"]
    centroid = segment["avg_spectral_centroid"]
    edge = segment["edge_density"]

    is_silent = rms < 0.01
    is_quiet = rms < 0.05
    is_static = static > 0.9
    is_mostly_static = static > 0.7
    is_low_motion = motion < 0.3
    has_music = centroid > 1000 and rms > 0.05
    has_text_overlay = edge > 0.1
    is_near_start = start < 90.0
    is_near_end = end > video_duration - 90.0

    # === Dead air: silence + no motion + static ===
    if is_silent and is_low_motion and is_static:
        return {"label": "dead_air", "confidence": 0.95,
                "reason": "Silence with no motion and static frame"}

    # === Intro detection (relaxed) ===
    # Pattern 1: Static/text near start without speech
    if is_near_start and not has_speech and is_mostly_static and has_text_overlay:
        return {"label": "intro", "confidence": 0.85,
                "reason": "Static frame with text overlay near video start"}
    # Pattern 2: Music without speech near start (animated intro)
    if is_near_start and has_music and not has_speech and duration < 90:
        return {"label": "intro", "confidence": 0.80,
                "reason": "Music without speech near video start"}
    # Pattern 3: Short non-speech segment at very start
    if start == 0 and not has_speech and duration < 30 and is_quiet:
        return {"label": "intro", "confidence": 0.70,
                "reason": "Short quiet segment at video start"}

    # === Outro detection (relaxed) ===
    # Pattern 1: Static/text near end without speech
    if is_near_end and not has_speech and is_mostly_static and has_text_overlay:
        return {"label": "outro", "confidence": 0.85,
                "reason": "Static frame with text overlay near video end"}
    # Pattern 2: Music without speech near end
    if is_near_end and has_music and not has_speech and duration < 90:
        return {"label": "outro", "confidence": 0.80,
                "reason": "Music without speech near video end"}
    # Pattern 3: Quiet/silent end of video
    if end >= video_duration - 5 and not has_speech and is_quiet and duration < 60:
        return {"label": "outro", "confidence": 0.70,
                "reason": "Quiet segment at video end"}

    # === Ad vs Transition: music + no speech, distinguished by duration ===
    # Ads are typically 15s+ with music/visuals; transitions are short bumpers
    if has_music and not has_speech and not is_static:
        if duration >= 15 and not is_near_start and not is_near_end:
            return {"label": "sponsorship", "confidence": 0.75,
                    "reason": "Extended music-only visual segment mid-video (likely ad)"}
        return {"label": "transition", "confidence": 0.75,
                "reason": "Short music segment without speech (transition)"}

    # === Transcript-based keyword detection ===
    if transcript:
        sponsor_hits = _match_keywords(transcript, SPONSOR_KEYWORDS)
        if sponsor_hits >= 2:
            return {"label": "sponsorship", "confidence": min(0.6 + sponsor_hits * 0.1, 0.95),
                    "reason": f"Transcript contains {sponsor_hits} sponsorship keyword matches"}

        promo_hits = _match_keywords(transcript, SELF_PROMO_KEYWORDS)
        if promo_hits >= 2:
            return {"label": "self_promotion", "confidence": min(0.6 + promo_hits * 0.1, 0.90),
                    "reason": f"Transcript contains {promo_hits} self-promotion keyword matches"}

        recap_hits = _match_keywords(transcript, RECAP_KEYWORDS)
        if recap_hits >= 2:
            return {"label": "recap", "confidence": min(0.5 + recap_hits * 0.1, 0.80),
                    "reason": f"Transcript contains {recap_hits} recap keyword matches"}

    # === Self-promo near end with speech (common pattern) ===
    if is_near_end and has_speech and transcript:
        promo_hits = _match_keywords(transcript, SELF_PROMO_KEYWORDS)
        if promo_hits >= 1:
            return {"label": "self_promotion", "confidence": 0.70,
                    "reason": "Speech near video end with self-promotion keywords"}

    return None


def _extract_segment_frames(video_path: str, start: float, end: float, num_frames: int = 2) -> list[str]:
    """Extract representative frames from a segment and return as base64 JPEG strings."""
    import base64
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = end - start

    # Sample at start and middle of segment
    if num_frames == 1:
        sample_times = [start + duration * 0.5]
    else:
        sample_times = [start + duration * 0.2, start + duration * 0.7]

    frames_b64 = []
    for t in sample_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Resize to save tokens (512px wide)
            h, w = frame.shape[:2]
            new_w = 512
            new_h = int(h * new_w / w)
            frame = cv2.resize(frame, (new_w, new_h))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))

    cap.release()
    return frames_b64


def classify_with_vision(
    segments: list[dict],
    unclassified_indices: list[int],
    transcript_map: dict[int, str],
    video_path: str,
) -> list[dict]:
    """Classify segments using a vision LLM that can see frames + transcript."""
    if not unclassified_indices:
        return []

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        import click
        click.echo("  No OPENROUTER_API_KEY set — skipping vision classification")
        return [
            {"index": idx, "label": "core_content", "confidence": 0.5,
             "reason": "No API key, defaulted to content"}
            for idx in unclassified_indices
        ]

    import click
    results = []

    # Batch 5 segments per request (each has 2 images, so 10 images per call)
    for batch_start in range(0, len(unclassified_indices), 5):
        batch_indices = unclassified_indices[batch_start : batch_start + 5]
        click.echo(f"  Vision classifying segments {batch_indices}...")

        # Build multimodal content array
        content = []
        content.append({
            "type": "text",
            "text": """You are analyzing segments of a video. For each segment, I'll show you 2 representative frames and the transcript (if any speech was detected).

Classify each segment as one of:
- core_content: the main material the viewer came to watch
- intro: opening sequence, title cards, animated logos, theme music
- outro: closing sequence, end cards, credits
- sponsorship: paid ads, product promotions, discount codes, "brought to you by"
- self_promotion: subscribe/like reminders, merch plugs, social media callouts, Patreon
- recap: summary of previously covered material
- transition: brief interstitial screens, bumpers between sections
- dead_air: silence, inactivity, holding screens
- filler: unrelated tangents, low-information content

Respond with ONLY a JSON array (no markdown fencing):
[{"index": <N>, "label": "<label>", "confidence": <0.0-1.0>, "reason": "<one line>"}]

Here are the segments:
"""
        })

        for idx in batch_indices:
            seg = segments[idx]
            transcript = transcript_map.get(idx, "(no speech detected)")

            content.append({
                "type": "text",
                "text": f"\n--- Segment {idx} [{seg['start']:.1f}s - {seg['end']:.1f}s] ---\nTranscript: {transcript}\nFrames:"
            })

            frames = _extract_segment_frames(video_path, seg["start"], seg["end"])
            for frame_b64 in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}"
                    }
                })

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.5-flash",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": content}],
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            response_text = data["choices"][0]["message"]["content"].strip()

            # Strip markdown code fencing if present
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

            batch_results = json.loads(response_text)
            results.extend(batch_results)

        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            click.echo(f"  Vision batch failed: {e}")
            for idx in batch_indices:
                results.append({
                    "index": idx, "label": "core_content",
                    "confidence": 0.5, "reason": "Vision classification failed, defaulting to content",
                })

    return results
