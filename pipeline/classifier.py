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
    """Apply rule-based heuristics for only the most obvious cases.

    Only classifies dead_air (silence + no motion). Everything else
    is deferred to the vision model for better accuracy.

    Returns {"label": str, "confidence": float, "reason": str} or None.
    """
    rms = segment["avg_rms"]
    motion = segment["avg_motion"]
    static = segment["static_score"]

    is_silent = rms < 0.01
    is_static = static > 0.9
    is_low_motion = motion < 0.3

    # Dead air is the one case where signal analysis is unambiguous
    if is_silent and is_low_motion and is_static:
        return {"label": "dead_air", "confidence": 0.95,
                "reason": "Silence with no motion and static frame"}

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
                    "temperature": 0,
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

            # Try parsing, with fallback for common LLM JSON issues
            try:
                batch_results = json.loads(response_text)
            except json.JSONDecodeError:
                # Try extracting JSON array from messy response
                match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if match:
                    # Fix trailing commas before ]
                    cleaned = re.sub(r",\s*\]", "]", match.group())
                    batch_results = json.loads(cleaned)
                else:
                    raise
            results.extend(batch_results)

        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            click.echo(f"  Vision batch failed: {e}")
            for idx in batch_indices:
                results.append({
                    "index": idx, "label": "core_content",
                    "confidence": 0.5, "reason": "Vision classification failed, defaulting to content",
                })

    return results
