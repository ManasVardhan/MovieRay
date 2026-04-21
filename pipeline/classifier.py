from __future__ import annotations

import json
import re
import anthropic

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
