"""Shared duration estimation and segment truncation utilities."""

from __future__ import annotations

from video_overview.config import ScriptSegment

# Average speaking rate: ~150 words per minute => ~2.5 words/sec
# Average word length ~5 chars => ~12.5 chars/sec
_CHARS_PER_SECOND = 12.5
_MIN_DURATION = 0.5  # seconds


def estimate_segment_duration(text: str) -> float:
    """Estimate the spoken duration of *text* in seconds.

    Uses an approximate speaking rate of 12.5 characters per second
    with a minimum of 0.5 seconds.
    """
    return max(_MIN_DURATION, len(text) / _CHARS_PER_SECOND)


def truncate_segments(
    segments: list[ScriptSegment],
    max_duration_minutes: float | None,
) -> list[ScriptSegment]:
    """Return a prefix of *segments* whose cumulative estimated duration
    stays within *max_duration_minutes*.

    - Truncates at segment boundaries only.
    - Always keeps at least one segment, even if it alone exceeds the limit.
    - If *max_duration_minutes* is ``None``, all segments are returned.
    """
    if max_duration_minutes is None:
        return list(segments)

    max_seconds = max_duration_minutes * 60.0
    cumulative = 0.0
    keep = 0

    for seg in segments:
        dur = estimate_segment_duration(seg.text)
        if cumulative + dur > max_seconds and keep >= 1:
            break
        cumulative += dur
        keep += 1

    return list(segments[:keep])
