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


# Average characters per word in running text (word + trailing space).
# This is deliberately aligned with _CHARS_PER_SECOND so the prompt
# budget and the post-generation truncation use the same underlying
# rate model: 12.5 chars/sec ÷ 6 chars/word ≈ 125 words/min.
_CHARS_PER_WORD_IN_TEXT = 6

# Derived speaking rate in words per minute, consistent with the
# character-based estimator used by estimate_segment_duration().
_WORDS_PER_MINUTE = _CHARS_PER_SECOND * 60.0 / _CHARS_PER_WORD_IN_TEXT

# Approximate average duration per segment (seconds)
_SECONDS_PER_SEGMENT = 30


def compute_duration_budget(
    max_duration_minutes: float | None,
    max_segments_cap: int = 20,
) -> dict | None:
    """Derive an approximate word and segment budget from a duration limit.

    The word budget is derived from the same character-based speaking
    rate used by :func:`estimate_segment_duration` so that a script
    meeting the advertised budget is unlikely to be truncated
    afterwards.

    Args:
        max_duration_minutes: Target maximum duration in minutes.
            Returns ``None`` when this is ``None``.
        max_segments_cap: Hard upper bound on the segment count
            regardless of duration.

    Returns:
        A dict with ``max_words``, ``max_segments``, and
        ``target_minutes``; or ``None`` if *max_duration_minutes* is
        ``None``.

    Raises:
        ValueError: If *max_duration_minutes* is non-positive.
    """
    if max_duration_minutes is None:
        return None

    if max_duration_minutes <= 0:
        raise ValueError(
            f"max_duration_minutes must be positive, got {max_duration_minutes}"
        )

    max_words = int(max_duration_minutes * _WORDS_PER_MINUTE)
    max_seconds = max_duration_minutes * 60.0
    derived_segments = max(2, int(max_seconds / _SECONDS_PER_SEGMENT))
    max_segments = min(derived_segments, max_segments_cap)

    return {
        "max_words": max_words,
        "max_segments": max_segments,
        "target_minutes": max_duration_minutes,
    }


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

    if max_duration_minutes <= 0:
        raise ValueError(
            f"max_duration_minutes must be positive, got {max_duration_minutes}"
        )

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
