"""Tests for the shared duration estimation and segment truncation utilities."""

from __future__ import annotations

import pytest

from video_overview.config import ScriptSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(text: str, speaker: str = "Host") -> ScriptSegment:
    """Create a ScriptSegment with the given text."""
    return ScriptSegment(speaker=speaker, text=text, visual_prompt="prompt")


# ---------------------------------------------------------------------------
# Tests: estimate_segment_duration
# ---------------------------------------------------------------------------


class TestEstimateSegmentDuration:
    """Unit tests for the shared duration estimator."""

    def test_returns_float(self):
        from video_overview.duration import estimate_segment_duration

        result = estimate_segment_duration("Hello world")
        assert isinstance(result, float)

    def test_proportional_to_text_length(self):
        from video_overview.duration import estimate_segment_duration

        short = estimate_segment_duration("Hi")
        long = estimate_segment_duration("A" * 1000)
        assert long > short

    def test_minimum_duration_is_half_second(self):
        from video_overview.duration import estimate_segment_duration

        result = estimate_segment_duration("")
        assert result >= 0.5

    def test_uses_chars_per_second_rate(self):
        """100 chars at 12.5 chars/sec = 8.0 seconds."""
        from video_overview.duration import estimate_segment_duration

        result = estimate_segment_duration("A" * 100)
        assert result == pytest.approx(8.0)

    def test_short_text_clamps_to_minimum(self):
        """Very short text should return 0.5s minimum."""
        from video_overview.duration import estimate_segment_duration

        result = estimate_segment_duration("Hi")
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: truncate_segments
# ---------------------------------------------------------------------------


class TestTruncateSegments:
    """Unit tests for segment truncation logic."""

    def test_no_truncation_when_within_limit(self):
        """All segments kept when total duration is within the limit."""
        from video_overview.duration import truncate_segments

        # Each segment ~2.0s, 3 segments = ~6s, limit = 1 minute
        segments = [_seg("A" * 25) for _ in range(3)]
        result = truncate_segments(segments, max_duration_minutes=1)
        assert len(result) == 3

    def test_truncates_when_over_limit(self):
        """Segments are truncated when total exceeds limit."""
        from video_overview.duration import truncate_segments

        # Each "A" * 750 => 750/12.5 = 60s = 1 minute per segment
        segments = [_seg("A" * 750) for _ in range(5)]
        # Limit to 2 minutes => should keep 2 segments
        result = truncate_segments(segments, max_duration_minutes=2)
        assert len(result) == 2

    def test_always_keeps_at_least_one_segment(self):
        """Even if first segment exceeds limit, keep it."""
        from video_overview.duration import truncate_segments

        # One very long segment: 10000 chars = 800s = 13+ minutes
        segments = [_seg("A" * 10000)]
        result = truncate_segments(segments, max_duration_minutes=1)
        assert len(result) == 1

    def test_keeps_at_least_one_even_with_multiple(self):
        """When all segments are too long, keep only the first."""
        from video_overview.duration import truncate_segments

        # Each segment = 750 chars = 60s. Limit = 0.5 minutes = 30s.
        # First segment alone exceeds, but we must keep it.
        segments = [_seg("A" * 750) for _ in range(5)]
        result = truncate_segments(segments, max_duration_minutes=0.5)
        assert len(result) == 1

    def test_preserves_original_order(self):
        """Truncation keeps segments in original order."""
        from video_overview.duration import truncate_segments

        segments = [_seg(f"Segment {i} " * 5) for i in range(10)]
        result = truncate_segments(segments, max_duration_minutes=100)
        for i, seg in enumerate(result):
            assert f"Segment {i}" in seg.text

    def test_exact_boundary_keeps_segment(self):
        """When cumulative duration exactly hits the limit, include that segment."""
        from video_overview.duration import truncate_segments

        # Each segment = 75 chars = 6s. 10 segments = 60s = 1 minute.
        segments = [_seg("A" * 75) for _ in range(10)]
        result = truncate_segments(segments, max_duration_minutes=1)
        assert len(result) == 10

    def test_returns_list_of_script_segments(self):
        from video_overview.duration import truncate_segments

        segments = [_seg("Hello world")]
        result = truncate_segments(segments, max_duration_minutes=10)
        assert all(isinstance(s, ScriptSegment) for s in result)

    def test_does_not_modify_input_list(self):
        """Truncation should not mutate the input list."""
        from video_overview.duration import truncate_segments

        segments = [_seg("A" * 750) for _ in range(5)]
        original_len = len(segments)
        truncate_segments(segments, max_duration_minutes=2)
        assert len(segments) == original_len

    def test_none_max_duration_returns_all(self):
        """When max_duration_minutes is None, return all segments."""
        from video_overview.duration import truncate_segments

        segments = [_seg("A" * 750) for _ in range(5)]
        result = truncate_segments(segments, max_duration_minutes=None)
        assert len(result) == 5

    def test_zero_max_duration_raises_value_error(self):
        """Zero max_duration_minutes is invalid."""
        from video_overview.duration import truncate_segments

        segments = [_seg("Hello")]
        with pytest.raises(ValueError, match="positive"):
            truncate_segments(segments, max_duration_minutes=0)

    def test_negative_max_duration_raises_value_error(self):
        """Negative max_duration_minutes is invalid."""
        from video_overview.duration import truncate_segments

        segments = [_seg("Hello")]
        with pytest.raises(ValueError, match="positive"):
            truncate_segments(segments, max_duration_minutes=-5)


# ---------------------------------------------------------------------------
# Tests: Exact kept segment counts for various durations
# ---------------------------------------------------------------------------


class TestExactKeptSegmentCounts:
    """Assert exact segment counts for specific duration limits."""

    def _make_segments(self, n: int, chars_per_seg: int = 750) -> list[ScriptSegment]:
        """Each segment has chars_per_seg chars => chars_per_seg/12.5 seconds."""
        return [_seg("X" * chars_per_seg) for _ in range(n)]

    def test_10_segments_60s_each_limit_5min_keeps_5(self):
        """10 segments * 60s = 600s. Limit 5min=300s => keep 5."""
        from video_overview.duration import truncate_segments

        segments = self._make_segments(10, 750)  # 750/12.5=60s each
        result = truncate_segments(segments, max_duration_minutes=5)
        assert len(result) == 5

    def test_10_segments_60s_each_limit_3min_keeps_3(self):
        """10 segments * 60s = 600s. Limit 3min=180s => keep 3."""
        from video_overview.duration import truncate_segments

        segments = self._make_segments(10, 750)
        result = truncate_segments(segments, max_duration_minutes=3)
        assert len(result) == 3

    def test_10_segments_60s_each_limit_1min_keeps_1(self):
        """10 segments * 60s = 600s. Limit 1min=60s => keep 1."""
        from video_overview.duration import truncate_segments

        segments = self._make_segments(10, 750)
        result = truncate_segments(segments, max_duration_minutes=1)
        assert len(result) == 1

    def test_10_segments_60s_each_limit_10min_keeps_all(self):
        """10 segments * 60s = 600s. Limit 10min=600s => keep 10."""
        from video_overview.duration import truncate_segments

        segments = self._make_segments(10, 750)
        result = truncate_segments(segments, max_duration_minutes=10)
        assert len(result) == 10

    def test_20_segments_30s_each_limit_5min_keeps_10(self):
        """20 segments * 30s = 600s. Limit 5min=300s => keep 10."""
        from video_overview.duration import truncate_segments

        segments = self._make_segments(20, 375)  # 375/12.5=30s each
        result = truncate_segments(segments, max_duration_minutes=5)
        assert len(result) == 10

    def test_5_segments_120s_each_limit_5min_keeps_2(self):
        """5 segments * 120s = 600s. Limit 5min=300s => keep 2."""
        from video_overview.duration import truncate_segments

        segments = self._make_segments(5, 1500)  # 1500/12.5=120s each
        result = truncate_segments(segments, max_duration_minutes=5)
        assert len(result) == 2

    def test_mixed_length_segments(self):
        """Mixed lengths: 30s, 60s, 30s, 60s, 30s. Limit 3min=180s.
        Cumulative: 30, 90, 120, 180 => keep 4."""
        from video_overview.duration import truncate_segments

        segments = [
            _seg("X" * 375),  # 30s
            _seg("X" * 750),  # 60s
            _seg("X" * 375),  # 30s
            _seg("X" * 750),  # 60s
            _seg("X" * 375),  # 30s
        ]
        result = truncate_segments(segments, max_duration_minutes=3)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Tests: compute_duration_budget
# ---------------------------------------------------------------------------


class TestComputeDurationBudget:
    """Unit tests for the duration budget calculator."""

    def test_returns_dict_with_expected_keys(self):
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=5)
        assert "max_words" in budget
        assert "max_segments" in budget
        assert "target_minutes" in budget

    def test_target_minutes_matches_input(self):
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=7)
        assert budget["target_minutes"] == 7

    def test_max_words_proportional_to_duration(self):
        """At ~125 words/minute (aligned with 12.5 chars/sec), 5 min => 625."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=5)
        assert budget["max_words"] == 625

    def test_max_words_for_10_minutes(self):
        """10 minutes => ~1250 words."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=10)
        assert budget["max_words"] == 1250

    def test_max_words_for_1_minute(self):
        """1 minute => ~125 words."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=1)
        assert budget["max_words"] == 125

    def test_max_segments_scales_with_duration(self):
        """Longer duration allows more segments."""
        from video_overview.duration import compute_duration_budget

        short = compute_duration_budget(max_duration_minutes=2)
        long = compute_duration_budget(max_duration_minutes=10)
        assert long["max_segments"] > short["max_segments"]

    def test_max_segments_minimum_is_2(self):
        """Even very short durations should allow at least 2 segments."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=1)
        assert budget["max_segments"] >= 2

    def test_max_segments_capped_at_given_cap(self):
        """Segments should not exceed the explicit max_segments cap."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=60, max_segments_cap=15)
        assert budget["max_segments"] <= 15

    def test_none_duration_returns_none(self):
        """None max_duration_minutes should return None budget."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=None)
        assert budget is None

    def test_zero_raises_value_error(self):
        from video_overview.duration import compute_duration_budget

        with pytest.raises(ValueError, match="positive"):
            compute_duration_budget(max_duration_minutes=0)

    def test_negative_raises_value_error(self):
        from video_overview.duration import compute_duration_budget

        with pytest.raises(ValueError, match="positive"):
            compute_duration_budget(max_duration_minutes=-3)

    def test_all_values_are_integers(self):
        """max_words and max_segments should be integers."""
        from video_overview.duration import compute_duration_budget

        budget = compute_duration_budget(max_duration_minutes=5)
        assert isinstance(budget["max_words"], int)
        assert isinstance(budget["max_segments"], int)


class TestBudgetTruncationAlignment:
    """Verify that the prompt budget and truncation model are aligned.

    A script that exactly meets the advertised word budget should NOT
    be truncated by truncate_segments().
    """

    def test_script_at_word_budget_not_truncated(self):
        """Segments totalling max_words survive truncation."""
        from video_overview.duration import (
            compute_duration_budget,
            truncate_segments,
        )

        for minutes in (1, 3, 5, 10):
            budget = compute_duration_budget(max_duration_minutes=minutes)
            # Build segments whose total word count equals the budget.
            # Use ~6 chars per word (including space) to mirror real text.
            total_words = budget["max_words"]
            n_segments = budget["max_segments"]
            words_per_seg = total_words // n_segments
            segments = [
                _seg(" ".join(["hello"] * words_per_seg)) for _ in range(n_segments)
            ]
            result = truncate_segments(segments, max_duration_minutes=minutes)
            assert len(result) == n_segments, (
                f"At {minutes} min: expected {n_segments} segments "
                f"but got {len(result)}"
            )
