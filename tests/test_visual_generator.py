"""Tests for VisualGenerator."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_overview.config import Script, ScriptSegment
from video_overview.visuals import VisualGenerationError, VisualGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(
    segments: list[tuple[str, str, str]], title: str = "Test"
) -> Script:
    """Build a Script with the given (speaker, text, visual_prompt) triples."""
    return Script(
        title=title,
        segments=[
            ScriptSegment(speaker=s, text=t, visual_prompt=vp)
            for s, t, vp in segments
        ],
    )


def _make_image_response(
    data: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
    mime_type: str = "image/png",
) -> MagicMock:
    """Build a mock Gemini image generation response."""
    inline_data = MagicMock()
    inline_data.data = data
    inline_data.mime_type = mime_type

    part = MagicMock()
    part.inline_data = inline_data

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_no_image_response() -> MagicMock:
    """Build a mock response with no inline_data (text-only parts)."""
    part = MagicMock()
    part.inline_data = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _cache_key(visual_prompt: str) -> str:
    """Compute expected cache key for a visual_prompt."""
    return hashlib.md5(visual_prompt.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_key() -> str:
    return "test-api-key-12345"


@pytest.fixture()
def generator(api_key: str) -> VisualGenerator:
    return VisualGenerator(api_key=api_key)


@pytest.fixture()
def three_segment_script() -> Script:
    return _make_script([
        ("Host", "Welcome to the overview.", "A welcome banner diagram"),
        ("Expert", "Let me explain.", "Architecture diagram of the system"),
        ("Host", "That's fascinating.", "Summary infographic"),
    ])


@pytest.fixture()
def empty_script() -> Script:
    return Script(title="Empty", segments=[])


@pytest.fixture()
def duplicate_prompt_script() -> Script:
    """Script where two segments share the same visual_prompt."""
    return _make_script([
        ("Host", "First segment.", "Shared diagram prompt"),
        ("Expert", "Second segment.", "Unique diagram prompt"),
        ("Host", "Third segment.", "Shared diagram prompt"),
    ])


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


class TestAPIKeyValidation:
    def test_empty_api_key_raises_error(self):
        """Empty API key should raise VisualGenerationError."""
        with pytest.raises(VisualGenerationError, match="API key"):
            VisualGenerator(api_key="")

    def test_none_api_key_raises_error(self):
        """None API key should raise VisualGenerationError."""
        with pytest.raises(VisualGenerationError, match="API key"):
            VisualGenerator(api_key=None)


# ---------------------------------------------------------------------------
# Empty script handling
# ---------------------------------------------------------------------------


class TestEmptyScript:
    def test_empty_script_returns_empty_list(
        self, generator, empty_script, tmp_path
    ):
        """Empty script should return an empty list of paths."""
        result = asyncio.run(
            generator.generate(empty_script, tmp_path)
        )
        assert result == []


# ---------------------------------------------------------------------------
# Image generation for 3 segments
# ---------------------------------------------------------------------------


class TestImageGeneration:
    def test_generates_images_for_three_segments(
        self, generator, three_segment_script, tmp_path, mocker
    ):
        """Should generate one image per segment."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(
            generator.generate(three_segment_script, tmp_path)
        )

        assert len(result) == 3
        assert all(isinstance(p, Path) for p in result)
        assert all(p.exists() for p in result)
        assert all(p.suffix == ".png" for p in result)

    def test_images_ordered_by_segment(
        self, generator, three_segment_script, tmp_path, mocker
    ):
        """Returned paths should be ordered by segment index."""
        image_data = [
            b"\x89PNG_seg0" + b"\x00" * 100,
            b"\x89PNG_seg1" + b"\x00" * 100,
            b"\x89PNG_seg2" + b"\x00" * 100,
        ]
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_image_response(data=d) for d in image_data
        ]
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(
            generator.generate(three_segment_script, tmp_path)
        )

        # Each returned path should match the hash-based cache key
        for i, seg in enumerate(three_segment_script.segments):
            expected_hash = _cache_key(seg.visual_prompt)
            assert expected_hash in str(result[i])

    def test_api_called_with_correct_prompt(
        self, generator, tmp_path, mocker
    ):
        """API prompt should request 16:9 informative diagram."""
        script = _make_script([
            ("Host", "Hello.", "A welcome banner"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        asyncio.run(generator.generate(script, tmp_path))

        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs["contents"]
        assert "16:9" in prompt
        assert "A welcome banner" in prompt

    def test_response_modalities_include_image(
        self, generator, tmp_path, mocker
    ):
        """Config should specify IMAGE in response_modalities."""
        script = _make_script([
            ("Host", "Hello.", "A welcome banner"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        asyncio.run(generator.generate(script, tmp_path))

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert "IMAGE" in config.response_modalities


# ---------------------------------------------------------------------------
# Image extraction from response
# ---------------------------------------------------------------------------


class TestImageExtraction:
    def test_extracts_image_from_inline_data(
        self, generator, tmp_path, mocker
    ):
        """Image bytes from response inline_data should be saved to PNG."""
        image_bytes = b"\x89PNG" + b"\xDE\xAD" * 50
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response(data=image_bytes)
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        saved_data = result[0].read_bytes()
        assert saved_data == image_bytes

    def test_response_with_no_image_parts_triggers_fallback(
        self, generator, tmp_path, mocker
    ):
        """Response with no inline_data should trigger fallback image."""
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_no_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        assert len(result) == 1
        # FFmpeg should have been called for fallback
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# Concurrent generation with semaphore
# ---------------------------------------------------------------------------


class TestConcurrentGeneration:
    def test_concurrent_generation_with_semaphore(
        self, generator, three_segment_script, tmp_path, mocker
    ):
        """Should use asyncio.gather with semaphore for concurrency."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(
            generator.generate(three_segment_script, tmp_path)
        )

        # All 3 segments should have been generated
        assert len(result) == 3
        # API should have been called 3 times (one per unique prompt)
        assert mock_client.models.generate_content.call_count == 3

    def test_semaphore_limits_concurrency(
        self, generator, tmp_path, mocker
    ):
        """Semaphore should limit to max 3 concurrent generations."""
        # Create a script with 6 segments (more than semaphore limit)
        script = _make_script([
            ("Host", f"Segment {i}.", f"Unique prompt {i}")
            for i in range(6)
        ])

        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        original_to_thread = asyncio.to_thread

        async def tracked_to_thread(func, *args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent:
                    max_concurrent = concurrent_count
            try:
                return await original_to_thread(func, *args, **kwargs)
            finally:
                async with lock:
                    concurrent_count -= 1

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.visuals.generator.asyncio.to_thread",
            side_effect=tracked_to_thread,
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        assert len(result) == 6
        # Max concurrent should not exceed 3
        assert max_concurrent <= 3


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_hit_skips_api_call(
        self, generator, tmp_path, mocker
    ):
        """Cached image should be returned without API call."""
        script = _make_script([
            ("Host", "Hello.", "Cached diagram"),
        ])

        # Pre-populate cache
        visuals_dir = tmp_path / "visuals"
        visuals_dir.mkdir(parents=True)
        cache_hash = _cache_key("Cached diagram")
        cached_file = visuals_dir / f"{cache_hash}.png"
        cached_file.write_bytes(b"\x89PNG_cached")

        mock_client = MagicMock()
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        assert len(result) == 1
        assert result[0] == cached_file
        # API should NOT have been called
        mock_client.models.generate_content.assert_not_called()

    def test_cache_miss_calls_api(
        self, generator, tmp_path, mocker
    ):
        """New visual_prompt should trigger API call."""
        script = _make_script([
            ("Host", "Hello.", "Brand new diagram"),
        ])

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        assert len(result) == 1
        mock_client.models.generate_content.assert_called_once()

    def test_duplicate_prompts_reuse_cache(
        self, generator, duplicate_prompt_script, tmp_path, mocker
    ):
        """Segments with same visual_prompt should reuse the cached image."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(
            generator.generate(duplicate_prompt_script, tmp_path)
        )

        assert len(result) == 3
        # Only 2 unique prompts, so API called twice
        assert mock_client.models.generate_content.call_count == 2
        # First and third paths should be the same (same prompt)
        assert result[0] == result[2]

    def test_hash_based_cache_key(self, generator):
        """Cache key should be MD5 hash of visual_prompt."""
        prompt = "A welcome banner diagram"
        expected = hashlib.md5(prompt.encode()).hexdigest()
        assert expected == _cache_key(prompt)

    def test_duplicate_prompts_fallback_uses_segment_text(
        self, generator, tmp_path, mocker
    ):
        """When API fails, segments with same prompt get per-segment fallback."""
        script = _make_script([
            ("Host", "First segment text.", "Shared prompt"),
            ("Expert", "Second segment text.", "Shared prompt"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        assert len(result) == 2
        # Segments with different text should get different fallback paths
        assert result[0] != result[1]
        # FFmpeg called twice (once per unique segment text)
        assert mock_run.call_count == 2
        # API should only be called once; second segment skips the
        # redundant call because the prompt is already marked as failed.
        assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# Fallback on API failure
# ---------------------------------------------------------------------------


class TestFallbackOnFailure:
    def test_api_failure_creates_fallback_image(
        self, generator, tmp_path, mocker
    ):
        """API failure should create fallback text-on-background image."""
        script = _make_script([
            ("Host", "Hello.", "Failing diagram"),
        ])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        result = asyncio.run(generator.generate(script, tmp_path))

        assert len(result) == 1
        # FFmpeg should have been called for fallback
        assert mock_run.call_count == 1

    def test_fallback_ffmpeg_command_construction(
        self, generator, tmp_path, mocker
    ):
        """Fallback should use correct ffmpeg command for text slide."""
        script = _make_script([
            ("Host", "Hello world.", "Failing diagram"),
        ])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        asyncio.run(generator.generate(script, tmp_path))

        ffmpeg_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(ffmpeg_cmd)

        # Should contain key ffmpeg parameters
        assert "ffmpeg" in cmd_str
        assert "1920x1080" in cmd_str or "1920:1080" in cmd_str
        assert "#1a1a2e" in cmd_str or "0x1a1a2e" in cmd_str
        assert "drawtext" in cmd_str
        assert "fontcolor=white" in cmd_str
        assert "Hello world." in cmd_str or "Hello world" in cmd_str

    def test_fallback_uses_dark_background(
        self, generator, tmp_path, mocker
    ):
        """Fallback image should use #1a1a2e background color."""
        script = _make_script([
            ("Host", "Test.", "Failing diagram"),
        ])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        asyncio.run(generator.generate(script, tmp_path))

        ffmpeg_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(ffmpeg_cmd)
        assert "1a1a2e" in cmd_str

    def test_fallback_escapes_special_chars(
        self, generator, tmp_path, mocker
    ):
        """Special chars in text should be escaped for ffmpeg drawtext."""
        script = _make_script([
            ("Host", "It's a test: 100%!", "Failing diagram"),
        ])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        asyncio.run(generator.generate(script, tmp_path))

        # Should not raise and should have called ffmpeg
        assert mock_run.call_count == 1
        ffmpeg_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(ffmpeg_cmd)
        # Colons and special chars should be escaped
        assert "drawtext" in cmd_str


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class TestModelConfiguration:
    def test_uses_default_model(
        self, generator, tmp_path, mocker
    ):
        """Should default to gemini-2.0-flash-exp."""
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        asyncio.run(generator.generate(script, tmp_path))

        call_kwargs = mock_client.models.generate_content.call_args
        model = call_kwargs.kwargs["model"]
        assert model == "gemini-2.0-flash-exp"

    def test_custom_model(self, api_key, tmp_path, mocker):
        """Should use a custom model when specified."""
        gen = VisualGenerator(api_key=api_key, model="gemini-custom")
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        asyncio.run(gen.generate(script, tmp_path))

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-custom"

    def test_image_config_has_aspect_ratio(
        self, generator, tmp_path, mocker
    ):
        """Config should include image_config with 16:9 aspect_ratio."""
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        asyncio.run(generator.generate(script, tmp_path))

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.image_config is not None
        assert config.image_config.aspect_ratio == "16:9"


# ---------------------------------------------------------------------------
# Fallback ffmpeg errors
# ---------------------------------------------------------------------------


class TestFallbackFFmpegErrors:
    def test_ffmpeg_not_found_raises_error(
        self, generator, tmp_path, mocker
    ):
        """Missing ffmpeg during fallback should raise VisualGenerationError."""
        script = _make_script([
            ("Host", "Hello.", "Failing diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            side_effect=FileNotFoundError("ffmpeg not found"),
        )

        with pytest.raises(VisualGenerationError, match="not installed"):
            asyncio.run(generator.generate(script, tmp_path))

    def test_ffmpeg_nonzero_exit_raises_error(
        self, generator, tmp_path, mocker
    ):
        """Non-zero ffmpeg exit code should raise VisualGenerationError."""
        script = _make_script([
            ("Host", "Hello.", "Failing diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API error"
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="error"),
        )

        with pytest.raises(VisualGenerationError, match="ffmpeg fallback failed"):
            asyncio.run(generator.generate(script, tmp_path))


# ---------------------------------------------------------------------------
# Cache directory creation
# ---------------------------------------------------------------------------


class TestCacheDirectory:
    def test_creates_visuals_subdirectory(
        self, generator, tmp_path, mocker
    ):
        """Should create a 'visuals' subdirectory under cache_dir."""
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        asyncio.run(generator.generate(script, tmp_path))

        visuals_dir = tmp_path / "visuals"
        assert visuals_dir.exists()
        assert visuals_dir.is_dir()

    def test_creates_nested_cache_dir(
        self, generator, tmp_path, mocker
    ):
        """Should create cache_dir and visuals subdir if they don't exist."""
        script = _make_script([
            ("Host", "Hello.", "A diagram"),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        nested_cache = tmp_path / "deep" / "nested" / "cache"
        assert not nested_cache.exists()

        asyncio.run(generator.generate(script, nested_cache))

        assert (nested_cache / "visuals").exists()


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_list_of_paths(
        self, generator, three_segment_script, tmp_path, mocker
    ):
        """generate() should return list[Path]."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            _make_image_response()
        )
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        result = asyncio.run(
            generator.generate(three_segment_script, tmp_path)
        )

        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)
        assert len(result) == len(three_segment_script.segments)
