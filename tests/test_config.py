"""Tests for the configuration module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from video_overview.config import (
    OverviewConfig,
    OverviewResult,
    Script,
    ScriptSegment,
)

# ---------------------------------------------------------------------------
# ScriptSegment
# ---------------------------------------------------------------------------


class TestScriptSegment:
    """Tests for ScriptSegment model."""

    def test_creation(self):
        seg = ScriptSegment(speaker="Host", text="Hello", visual_prompt="A sunset")
        assert seg.speaker == "Host"
        assert seg.text == "Hello"
        assert seg.visual_prompt == "A sunset"

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            ScriptSegment(speaker="Host", text="Hello")  # missing visual_prompt

    def test_empty_speaker_rejected(self):
        with pytest.raises(ValidationError):
            ScriptSegment(speaker="", text="Hello", visual_prompt="A sunset")

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            ScriptSegment(speaker="Host", text="", visual_prompt="A sunset")

    def test_empty_visual_prompt_rejected(self):
        with pytest.raises(ValidationError):
            ScriptSegment(speaker="Host", text="Hello", visual_prompt="")


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


class TestScript:
    """Tests for Script model."""

    def test_creation_with_segments(self):
        seg = ScriptSegment(speaker="Host", text="Hi", visual_prompt="wave")
        script = Script(title="My Script", segments=[seg])
        assert script.title == "My Script"
        assert len(script.segments) == 1
        assert script.segments[0].speaker == "Host"

    def test_empty_segments_list(self):
        script = Script(title="Empty", segments=[])
        assert script.segments == []

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            Script(segments=[])  # missing title

    def test_missing_segments_raises(self):
        with pytest.raises(ValidationError):
            Script(title="No segments")  # missing segments

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            Script(title="", segments=[])


# ---------------------------------------------------------------------------
# OverviewResult
# ---------------------------------------------------------------------------


class TestOverviewResult:
    """Tests for OverviewResult model."""

    def test_creation(self, tmp_path):
        result = OverviewResult(
            output_path=tmp_path / "out.mp4",
            duration_seconds=123.4,
            segments_count=5,
        )
        assert result.output_path == tmp_path / "out.mp4"
        assert result.duration_seconds == 123.4
        assert result.segments_count == 5

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            OverviewResult(output_path="/tmp/x.mp4", duration_seconds=1.0)

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            OverviewResult(
                output_path="/tmp/x.mp4", duration_seconds=-1.0, segments_count=1
            )

    def test_negative_segments_count_rejected(self):
        with pytest.raises(ValidationError):
            OverviewResult(
                output_path="/tmp/x.mp4", duration_seconds=1.0, segments_count=-1
            )

    def test_zero_values_accepted(self):
        result = OverviewResult(
            output_path="/tmp/x.mp4", duration_seconds=0.0, segments_count=0
        )
        assert result.duration_seconds == 0.0
        assert result.segments_count == 0


# ---------------------------------------------------------------------------
# OverviewConfig – defaults
# ---------------------------------------------------------------------------


class TestOverviewConfigDefaults:
    """Tests for OverviewConfig default values."""

    def test_creation_with_defaults(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        output = tmp_path / "output.mp4"

        cfg = OverviewConfig(source_dir=source, output=output, topic="Testing")

        assert cfg.source_dir == source
        assert cfg.output == output
        assert cfg.topic == "Testing"
        assert cfg.include == ["*"]
        assert cfg.exclude == []
        assert cfg.mode == "conversation"
        assert cfg.format == "video"
        assert cfg.host_voice == "Aoede"
        assert cfg.expert_voice == "Charon"
        assert cfg.narrator_voice == "Kore"
        assert cfg.max_duration_minutes == 10
        assert cfg.llm_backend == "claude"

    def test_cache_dir_defaults_to_source_subdir(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        output = tmp_path / "output.mp4"

        cfg = OverviewConfig(source_dir=source, output=output, topic="Testing")
        assert cfg.cache_dir == source / ".video_overview_cache"


# ---------------------------------------------------------------------------
# OverviewConfig – all fields specified
# ---------------------------------------------------------------------------


class TestOverviewConfigAllFields:
    """Tests for OverviewConfig with all fields specified."""

    def test_all_fields(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        output = tmp_path / "output.mp4"
        cache = tmp_path / "my_cache"

        cfg = OverviewConfig(
            source_dir=source,
            output=output,
            topic="Full Config",
            include=["*.py", "*.md"],
            exclude=["__pycache__"],
            mode="narration",
            format="audio",
            host_voice="CustomHost",
            expert_voice="CustomExpert",
            narrator_voice="CustomNarrator",
            max_duration_minutes=30,
            llm_backend="codex",
            cache_dir=cache,
        )

        assert cfg.include == ["*.py", "*.md"]
        assert cfg.exclude == ["__pycache__"]
        assert cfg.mode == "narration"
        assert cfg.format == "audio"
        assert cfg.host_voice == "CustomHost"
        assert cfg.expert_voice == "CustomExpert"
        assert cfg.narrator_voice == "CustomNarrator"
        assert cfg.max_duration_minutes == 30
        assert cfg.llm_backend == "codex"
        assert cfg.cache_dir == cache


# ---------------------------------------------------------------------------
# OverviewConfig – source_dir validation
# ---------------------------------------------------------------------------


class TestOverviewConfigSourceDirValidation:
    """Tests for source_dir validation."""

    def test_source_dir_must_exist(self, tmp_path):
        with pytest.raises(ValidationError, match="source_dir"):
            OverviewConfig(
                source_dir=tmp_path / "nonexistent",
                output=tmp_path / "out.mp4",
                topic="Test",
            )

    def test_source_dir_must_be_directory(self, tmp_path):
        # Create a file, not a directory
        f = tmp_path / "afile.txt"
        f.write_text("hi")
        with pytest.raises(ValidationError, match="source_dir"):
            OverviewConfig(
                source_dir=f,
                output=tmp_path / "out.mp4",
                topic="Test",
            )

    def test_source_dir_string_coerced_to_path(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        cfg = OverviewConfig(
            source_dir=str(source),
            output=tmp_path / "out.mp4",
            topic="Test",
        )
        assert isinstance(cfg.source_dir, Path)


# ---------------------------------------------------------------------------
# OverviewConfig – output path validation
# ---------------------------------------------------------------------------


class TestOverviewConfigOutputValidation:
    """Tests for output path validation."""

    def test_output_string_coerced_to_path(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        cfg = OverviewConfig(
            source_dir=source,
            output=str(tmp_path / "out.mp4"),
            topic="Test",
        )
        assert isinstance(cfg.output, Path)

    def test_output_parent_must_exist(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError, match="output"):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "nonexistent_dir" / "out.mp4",
                topic="Test",
            )

    def test_output_rejects_existing_directory(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        out_dir = tmp_path / "out_dir"
        out_dir.mkdir()
        with pytest.raises(ValidationError, match="output"):
            OverviewConfig(
                source_dir=source,
                output=out_dir,
                topic="Test",
            )


# ---------------------------------------------------------------------------
# OverviewConfig – Literal validation
# ---------------------------------------------------------------------------


class TestOverviewConfigLiteralValidation:
    """Tests for mode and format literal validation."""

    def test_invalid_mode_rejected(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                mode="podcast",  # invalid
            )

    def test_invalid_format_rejected(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                format="gif",  # invalid
            )

    def test_invalid_llm_backend_rejected(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                llm_backend="gpt",  # invalid
            )

    def test_negative_max_duration_rejected(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                max_duration_minutes=-5,
            )

    def test_zero_max_duration_rejected(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                max_duration_minutes=0,
            )

    def test_extra_fields_rejected(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        with pytest.raises(ValidationError):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                narrtor_voice="Typo",  # misspelled field
            )


# ---------------------------------------------------------------------------
# OverviewConfig – GEMINI_API_KEY / GOOGLE_API_KEY env var loading
# ---------------------------------------------------------------------------


class TestOverviewConfigGeminiApiKey:
    """Tests for gemini_api_key property and env var loading."""

    def test_gemini_api_key_from_env(self, tmp_path, monkeypatch):
        source = tmp_path / "src_dir"
        source.mkdir()
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        cfg = OverviewConfig(
            source_dir=source, output=tmp_path / "out.mp4", topic="Test"
        )
        assert cfg.gemini_api_key == "test-gemini-key"

    def test_google_api_key_fallback(self, tmp_path, monkeypatch):
        source = tmp_path / "src_dir"
        source.mkdir()
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

        cfg = OverviewConfig(
            source_dir=source, output=tmp_path / "out.mp4", topic="Test"
        )
        assert cfg.gemini_api_key == "test-google-key"

    def test_gemini_takes_precedence_over_google(self, tmp_path, monkeypatch):
        source = tmp_path / "src_dir"
        source.mkdir()
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-wins")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-loses")

        cfg = OverviewConfig(
            source_dir=source, output=tmp_path / "out.mp4", topic="Test"
        )
        assert cfg.gemini_api_key == "gemini-wins"

    def test_no_api_key_returns_none(self, tmp_path, monkeypatch):
        source = tmp_path / "src_dir"
        source.mkdir()
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        cfg = OverviewConfig(
            source_dir=source, output=tmp_path / "out.mp4", topic="Test"
        )
        assert cfg.gemini_api_key is None


# ---------------------------------------------------------------------------
# OverviewConfig – cache_dir
# ---------------------------------------------------------------------------


class TestOverviewConfigCacheDir:
    """Tests for cache_dir behaviour."""

    def test_explicit_cache_dir(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        custom_cache = tmp_path / "custom_cache"

        cfg = OverviewConfig(
            source_dir=source,
            output=tmp_path / "out.mp4",
            topic="Test",
            cache_dir=custom_cache,
        )
        assert cfg.cache_dir == custom_cache

    def test_cache_dir_none_defaults(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()

        cfg = OverviewConfig(
            source_dir=source,
            output=tmp_path / "out.mp4",
            topic="Test",
        )
        assert cfg.cache_dir == source / ".video_overview_cache"

    def test_cache_dir_rejects_existing_file(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        cache_file = tmp_path / "not_a_dir"
        cache_file.write_text("I am a file")

        with pytest.raises(ValidationError, match="cache_dir"):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
                cache_dir=cache_file,
            )

    def test_cache_dir_accepts_existing_directory(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        cache_dir = tmp_path / "existing_cache"
        cache_dir.mkdir()

        cfg = OverviewConfig(
            source_dir=source,
            output=tmp_path / "out.mp4",
            topic="Test",
            cache_dir=cache_dir,
        )
        assert cfg.cache_dir == cache_dir

    def test_default_cache_dir_rejects_existing_file(self, tmp_path):
        source = tmp_path / "src_dir"
        source.mkdir()
        # Create a file at the default cache_dir path
        default_cache = source / ".video_overview_cache"
        default_cache.write_text("I am a file, not a directory")

        with pytest.raises(ValidationError, match="cache_dir"):
            OverviewConfig(
                source_dir=source,
                output=tmp_path / "out.mp4",
                topic="Test",
            )


# ---------------------------------------------------------------------------
# Exports from top-level package
# ---------------------------------------------------------------------------


class TestExports:
    """Verify models are exported from the top-level package."""

    def test_overview_config_exported(self):
        from video_overview import OverviewConfig as OC

        assert OC is OverviewConfig

    def test_script_segment_exported(self):
        from video_overview import ScriptSegment as SS

        assert SS is ScriptSegment

    def test_script_exported(self):
        from video_overview import Script as S

        assert S is Script

    def test_overview_result_exported(self):
        from video_overview import OverviewResult as OR

        assert OR is OverviewResult
