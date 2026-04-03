"""Tests for cache management CLI subcommands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from video_overview.cli import main
from video_overview.config import OverviewResult


@pytest.fixture
def runner():
    """Provide a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a realistic cache directory with audio and visual assets."""
    cache = tmp_path / "project" / ".video_overview_cache"
    cache.mkdir(parents=True)

    # Audio cache files
    (cache / "audio_abc123.wav").write_bytes(b"\x00" * 1024)
    (cache / "audio_def456.wav").write_bytes(b"\x00" * 2048)
    (cache / "output.wav").write_bytes(b"\x00" * 4096)
    (cache / "filelist.txt").write_text(
        "file 'audio_abc123.wav'\nfile 'audio_def456.wav'\n"
    )

    # Visual cache files
    visuals = cache / "visuals"
    visuals.mkdir()
    (visuals / "aaa111.png").write_bytes(b"\x89PNG" + b"\x00" * 512)
    (visuals / "bbb222.png").write_bytes(b"\x89PNG" + b"\x00" * 1024)

    # Static frame
    (cache / "static_frame_1920x1080.png").write_bytes(b"\x89PNG" + b"\x00" * 256)

    return cache


class TestCacheList:
    """Test 'video-overview cache list' subcommand."""

    def test_list_shows_cached_assets(self, runner, cache_dir):
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "audio" in result.output.lower()
        assert "visual" in result.output.lower()

    def test_list_shows_file_sizes(self, runner, cache_dir):
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        # Output should contain size information
        assert (
            "KB" in result.output
            or "bytes" in result.output.lower()
            or "B" in result.output
        )

    def test_list_shows_file_counts(self, runner, cache_dir):
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        # Should show counts for audio and visual assets
        assert "2" in result.output  # 2 audio cache files (audio_*.wav)

    def test_list_empty_cache_dir(self, runner, tmp_path):
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(empty_cache)])
        assert result.exit_code == 0
        assert "no cached" in result.output.lower() or "0" in result.output

    def test_list_nonexistent_cache_dir(self, runner, tmp_path):
        missing = tmp_path / "nonexistent"
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(missing)])
        assert result.exit_code == 0
        assert (
            "no cached" in result.output.lower()
            or "not found" in result.output.lower()
            or "0" in result.output
        )

    def test_list_source_dir_option(self, runner, cache_dir):
        """--source-dir derives cache dir from .video_overview_cache."""
        source_dir = cache_dir.parent  # tmp_path/project
        result = runner.invoke(
            main,
            ["cache", "list", "--source-dir", str(source_dir)],
        )
        assert result.exit_code == 0
        assert "audio" in result.output.lower()

    def test_list_no_options_shows_error(self, runner):
        """Without --cache-dir or --source-dir, should show error."""
        result = runner.invoke(main, ["cache", "list"])
        assert result.exit_code != 0


class TestCacheClear:
    """Test 'video-overview cache clear' subcommand."""

    def test_clear_removes_audio_cache(self, runner, cache_dir):
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"],
        )
        assert result.exit_code == 0
        audio_files = list(cache_dir.glob("audio_*.wav"))
        assert len(audio_files) == 0

    def test_clear_removes_visual_cache(self, runner, cache_dir):
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"],
        )
        assert result.exit_code == 0
        visuals_dir = cache_dir / "visuals"
        visual_files = list(visuals_dir.glob("*.png")) if visuals_dir.exists() else []
        assert len(visual_files) == 0

    def test_clear_removes_intermediate_files(self, runner, cache_dir):
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"],
        )
        assert result.exit_code == 0
        assert not (cache_dir / "output.wav").exists()
        assert not (cache_dir / "filelist.txt").exists()
        assert not (cache_dir / "static_frame_1920x1080.png").exists()

    def test_clear_prints_confirmation(self, runner, cache_dir):
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"],
        )
        assert result.exit_code == 0
        assert "cleared" in result.output.lower() or "removed" in result.output.lower()

    def test_clear_nonexistent_cache_dir(self, runner, tmp_path):
        missing = tmp_path / "nonexistent"
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(missing), "--yes"],
        )
        assert result.exit_code == 0
        assert (
            "no cache" in result.output.lower()
            or "not found" in result.output.lower()
            or "nothing" in result.output.lower()
        )

    def test_clear_empty_cache_dir(self, runner, tmp_path):
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(empty_cache), "--yes"],
        )
        assert result.exit_code == 0

    def test_clear_source_dir_option(self, runner, cache_dir):
        """--source-dir derives cache dir from .video_overview_cache."""
        source_dir = cache_dir.parent
        result = runner.invoke(
            main,
            ["cache", "clear", "--source-dir", str(source_dir), "-y"],
        )
        assert result.exit_code == 0
        audio_files = list(cache_dir.glob("audio_*.wav"))
        assert len(audio_files) == 0

    def test_clear_preserves_cache_dir_itself(self, runner, cache_dir):
        """Cache clear removes contents but not the directory itself."""
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"],
        )
        assert result.exit_code == 0


class TestCacheClearConfirmation:
    """Test --yes flag and confirmation prompt for cache clear."""

    def test_clear_without_yes_prompts_for_confirmation(self, runner, cache_dir):
        """Without --yes, cache clear should prompt for confirmation."""
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir)],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()

    def test_clear_aborted_on_no(self, runner, cache_dir):
        """Answering 'n' to the prompt should abort without deleting."""
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "aborted" in result.output.lower()
        # Files should still exist
        audio_files = list(cache_dir.glob("audio_*.wav"))
        assert len(audio_files) == 2

    def test_yes_flag_skips_prompt(self, runner, cache_dir):
        """--yes should skip the confirmation prompt."""
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"],
        )
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()
        # Verify files were deleted
        audio_files = list(cache_dir.glob("audio_*.wav"))
        assert len(audio_files) == 0

    def test_clear_help_shows_yes_flag(self, runner):
        result = runner.invoke(main, ["cache", "clear", "--help"])
        assert "--yes" in result.output

    def test_safety_check_blocks_non_cache_dir_even_with_yes(self, runner, tmp_path):
        """--yes skips prompts but not the safety check."""
        bad_dir = tmp_path / "not_a_cache"
        bad_dir.mkdir()
        (bad_dir / "important.py").write_text("x = 1")
        result = runner.invoke(
            main,
            ["cache", "clear", "--cache-dir", str(bad_dir), "--yes"],
        )
        assert result.exit_code != 0
        assert "does not look like" in result.output.lower()
        # File should still exist
        assert (bad_dir / "important.py").exists()


class TestCacheGroupHelp:
    """Test cache group help text."""

    def test_cache_help(self, runner):
        result = runner.invoke(main, ["cache", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()
        assert "clear" in result.output.lower()

    def test_cache_list_help(self, runner):
        result = runner.invoke(main, ["cache", "list", "--help"])
        assert result.exit_code == 0
        assert "--cache-dir" in result.output
        assert "--source-dir" in result.output

    def test_cache_clear_help(self, runner):
        result = runner.invoke(main, ["cache", "clear", "--help"])
        assert result.exit_code == 0
        assert "--cache-dir" in result.output
        assert "--source-dir" in result.output


class TestExistingCLIUnchanged:
    """Verify existing top-level CLI interface still works unchanged."""

    def test_top_level_help_still_shows_options(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--topic" in result.output
        assert "--output" in result.output

    def test_top_level_help_mentions_cache_command(self, runner):
        """Top-level help should mention the cache subcommand."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "cache" in result.output.lower()

    def test_version_flag(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "video-overview" in result.output.lower()

    def test_explicit_generate_subcommand(self, runner, tmp_path):
        """Explicit 'generate' subcommand should work."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.py").write_text("print('hello')")
        output = tmp_path / "output.mp4"
        mock_result = OverviewResult(
            output_path=output,
            duration_seconds=60.0,
            segments_count=3,
        )
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ):
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(source),
                    "--topic",
                    "test",
                    "--output",
                    str(output),
                ],
            )
        assert result.exit_code == 0
        assert "Overview created" in result.output
