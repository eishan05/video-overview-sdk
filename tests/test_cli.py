"""Tests for the video-overview CLI."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from video_overview.audio.generator import AudioGenerationError
from video_overview.cli import main
from video_overview.config import OverviewResult
from video_overview.script.generator import ScriptGenerationError
from video_overview.video.assembler import VideoAssemblyError
from video_overview.visuals.generator import VisualGenerationError


@pytest.fixture
def runner():
    """Provide a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def source_dir(tmp_path):
    """Create a temporary source directory."""
    d = tmp_path / "source"
    d.mkdir()
    (d / "file.py").write_text("print('hello')")
    return d


@pytest.fixture
def output_file(tmp_path):
    """Provide a valid output file path (parent exists, file does not)."""
    return tmp_path / "output.mp4"


@pytest.fixture
def mock_result(output_file):
    """Provide a mock OverviewResult."""
    return OverviewResult(
        output_path=output_file,
        duration_seconds=120.5,
        segments_count=5,
    )


class TestBasicInvocation:
    """Test basic CLI invocation with required args."""

    def test_basic_invocation_succeeds(
        self, runner, source_dir, output_file, mock_result
    ):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "Python basics",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 0

    def test_basic_invocation_prints_summary(
        self, runner, source_dir, output_file, mock_result
    ):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "-t",
                    "Python basics",
                    "-o",
                    str(output_file),
                ],
            )
        assert str(output_file) in result.output
        assert "120.5" in result.output
        assert "5 segments" in result.output

    def test_calls_create_overview_with_config(
        self, runner, source_dir, output_file, mock_result
    ):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ) as mock_fn:
            runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "Python",
                    "--output",
                    str(output_file),
                ],
            )
        mock_fn.assert_called_once()
        config = mock_fn.call_args[1]["config"]
        assert config.source_dir == source_dir
        assert config.topic == "Python"
        assert config.output == output_file


class TestAllOptionsSpecified:
    """Test CLI with all options specified."""

    def test_all_options_passed_to_config(
        self, runner, source_dir, tmp_path, mock_result
    ):
        audio_output = tmp_path / "output.mp3"
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ) as mock_fn:
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "Advanced Python",
                    "--output",
                    str(audio_output),
                    "--include",
                    "*.py",
                    "--include",
                    "*.md",
                    "--exclude",
                    "*.pyc",
                    "--exclude",
                    "__pycache__",
                    "--mode",
                    "narration",
                    "--format",
                    "audio",
                    "--host-voice",
                    "Puck",
                    "--expert-voice",
                    "Kore",
                    "--narrator-voice",
                    "Aoede",
                    "--llm",
                    "codex",
                    "--max-duration",
                    "20",
                ],
            )
        assert result.exit_code == 0
        config = mock_fn.call_args[1]["config"]
        assert config.include == ["*.py", "*.md"]
        assert config.exclude == ["*.pyc", "__pycache__"]
        assert config.mode == "narration"
        assert config.format == "audio"
        assert config.host_voice == "Puck"
        assert config.expert_voice == "Kore"
        assert config.narrator_voice == "Aoede"
        assert config.llm_backend == "codex"
        assert config.max_duration_minutes == 20


class TestMissingRequiredArgs:
    """Test that missing required args produce errors."""

    def test_missing_topic_fails(self, runner, source_dir, output_file):
        result = runner.invoke(
            main,
            [str(source_dir), "--output", str(output_file)],
        )
        assert result.exit_code != 0
        combined = result.output.lower()
        if result.exception:
            combined += str(result.exception).lower()
        assert "topic" in combined

    def test_missing_output_fails(self, runner, source_dir):
        result = runner.invoke(
            main,
            [str(source_dir), "--topic", "test"],
        )
        assert result.exit_code != 0
        combined = result.output.lower()
        if result.exception:
            combined += str(result.exception).lower()
        assert "output" in combined

    def test_missing_source_dir_fails(self, runner, output_file):
        result = runner.invoke(
            main,
            ["--topic", "test", "--output", str(output_file)],
        )
        assert result.exit_code != 0


class TestInvalidChoices:
    """Test that invalid choice values produce errors."""

    def test_invalid_mode_fails(self, runner, source_dir, output_file):
        result = runner.invoke(
            main,
            [
                str(source_dir),
                "--topic",
                "test",
                "--output",
                str(output_file),
                "--mode",
                "invalid_mode",
            ],
        )
        assert result.exit_code != 0
        out_lower = result.output.lower()
        assert "invalid_mode" in out_lower or "invalid" in out_lower

    def test_invalid_format_fails(self, runner, source_dir, output_file):
        result = runner.invoke(
            main,
            [
                str(source_dir),
                "--topic",
                "test",
                "--output",
                str(output_file),
                "--format",
                "pdf",
            ],
        )
        assert result.exit_code != 0
        out_lower = result.output.lower()
        assert "pdf" in out_lower or "invalid" in out_lower


class TestInputValidation:
    """Test Click-level validation for paths and ranges."""

    def test_nonexistent_source_dir_fails(self, runner, tmp_path, output_file):
        bad_dir = str(tmp_path / "does_not_exist")
        result = runner.invoke(
            main,
            [
                bad_dir,
                "--topic",
                "test",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_zero_max_duration_fails(self, runner, source_dir, output_file):
        result = runner.invoke(
            main,
            [
                str(source_dir),
                "--topic",
                "test",
                "--output",
                str(output_file),
                "--max-duration",
                "0",
            ],
        )
        assert result.exit_code != 0

    def test_negative_max_duration_fails(self, runner, source_dir, output_file):
        result = runner.invoke(
            main,
            [
                str(source_dir),
                "--topic",
                "test",
                "--output",
                str(output_file),
                "--max-duration",
                "-5",
            ],
        )
        assert result.exit_code != 0

    def test_nonexistent_output_parent_fails(self, runner, source_dir, tmp_path):
        bad_output = str(tmp_path / "missing_dir" / "output.mp4")
        result = runner.invoke(
            main,
            [
                str(source_dir),
                "--topic",
                "test",
                "--output",
                bad_output,
            ],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()


class TestMultiplePatterns:
    """Test multiple --include and --exclude patterns."""

    def test_multiple_include_patterns(
        self, runner, source_dir, output_file, mock_result
    ):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ) as mock_fn:
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                    "-i",
                    "*.py",
                    "-i",
                    "*.js",
                    "-i",
                    "*.md",
                ],
            )
        assert result.exit_code == 0
        config = mock_fn.call_args[1]["config"]
        assert config.include == ["*.py", "*.js", "*.md"]

    def test_multiple_exclude_patterns(
        self, runner, source_dir, output_file, mock_result
    ):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ) as mock_fn:
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                    "-e",
                    "*.pyc",
                    "-e",
                    "node_modules",
                ],
            )
        assert result.exit_code == 0
        config = mock_fn.call_args[1]["config"]
        assert config.exclude == ["*.pyc", "node_modules"]

    def test_default_include_is_wildcard(
        self, runner, source_dir, output_file, mock_result
    ):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ) as mock_fn:
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 0
        config = mock_fn.call_args[1]["config"]
        assert config.include == ["*"]


class TestKeyboardInterrupt:
    """Test that KeyboardInterrupt is handled gracefully."""

    def test_keyboard_interrupt_exits_130(self, runner, source_dir, output_file):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=KeyboardInterrupt,
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 130
        assert "interrupted" in result.output.lower()


class TestErrorHandling:
    """Test that errors from create_overview are handled gracefully."""

    def test_value_error_displays_message(self, runner, source_dir, output_file):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=ValueError("Gemini API key is required"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "Gemini API key is required" in result.output

    def test_runtime_error_displays_message(self, runner, source_dir, output_file):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=RuntimeError("ffmpeg not found"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "ffmpeg not found" in result.output

    def test_os_error_displays_message(self, runner, source_dir, output_file):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=OSError("Permission denied"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "Permission denied" in result.output

    def test_audio_generation_error_displays_message(
        self, runner, source_dir, output_file
    ):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=AudioGenerationError("audio failed"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "audio failed" in result.output

    def test_script_generation_error_displays_message(
        self, runner, source_dir, output_file
    ):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=ScriptGenerationError("script err"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "script err" in result.output

    def test_video_assembly_error_displays_message(
        self, runner, source_dir, output_file
    ):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=VideoAssemblyError("assembly err"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "assembly err" in result.output

    def test_visual_generation_error_displays_message(
        self, runner, source_dir, output_file
    ):
        with patch(
            "video_overview.cli.create_overview",
            side_effect=VisualGenerationError("visual err"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        assert result.exit_code == 1
        assert "visual err" in result.output

    def test_unexpected_error_propagates(self, runner, source_dir, output_file):
        """Unexpected exceptions are not swallowed."""
        with patch(
            "video_overview.cli.create_overview",
            side_effect=TypeError("unexpected"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        # TypeError is not caught, so Click reports it
        assert result.exit_code != 0
        assert result.exception is not None


class TestResultSummary:
    """Test that result summary is printed correctly."""

    def test_summary_format(self, runner, source_dir, output_file, mock_result):
        with patch(
            "video_overview.cli.create_overview",
            return_value=mock_result,
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                ],
            )
        expected = f"Overview created: {output_file} (120.5s, 5 segments)"
        assert expected in result.output


class TestIncludePatternMatchesNothing:
    """Test CLI behavior when include patterns match no files."""

    def test_include_no_match_exits_with_error(self, runner, source_dir, output_file):
        """When --include patterns match no files, CLI should exit 1 with message."""
        with patch(
            "video_overview.cli.create_overview",
            side_effect=ValueError("No files matched include patterns: *.rs"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                    "--include",
                    "*.rs",
                ],
            )
        assert result.exit_code == 1
        assert "include" in result.output.lower() or "no files" in result.output.lower()

    def test_include_no_match_does_not_call_script_gen(
        self, runner, source_dir, output_file
    ):
        """When include patterns match nothing, script generation should not run."""
        with patch(
            "video_overview.cli.create_overview",
            side_effect=ValueError("No files matched include patterns: *.rs"),
        ):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                    "--include",
                    "*.rs",
                ],
            )
        assert result.exit_code == 1
        assert "include" in result.output.lower() or "no files" in result.output.lower()


class TestHelpText:
    """Test that --help works."""

    def test_help_shows_usage(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "source_dir" in result.output.lower()
        assert "--topic" in result.output
        assert "--output" in result.output

    def test_help_shows_all_options(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "--include" in result.output
        assert "--exclude" in result.output
        assert "--mode" in result.output
        assert "--format" in result.output
        assert "--host-voice" in result.output
        assert "--expert-voice" in result.output
        assert "--narrator-voice" in result.output
        assert "--llm" in result.output
        assert "--max-duration" in result.output
