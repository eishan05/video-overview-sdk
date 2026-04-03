"""Tests for structured logging across all generators and the CLI --verbose flag."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from video_overview.audio.generator import AudioGenerator
from video_overview.cli import main
from video_overview.config import OverviewResult, Script, ScriptSegment
from video_overview.content.reader import ContentReader
from video_overview.script.generator import ScriptGenerator
from video_overview.visuals.generator import VisualGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(segments: list[tuple[str, str, str]], title: str = "Test") -> Script:
    return Script(
        title=title,
        segments=[
            ScriptSegment(speaker=s, text=t, visual_prompt=vp) for s, t, vp in segments
        ],
    )


def _make_wav_bytes(num_samples: int = 100, sample_rate: int = 24000) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _mock_audio_response(data: bytes, mime_type: str = "audio/wav") -> MagicMock:
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


def _make_image_response(
    data: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
) -> MagicMock:
    inline_data = MagicMock()
    inline_data.data = data
    inline_data.mime_type = "image/png"
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
    part = MagicMock()
    part.inline_data = None
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# ContentReader logging
# ---------------------------------------------------------------------------


class TestContentReaderLogging:
    """Verify ContentReader emits structured log messages."""

    def test_logs_skipped_binary_extension(self, tmp_path, caplog):
        """Files skipped due to binary extension should be logged."""
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "main.py").write_text("x = 1")

        with caplog.at_level(logging.DEBUG, logger="video_overview.content.reader"):
            ContentReader().read(tmp_path)

        assert any(
            "image.png" in r.message and "skip" in r.message.lower()
            for r in caplog.records
        )

    def test_logs_skipped_directory(self, tmp_path, caplog):
        """Files inside always-skipped directories should be logged."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_bytes(b"\x00\x01")
        (tmp_path / "main.py").write_text("x = 1")

        with caplog.at_level(logging.DEBUG, logger="video_overview.content.reader"):
            ContentReader().read(tmp_path)

        # The directory skip should be logged
        assert any("__pycache__" in r.message for r in caplog.records)

    def test_logs_unicode_decode_failure(self, tmp_path, caplog):
        """Files that fail UTF-8 decode should be logged."""
        # Create a file with invalid UTF-8 bytes (but no null byte so not binary)
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x80\x81" + b"x" * 100)
        (tmp_path / "good.py").write_text("x = 1")

        with caplog.at_level(logging.DEBUG, logger="video_overview.content.reader"):
            ContentReader().read(tmp_path)

        assert any(
            "bad.txt" in r.message and "unicode" in r.message.lower()
            for r in caplog.records
        )

    def test_logs_gitignore_parse_failure(self, tmp_path, caplog):
        """A .gitignore that cannot be read should be logged."""
        gitignore = tmp_path / ".gitignore"
        # Write invalid UTF-8 as the .gitignore content
        gitignore.write_bytes(b"\xff\xfe\x80\x81")
        (tmp_path / "main.py").write_text("x = 1")

        with caplog.at_level(logging.WARNING, logger="video_overview.content.reader"):
            ContentReader().read(tmp_path)

        assert any("gitignore" in r.message.lower() for r in caplog.records)

    def test_logs_skipped_binary_content(self, tmp_path, caplog):
        """Files with binary content (null bytes) should be logged."""
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02\x03")
        (tmp_path / "main.py").write_text("x = 1")

        with caplog.at_level(logging.DEBUG, logger="video_overview.content.reader"):
            ContentReader().read(tmp_path)

        assert any(
            "data.bin" in r.message and "binary" in r.message.lower()
            for r in caplog.records
        )

    def test_reader_has_module_logger(self):
        """ContentReader module should define a module-level logger."""
        from video_overview.content import reader

        assert hasattr(reader, "logger")
        assert isinstance(reader.logger, logging.Logger)
        assert reader.logger.name == "video_overview.content.reader"


# ---------------------------------------------------------------------------
# AudioGenerator logging
# ---------------------------------------------------------------------------


class TestAudioGeneratorLogging:
    """Verify AudioGenerator logs TTS retry attempts."""

    def test_logs_retry_attempts(self, tmp_path, caplog, mocker):
        """Each TTS retry attempt should be logged."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("API temporarily unavailable"),
            _mock_audio_response(wav_data),
        ]
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch("video_overview.audio.generator.time.sleep")

        script = _make_script([("Narrator", "Hello.", "Visual")])
        gen = AudioGenerator(api_key="test-key")

        with caplog.at_level(logging.WARNING, logger="video_overview.audio.generator"):
            gen.generate(
                script=script,
                host_voice="Aoede",
                expert_voice="Charon",
                narrator_voice="Kore",
                cache_dir=tmp_path,
            )

        assert any("retry" in r.message.lower() for r in caplog.records)

    def test_audio_generator_has_module_logger(self):
        """AudioGenerator module should define a module-level logger."""
        from video_overview.audio import generator

        assert hasattr(generator, "logger")
        assert isinstance(generator.logger, logging.Logger)
        assert generator.logger.name == "video_overview.audio.generator"


# ---------------------------------------------------------------------------
# VisualGenerator logging
# ---------------------------------------------------------------------------


class TestVisualGeneratorLogging:
    """Verify VisualGenerator logs cache hits/misses and fallbacks."""

    def test_logs_cache_hit(self, tmp_path, caplog, mocker):
        """Cache hits should be logged."""
        import hashlib

        script = _make_script([("Host", "Hello.", "Cached diagram")])
        visuals_dir = tmp_path / "visuals"
        visuals_dir.mkdir(parents=True)
        prompt_hash = hashlib.md5("Cached diagram".encode()).hexdigest()
        cached_file = visuals_dir / f"{prompt_hash}.png"
        cached_file.write_bytes(b"\x89PNG_cached")

        mock_client = MagicMock()
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        gen = VisualGenerator(api_key="test-key")
        with caplog.at_level(logging.DEBUG, logger="video_overview.visuals.generator"):
            asyncio.run(gen.generate(script, tmp_path))

        assert any(
            "cache" in r.message.lower() and "hit" in r.message.lower()
            for r in caplog.records
        )

    def test_logs_cache_miss(self, tmp_path, caplog, mocker):
        """Cache misses should be logged."""
        script = _make_script([("Host", "Hello.", "New diagram")])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_image_response()
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )

        gen = VisualGenerator(api_key="test-key")
        with caplog.at_level(logging.DEBUG, logger="video_overview.visuals.generator"):
            asyncio.run(gen.generate(script, tmp_path))

        assert any(
            "cache" in r.message.lower() and "miss" in r.message.lower()
            for r in caplog.records
        )

    def test_logs_fallback_on_api_failure(self, tmp_path, caplog, mocker):
        """Fallback to ffmpeg should be logged."""
        script = _make_script([("Host", "Hello.", "Failing diagram")])
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mocker.patch(
            "video_overview.visuals.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.visuals.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        gen = VisualGenerator(api_key="test-key")
        with caplog.at_level(logging.INFO, logger="video_overview.visuals.generator"):
            asyncio.run(gen.generate(script, tmp_path))

        assert any("fallback" in r.message.lower() for r in caplog.records)

    def test_visual_generator_has_module_logger(self):
        """VisualGenerator module should define a module-level logger."""
        from video_overview.visuals import generator

        assert hasattr(generator, "logger")
        assert isinstance(generator.logger, logging.Logger)
        assert generator.logger.name == "video_overview.visuals.generator"


# ---------------------------------------------------------------------------
# ScriptGenerator logging
# ---------------------------------------------------------------------------


class TestScriptGeneratorLogging:
    """Verify ScriptGenerator logs key events."""

    def test_logs_llm_invocation(self, caplog, mocker):
        """LLM invocation should be logged."""
        valid_response = {
            "title": "Test",
            "segments": [
                {
                    "speaker": "Host",
                    "text": "Hello.",
                    "visual_prompt": "A visual",
                },
            ],
        }
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(valid_response),
                stderr="",
            ),
        )

        gen = ScriptGenerator()
        content_bundle = {
            "directory_structure": "test/\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }

        with caplog.at_level(logging.DEBUG, logger="video_overview.script.generator"):
            gen.generate(
                content_bundle=content_bundle,
                topic="Test",
                mode="conversation",
                llm_backend="claude",
            )

        assert any(
            "claude" in r.message.lower() or "llm" in r.message.lower()
            for r in caplog.records
        )

    def test_script_generator_has_module_logger(self):
        """ScriptGenerator module should define a module-level logger."""
        from video_overview.script import generator

        assert hasattr(generator, "logger")
        assert isinstance(generator.logger, logging.Logger)
        assert generator.logger.name == "video_overview.script.generator"


# ---------------------------------------------------------------------------
# CLI --verbose flag
# ---------------------------------------------------------------------------


class TestCLIVerboseFlag:
    """Verify --verbose / -v CLI flag controls log level."""

    def test_verbose_flag_accepted(self, tmp_path):
        """--verbose flag should be accepted without error."""
        runner = CliRunner()
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("x = 1")
        output_file = tmp_path / "out.mp4"

        mock_result = OverviewResult(
            output_path=output_file,
            duration_seconds=60.0,
            segments_count=3,
        )

        with patch("video_overview.cli.create_overview", return_value=mock_result):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "--topic",
                    "test",
                    "--output",
                    str(output_file),
                    "--verbose",
                ],
            )
        assert result.exit_code == 0

    def test_short_verbose_flag_accepted(self, tmp_path):
        """-v flag should be accepted as shorthand for --verbose."""
        runner = CliRunner()
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("x = 1")
        output_file = tmp_path / "out.mp4"

        mock_result = OverviewResult(
            output_path=output_file,
            duration_seconds=60.0,
            segments_count=3,
        )

        with patch("video_overview.cli.create_overview", return_value=mock_result):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "-t",
                    "test",
                    "-o",
                    str(output_file),
                    "-v",
                ],
            )
        assert result.exit_code == 0

    def test_verbose_in_help(self):
        """--verbose should appear in --help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--verbose" in result.output or "-v" in result.output

    def test_default_no_verbose(self, tmp_path):
        """Without --verbose, root log level should remain at WARNING."""
        runner = CliRunner()
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("x = 1")
        output_file = tmp_path / "out.mp4"

        mock_result = OverviewResult(
            output_path=output_file,
            duration_seconds=60.0,
            segments_count=3,
        )

        # Track what basicConfig gets called with
        log_levels = []

        def capture_and_run(**kwargs):
            log_levels.append(logging.getLogger().level)
            return mock_result

        with patch("video_overview.cli.create_overview", side_effect=capture_and_run):
            result = runner.invoke(
                main,
                [
                    str(source_dir),
                    "-t",
                    "test",
                    "-o",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        # Root logger should be at WARNING (30) or higher
        if log_levels:
            assert log_levels[0] >= logging.WARNING


# ---------------------------------------------------------------------------
# Library consumers can enable logging without CLI
# ---------------------------------------------------------------------------


class TestLibraryLogging:
    """Library consumers should be able to enable logging via standard API."""

    def test_loggers_are_children_of_video_overview(self):
        """All loggers should be under the video_overview namespace."""
        from video_overview.audio import generator as audio_gen
        from video_overview.content import reader
        from video_overview.script import generator as script_gen
        from video_overview.visuals import generator as visual_gen

        assert reader.logger.name.startswith("video_overview.")
        assert audio_gen.logger.name.startswith("video_overview.")
        assert visual_gen.logger.name.startswith("video_overview.")
        assert script_gen.logger.name.startswith("video_overview.")

    def test_no_handlers_on_library_loggers(self):
        """Library loggers should not have handlers added."""
        from video_overview.audio import generator as audio_gen
        from video_overview.content import reader
        from video_overview.script import generator as script_gen
        from video_overview.visuals import generator as visual_gen

        # Module loggers themselves should not add handlers
        for mod_logger in [
            reader.logger,
            audio_gen.logger,
            visual_gen.logger,
            script_gen.logger,
        ]:
            assert len(mod_logger.handlers) == 0, (
                f"Logger {mod_logger.name} should not have handlers attached"
            )
