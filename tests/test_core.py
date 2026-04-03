"""Tests for the core orchestrator (create_overview)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from video_overview.config import (
    OverviewConfig,
    OverviewResult,
    Script,
    ScriptSegment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_script(n_segments: int = 3) -> Script:
    """Create a simple Script with the given number of segments."""
    segments = [
        ScriptSegment(
            speaker="Host" if i % 2 == 0 else "Expert",
            text=f"Segment {i} text content here.",
            visual_prompt=f"Visual prompt for segment {i}",
        )
        for i in range(n_segments)
    ]
    return Script(title="Test Overview", segments=segments)


def _make_config(tmp_path: Path, **overrides) -> OverviewConfig:
    """Create a minimal valid OverviewConfig."""
    source_dir = tmp_path / "source"
    source_dir.mkdir(exist_ok=True)
    output = tmp_path / "output.mp4"
    defaults = dict(
        source_dir=source_dir,
        output=output,
        topic="Test Topic",
    )
    defaults.update(overrides)
    return OverviewConfig(**defaults)


@pytest.fixture()
def tmp_source(tmp_path):
    """Create a temporary source directory and output path."""
    source = tmp_path / "source"
    source.mkdir()
    return tmp_path


@pytest.fixture()
def mock_content_reader():
    reader = MagicMock()
    reader.read.return_value = {
        "directory_structure": "src/\n  main.py\n",
        "files": [
            {
                "path": "main.py",
                "content": "print('hello')",
                "language": "python",
            }
        ],
        "total_files": 1,
        "total_chars": 14,
    }
    return reader


@pytest.fixture()
def mock_script_generator():
    gen = MagicMock()
    gen.generate.return_value = _make_script()
    return gen


@pytest.fixture()
def mock_audio_generator_cls():
    """Return a class-like callable that produces a mock AudioGenerator."""
    instance = MagicMock()
    instance.generate.return_value = (
        Path("/tmp/cache/output.wav"),
        [2.0, 3.0, 2.5],
    )
    cls = MagicMock(return_value=instance)
    return cls, instance


@pytest.fixture()
def mock_visual_generator_cls():
    """Return a class-like callable that produces a mock VisualGenerator."""
    instance = MagicMock()
    # VisualGenerator.generate is async
    instance.generate = AsyncMock(
        return_value=[
            Path("/tmp/cache/visuals/img0.png"),
            Path("/tmp/cache/visuals/img1.png"),
            Path("/tmp/cache/visuals/img2.png"),
        ]
    )
    cls = MagicMock(return_value=instance)
    return cls, instance


@pytest.fixture()
def mock_video_assembler_cls():
    """Return a class-like callable that produces a mock VideoAssembler."""
    instance = MagicMock()
    instance.assemble.return_value = Path("/tmp/output.mp4")
    cls = MagicMock(return_value=instance)
    return cls, instance


@pytest.fixture()
def all_mocks(
    mock_content_reader,
    mock_script_generator,
    mock_audio_generator_cls,
    mock_visual_generator_cls,
    mock_video_assembler_cls,
):
    """Patch all sub-components and return their mocks."""
    audio_cls, audio_inst = mock_audio_generator_cls
    visual_cls, visual_inst = mock_visual_generator_cls
    assembler_cls, assembler_inst = mock_video_assembler_cls

    with (
        patch("video_overview.core.ContentReader", return_value=mock_content_reader),
        patch(
            "video_overview.core.ScriptGenerator",
            return_value=mock_script_generator,
        ),
        patch("video_overview.core.AudioGenerator", audio_cls),
        patch("video_overview.core.VisualGenerator", visual_cls),
        patch("video_overview.core.VideoAssembler", assembler_cls),
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-123"}),
    ):
        yield {
            "content_reader": mock_content_reader,
            "script_generator": mock_script_generator,
            "audio_cls": audio_cls,
            "audio_inst": audio_inst,
            "visual_cls": visual_cls,
            "visual_inst": visual_inst,
            "assembler_cls": assembler_cls,
            "assembler_inst": assembler_inst,
        }


# ---------------------------------------------------------------------------
# Tests: Full pipeline (video mode)
# ---------------------------------------------------------------------------


class TestFullPipelineVideoMode:
    """End-to-end pipeline in video mode with all mocks."""

    def test_returns_overview_result(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        result = create_overview(config=config)

        assert isinstance(result, OverviewResult)

    def test_calls_content_reader(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        reader = all_mocks["content_reader"]
        reader.read.assert_called_once_with(
            source_dir=config.source_dir,
            include=config.include,
            exclude=config.exclude,
        )

    def test_calls_script_generator(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        gen = all_mocks["script_generator"]
        gen.generate.assert_called_once()
        call_kwargs = gen.generate.call_args
        assert call_kwargs.kwargs["topic"] == "Test Topic"
        assert call_kwargs.kwargs["mode"] == "conversation"

    def test_passes_max_duration_to_script_generator(self, tmp_source, all_mocks):
        """create_overview should pass max_duration_minutes to ScriptGenerator."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", max_duration_minutes=7)
        create_overview(config=config)

        gen = all_mocks["script_generator"]
        call_kwargs = gen.generate.call_args
        assert call_kwargs.kwargs["max_duration_minutes"] == 7

    def test_calls_audio_generator(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        all_mocks["audio_cls"].assert_called_once()
        all_mocks["audio_inst"].generate.assert_called_once()

    def test_calls_visual_generator(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        all_mocks["visual_cls"].assert_called_once()
        all_mocks["visual_inst"].generate.assert_called_once()

    def test_calls_video_assembler(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        all_mocks["assembler_cls"].assert_called_once()
        all_mocks["assembler_inst"].assemble.assert_called_once()

    def test_pipeline_order(self, tmp_source, all_mocks):
        """Verify the pipeline runs in correct order:
        read -> script -> audio+visuals -> assemble."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        call_order = []

        reader = all_mocks["content_reader"]

        def track_read(*a, **kw):
            call_order.append("read")
            return {
                "directory_structure": "src/\n  main.py\n",
                "files": [
                    {
                        "path": "main.py",
                        "content": "print('hello')",
                        "language": "python",
                    }
                ],
                "total_files": 1,
                "total_chars": 14,
            }

        reader.read.side_effect = track_read

        gen = all_mocks["script_generator"]

        def track_generate(*a, **kw):
            call_order.append("script")
            return _make_script()

        gen.generate.side_effect = track_generate

        audio_inst = all_mocks["audio_inst"]

        def track_audio(*a, **kw):
            call_order.append("audio")
            return (Path("/tmp/cache/output.wav"), [2.0, 3.0, 2.5])

        audio_inst.generate.side_effect = track_audio

        visual_inst = all_mocks["visual_inst"]

        async def track_visual(*a, **kw):
            call_order.append("visual")
            return [Path("/tmp/cache/visuals/img0.png")] * 3

        visual_inst.generate = AsyncMock(side_effect=track_visual)

        assembler_inst = all_mocks["assembler_inst"]

        def track_assemble(*a, **kw):
            call_order.append("assemble")
            return Path("/tmp/output.mp4")

        assembler_inst.assemble.side_effect = track_assemble

        create_overview(config=config)

        # read and script must come before audio/visual
        assert call_order.index("read") < call_order.index("audio")
        assert call_order.index("script") < call_order.index("audio")
        assert call_order.index("script") < call_order.index("visual")
        # assemble must come last
        assert call_order.index("assemble") > call_order.index("audio")
        assert call_order.index("assemble") > call_order.index("visual")


# ---------------------------------------------------------------------------
# Tests: Full pipeline (audio mode)
# ---------------------------------------------------------------------------


class TestFullPipelineAudioMode:
    """Audio-only mode skips visuals and video assembly."""

    def test_skips_visual_generation(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        all_mocks["visual_inst"].generate.assert_not_called()

    def test_assembles_as_audio_not_video(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        call_kwargs = all_mocks["assembler_inst"].assemble.call_args
        assert call_kwargs.kwargs["format"] == "audio"

    def test_still_calls_content_reader(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        all_mocks["content_reader"].read.assert_called_once()

    def test_still_calls_script_generator(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        all_mocks["script_generator"].generate.assert_called_once()

    def test_still_calls_audio_generator(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        all_mocks["audio_inst"].generate.assert_called_once()

    def test_returns_overview_result(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        result = create_overview(config=config)

        assert isinstance(result, OverviewResult)

    def test_wav_output_skips_assembler(self, tmp_source, all_mocks):
        """WAV audio output uses shutil.copy2, not VideoAssembler."""
        from video_overview.core import create_overview

        out = tmp_source / "output.wav"
        config = _make_config(tmp_source, format="audio", output=out)
        with patch("video_overview.core.shutil.copy2"):
            create_overview(config=config)

        # Assembler should NOT be instantiated for WAV output
        all_mocks["assembler_cls"].assert_not_called()

    def test_mp3_output_uses_assembler(self, tmp_source, all_mocks):
        """Non-WAV audio output routes through VideoAssembler."""
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        all_mocks["assembler_cls"].assert_called_once()

    def test_wav_output_same_path_no_error(self, tmp_source, all_mocks):
        """When output path equals the cache WAV path, no
        SameFileError should occur."""
        from video_overview.core import create_overview

        cache_dir = tmp_source / "source" / ".video_overview_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_wav = cache_dir / "output.wav"
        # AudioGenerator returns the cache wav path
        all_mocks["audio_inst"].generate.return_value = (
            cache_wav,
            [2.0, 3.0],
        )
        config = _make_config(tmp_source, format="audio", output=cache_wav)
        # Should not raise SameFileError
        result = create_overview(config=config)
        assert result.output_path == cache_wav


# ---------------------------------------------------------------------------
# Tests: Config creation
# ---------------------------------------------------------------------------


class TestConfigCreation:
    """create_overview accepts both OverviewConfig and keyword args."""

    def test_accepts_config_object(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source)
        result = create_overview(config=config)
        assert isinstance(result, OverviewResult)

    def test_accepts_keyword_arguments(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        source_dir = tmp_source / "source"
        result = create_overview(
            source_dir=source_dir,
            output=tmp_source / "output.mp4",
            topic="Keyword Topic",
        )
        assert isinstance(result, OverviewResult)

    def test_kwargs_passed_to_config(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        source_dir = tmp_source / "source"
        create_overview(
            source_dir=source_dir,
            output=tmp_source / "output.mp4",
            topic="Keyword Topic",
            mode="narration",
        )
        gen = all_mocks["script_generator"]
        call_kwargs = gen.generate.call_args.kwargs
        assert call_kwargs["mode"] == "narration"

    def test_config_takes_precedence_over_kwargs(self, tmp_source, all_mocks):
        """If both config and kwargs are given, config wins."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, topic="Config Topic")
        result = create_overview(config=config, topic="Kwargs Topic")
        assert isinstance(result, OverviewResult)
        # The script generator should get the config's topic
        gen = all_mocks["script_generator"]
        call_kwargs = gen.generate.call_args.kwargs
        assert call_kwargs["topic"] == "Config Topic"


# ---------------------------------------------------------------------------
# Tests: Progress output
# ---------------------------------------------------------------------------


class TestProgressMessages:
    """Progress is printed to stderr at each stage."""

    def test_progress_to_stderr(self, tmp_source, all_mocks, capsys):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        captured = capsys.readouterr()
        stderr = captured.err

        stderr_lower = stderr.lower()
        assert "reading content" in stderr_lower
        assert "generating script" in stderr_lower
        assert "generating audio" in stderr_lower
        assert "visuals" in stderr_lower or "visual" in stderr_lower
        assert "assembling" in stderr_lower

    def test_audio_mode_no_visual_progress(self, tmp_source, all_mocks, capsys):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out)
        create_overview(config=config)

        captured = capsys.readouterr()
        stderr = captured.err

        # Audio mode should not mention visuals or assembling video
        assert "generating visuals" not in stderr.lower()
        assert "assembling video" not in stderr.lower()


# ---------------------------------------------------------------------------
# Tests: API key validation
# ---------------------------------------------------------------------------


class TestAPIKeyValidation:
    """Missing API key produces a clear error."""

    def test_missing_gemini_api_key(self, tmp_source):
        from video_overview.core import create_overview

        config = _make_config(tmp_source)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="(?i)api.key"):
                create_overview(config=config)


# ---------------------------------------------------------------------------
# Tests: Empty source content (fail fast)
# ---------------------------------------------------------------------------


class TestEmptySourceContent:
    """create_overview should raise ValueError when the source directory
    yields zero readable files -- covering empty directories, binary-only
    directories, and exclude-all filter combinations."""

    def test_raises_on_empty_directory(self, tmp_source, all_mocks):
        """Empty source directory should raise ValueError before script gen."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)

        with pytest.raises(ValueError, match="(?i)no readable files"):
            create_overview(config=config)

    def test_raises_on_binary_only_directory(self, tmp_source, all_mocks):
        """Directory containing only binary files should raise ValueError."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n  image.png\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)

        with pytest.raises(ValueError, match="(?i)no readable files"):
            create_overview(config=config)

    def test_raises_on_exclude_all(self, tmp_source, all_mocks):
        """When exclude filters remove all files, should raise ValueError."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n  main.py\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)

        with pytest.raises(ValueError, match="(?i)no readable files"):
            create_overview(config=config)

    def test_script_generator_not_called_on_empty_content(self, tmp_source, all_mocks):
        """Script generation should not be attempted with empty content."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)

        with pytest.raises(ValueError):
            create_overview(config=config)

        all_mocks["script_generator"].generate.assert_not_called()

    def test_error_message_mentions_source_dir(self, tmp_source, all_mocks):
        """The error message should mention the source directory."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)

        with pytest.raises(ValueError, match=str(config.source_dir)):
            create_overview(config=config)

    def test_raises_on_zero_total_chars(self, tmp_source, all_mocks):
        """Files present but all empty (zero chars) should raise ValueError."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n  empty.py\n",
            "files": [
                {
                    "path": "empty.py",
                    "content": "",
                    "language": "python",
                }
            ],
            "total_files": 1,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)

        with pytest.raises(ValueError, match="(?i)no readable files"):
            create_overview(config=config)

    def test_cache_dir_not_created_on_empty_content(self, tmp_source, all_mocks):
        """Cache directory should not be created when content is empty."""
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.return_value = {
            "directory_structure": "source/\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        config = _make_config(tmp_source)
        cache_dir = config.cache_dir

        # Remove cache dir if it exists
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)

        with pytest.raises(ValueError):
            create_overview(config=config)

        assert not cache_dir.exists()

    def test_nonzero_total_files_does_not_raise(self, tmp_source, all_mocks):
        """Normal content bundles (total_files > 0) should proceed normally."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source)
        # all_mocks provides a default bundle with total_files=1
        result = create_overview(config=config)
        assert isinstance(result, OverviewResult)


# ---------------------------------------------------------------------------
# Tests: Error propagation
# ---------------------------------------------------------------------------


class TestContentReaderError:
    def test_content_reader_error_propagation(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        all_mocks["content_reader"].read.side_effect = FileNotFoundError("not found")
        config = _make_config(tmp_source)

        with pytest.raises(FileNotFoundError, match="not found"):
            create_overview(config=config)


class TestScriptGeneratorError:
    def test_script_generator_error_propagation(self, tmp_source, all_mocks):
        from video_overview.core import create_overview
        from video_overview.script.generator import ScriptGenerationError

        all_mocks["script_generator"].generate.side_effect = ScriptGenerationError(
            "LLM failed"
        )
        config = _make_config(tmp_source)

        with pytest.raises(ScriptGenerationError, match="LLM failed"):
            create_overview(config=config)


class TestAudioGeneratorError:
    def test_audio_generator_error_propagation(self, tmp_source, all_mocks):
        from video_overview.audio.generator import AudioGenerationError
        from video_overview.core import create_overview

        all_mocks["audio_inst"].generate.side_effect = AudioGenerationError(
            "TTS failed"
        )
        config = _make_config(tmp_source)

        with pytest.raises(AudioGenerationError, match="TTS failed"):
            create_overview(config=config)


class TestVisualGeneratorError:
    def test_visual_generator_error_propagation(self, tmp_source, all_mocks):
        from video_overview.core import create_overview
        from video_overview.visuals.generator import VisualGenerationError

        all_mocks["visual_inst"].generate = AsyncMock(
            side_effect=VisualGenerationError("Image gen failed")
        )
        config = _make_config(tmp_source, format="video")

        with pytest.raises(VisualGenerationError, match="Image gen failed"):
            create_overview(config=config)


class TestVideoAssemblerError:
    def test_video_assembler_error_propagation(self, tmp_source, all_mocks):
        from video_overview.core import create_overview
        from video_overview.video.assembler import VideoAssemblyError

        all_mocks["assembler_inst"].assemble.side_effect = VideoAssemblyError(
            "ffmpeg broken"
        )
        config = _make_config(tmp_source, format="video")

        with pytest.raises(VideoAssemblyError, match="ffmpeg broken"):
            create_overview(config=config)


# ---------------------------------------------------------------------------
# Tests: Concurrent audio + visual generation
# ---------------------------------------------------------------------------


class TestConcurrentGeneration:
    """Audio and visual generation run concurrently via asyncio."""

    def test_audio_and_visual_both_execute(self, tmp_source, all_mocks):
        """Verify both audio and visual generation are invoked."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        all_mocks["audio_inst"].generate.assert_called_once()
        all_mocks["visual_inst"].generate.assert_called_once()

    def test_uses_asyncio_for_concurrency(self, tmp_source, all_mocks):
        """Verify the orchestrator dispatches audio and visuals
        through the async helper, which uses asyncio.gather for
        concurrent execution."""
        import threading

        from video_overview.core import create_overview

        audio_thread_id = None
        visual_thread_id = None

        def audio_generate(*a, **kw):
            nonlocal audio_thread_id
            audio_thread_id = threading.current_thread().ident
            return (
                Path("/tmp/cache/output.wav"),
                [2.0, 3.0, 2.5],
            )

        all_mocks["audio_inst"].generate.side_effect = audio_generate

        async def visual_generate(*a, **kw):
            nonlocal visual_thread_id
            visual_thread_id = threading.current_thread().ident
            return [Path("/tmp/cache/visuals/img0.png")] * 3

        all_mocks["visual_inst"].generate = AsyncMock(side_effect=visual_generate)

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        # Both must have executed
        assert audio_thread_id is not None
        assert visual_thread_id is not None
        # Audio runs in executor (different thread), visual runs
        # in the event loop thread -- they should differ, proving
        # concurrent dispatch via asyncio
        assert audio_thread_id != visual_thread_id


class TestRunAsyncFallback:
    """_run_async falls back to a thread when already in a loop."""

    @pytest.mark.asyncio
    async def test_run_async_inside_event_loop(self):
        """Calling _run_async from inside a running event loop
        should succeed by spawning a new thread."""
        from video_overview.core import _run_async

        async def dummy():
            return 42

        # We are inside the pytest-asyncio event loop here
        result = _run_async(dummy())
        assert result == 42


# ---------------------------------------------------------------------------
# Tests: OverviewResult populated correctly
# ---------------------------------------------------------------------------


class TestOverviewResultPopulation:
    def test_result_has_output_path(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        all_mocks["assembler_inst"].assemble.return_value = config.output
        result = create_overview(config=config)

        assert result.output_path == config.output

    def test_result_has_duration_seconds(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        # Durations: [2.0, 3.0, 2.5] => sum = 7.5
        result = create_overview(config=config)

        assert result.duration_seconds == pytest.approx(7.5)

    def test_result_has_segments_count(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        result = create_overview(config=config)

        assert result.segments_count == 3

    def test_audio_mode_result(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        audio_path = Path("/tmp/cache/output.wav")
        all_mocks["audio_inst"].generate.return_value = (
            audio_path,
            [1.0, 2.0, 3.0],
        )

        out = tmp_source / "output.mp3"
        # Assembler returns the final output path
        all_mocks["assembler_inst"].assemble.return_value = out
        config = _make_config(tmp_source, format="audio", output=out)
        result = create_overview(config=config)

        assert result.output_path == out
        assert result.duration_seconds == pytest.approx(6.0)
        assert result.segments_count == 3


# ---------------------------------------------------------------------------
# Tests: Cache directory creation
# ---------------------------------------------------------------------------


class TestCacheDirectory:
    def test_cache_dir_created(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        assert config.cache_dir.exists()
        assert config.cache_dir.is_dir()

    def test_explicit_cache_dir(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        custom_cache = tmp_source / "my_cache"
        config = _make_config(tmp_source, cache_dir=custom_cache, format="video")
        create_overview(config=config)

        assert custom_cache.exists()
        assert custom_cache.is_dir()


# ---------------------------------------------------------------------------
# Tests: __init__.py exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_create_overview_exported(self):
        from video_overview import create_overview

        assert callable(create_overview)

    def test_overview_config_exported(self):
        from video_overview import OverviewConfig

        assert OverviewConfig is not None


# ---------------------------------------------------------------------------
# Tests: max_duration_minutes truncation in create_overview
# ---------------------------------------------------------------------------


class TestMaxDurationTruncation:
    """create_overview truncates the script segment list to stay within
    config.max_duration_minutes."""

    @staticmethod
    def _make_long_script(n_segments: int = 10) -> Script:
        """Create a Script where each segment is ~60s (750 chars at 12.5 c/s)."""
        segments = [
            ScriptSegment(
                speaker="Host" if i % 2 == 0 else "Expert",
                text="X" * 750,
                visual_prompt=f"Visual prompt for segment {i}",
            )
            for i in range(n_segments)
        ]
        return Script(title="Long Overview", segments=segments)

    def test_truncates_segments_for_video_mode(self, tmp_source, all_mocks):
        """10 segments * 60s = 600s.  max_duration_minutes=3 => keep 3."""
        from video_overview.core import create_overview

        all_mocks["script_generator"].generate.return_value = self._make_long_script(10)
        # Audio mock must return durations matching the *truncated* segment count
        all_mocks["audio_inst"].generate.return_value = (
            Path("/tmp/cache/output.wav"),
            [60.0] * 3,
        )
        all_mocks["visual_inst"].generate = AsyncMock(
            return_value=[Path(f"/tmp/cache/visuals/img{i}.png") for i in range(3)]
        )

        config = _make_config(tmp_source, format="video", max_duration_minutes=3)
        result = create_overview(config=config)

        # The script passed to audio generator should have only 3 segments
        audio_call = all_mocks["audio_inst"].generate.call_args
        audio_script = audio_call.kwargs.get("script") or audio_call.args[0]
        assert len(audio_script.segments) == 3

        # The script passed to visual generator should also have only 3 segments
        visual_call = all_mocks["visual_inst"].generate.call_args
        visual_script = visual_call.kwargs.get("script") or visual_call.args[0]
        assert len(visual_script.segments) == 3

        assert result.segments_count == 3

    def test_truncates_segments_for_audio_mode(self, tmp_source, all_mocks):
        """Audio mode also truncates segments."""
        from video_overview.core import create_overview

        all_mocks["script_generator"].generate.return_value = self._make_long_script(10)
        all_mocks["audio_inst"].generate.return_value = (
            Path("/tmp/cache/output.wav"),
            [60.0] * 5,
        )

        out = tmp_source / "output.mp3"
        config = _make_config(
            tmp_source, format="audio", output=out, max_duration_minutes=5
        )
        result = create_overview(config=config)

        audio_call = all_mocks["audio_inst"].generate.call_args
        script_arg = audio_call.kwargs.get("script") or audio_call.args[0]
        assert len(script_arg.segments) == 5
        assert result.segments_count == 5

    def test_no_truncation_when_within_limit(self, tmp_source, all_mocks):
        """All segments kept if total estimated duration is within limit."""
        from video_overview.core import create_overview

        all_mocks["script_generator"].generate.return_value = _make_script(3)
        config = _make_config(tmp_source, format="video", max_duration_minutes=10)
        result = create_overview(config=config)

        assert result.segments_count == 3

    def test_always_keeps_at_least_one_segment(self, tmp_source, all_mocks):
        """Even when all segments exceed the limit, keep the first one."""
        from video_overview.core import create_overview

        # 5 segments, each 10000 chars = 800s = 13+ minutes
        # Limit = 1 minute, so even the first exceeds, but we keep it
        long_script = Script(
            title="Long",
            segments=[
                ScriptSegment(
                    speaker="Host" if i % 2 == 0 else "Expert",
                    text="X" * 10000,
                    visual_prompt=f"prompt {i}",
                )
                for i in range(5)
            ],
        )
        all_mocks["script_generator"].generate.return_value = long_script
        all_mocks["audio_inst"].generate.return_value = (
            Path("/tmp/cache/output.wav"),
            [800.0],
        )
        all_mocks["visual_inst"].generate = AsyncMock(
            return_value=[Path("/tmp/cache/visuals/img0.png")]
        )

        config = _make_config(tmp_source, format="video", max_duration_minutes=1)
        result = create_overview(config=config)

        # Only 1 segment kept despite 5 being available
        assert result.segments_count == 1
        audio_call = all_mocks["audio_inst"].generate.call_args
        script_arg = audio_call.kwargs.get("script") or audio_call.args[0]
        assert len(script_arg.segments) == 1

    def test_exact_kept_count_5min_limit(self, tmp_source, all_mocks):
        """10 segments * 60s each, limit 5min => keep exactly 5."""
        from video_overview.core import create_overview

        all_mocks["script_generator"].generate.return_value = self._make_long_script(10)
        all_mocks["audio_inst"].generate.return_value = (
            Path("/tmp/cache/output.wav"),
            [60.0] * 5,
        )
        all_mocks["visual_inst"].generate = AsyncMock(
            return_value=[Path(f"/tmp/cache/visuals/img{i}.png") for i in range(5)]
        )

        config = _make_config(tmp_source, format="video", max_duration_minutes=5)
        create_overview(config=config)

        audio_call = all_mocks["audio_inst"].generate.call_args
        audio_script = audio_call.kwargs.get("script") or audio_call.args[0]
        assert len(audio_script.segments) == 5

        visual_call = all_mocks["visual_inst"].generate.call_args
        visual_script = visual_call.kwargs.get("script") or visual_call.args[0]
        assert len(visual_script.segments) == 5


# ---------------------------------------------------------------------------
# Tests: skip_visuals mode
# ---------------------------------------------------------------------------


class TestSkipVisualsMode:
    """When skip_visuals=True, visual generation is skipped and a static
    dark frame is synthesised locally."""

    def test_skips_visual_generator(self, tmp_source, all_mocks):
        """VisualGenerator should not be called when skip_visuals=True."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", skip_visuals=True)
        # The static frame generator needs to return a valid path
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = Path("/tmp/cache/static_frame.png")
            create_overview(config=config)

        all_mocks["visual_inst"].generate.assert_not_called()

    def test_creates_static_frame(self, tmp_source, all_mocks):
        """A static dark frame should be created when skip_visuals=True."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", skip_visuals=True)
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = Path("/tmp/cache/static_frame.png")
            create_overview(config=config)

        mock_frame.assert_called_once()

    def test_static_frame_passed_to_assembler(self, tmp_source, all_mocks):
        """The assembler should receive one static frame replicated for
        each segment."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", skip_visuals=True)
        frame_path = Path("/tmp/cache/static_frame.png")
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = frame_path
            create_overview(config=config)

        call_kwargs = all_mocks["assembler_inst"].assemble.call_args
        image_paths = call_kwargs.kwargs.get("image_paths") or call_kwargs.args[1]
        # One static frame per segment (3 segments from _make_script)
        assert len(image_paths) == 3
        assert all(p == frame_path for p in image_paths)

    def test_returns_overview_result(self, tmp_source, all_mocks):
        """skip_visuals pipeline should still return a valid result."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", skip_visuals=True)
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = Path("/tmp/cache/static_frame.png")
            result = create_overview(config=config)

        assert isinstance(result, OverviewResult)

    def test_still_generates_audio(self, tmp_source, all_mocks):
        """Audio generation should still occur when skip_visuals=True."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", skip_visuals=True)
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = Path("/tmp/cache/static_frame.png")
            create_overview(config=config)

        all_mocks["audio_inst"].generate.assert_called_once()

    def test_progress_message_mentions_skip(self, tmp_source, all_mocks, capsys):
        """Progress should indicate visuals are being skipped."""
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", skip_visuals=True)
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = Path("/tmp/cache/static_frame.png")
            create_overview(config=config)

        captured = capsys.readouterr()
        stderr = captured.err.lower()
        assert "skip" in stderr or "static" in stderr


class TestCreateStaticFrame:
    """Tests for the _create_static_frame helper."""

    def test_creates_png_file(self, tmp_path):
        """The helper should produce a PNG file via ffmpeg (mocked)."""
        from video_overview.core import _create_static_frame

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("video_overview.core.subprocess.run", return_value=mock_result):
            frame_path = _create_static_frame(
                cache_dir=tmp_path,
                width=1920,
                height=1080,
            )
        assert frame_path.suffix == ".png"
        assert frame_path.parent == tmp_path

    def test_ffmpeg_called_with_correct_dimensions(self, tmp_path):
        """The ffmpeg command should use the requested width and height."""
        from video_overview.core import _create_static_frame

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch(
            "video_overview.core.subprocess.run", return_value=mock_result
        ) as mock_run:
            _create_static_frame(cache_dir=tmp_path, width=1280, height=720)

        cmd = mock_run.call_args[0][0]
        # The lavfi color source should contain the dimensions
        color_arg = [a for a in cmd if "color=" in a][0]
        assert "1280x720" in color_arg

    def test_reuses_cached_frame(self, tmp_path):
        """If the frame already exists, ffmpeg should not be called again."""
        from video_overview.core import _create_static_frame

        # Pre-create the cache file
        expected = tmp_path / "static_frame_1920x1080.png"
        expected.write_bytes(b"fake png")

        with patch("video_overview.core.subprocess.run") as mock_run:
            frame_path = _create_static_frame(
                cache_dir=tmp_path, width=1920, height=1080
            )
        mock_run.assert_not_called()
        assert frame_path == expected

    def test_frame_dimensions_in_filename(self, tmp_path):
        """Frame filename should encode the dimensions for cache correctness."""
        from video_overview.core import _create_static_frame

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("video_overview.core.subprocess.run", return_value=mock_result):
            frame_path = _create_static_frame(
                cache_dir=tmp_path, width=1280, height=720
            )
        assert "1280x720" in frame_path.name

    def test_ffmpeg_not_found_raises_video_assembly_error(self, tmp_path):
        """FileNotFoundError from ffmpeg should raise VideoAssemblyError."""
        from video_overview.core import _create_static_frame
        from video_overview.video.assembler import VideoAssemblyError

        with patch(
            "video_overview.core.subprocess.run",
            side_effect=FileNotFoundError("ffmpeg not found"),
        ):
            with pytest.raises(VideoAssemblyError, match="ffmpeg"):
                _create_static_frame(cache_dir=tmp_path, width=1920, height=1080)

    def test_ffmpeg_timeout_raises_video_assembly_error(self, tmp_path):
        """TimeoutExpired from ffmpeg should raise VideoAssemblyError."""
        import subprocess

        from video_overview.core import _create_static_frame
        from video_overview.video.assembler import VideoAssemblyError

        with patch(
            "video_overview.core.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30),
        ):
            with pytest.raises(VideoAssemblyError, match="timed out"):
                _create_static_frame(cache_dir=tmp_path, width=1920, height=1080)

    def test_ffmpeg_nonzero_exit_raises_video_assembly_error(self, tmp_path):
        """Non-zero ffmpeg exit code should raise VideoAssemblyError."""
        from video_overview.core import _create_static_frame
        from video_overview.video.assembler import VideoAssemblyError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"
        with patch("video_overview.core.subprocess.run", return_value=mock_result):
            with pytest.raises(VideoAssemblyError, match="ffmpeg failed"):
                _create_static_frame(cache_dir=tmp_path, width=1920, height=1080)


# ---------------------------------------------------------------------------
# Tests: VideoAssembler receives video config from OverviewConfig
# ---------------------------------------------------------------------------


class TestVideoAssemblerConfigThreading:
    """Verify core.py threads video config fields to VideoAssembler constructor."""

    def test_default_config_passes_defaults_to_assembler(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video")
        create_overview(config=config)

        all_mocks["assembler_cls"].assert_called_once_with(
            width=1920,
            height=1080,
            fps=30,
            crossfade_seconds=0.5,
            ken_burns_zoom_percent=5.0,
        )

    def test_custom_config_passes_custom_to_assembler(self, tmp_source, all_mocks):
        from video_overview.core import create_overview

        config = _make_config(
            tmp_source,
            format="video",
            video_width=1280,
            video_height=720,
            video_fps=24,
            crossfade_seconds=1.0,
            ken_burns_zoom_percent=10.0,
        )
        create_overview(config=config)

        all_mocks["assembler_cls"].assert_called_once_with(
            width=1280,
            height=720,
            fps=24,
            crossfade_seconds=1.0,
            ken_burns_zoom_percent=10.0,
        )


# ---------------------------------------------------------------------------
# Tests: no_cache threading
# ---------------------------------------------------------------------------


class TestNoCacheThreading:
    """Verify no_cache is passed through to both audio and visual generators."""

    def test_no_cache_true_passed_to_audio_generator_video_mode(
        self, tmp_source, all_mocks
    ):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", no_cache=True)
        create_overview(config=config)

        audio_call = all_mocks["audio_inst"].generate.call_args
        assert audio_call.kwargs.get("no_cache") is True

    def test_no_cache_true_passed_to_visual_generator_video_mode(
        self, tmp_source, all_mocks
    ):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", no_cache=True)
        create_overview(config=config)

        visual_call = all_mocks["visual_inst"].generate.call_args
        assert visual_call.kwargs.get("no_cache") is True

    def test_no_cache_false_passed_to_audio_generator_video_mode(
        self, tmp_source, all_mocks
    ):
        from video_overview.core import create_overview

        config = _make_config(tmp_source, format="video", no_cache=False)
        create_overview(config=config)

        audio_call = all_mocks["audio_inst"].generate.call_args
        assert audio_call.kwargs.get("no_cache") is False

    def test_no_cache_true_passed_to_audio_generator_audio_mode(
        self, tmp_source, all_mocks
    ):
        from video_overview.core import create_overview

        out = tmp_source / "output.mp3"
        config = _make_config(tmp_source, format="audio", output=out, no_cache=True)
        create_overview(config=config)

        audio_call = all_mocks["audio_inst"].generate.call_args
        assert audio_call.kwargs.get("no_cache") is True

    def test_no_cache_true_passed_to_audio_in_skip_visuals_mode(
        self, tmp_source, all_mocks
    ):
        from video_overview.core import create_overview

        config = _make_config(
            tmp_source, format="video", skip_visuals=True, no_cache=True
        )
        with patch("video_overview.core._create_static_frame") as mock_frame:
            mock_frame.return_value = Path("/tmp/cache/static_frame.png")
            create_overview(config=config)

        audio_call = all_mocks["audio_inst"].generate.call_args
        assert audio_call.kwargs.get("no_cache") is True
