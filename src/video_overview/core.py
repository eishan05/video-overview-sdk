"""Core orchestrator for generating video/audio overviews."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from video_overview.audio.generator import AudioGenerator
from video_overview.config import OverviewConfig, OverviewResult, Script
from video_overview.content.reader import ContentReader
from video_overview.duration import truncate_segments
from video_overview.script.generator import ScriptGenerator
from video_overview.video.assembler import VideoAssembler, VideoAssemblyError
from video_overview.visuals.generator import VisualGenerator


def _progress(message: str) -> None:
    """Print a progress message to stderr."""
    print(message, file=sys.stderr)


def _create_static_frame(
    cache_dir: Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Create a single static dark frame image using ffmpeg.

    The frame is cached in *cache_dir* so repeated calls with the same
    dimensions reuse the existing file.

    Args:
        cache_dir: Directory to store the generated frame.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path to the generated PNG file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = cache_dir / f"static_frame_{width}x{height}.png"

    if frame_path.exists():
        return frame_path

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=#1a1a2e:s={width}x{height}:d=1",
        "-frames:v",
        "1",
        str(frame_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as exc:
        raise VideoAssemblyError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoAssemblyError("ffmpeg timed out creating static frame") from exc
    except OSError as exc:
        raise VideoAssemblyError(f"ffmpeg execution error: {exc}") from exc

    if result.returncode != 0:
        raise VideoAssemblyError(
            f"ffmpeg failed creating static frame "
            f"(exit code {result.returncode}): {result.stderr}"
        )
    return frame_path


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    Uses ``asyncio.run()`` when no event loop is active.  When called
    from inside a running loop (e.g. Jupyter notebooks, async test
    runners), falls back to running in a new thread with its own loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # Already inside a running loop -- run in a separate thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


async def _run_audio_and_visuals(
    audio_gen: AudioGenerator,
    visual_gen: VisualGenerator,
    script: Script,
    config: OverviewConfig,
) -> tuple[tuple[Path, list[float]], list[Path]]:
    """Run audio and visual generation concurrently."""
    loop = asyncio.get_event_loop()

    # Run audio generation in a thread (it's synchronous)
    audio_task = loop.run_in_executor(
        None,
        lambda: audio_gen.generate(
            script=script,
            host_voice=config.host_voice,
            expert_voice=config.expert_voice,
            narrator_voice=config.narrator_voice,
            cache_dir=config.cache_dir,
            max_tokens_per_batch=config.max_tokens_per_batch,
            max_segments_per_batch=config.max_segments_per_batch,
            max_attempts=config.audio_max_attempts,
        ),
    )

    # Visual generation is already async
    visual_task = visual_gen.generate(
        script=script,
        cache_dir=config.cache_dir,
    )

    audio_result, visual_result = await asyncio.gather(audio_task, visual_task)
    return audio_result, visual_result


def create_overview(config: OverviewConfig | None = None, **kwargs) -> OverviewResult:
    """Create a video or audio overview from source content.

    Accepts either an ``OverviewConfig`` object or keyword arguments
    that will be forwarded to ``OverviewConfig()``.  If both are
    provided, the ``config`` object takes precedence and kwargs are
    ignored.

    Pipeline:
        1. Validate config and API keys
        2. Create cache directory
        3. Read content from source directory
        4. Generate script from content
        5. Generate audio and visuals concurrently (video mode)
           or audio only (audio mode)
        6. Assemble final output (video mode only)
        7. Return OverviewResult

    Args:
        config: An ``OverviewConfig`` instance.  Mutually preferred
            over ``**kwargs``.
        **kwargs: Keyword arguments forwarded to ``OverviewConfig()``
            when ``config`` is not provided.

    Returns:
        An ``OverviewResult`` with output path, duration, and
        segment count.

    Raises:
        ValueError: If required API keys are missing.
        Various sub-component errors propagate unchanged.
    """
    # ---- 1. Build / validate config ----
    if config is None:
        config = OverviewConfig(**kwargs)

    # Validate API key
    api_key = config.gemini_api_key
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Set the GEMINI_API_KEY or "
            "GOOGLE_API_KEY environment variable."
        )

    # ---- 2. Create cache directory ----
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Read content ----
    _progress("Reading content from source directory...")
    reader = ContentReader()
    content_bundle = reader.read(
        source_dir=config.source_dir,
        include=config.include,
        exclude=config.exclude,
    )

    # ---- 4. Generate script ----
    _progress("Generating script...")
    script_gen = ScriptGenerator()
    script = script_gen.generate(
        content_bundle=content_bundle,
        topic=config.topic,
        mode=config.mode,
        llm_backend=config.llm_backend,
    )

    # ---- 4b. Truncate script to honour max_duration_minutes ----
    truncated = truncate_segments(script.segments, config.max_duration_minutes)
    if len(truncated) < len(script.segments):
        _progress(
            f"Truncated script from {len(script.segments)} to "
            f"{len(truncated)} segments to fit within "
            f"{config.max_duration_minutes} minute(s)."
        )
        script = Script(title=script.title, segments=truncated)

    # ---- 5. Generate audio (+ visuals for video mode) ----
    if config.format == "video":
        if config.skip_visuals:
            _progress("Generating audio (skipping visual generation)...")
            audio_gen = AudioGenerator(api_key=api_key)
            audio_result = audio_gen.generate(
                script=script,
                host_voice=config.host_voice,
                expert_voice=config.expert_voice,
                narrator_voice=config.narrator_voice,
                cache_dir=config.cache_dir,
                max_tokens_per_batch=config.max_tokens_per_batch,
                max_segments_per_batch=config.max_segments_per_batch,
                max_attempts=config.audio_max_attempts,
            )
            audio_path, segment_durations = audio_result

            _progress("Creating static frame...")
            static_frame = _create_static_frame(
                cache_dir=config.cache_dir,
                width=config.video_width,
                height=config.video_height,
            )
            image_paths = [static_frame] * len(script.segments)
        else:
            _progress("Generating audio and visuals concurrently...")
            audio_gen = AudioGenerator(api_key=api_key)
            visual_gen = VisualGenerator(api_key=api_key)

            audio_result, image_paths = _run_async(
                _run_audio_and_visuals(audio_gen, visual_gen, script, config)
            )
            audio_path, segment_durations = audio_result

        # ---- 6. Assemble video ----
        _progress("Assembling video...")
        assembler = VideoAssembler(
            width=config.video_width,
            height=config.video_height,
            fps=config.video_fps,
            crossfade_seconds=config.crossfade_seconds,
            ken_burns_zoom_percent=config.ken_burns_zoom_percent,
        )
        output_path = assembler.assemble(
            audio_path=audio_path,
            image_paths=image_paths,
            segment_durations=segment_durations,
            output_path=config.output,
            format=config.format,
        )
    else:
        # Audio-only mode
        _progress("Generating audio...")
        audio_gen = AudioGenerator(api_key=api_key)
        audio_result = audio_gen.generate(
            script=script,
            host_voice=config.host_voice,
            expert_voice=config.expert_voice,
            narrator_voice=config.narrator_voice,
            cache_dir=config.cache_dir,
            max_tokens_per_batch=config.max_tokens_per_batch,
            max_segments_per_batch=config.max_segments_per_batch,
            max_attempts=config.audio_max_attempts,
        )
        audio_path, segment_durations = audio_result

        # Convert/copy to the user-specified output path
        _progress("Writing audio output...")
        if config.output.suffix.lower() == ".wav":
            # Simple copy -- no ffmpeg needed
            if audio_path.resolve() != config.output.resolve():
                shutil.copy2(audio_path, config.output)
            output_path = config.output
        else:
            # Needs format conversion (e.g. WAV -> MP3)
            assembler = VideoAssembler(
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                crossfade_seconds=config.crossfade_seconds,
                ken_burns_zoom_percent=config.ken_burns_zoom_percent,
            )
            output_path = assembler.assemble(
                audio_path=audio_path,
                image_paths=[],
                segment_durations=segment_durations,
                output_path=config.output,
                format="audio",
            )

    # ---- 7. Build and return result ----
    _progress("Done.")
    return OverviewResult(
        output_path=output_path,
        duration_seconds=sum(segment_durations),
        segments_count=len(script.segments),
    )
