"""Core orchestrator for generating video/audio overviews."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from video_overview.audio.generator import AudioGenerator
from video_overview.config import OverviewConfig, OverviewResult, Script
from video_overview.content.reader import ContentReader
from video_overview.script.generator import ScriptGenerator
from video_overview.video.assembler import VideoAssembler
from video_overview.visuals.generator import VisualGenerator


def _progress(message: str) -> None:
    """Print a progress message to stderr."""
    print(message, file=sys.stderr)


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
        ),
    )

    # Visual generation is already async
    visual_task = visual_gen.generate(
        script=script,
        cache_dir=config.cache_dir,
    )

    audio_result, visual_result = await asyncio.gather(
        audio_task, visual_task
    )
    return audio_result, visual_result


def create_overview(
    config: OverviewConfig | None = None, **kwargs
) -> OverviewResult:
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

    # ---- 5. Generate audio (+ visuals for video mode) ----
    if config.format == "video":
        _progress("Generating audio and visuals concurrently...")
        audio_gen = AudioGenerator(api_key=api_key)
        visual_gen = VisualGenerator(api_key=api_key)

        audio_result, image_paths = asyncio.run(
            _run_audio_and_visuals(audio_gen, visual_gen, script, config)
        )
        audio_path, segment_durations = audio_result

        # ---- 6. Assemble video ----
        _progress("Assembling video...")
        assembler = VideoAssembler()
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
        )
        audio_path, segment_durations = audio_result
        output_path = audio_path

    # ---- 7. Build and return result ----
    _progress("Done.")
    return OverviewResult(
        output_path=output_path,
        duration_seconds=sum(segment_durations),
        segments_count=len(script.segments),
    )
