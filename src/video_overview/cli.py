"""CLI entry point for video-overview."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import click

from video_overview import __version__
from video_overview.audio.generator import AudioGenerationError
from video_overview.config import OverviewConfig
from video_overview.core import create_overview
from video_overview.script.generator import ScriptGenerationError
from video_overview.video.assembler import VideoAssemblyError
from video_overview.visuals.generator import VisualGenerationError

#: Exception types that represent user-facing operational errors.
_HANDLED_ERRORS = (
    ValueError,
    RuntimeError,
    OSError,
    AudioGenerationError,
    ScriptGenerationError,
    VideoAssemblyError,
    VisualGenerationError,
)

#: Default cache directory name (relative to source_dir).
_CACHE_DIR_NAME = ".video_overview_cache"


def _validate_output_parent(
    ctx: click.Context,
    param: click.Parameter,
    value: str,
) -> str:
    """Ensure the parent directory of ``--output`` exists."""
    parent = Path(value).parent
    if not parent.exists():
        raise click.BadParameter(
            f"output parent directory does not exist: {parent}",
            ctx=ctx,
            param=param,
        )
    return value


class _DefaultCommandGroup(click.Group):
    """A click Group that falls back to a default command.

    When the first argument is not a registered subcommand, the group
    delegates to the ``generate`` command so that the existing
    ``video-overview SOURCE_DIR --topic ...`` interface keeps working.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Route to default command when the first arg is not a subcommand."""
        # --version is handled by the group itself (click.version_option).
        if args and args[0] == "--version":
            return super().parse_args(ctx, args)
        # Dynamically check registered commands so that explicit
        # `video-overview generate ...` works alongside `cache` etc.
        registered = set(self.list_commands(ctx))
        # For everything else, if the first arg is not a known subcommand,
        # prepend 'generate' so Click routes to the default command.
        # This includes --help (so `video-overview --help` shows generate help)
        # and positional args (so `video-overview ./src ...` still works).
        if args and args[0] not in registered:
            args = ["generate"] + list(args)
        # If no args at all, also route to generate (which will show its help
        # due to missing required arguments).
        if not args:
            args = ["generate"]
        return super().parse_args(ctx, args)


@click.group(cls=_DefaultCommandGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="video-overview")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Generate video or audio overviews from source content."""
    # If invoked without any subcommand, show help.
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command(epilog="Additional commands: cache (see 'video-overview cache --help')")
@click.argument(
    "source_dir",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--topic",
    "-t",
    required=True,
    help="Topic of the overview to generate.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    callback=_validate_output_parent,
    help="Output file path for the generated overview.",
)
@click.option(
    "--include",
    "-i",
    multiple=True,
    default=["*"],
    help=("Glob patterns for files to include. Can be specified multiple times."),
)
@click.option(
    "--exclude",
    "-e",
    multiple=True,
    default=[],
    help=("Glob patterns for files to exclude. Can be specified multiple times."),
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["conversation", "narration"], case_sensitive=False),
    default="conversation",
    help="Script mode: conversation or narration.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["video", "audio"], case_sensitive=False),
    default="video",
    help="Output format: video or audio.",
)
@click.option(
    "--host-voice",
    default="Aoede",
    help="Voice for the host speaker (conversation mode).",
)
@click.option(
    "--expert-voice",
    default="Charon",
    help="Voice for the expert speaker (conversation mode).",
)
@click.option(
    "--narrator-voice",
    default="Kore",
    help="Voice for the narrator (narration mode).",
)
@click.option(
    "--llm",
    type=click.Choice(["claude", "codex"], case_sensitive=False),
    default="claude",
    help="LLM backend to use for script generation.",
)
@click.option(
    "--max-duration",
    type=click.IntRange(min=1),
    default=10,
    help="Maximum duration in minutes.",
)
@click.option(
    "--skip-visuals",
    is_flag=True,
    default=False,
    help="Skip Gemini image generation; use a static dark frame (video format only).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Skip reading cached assets and always regenerate.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging (INFO level).",
)
def generate(
    source_dir: str,
    topic: str,
    output: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    mode: str,
    output_format: str,
    host_voice: str,
    expert_voice: str,
    narrator_voice: str,
    llm: str,
    max_duration: int,
    skip_visuals: bool,
    no_cache: bool,
    verbose: bool,
) -> None:
    """Generate a video or audio overview from source content."""
    # Configure logging based on --verbose flag.
    # Default is WARNING; --verbose sets to INFO.
    # Library consumers can configure logging independently.
    log_level = logging.INFO if verbose else logging.WARNING
    root = logging.getLogger()
    if not root.handlers:
        # No handlers yet — add a default stderr handler.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(name)s: %(levelname)s: %(message)s"))
        root.addHandler(handler)
    # Always set the effective level so --verbose takes effect
    # even when the embedding process pre-configured logging.
    root.setLevel(log_level)

    try:
        config = OverviewConfig(
            source_dir=source_dir,
            output=output,
            topic=topic,
            include=list(include),
            exclude=list(exclude),
            mode=mode,
            format=output_format,
            host_voice=host_voice,
            expert_voice=expert_voice,
            narrator_voice=narrator_voice,
            llm_backend=llm,
            max_duration_minutes=max_duration,
            skip_visuals=skip_visuals,
            no_cache=no_cache,
        )
        result = create_overview(config=config)
        click.echo(
            f"Overview created: {result.output_path} "
            f"({result.duration_seconds}s, "
            f"{result.segments_count} segments)"
        )
    except KeyboardInterrupt:
        click.echo("Interrupted")
        sys.exit(130)
    except _HANDLED_ERRORS as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# Cache management subcommands
# ------------------------------------------------------------------


def _resolve_cache_dir(
    cache_dir: str | None,
    source_dir: str | None,
) -> Path | None:
    """Resolve the cache directory from CLI options.

    Returns ``None`` when neither option is provided.
    """
    if cache_dir:
        return Path(cache_dir)
    if source_dir:
        return Path(source_dir) / _CACHE_DIR_NAME
    return None


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


@main.group()
def cache() -> None:
    """Manage cached audio and visual assets."""


@cache.command("list")
@click.option(
    "--cache-dir",
    default=None,
    help="Path to the cache directory.",
)
@click.option(
    "--source-dir",
    default=None,
    help="Source directory (cache is at <source-dir>/.video_overview_cache).",
)
def cache_list(cache_dir: str | None, source_dir: str | None) -> None:
    """Show cached audio and visual assets with sizes."""
    resolved = _resolve_cache_dir(cache_dir, source_dir)
    if resolved is None:
        raise click.UsageError(
            "Provide --cache-dir or --source-dir to locate the cache."
        )

    if not resolved.is_dir():
        click.echo("No cached assets found (cache directory does not exist).")
        return

    # Collect audio cache files (audio_*.wav)
    audio_files = sorted(resolved.glob("audio_*.wav"))
    audio_total = sum(f.stat().st_size for f in audio_files)

    # Collect visual cache files (visuals/*.png)
    visuals_dir = resolved / "visuals"
    visual_files = sorted(visuals_dir.glob("*.png")) if visuals_dir.is_dir() else []
    visual_total = sum(f.stat().st_size for f in visual_files)

    # Collect other intermediate files
    other_patterns = ["output.wav", "filelist.txt", "static_frame_*.png"]
    other_files: list[Path] = []
    for pattern in other_patterns:
        other_files.extend(resolved.glob(pattern))
    other_total = sum(f.stat().st_size for f in other_files)

    total_files = len(audio_files) + len(visual_files) + len(other_files)
    total_size = audio_total + visual_total + other_total

    if total_files == 0:
        click.echo("No cached assets found.")
        return

    click.echo(f"Cache directory: {resolved}")
    click.echo()
    click.echo(
        f"  Audio assets:  {len(audio_files)} files  ({_format_size(audio_total)})"
    )
    click.echo(
        f"  Visual assets: {len(visual_files)} files  ({_format_size(visual_total)})"
    )
    if other_files:
        click.echo(
            f"  Other files:   {len(other_files)} files  ({_format_size(other_total)})"
        )
    click.echo()
    click.echo(f"  Total: {total_files} files ({_format_size(total_size)})")


@cache.command("clear")
@click.option(
    "--cache-dir",
    default=None,
    help="Path to the cache directory.",
)
@click.option(
    "--source-dir",
    default=None,
    help="Source directory (cache is at <source-dir>/.video_overview_cache).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def cache_clear(
    cache_dir: str | None,
    source_dir: str | None,
    yes: bool,
) -> None:
    """Delete all cached audio and visual assets."""
    resolved = _resolve_cache_dir(cache_dir, source_dir)
    if resolved is None:
        raise click.UsageError(
            "Provide --cache-dir or --source-dir to locate the cache."
        )

    if not resolved.is_dir():
        click.echo("No cache directory found. Nothing to clear.")
        return

    # Safety check: ensure the directory looks like a cache directory
    # (contains expected cache artifacts or is named correctly).
    # This check is always enforced, even with --yes, to prevent
    # accidental deletion of non-cache directories in scripts.
    _looks_like_cache = (
        resolved.name == _CACHE_DIR_NAME
        or any(resolved.glob("audio_*.wav"))
        or (resolved / "visuals").is_dir()
    )
    if not _looks_like_cache:
        # Empty directories are safe to clear regardless.
        if any(resolved.iterdir()):
            click.echo(
                f"Error: {resolved} does not look like a "
                f"video-overview cache directory. Aborting."
            )
            sys.exit(1)

    if not yes:
        if not click.confirm(f"Delete all cached files in {resolved}?"):
            click.echo("Aborted.")
            return

    # Remove all contents of the cache directory
    removed_count = 0
    removed_bytes = 0

    for item in list(resolved.iterdir()):
        if item.is_dir():
            dir_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            dir_count = sum(1 for f in item.rglob("*") if f.is_file())
            shutil.rmtree(item)
            removed_count += dir_count
            removed_bytes += dir_size
        else:
            removed_bytes += item.stat().st_size
            removed_count += 1
            item.unlink()

    click.echo(
        f"Cleared {removed_count} cached files "
        f"({_format_size(removed_bytes)}) from {resolved}"
    )


if __name__ == "__main__":
    main()
