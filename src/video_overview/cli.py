"""CLI entry point for video-overview."""

from __future__ import annotations

import sys

import click

from video_overview import __version__
from video_overview.config import OverviewConfig
from video_overview.core import create_overview


@click.command()
@click.version_option(version=__version__, prog_name="video-overview")
@click.argument("source_dir")
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
    help="Output file path for the generated overview.",
)
@click.option(
    "--include",
    "-i",
    multiple=True,
    default=["*"],
    help="Glob patterns for files to include. Can be specified multiple times.",
)
@click.option(
    "--exclude",
    "-e",
    multiple=True,
    default=[],
    help="Glob patterns for files to exclude. Can be specified multiple times.",
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
    help="Voice for the host speaker.",
)
@click.option(
    "--expert-voice",
    default="Charon",
    help="Voice for the expert speaker.",
)
@click.option(
    "--llm",
    type=click.Choice(["claude", "codex"], case_sensitive=False),
    default="claude",
    help="LLM backend to use for script generation.",
)
@click.option(
    "--max-duration",
    type=int,
    default=10,
    help="Maximum duration in minutes.",
)
def main(
    source_dir: str,
    topic: str,
    output: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    mode: str,
    output_format: str,
    host_voice: str,
    expert_voice: str,
    llm: str,
    max_duration: int,
) -> None:
    """Generate a video or audio overview from source content."""
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
            llm_backend=llm,
            max_duration_minutes=max_duration,
        )
        result = create_overview(config=config)
        click.echo(
            f"Overview created: {result.output_path} "
            f"({result.duration_seconds}s, {result.segments_count} segments)"
        )
    except KeyboardInterrupt:
        click.echo("Interrupted")
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
