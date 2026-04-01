"""CLI entry point for video-overview."""

import click

from video_overview import __version__


@click.group()
@click.version_option(version=__version__, prog_name="video-overview")
def main():
    """Video Overview SDK - Generate video overviews with AI."""


if __name__ == "__main__":
    main()
