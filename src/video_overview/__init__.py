"""Video Overview SDK - Generate video overviews with AI."""

from importlib.metadata import PackageNotFoundError, version

from video_overview import audio, content, script, video, visuals

try:
    __version__ = version("video-overview")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "__version__",
    "audio",
    "content",
    "script",
    "video",
    "visuals",
]
