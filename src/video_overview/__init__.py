"""Video Overview SDK - Generate video overviews with AI."""

from importlib.metadata import version

from video_overview import audio, content, script, video, visuals

__version__ = version("video-overview")

__all__ = [
    "__version__",
    "audio",
    "content",
    "script",
    "video",
    "visuals",
]
