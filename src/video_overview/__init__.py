"""Video Overview SDK - Generate video overviews with AI."""

from importlib.metadata import PackageNotFoundError, version

from video_overview import audio, content, script, video, visuals
from video_overview.config import (
    OverviewConfig,
    OverviewResult,
    Script,
    ScriptSegment,
)

try:
    __version__ = version("video-overview")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "__version__",
    "OverviewConfig",
    "OverviewResult",
    "Script",
    "ScriptSegment",
    "audio",
    "content",
    "script",
    "video",
    "visuals",
]
