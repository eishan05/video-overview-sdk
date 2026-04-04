"""Video Overview SDK - Generate video overviews with AI."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version

# Eagerly import items that do NOT depend on google-genai.
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
    "create_overview",
    "script",
    "video",
    "visuals",
]

# ---------------------------------------------------------------------------
# Lazy attribute loading
# ---------------------------------------------------------------------------
# Submodules and ``create_overview`` are loaded on first access so that
# ``import video_overview`` does not transitively pull in google-genai
# (required only by audio and visuals generators).

_SUBMODULES = frozenset({"audio", "content", "script", "video", "visuals"})


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"video_overview.{name}")
        globals()[name] = module
        return module

    if name == "create_overview":
        from video_overview.core import create_overview

        globals()["create_overview"] = create_overview
        return create_overview

    raise AttributeError(f"module 'video_overview' has no attribute {name!r}")
