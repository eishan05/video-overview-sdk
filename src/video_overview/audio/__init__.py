"""Audio processing subpackage."""

from __future__ import annotations

__all__ = ["AudioGenerationError", "AudioGenerator"]


def __getattr__(name: str):
    if name in __all__:
        from video_overview.audio.generator import (
            AudioGenerationError,
            AudioGenerator,
        )

        globals()["AudioGenerationError"] = AudioGenerationError
        globals()["AudioGenerator"] = AudioGenerator
        return globals()[name]

    raise AttributeError(f"module 'video_overview.audio' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
