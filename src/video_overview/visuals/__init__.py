"""Visual asset generation subpackage."""

from __future__ import annotations

__all__ = ["VisualGenerationError", "VisualGenerator"]


def __getattr__(name: str):
    if name in __all__:
        from video_overview.visuals.generator import (
            VisualGenerationError,
            VisualGenerator,
        )

        globals()["VisualGenerationError"] = VisualGenerationError
        globals()["VisualGenerator"] = VisualGenerator
        return globals()[name]

    raise AttributeError(f"module 'video_overview.visuals' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
