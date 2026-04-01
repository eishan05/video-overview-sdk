"""Configuration models for the Video Overview SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class ScriptSegment(BaseModel):
    """A single segment of a generated script."""

    speaker: str
    text: str
    visual_prompt: str


class Script(BaseModel):
    """A complete script composed of ordered segments."""

    title: str
    segments: list[ScriptSegment]


class OverviewResult(BaseModel):
    """Result metadata from generating a video/audio overview."""

    output_path: Path
    duration_seconds: float
    segments_count: int


class OverviewConfig(BaseModel):
    """Main configuration for generating a video overview."""

    source_dir: Path
    output: Path
    topic: str
    include: list[str] = ["*"]
    exclude: list[str] = []
    mode: Literal["conversation", "narration"] = "conversation"
    format: Literal["video", "audio"] = "video"
    host_voice: str = "Aoede"
    expert_voice: str = "Charon"
    narrator_voice: str = "Kore"
    max_duration_minutes: int = 10
    llm_backend: Literal["claude", "codex"] = "claude"
    cache_dir: Optional[Path] = None

    @field_validator("source_dir")
    @classmethod
    def _validate_source_dir(cls, v: Path) -> Path:
        v = Path(v)
        if not v.exists():
            raise ValueError(f"source_dir does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"source_dir is not a directory: {v}")
        return v

    @field_validator("output")
    @classmethod
    def _validate_output(cls, v: Path) -> Path:
        v = Path(v)
        if not v.parent.exists():
            raise ValueError(
                f"output parent directory does not exist: {v.parent}"
            )
        return v

    @model_validator(mode="after")
    def _set_cache_dir_default(self) -> OverviewConfig:
        if self.cache_dir is None:
            self.cache_dir = self.source_dir / ".video_overview_cache"
        return self

    @property
    def gemini_api_key(self) -> str | None:
        """Read the Gemini/Google API key from environment variables.

        Checks ``GEMINI_API_KEY`` first, falling back to ``GOOGLE_API_KEY``.
        Returns ``None`` if neither is set.
        """
        return os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
