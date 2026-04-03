"""Configuration models for the Video Overview SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)


class ScriptSegment(BaseModel):
    """A single segment of a generated script."""

    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    visual_prompt: str = Field(min_length=1)


class Script(BaseModel):
    """A complete script composed of ordered segments."""

    title: str = Field(min_length=1)
    segments: list[ScriptSegment] = Field(min_length=1)


class OverviewResult(BaseModel):
    """Result metadata from generating a video/audio overview."""

    output_path: Path
    duration_seconds: NonNegativeFloat
    segments_count: NonNegativeInt


class OverviewConfig(BaseModel):
    """Main configuration for generating a video overview."""

    model_config = ConfigDict(extra="forbid")

    source_dir: Path
    output: Path
    topic: str
    include: list[str] = Field(default=["*"])
    exclude: list[str] = Field(default=[])
    mode: Literal["conversation", "narration"] = "conversation"
    format: Literal["video", "audio"] = "video"
    host_voice: str = "Aoede"
    expert_voice: str = "Charon"
    narrator_voice: str = "Kore"
    max_duration_minutes: PositiveInt = 10
    llm_backend: Literal["claude", "codex"] = "claude"
    cache_dir: Optional[Path] = None
    max_tokens_per_batch: PositiveInt = 8000
    max_segments_per_batch: PositiveInt = 13
    audio_max_attempts: PositiveInt = 3
    video_width: PositiveInt = 1920
    video_height: PositiveInt = 1080
    video_fps: PositiveInt = 30
    crossfade_seconds: NonNegativeFloat = 0.5
    ken_burns_zoom_percent: NonNegativeFloat = 5.0

    @field_validator("video_width", "video_height")
    @classmethod
    def _validate_even_dimensions(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(
                f"Video dimensions must be even for libx264 encoding, got {v}"
            )
        return v

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
            raise ValueError(f"output parent directory does not exist: {v.parent}")
        if v.exists() and v.is_dir():
            raise ValueError(
                f"output must be a file path, not an existing directory: {v}"
            )
        return v

    @field_validator("cache_dir")
    @classmethod
    def _validate_cache_dir(cls, v: Path | None) -> Path | None:
        if v is not None:
            v = Path(v)
            if v.exists() and not v.is_dir():
                raise ValueError(f"cache_dir exists but is not a directory: {v}")
        return v

    @staticmethod
    def _check_cache_dir_is_not_file(path: Path) -> None:
        """Raise if *path* exists and is not a directory."""
        if path.exists() and not path.is_dir():
            raise ValueError(f"cache_dir exists but is not a directory: {path}")

    @model_validator(mode="after")
    def _validate_output_format_match(self) -> OverviewConfig:
        """Validate that output file extension matches the chosen format."""
        suffix = self.output.suffix.lower()
        if self.format == "video" and suffix not in (".mp4",):
            raise ValueError(f"Video format requires .mp4 output, got '{suffix}'")
        if self.format == "audio" and suffix not in (".mp3", ".wav"):
            raise ValueError(
                f"Audio format requires .mp3 or .wav output, got '{suffix}'"
            )
        return self

    @model_validator(mode="after")
    def _set_cache_dir_default(self) -> OverviewConfig:
        if self.cache_dir is None:
            default = self.source_dir / ".video_overview_cache"
            self._check_cache_dir_is_not_file(default)
            self.cache_dir = default
        return self

    @property
    def gemini_api_key(self) -> str | None:
        """Read the Gemini/Google API key from environment variables.

        Checks ``GEMINI_API_KEY`` first, falling back to ``GOOGLE_API_KEY``.
        Returns ``None`` if neither is set.
        """
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
