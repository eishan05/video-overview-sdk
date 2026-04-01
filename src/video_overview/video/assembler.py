"""Video assembler using ffmpeg for audio/video output."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from video_overview.config import ScriptSegment

_FPS = 30
_WIDTH = 1920
_HEIGHT = 1080
_CROSSFADE_DURATION = 0.5
_MAX_ZOOM = 1.05


class VideoAssemblyError(Exception):
    """Raised when video assembly fails."""


class VideoAssembler:
    """Assembles audio and images into video or audio output using ffmpeg."""

    def __init__(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise VideoAssemblyError(
                "ffmpeg is not installed or not on PATH. "
                "Install ffmpeg to use VideoAssembler."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        audio_path: Path,
        image_paths: list[Path],
        segment_durations: list[float],
        output_path: Path,
        format: str = "video",
    ) -> Path:
        """Assemble audio and images into the specified output format.

        Args:
            audio_path: Path to the input WAV audio file.
            image_paths: List of image file paths (one per segment).
            segment_durations: Duration in seconds for each segment.
            output_path: Path for the output file.
            format: Output format - "audio" or "video".

        Returns:
            The output file path.

        Raises:
            VideoAssemblyError: On validation errors or ffmpeg failures.
        """
        if format == "audio":
            return self._assemble_audio(audio_path, output_path)
        elif format == "video":
            return self._assemble_video(
                audio_path, image_paths, segment_durations, output_path
            )
        else:
            raise VideoAssemblyError(
                f"Unsupported format: {format!r}. Use 'audio' or 'video'."
            )

    def _estimate_segment_durations(
        self,
        segments: list[ScriptSegment],
        total_audio_duration: float,
    ) -> list[float]:
        """Estimate each segment's duration proportional to its text length.

        Args:
            segments: List of script segments.
            total_audio_duration: Total audio duration in seconds.

        Returns:
            List of estimated durations that sum to total_audio_duration.

        Raises:
            VideoAssemblyError: If segments list is empty.
        """
        if not segments:
            raise VideoAssemblyError(
                "Cannot estimate durations for an empty segments list."
            )

        total_chars = sum(len(seg.text) for seg in segments)
        if total_chars == 0:
            # Equal distribution if all texts are empty
            n = len(segments)
            return [total_audio_duration / n] * n

        return [
            (len(seg.text) / total_chars) * total_audio_duration for seg in segments
        ]

    # ------------------------------------------------------------------
    # Audio assembly
    # ------------------------------------------------------------------

    def _assemble_audio(self, audio_path: Path, output_path: Path) -> Path:
        """Convert WAV to MP3 or copy if output is WAV."""
        if output_path.suffix.lower() == ".wav":
            shutil.copy2(audio_path, output_path)
            return output_path

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            str(output_path),
        ]
        self._run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Video assembly
    # ------------------------------------------------------------------

    def _assemble_video(
        self,
        audio_path: Path,
        image_paths: list[Path],
        segment_durations: list[float],
        output_path: Path,
    ) -> Path:
        """Create an MP4 video from images with Ken Burns effect and audio."""
        if not image_paths:
            raise VideoAssemblyError(
                "At least one image is required for video assembly."
            )

        if len(image_paths) != len(segment_durations):
            raise VideoAssemblyError(
                f"Count mismatch: {len(image_paths)} images but "
                f"{len(segment_durations)} durations provided."
            )

        # Validate durations
        _MIN_DURATION = 1.0 / _FPS  # at least one frame
        for i, d in enumerate(segment_durations):
            if d <= 0:
                raise VideoAssemblyError(
                    f"Segment {i} has non-positive duration ({d}s). "
                    "All durations must be positive."
                )
            if d < _MIN_DURATION:
                raise VideoAssemblyError(
                    f"Segment {i} duration ({d}s) is too short to produce "
                    f"even one frame at {_FPS}fps."
                )

        n = len(image_paths)
        if n > 1:
            for i, d in enumerate(segment_durations):
                if d <= _CROSSFADE_DURATION:
                    raise VideoAssemblyError(
                        f"Segment {i} duration ({d}s) must be greater than "
                        f"the crossfade duration ({_CROSSFADE_DURATION}s)."
                    )

        # Compute effective durations for each image input.
        # Each xfade overlap shortens the timeline by _CROSSFADE_DURATION.
        # To keep the total video timeline equal to sum(segment_durations),
        # we extend each interior clip by the crossfade overlap it participates in.
        effective_durations = list(segment_durations)
        if n > 1:
            # First clip is extended by one overlap (right side)
            effective_durations[0] += _CROSSFADE_DURATION
            # Interior clips are extended by two overlaps (left + right)
            for i in range(1, n - 1):
                effective_durations[i] += _CROSSFADE_DURATION
            # Last clip needs no extension (its left side overlap is covered
            # by the preceding clip's extension)

        filter_complex = self._build_filter_complex(
            image_paths, segment_durations, effective_durations
        )

        # Build the ffmpeg command
        cmd: list[str] = ["ffmpeg", "-y"]

        # Add audio input first
        cmd.extend(["-i", str(audio_path)])

        # Add each image as a looped input with its effective duration
        for i, (img_path, eff_dur) in enumerate(zip(image_paths, effective_durations)):
            cmd.extend(["-loop", "1", "-t", str(eff_dur), "-i", str(img_path)])

        # Add filter complex
        cmd.extend(["-filter_complex", filter_complex])

        # Output settings
        cmd.extend(
            [
                "-map",
                "[vout]",
                "-map",
                "0:a",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                "-r",
                str(_FPS),
                str(output_path),
            ]
        )

        self._run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Filter complex construction
    # ------------------------------------------------------------------

    def _build_filter_complex(
        self,
        image_paths: list[Path],
        segment_durations: list[float],
        effective_durations: list[float],
    ) -> str:
        """Build the ffmpeg filter_complex string.

        For each image:
        - Scale to 1920x1080 with padding to maintain aspect ratio
        - Apply zoompan for Ken Burns effect (5% zoom over duration)

        Between images:
        - Apply xfade transitions with 0.5s crossfade

        Args:
            image_paths: Image file paths.
            segment_durations: Original segment durations (for offset math).
            effective_durations: Extended durations accounting for xfade overlap
                (used for zoompan frame counts).
        """
        filters: list[str] = []
        n = len(image_paths)

        # Build zoompan filter for each image.
        # Use math.ceil for frame counts to ensure each clip is at least as
        # long as the requested duration. Derive xfade offsets from these
        # frame-aligned durations so the timeline stays consistent.
        # Input indices: 0 = audio, 1..n = images
        frame_aligned_durations: list[float] = []
        for i in range(n):
            input_idx = i + 1  # offset by 1 because audio is input 0
            frames = max(1, math.ceil(_FPS * effective_durations[i]))
            frame_aligned_durations.append(frames / _FPS)
            zoom_increment = (_MAX_ZOOM - 1.0) / frames

            filters.append(
                f"[{input_idx}:v]"
                f"scale={_WIDTH}:{_HEIGHT}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={_WIDTH}:{_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"zoompan=z='min(zoom+{zoom_increment:.6f},{_MAX_ZOOM})':"
                f"d={frames}:s={_WIDTH}x{_HEIGHT}:fps={_FPS}"
                f"[v{i}]"
            )

        if n == 1:
            # Single image: just map it to output
            filters.append("[v0]copy[vout]")
        else:
            # Chain xfade transitions between consecutive images.
            # Use frame-aligned durations for offset calculation so the
            # timeline matches the actual zoompan output lengths.
            offset = frame_aligned_durations[0] - _CROSSFADE_DURATION
            prev_label = "v0"

            for i in range(1, n):
                if i == n - 1:
                    out_label = "vout"
                else:
                    out_label = f"xf{i - 1}"

                filters.append(
                    f"[{prev_label}][v{i}]"
                    f"xfade=transition=fade:"
                    f"duration={_CROSSFADE_DURATION}:"
                    f"offset={offset:.3f}"
                    f"[{out_label}]"
                )

                prev_label = out_label
                if i < n - 1:
                    offset += frame_aligned_durations[i] - _CROSSFADE_DURATION

        return ";".join(filters)

    # ------------------------------------------------------------------
    # FFmpeg execution
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        """Run an ffmpeg command and raise on failure."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as exc:
            raise VideoAssemblyError("ffmpeg is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise VideoAssemblyError("ffmpeg command timed out") from exc
        except OSError as exc:
            raise VideoAssemblyError(f"ffmpeg execution error: {exc}") from exc

        if result.returncode != 0:
            raise VideoAssemblyError(
                f"ffmpeg failed (exit code {result.returncode}): {result.stderr}"
            )
