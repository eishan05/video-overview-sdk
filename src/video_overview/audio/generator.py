"""Audio generator using Gemini TTS API."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

from video_overview.config import Script
from video_overview.duration import estimate_segment_duration

logger = logging.getLogger(__name__)

_MODEL = "gemini-2.5-flash-preview-tts"
_BATCH_SIZE = 13  # Target ~10-15 segments per batch; TODO: also budget by token count
_MAX_RETRIES = 3
_BASE_DELAY = 1  # seconds
_CONVERSATION_SPEAKERS = {"Host", "Expert"}
_NARRATION_SPEAKERS = {"Narrator"}


class AudioGenerationError(Exception):
    """Raised when audio generation fails."""


class AudioGenerator:
    """Generates audio from a Script model using Gemini TTS."""

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise AudioGenerationError(
                "API key is required. Set GEMINI_API_KEY environment variable."
            )
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        script: Script,
        host_voice: str,
        expert_voice: str,
        narrator_voice: str,
        cache_dir: Path,
    ) -> tuple[Path, list[float]]:
        """Generate audio from a script.

        Args:
            script: The Script model containing segments.
            host_voice: Prebuilt voice ID for the Host speaker.
            expert_voice: Prebuilt voice ID for the Expert speaker.
            narrator_voice: Prebuilt voice ID for the Narrator speaker.
            cache_dir: Directory for intermediate and output files.

        Returns:
            A tuple of (audio_path, segment_durations) where audio_path
            is the final WAV file and segment_durations is a list of
            estimated durations (seconds) for each segment.

        Raises:
            AudioGenerationError: On API errors, file errors, or
                ffmpeg failures.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not script.segments:
            raise AudioGenerationError("Script must contain at least one segment.")

        client = genai.Client(api_key=self._api_key)

        # Determine mode from speaker names (not count).
        # Conversation uses Host/Expert; narration uses Narrator.
        unique_speakers = {seg.speaker for seg in script.segments}
        is_conversation = bool(unique_speakers & _CONVERSATION_SPEAKERS)
        is_narration = bool(unique_speakers & _NARRATION_SPEAKERS)

        if is_conversation and is_narration:
            raise AudioGenerationError(
                f"Script mixes conversation speakers "
                f"({_CONVERSATION_SPEAKERS}) with narration speakers "
                f"({_NARRATION_SPEAKERS}). Use one mode only."
            )

        unknown = unique_speakers - _CONVERSATION_SPEAKERS - _NARRATION_SPEAKERS
        if unknown:
            raise AudioGenerationError(
                f"Unknown speaker(s): {unknown}. "
                f"Allowed: {_CONVERSATION_SPEAKERS | _NARRATION_SPEAKERS}"
            )

        is_multi_speaker = is_conversation

        # Build voice map
        voice_map: dict[str, str] = {
            "Host": host_voice,
            "Expert": expert_voice,
            "Narrator": narrator_voice,
        }

        # Chunk segments into batches
        batches = self._chunk_segments(script.segments)

        # Generate audio for each batch
        chunk_paths: list[Path] = []
        for i, batch in enumerate(batches):
            prompt = self._build_prompt(batch)
            config = self._build_config(batch, is_multi_speaker, voice_map)
            audio_bytes = self._call_api_with_retry(client, prompt, config)
            chunk_path = cache_dir / f"chunk_{i:03d}.wav"
            chunk_path.write_bytes(audio_bytes)
            chunk_paths.append(chunk_path)

        # Concatenate or copy final output
        output_path = cache_dir / "output.wav"
        if len(chunk_paths) == 1:
            shutil.copy2(chunk_paths[0], output_path)
        else:
            self._concat_with_ffmpeg(chunk_paths, output_path, cache_dir)

        # Estimate segment durations
        durations = self._estimate_durations(script.segments)

        return output_path, durations

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_segments(segments: list, batch_size: int = _BATCH_SIZE) -> list[list]:
        """Split segments into batches of approximately batch_size."""
        batches = []
        for i in range(0, len(segments), batch_size):
            batches.append(segments[i : i + batch_size])
        return batches

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(segments: list) -> str:
        """Build TTS prompt text with speaker labels."""
        lines = []
        for seg in segments:
            lines.append(f"{seg.speaker}: {seg.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(
        segments: list,
        is_multi_speaker: bool,
        voice_map: dict[str, str],
    ) -> types.GenerateContentConfig:
        """Build the GenerateContentConfig for the TTS call.

        Uses MultiSpeakerVoiceConfig only when the batch actually
        contains 2+ distinct speakers. Falls back to single VoiceConfig
        when a batch has only one speaker (even in conversation mode).
        """
        batch_speakers = {seg.speaker for seg in segments}

        if is_multi_speaker and len(batch_speakers) > 1:
            speaker_voice_configs = [
                types.SpeakerVoiceConfig(
                    speaker=speaker,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_map.get(speaker, "Kore"),
                        ),
                    ),
                )
                for speaker in sorted(batch_speakers)
            ]
            speech_config = types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs,
                )
            )
        else:
            # Single speaker in this batch
            speaker = next(iter(batch_speakers))
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_map.get(speaker, "Kore"),
                    ),
                )
            )

        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )

    # ------------------------------------------------------------------
    # API call with retry
    # ------------------------------------------------------------------

    def _call_api_with_retry(
        self,
        client: genai.Client,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> bytes:
        """Call the Gemini TTS API with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=_MODEL,
                    contents=prompt,
                    config=config,
                )
                return self._extract_audio(response)
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2**attempt)
                    logger.warning(
                        "TTS API retry attempt %d/%d after error: %s (delay=%ds)",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        raise AudioGenerationError(
            f"API call failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_audio(response) -> bytes:
        """Extract audio bytes from the API response.

        Handles both audio/wav and audio/pcm mime types.
        Iterates through candidates and parts to find the audio payload.
        """
        if not response.candidates:
            raise AudioGenerationError("API response contains no candidates")

        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                has_audio = (
                    part.inline_data is not None
                    and part.inline_data.mime_type
                    and "audio" in part.inline_data.mime_type.lower()
                )
                if has_audio:
                    data = part.inline_data.data
                    mime_type = part.inline_data.mime_type

                    if "pcm" in mime_type.lower():
                        sample_rate = 24000
                        if "rate=" in mime_type:
                            try:
                                rate_str = mime_type.split("rate=")[1].split(";")[0]
                                sample_rate = int(rate_str)
                            except (IndexError, ValueError):
                                pass
                        data = _pcm_to_wav(data, sample_rate=sample_rate)

                    return data

        raise AudioGenerationError(
            "API response contains no audio data in any candidate/part"
        )

    # ------------------------------------------------------------------
    # FFmpeg concatenation
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_with_ffmpeg(
        chunk_paths: list[Path],
        output_path: Path,
        cache_dir: Path,
    ) -> None:
        """Concatenate WAV chunks using ffmpeg."""
        filelist = cache_dir / "filelist.txt"
        # Escape single quotes for ffmpeg concat format:
        # replace ' with '\'' (end quote, escaped quote, start quote)
        filelist.write_text(
            "\n".join(
                "file '{}'".format(str(p.resolve()).replace("'", "'\\''"))
                for p in chunk_paths
            )
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(filelist),
            "-c",
            "copy",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise AudioGenerationError(
                "ffmpeg is not installed or not on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioGenerationError("ffmpeg concatenation timed out") from exc
        except OSError as exc:
            raise AudioGenerationError(f"ffmpeg execution error: {exc}") from exc

        if result.returncode != 0:
            raise AudioGenerationError(f"ffmpeg concatenation failed: {result.stderr}")

    # ------------------------------------------------------------------
    # Duration estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_durations(segments: list) -> list[float]:
        """Estimate duration for each segment based on text length.

        Delegates to the shared ``estimate_segment_duration`` helper
        (speaking rate ~12.5 characters per second, 0.5 s minimum).
        """
        return [estimate_segment_duration(seg.text) for seg in segments]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """Add WAV header to raw PCM data."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
