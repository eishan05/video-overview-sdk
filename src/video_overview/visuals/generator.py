"""Visual generator using Gemini image generation API."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
from pathlib import Path

from google import genai
from google.genai import types

from video_overview.config import Script

_MODEL = "gemini-2.0-flash-exp"
_MAX_CONCURRENT = 3

logger = logging.getLogger(__name__)


class VisualGenerationError(Exception):
    """Raised when visual generation fails."""


class VisualGenerator:
    """Generates visual assets from a Script model using Gemini image generation."""

    def __init__(self, api_key: str | None, *, model: str = _MODEL) -> None:
        if not api_key:
            raise VisualGenerationError(
                "API key is required. Set GEMINI_API_KEY environment variable."
            )
        self._api_key = api_key
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        script: Script,
        cache_dir: Path,
    ) -> list[Path]:
        """Generate visual images for each segment in the script.

        Args:
            script: The Script model containing segments with visual_prompts.
            cache_dir: Directory for caching generated images.

        Returns:
            A list of image file paths ordered by segment.

        Raises:
            VisualGenerationError: On unrecoverable errors.
        """
        if not script.segments:
            return []

        cache_dir = Path(cache_dir)
        visuals_dir = cache_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        client = genai.Client(api_key=self._api_key)
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

        # Per-prompt locks prevent duplicate concurrent API calls for
        # the same visual_prompt, while still generating per-segment
        # fallback images (with segment-specific text) when the API fails.
        prompt_locks: dict[str, asyncio.Lock] = {}
        # Track prompts that have already failed so subsequent segments
        # with the same prompt skip the API call entirely.
        failed_prompts: set[str] = set()

        for seg in script.segments:
            if seg.visual_prompt not in prompt_locks:
                prompt_locks[seg.visual_prompt] = asyncio.Lock()

        tasks = [
            asyncio.create_task(
                self._generate_single(
                    client=client,
                    visual_prompt=seg.visual_prompt,
                    segment_text=seg.text,
                    cache_dir=cache_dir,
                    semaphore=semaphore,
                    prompt_lock=prompt_locks[seg.visual_prompt],
                    failed_prompts=failed_prompts,
                )
            )
            for seg in script.segments
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unwrap results; raise on unexpected failures.
        paths: list[Path] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                raise VisualGenerationError(
                    f"Visual generation failed for segment {i}: {result}"
                ) from result
            paths.append(result)

        return paths

    # ------------------------------------------------------------------
    # Single image generation
    # ------------------------------------------------------------------

    async def _generate_single(
        self,
        client: genai.Client,
        visual_prompt: str,
        segment_text: str,
        cache_dir: Path,
        semaphore: asyncio.Semaphore,
        prompt_lock: asyncio.Lock,
        failed_prompts: set[str],
    ) -> Path:
        """Generate a single image for a visual prompt.

        Checks cache first, then calls the API, with fallback to ffmpeg.
        The ``prompt_lock`` serialises tasks that share the same
        ``visual_prompt`` so only one makes the API call.
        The ``failed_prompts`` set is shared across all tasks for the
        current ``generate()`` invocation; once a prompt fails, subsequent
        tasks skip the API call and go straight to fallback.

        Successful API images are cached by visual_prompt hash.
        Fallback images use a separate key that includes segment_text
        so each segment gets its own text slide even when prompts match.
        """
        visuals_dir = cache_dir / "visuals"
        prompt_hash = hashlib.md5(visual_prompt.encode()).hexdigest()
        cached_path = visuals_dir / f"{prompt_hash}.png"

        # Check cache before acquiring any locks (fast path)
        if cached_path.exists():
            logger.info("Cache hit for visual prompt: %s", visual_prompt)
            return cached_path

        # Serialise per-prompt so duplicate prompts don't race
        async with prompt_lock:
            # Re-check after acquiring lock (another task may have
            # written the cache while we waited).
            if cached_path.exists():
                logger.info(
                    "Cache hit (post-lock) for visual prompt: %s", visual_prompt
                )
                return cached_path

            logger.info("Cache miss for visual prompt: %s", visual_prompt)

            # If this prompt already failed, skip API and go to fallback
            if visual_prompt not in failed_prompts:
                async with semaphore:
                    try:
                        image_bytes = await self._call_api(client, visual_prompt)
                        if image_bytes is not None:
                            cached_path.write_bytes(image_bytes)
                            return cached_path
                        # No image in response -- fall through to fallback
                        logger.warning(
                            "No image in API response for prompt: %s",
                            visual_prompt,
                        )
                    except Exception as exc:
                        logger.warning(
                            "API call failed for prompt %r: %s",
                            visual_prompt,
                            exc,
                        )
                    # Mark this prompt as failed so subsequent tasks skip it
                    failed_prompts.add(visual_prompt)

            # Fallback: use segment_text in the cache key so each
            # segment gets its own text slide even when prompts match.
            logger.info("Using fallback image for prompt: %s", visual_prompt)
            fallback_key = hashlib.md5(
                f"{visual_prompt}::{segment_text}".encode()
            ).hexdigest()
            fallback_path = visuals_dir / f"{fallback_key}.png"
            if not fallback_path.exists():
                self._create_fallback_image(segment_text, fallback_path)
            return fallback_path

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        client: genai.Client,
        visual_prompt: str,
    ) -> bytes | None:
        """Call Gemini image generation API.

        Returns image bytes or None if no image in response.
        """
        prompt = f"Create a 16:9 informative diagram or illustration: {visual_prompt}"

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
            ),
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=config,
        )

        # Extract image from response
        return self._extract_image(response)

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(response) -> bytes | None:
        """Extract image bytes from the API response.

        Returns None if no image part is found.
        """
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        return None

    # ------------------------------------------------------------------
    # Fallback image creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_fallback_image(text: str, output_path: Path) -> None:
        """Create a fallback image using ffmpeg with text on dark background.

        Raises:
            VisualGenerationError: If ffmpeg is unavailable or fails.
        """
        # Escape special characters for ffmpeg drawtext filter
        escaped_text = text
        for char in ("\\", "'", '"', ":", "%"):
            escaped_text = escaped_text.replace(char, f"\\{char}")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=#1a1a2e:s=1920x1080:d=1",
            "-vf",
            (
                f"drawtext=text='{escaped_text}'"
                f":fontsize=36"
                f":fontcolor=white"
                f":x=(w-text_w)/2"
                f":y=(h-text_h)/2"
            ),
            "-frames:v",
            "1",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise VisualGenerationError(
                "ffmpeg is not installed or not on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise VisualGenerationError("ffmpeg fallback timed out") from exc
        except OSError as exc:
            raise VisualGenerationError(f"ffmpeg execution error: {exc}") from exc

        if result.returncode != 0:
            raise VisualGenerationError(f"ffmpeg fallback failed: {result.stderr}")
