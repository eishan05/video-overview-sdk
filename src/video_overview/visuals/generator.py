"""Visual generator using Gemini image generation API."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
from pathlib import Path

from google import genai
from google.genai import types

from video_overview.config import Script

_MODEL = "gemini-2.0-flash-exp"
_MAX_CONCURRENT = 3


class VisualGenerationError(Exception):
    """Raised when visual generation fails."""


class VisualGenerator:
    """Generates visual assets from a Script model using Gemini image generation."""

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise VisualGenerationError(
                "API key is required. Set GEMINI_API_KEY environment variable."
            )
        self._api_key = api_key

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

        # Build tasks: deduplicate by visual_prompt to avoid redundant API calls.
        # We track a dict of prompt -> future so duplicate prompts share a result.
        prompt_tasks: dict[str, asyncio.Task[Path]] = {}
        segment_tasks: list[asyncio.Task[Path]] = []

        for seg in script.segments:
            prompt = seg.visual_prompt
            if prompt not in prompt_tasks:
                task = asyncio.create_task(
                    self._generate_single(
                        client=client,
                        visual_prompt=prompt,
                        segment_text=seg.text,
                        cache_dir=cache_dir,
                        semaphore=semaphore,
                    )
                )
                prompt_tasks[prompt] = task
            segment_tasks.append(prompt_tasks[prompt])

        # Await all unique tasks
        unique_tasks = list(prompt_tasks.values())
        await asyncio.gather(*unique_tasks, return_exceptions=True)

        # Collect results ordered by segment
        results: list[Path] = []
        for task in segment_tasks:
            result = task.result()
            results.append(result)

        return results

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
    ) -> Path:
        """Generate a single image for a visual prompt.

        Checks cache first, then calls the API, with fallback to ffmpeg.
        """
        visuals_dir = cache_dir / "visuals"
        cache_hash = hashlib.md5(visual_prompt.encode()).hexdigest()
        cached_path = visuals_dir / f"{cache_hash}.png"

        # Check cache
        if cached_path.exists():
            return cached_path

        async with semaphore:
            try:
                image_bytes = await self._call_api(
                    client, visual_prompt
                )
                if image_bytes is not None:
                    cached_path.write_bytes(image_bytes)
                    return cached_path
            except Exception:
                pass

            # Fallback: create text slide with ffmpeg
            self._create_fallback_image(
                segment_text, cached_path
            )
            return cached_path

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
        prompt = (
            f"Create a 16:9 informative diagram or illustration: "
            f"{visual_prompt}"
        )

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=_MODEL,
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
        """Create a fallback image using ffmpeg with text on dark background."""
        # Escape special characters for ffmpeg drawtext filter
        escaped_text = text
        for char in ("\\", "'", '"', ":", "%"):
            escaped_text = escaped_text.replace(char, f"\\{char}")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "color=c=#1a1a2e:s=1920x1080:d=1",
            "-vf",
            (
                f"drawtext=text='{escaped_text}'"
                f":fontsize=36"
                f":fontcolor=white"
                f":x=(w-text_w)/2"
                f":y=(h-text_h)/2"
            ),
            "-frames:v", "1",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # If ffmpeg also fails, create a minimal placeholder
            output_path.write_bytes(b"")
            return

        if result.returncode != 0:
            # Write empty file as last resort
            output_path.write_bytes(b"")
