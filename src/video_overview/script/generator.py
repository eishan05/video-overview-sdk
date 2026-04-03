"""Script generator using LLM backends (Claude CLI or Codex CLI)."""

from __future__ import annotations

import json
import logging
import subprocess

from pydantic import ValidationError

from video_overview.config import Script
from video_overview.duration import compute_duration_budget

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 300


class ScriptGenerationError(Exception):
    """Raised when script generation fails."""


class ScriptGenerator:
    """Generates educational scripts via LLM subprocess calls."""

    def generate(
        self,
        content_bundle: dict,
        topic: str,
        mode: str = "conversation",
        llm_backend: str = "claude",
        max_segments: int = 20,
        max_duration_minutes: float | None = None,
    ) -> Script:
        """Generate a script from a content bundle using an LLM backend.

        Args:
            content_bundle: Dictionary with ``directory_structure``, ``files``,
                ``total_files``, and ``total_chars`` keys as produced by
                :class:`ContentReader`.
            topic: The topic or title of the overview.
            mode: Either ``"conversation"`` (Host + Expert dialogue) or
                ``"narration"`` (single Narrator).
            llm_backend: Either ``"claude"`` or ``"codex"``.
            max_segments: Maximum number of segments in the script.
            max_duration_minutes: Optional target duration in minutes.
                When provided, a word and segment budget is derived and
                included in the LLM prompt so the model produces a
                right-sized script upfront.

        Returns:
            A validated :class:`Script` model.

        Raises:
            ScriptGenerationError: On subprocess failure, timeout, JSON
                parsing failure, or validation failure.
        """
        _VALID_MODES = ("conversation", "narration")
        if mode not in _VALID_MODES:
            raise ScriptGenerationError(
                f"Invalid mode {mode!r}; must be one of {_VALID_MODES}"
            )

        # Derive a duration budget and tighten max_segments if needed
        budget = compute_duration_budget(
            max_duration_minutes, max_segments_cap=max_segments
        )
        if budget is not None:
            max_segments = budget["max_segments"]

        prompt = self._build_prompt(content_bundle, topic, mode, max_segments, budget)
        logger.info("Invoking LLM backend %r for topic %r", llm_backend, topic)
        raw_output = self._call_llm(prompt, llm_backend)
        logger.debug("LLM response received (%d chars)", len(raw_output))
        script = self._parse_response(raw_output)

        # Enforce max_segments server-side (LLM may exceed the limit)
        if len(script.segments) > max_segments:
            raise ScriptGenerationError(
                f"LLM returned {len(script.segments)} segments, "
                f"exceeding the maximum of {max_segments}"
            )

        # Validate speaker names match the selected mode
        if mode == "conversation":
            allowed_speakers = {"Host", "Expert"}
        else:
            allowed_speakers = {"Narrator"}

        for seg in script.segments:
            if seg.speaker not in allowed_speakers:
                raise ScriptGenerationError(
                    f"Invalid speaker {seg.speaker!r} for "
                    f"{mode!r} mode; allowed: {allowed_speakers}"
                )

        return script

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        content_bundle: dict,
        topic: str,
        mode: str,
        max_segments: int,
        budget: dict | None = None,
    ) -> str:
        """Assemble the full prompt text."""
        if mode == "conversation":
            mode_instructions = (
                "Create an engaging, educational, NotebookLM-style "
                "conversational script between two speakers:\n"
                "  - Host: the curious interviewer who asks "
                "insightful questions\n"
                "  - Expert: the knowledgeable specialist who "
                "explains concepts clearly\n"
                "The dialogue should feel natural and engaging, "
                "as if two colleagues are having an enthusiastic "
                "discussion about the topic."
            )
        elif mode == "narration":
            mode_instructions = (
                "Create an informative narration script with a "
                "single speaker:\n"
                "  - Narrator: a clear, authoritative voice that "
                "guides the audience through the topic\n"
                "The narration should be educational, well-paced, "
                "and engaging."
            )
        else:
            # Should not reach here due to validation in generate()
            raise ScriptGenerationError(f"Invalid mode: {mode!r}")

        content_text = self._format_content_bundle(content_bundle)

        # Build optional duration budget section
        budget_section = ""
        if budget is not None:
            budget_section = (
                f"\nDURATION BUDGET:\n"
                f"- Target duration: {budget['target_minutes']} minute(s)\n"
                f"- Maximum total word count across all segments: "
                f"{budget['max_words']} words\n"
                f"- Aim for {budget['max_segments']} segments or fewer\n"
                f"- Keep each segment concise to stay within the word budget\n"
            )

        return (
            f"You are a world-class educational scriptwriter.\n\n"
            f"TOPIC: {topic}\n\n"
            f"MODE INSTRUCTIONS:\n{mode_instructions}\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return ONLY a JSON object (no markdown fences, no extra text) "
            f"matching this schema:\n"
            f"{{\n"
            f'  "title": "<engaging title for the overview>",\n'
            f'  "segments": [\n'
            f"    {{\n"
            f'      "speaker": "<speaker name>",\n'
            f'      "text": "<what the speaker says>",\n'
            f'      "visual_prompt": "<description of an informative diagram/'
            f'illustration to accompany this segment>"\n'
            f"    }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"CONSTRAINTS:\n"
            f"- Maximum {max_segments} segments\n"
            f"- Each segment must have speaker, text, and visual_prompt fields\n"
            f"- visual_prompt should describe an informative diagram or "
            f"illustration that helps explain the content visually\n"
            f"- Cover the most important aspects of the codebase\n"
            f"{budget_section}\n"
            f"CONTENT BUNDLE:\n{content_text}"
        )

    def _format_content_bundle(self, content_bundle: dict) -> str:
        """Format the content bundle into a string for the prompt."""
        parts: list[str] = []

        tree = content_bundle.get("directory_structure", "")
        if tree:
            parts.append(f"Directory structure:\n{tree}")

        files = content_bundle.get("files", [])
        if files:
            parts.append("Files:")
            for f in files:
                path = f.get("path", "unknown")
                lang = f.get("language", "text")
                content = f.get("content", "")
                parts.append(f"\n--- {path} ({lang}) ---\n{content}")
        else:
            parts.append("Files: (none)")

        total_files = content_bundle.get("total_files", 0)
        total_chars = content_bundle.get("total_chars", 0)
        parts.append(f"\nTotal files: {total_files}, Total characters: {total_chars}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, llm_backend: str) -> str:
        """Invoke the LLM CLI and return raw stdout."""
        if llm_backend == "claude":
            cmd = ["claude", "-p", prompt, "--output-format", "json"]
        elif llm_backend == "codex":
            cmd = ["codex", "exec", prompt]
        else:
            raise ScriptGenerationError(f"Unsupported LLM backend: {llm_backend}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ScriptGenerationError(
                f"LLM subprocess timed out after {_TIMEOUT_SECONDS} seconds"
            ) from exc
        except OSError as exc:
            raise ScriptGenerationError(
                f"Failed to execute LLM command: {exc}"
            ) from exc

        if result.returncode != 0:
            raise ScriptGenerationError(
                f"LLM subprocess exited with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )

        return result.stdout

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Script:
        """Parse raw LLM output into a validated Script model."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ScriptGenerationError(
                f"Failed to parse LLM response as JSON: {exc}"
            ) from exc

        # Handle Claude --output-format json wrapper
        if isinstance(data, dict) and data.get("type") == "result" and "result" in data:
            # Check for Claude error responses
            if data.get("is_error"):
                raise ScriptGenerationError(
                    f"Claude returned an error: {data.get('result', '')}"
                )
            inner = data["result"]
            if isinstance(inner, str):
                try:
                    data = json.loads(inner)
                except json.JSONDecodeError as exc:
                    raise ScriptGenerationError(
                        f"Failed to parse inner Claude JSON result: {exc}"
                    ) from exc

        try:
            script = Script.model_validate(data)
        except ValidationError as exc:
            raise ScriptGenerationError(
                f"Failed to validate script against schema: {exc}"
            ) from exc

        return script
