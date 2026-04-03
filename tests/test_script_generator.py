"""Tests for ScriptGenerator."""

from __future__ import annotations

import json
import subprocess

import pytest

from video_overview.config import Script, ScriptSegment
from video_overview.script import ScriptGenerationError, ScriptGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> ScriptGenerator:
    return ScriptGenerator()


@pytest.fixture()
def sample_content_bundle() -> dict:
    return {
        "directory_structure": "myproject/\n  src/\n    main.py\n  README.md\n",
        "files": [
            {
                "path": "README.md",
                "content": "# My Project\nA sample project.",
                "language": "markdown",
            },
            {
                "path": "src/main.py",
                "content": 'print("hello")',
                "language": "python",
            },
        ],
        "total_files": 2,
        "total_chars": 50,
    }


@pytest.fixture()
def valid_conversation_response() -> dict:
    return {
        "title": "Understanding My Project",
        "segments": [
            {
                "speaker": "Host",
                "text": "Welcome! Today we're looking at My Project.",
                "visual_prompt": "Title card showing 'My Project' logo",
            },
            {
                "speaker": "Expert",
                "text": "This is a simple Python project.",
                "visual_prompt": "Diagram of the project structure",
            },
        ],
    }


@pytest.fixture()
def valid_narration_response() -> dict:
    return {
        "title": "My Project Overview",
        "segments": [
            {
                "speaker": "Narrator",
                "text": "Let's explore My Project.",
                "visual_prompt": "Overview diagram of project architecture",
            },
        ],
    }


def _make_subprocess_result(response_dict: dict) -> subprocess.CompletedProcess:
    """Create a mock CompletedProcess with JSON stdout."""
    return subprocess.CompletedProcess(
        args=["claude", "-p", "...", "--output-format", "json"],
        returncode=0,
        stdout=json.dumps(response_dict),
        stderr="",
    )


def _make_claude_json_wrapper(response_dict: dict) -> subprocess.CompletedProcess:
    """Claude --output-format json wraps result in a JSON object."""
    wrapper = {
        "type": "result",
        "result": json.dumps(response_dict),
    }
    return subprocess.CompletedProcess(
        args=["claude", "-p", "...", "--output-format", "json"],
        returncode=0,
        stdout=json.dumps(wrapper),
        stderr="",
    )


# ---------------------------------------------------------------------------
# Conversation mode tests
# ---------------------------------------------------------------------------


class TestConversationMode:
    def test_prompt_includes_content_bundle(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Conversation prompt must include the content bundle."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        # args[0] is the command list, [2] is the prompt
        prompt = mock_run.call_args[0][0][2]
        assert "myproject/" in prompt
        assert "README.md" in prompt
        assert "main.py" in prompt

    def test_prompt_includes_two_speakers(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Conversation prompt must instruct for Host and Expert speakers."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        prompt = mock_run.call_args[0][0][2]
        assert "Host" in prompt
        assert "Expert" in prompt

    def test_returns_script_model(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """generate() must return a Script model."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        result = generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        assert isinstance(result, Script)
        assert result.title == "Understanding My Project"
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Host"
        assert result.segments[1].speaker == "Expert"


# ---------------------------------------------------------------------------
# Narration mode tests
# ---------------------------------------------------------------------------


class TestNarrationMode:
    def test_prompt_uses_narrator(
        self, generator, sample_content_bundle, valid_narration_response, mocker
    ):
        """Narration prompt must instruct for a single Narrator speaker."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_narration_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="narration",
            llm_backend="claude",
        )
        prompt = mock_run.call_args[0][0][2]
        assert "Narrator" in prompt

    def test_returns_narrator_segments(
        self, generator, sample_content_bundle, valid_narration_response, mocker
    ):
        """Narration mode result should have Narrator segments."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_narration_response),
        )
        result = generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="narration",
            llm_backend="claude",
        )
        assert isinstance(result, Script)
        assert all(s.speaker == "Narrator" for s in result.segments)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_valid_json_parsing(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Valid JSON should be parsed into a Script model."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        result = generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        assert isinstance(result, Script)
        for seg in result.segments:
            assert isinstance(seg, ScriptSegment)
            assert seg.speaker
            assert seg.text
            assert seg.visual_prompt

    def test_invalid_json_raises_error(self, generator, sample_content_bundle, mocker):
        """Invalid JSON response should raise ScriptGenerationError."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout="this is not json {{{",
                stderr="",
            ),
        )
        with pytest.raises(ScriptGenerationError, match="parse"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )

    def test_response_missing_required_fields(
        self, generator, sample_content_bundle, mocker
    ):
        """Response missing required fields should raise ScriptGenerationError."""
        bad_response = {"title": "Hello"}  # missing 'segments'
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(bad_response),
        )
        with pytest.raises(ScriptGenerationError, match="validat"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )

    def test_segment_missing_visual_prompt(
        self, generator, sample_content_bundle, mocker
    ):
        """Segment missing visual_prompt should raise ScriptGenerationError."""
        bad_response = {
            "title": "Test",
            "segments": [
                {"speaker": "Host", "text": "Hello"},  # missing visual_prompt
            ],
        }
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(bad_response),
        )
        with pytest.raises(ScriptGenerationError, match="validat"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )

    def test_claude_json_wrapper_parsed(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Claude --output-format json wraps result; we should handle it."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_claude_json_wrapper(valid_conversation_response),
        )
        result = generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        assert isinstance(result, Script)
        assert result.title == "Understanding My Project"


# ---------------------------------------------------------------------------
# Subprocess error handling
# ---------------------------------------------------------------------------


class TestSubprocessErrors:
    def test_timeout_raises_error(self, generator, sample_content_bundle, mocker):
        """Subprocess timeout should raise ScriptGenerationError."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300),
        )
        with pytest.raises(ScriptGenerationError, match="timed out"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )

    def test_nonzero_exit_code_raises_error(
        self, generator, sample_content_bundle, mocker
    ):
        """Non-zero exit code should raise ScriptGenerationError."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["claude"],
                returncode=1,
                stdout="",
                stderr="Some CLI error",
            ),
        )
        with pytest.raises(ScriptGenerationError, match="exit code"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )

    def test_subprocess_general_error(self, generator, sample_content_bundle, mocker):
        """General subprocess errors should raise ScriptGenerationError."""
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            side_effect=OSError("command not found"),
        )
        with pytest.raises(ScriptGenerationError, match="Failed to execute"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )


# ---------------------------------------------------------------------------
# Backend command construction
# ---------------------------------------------------------------------------


class TestBackendCommands:
    def test_claude_backend_command(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Claude backend should use 'claude -p ... --output-format json'."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert cmd[1] == "-p"
        # cmd[2] is the prompt
        assert cmd[3] == "--output-format"
        assert cmd[4] == "json"

    def test_codex_backend_command(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Codex backend should use 'codex exec ...'."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="codex",
        )
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        # cmd[2] is the prompt


# ---------------------------------------------------------------------------
# Max segments and prompt parameters
# ---------------------------------------------------------------------------


class TestPromptParameters:
    def test_max_segments_in_prompt(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """max_segments should appear in the prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_segments=15,
        )
        prompt = mock_run.call_args[0][0][2]
        assert "15" in prompt

    def test_default_max_segments(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Default max_segments of 20 should appear in prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        prompt = mock_run.call_args[0][0][2]
        assert "20" in prompt

    def test_topic_in_prompt(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """The topic should be included in the prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="Advanced Python Patterns",
            mode="conversation",
            llm_backend="claude",
        )
        prompt = mock_run.call_args[0][0][2]
        assert "Advanced Python Patterns" in prompt


# ---------------------------------------------------------------------------
# Empty content bundle
# ---------------------------------------------------------------------------


class TestEmptyContentBundle:
    def test_empty_content_bundle(self, generator, valid_conversation_response, mocker):
        """Empty content bundle should still generate (may produce minimal output)."""
        empty_bundle = {
            "directory_structure": "empty/\n",
            "files": [],
            "total_files": 0,
            "total_chars": 0,
        }
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        result = generator.generate(
            content_bundle=empty_bundle,
            topic="Empty Project",
            mode="conversation",
            llm_backend="claude",
        )
        assert isinstance(result, Script)
        # Verify prompt still includes the directory structure
        prompt = mock_run.call_args[0][0][2]
        assert "empty/" in prompt


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------


class TestTimeoutConfig:
    def test_timeout_is_300_seconds(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """subprocess.run should be called with timeout=300."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
        )
        assert mock_run.call_args[1]["timeout"] == 300


# ---------------------------------------------------------------------------
# Invalid mode handling (Codex review finding #2)
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_invalid_mode_raises_error(self, generator, sample_content_bundle, mocker):
        """Invalid mode should raise ScriptGenerationError."""
        with pytest.raises(ScriptGenerationError, match="Invalid mode"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="converstion",  # typo
                llm_backend="claude",
            )

    def test_empty_mode_raises_error(self, generator, sample_content_bundle, mocker):
        """Empty mode string should raise ScriptGenerationError."""
        with pytest.raises(ScriptGenerationError, match="Invalid mode"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="",
                llm_backend="claude",
            )


# ---------------------------------------------------------------------------
# Max segments enforcement (Codex review finding #1)
# ---------------------------------------------------------------------------


class TestMaxSegmentsEnforcement:
    def test_exceeding_max_segments_raises_error(
        self, generator, sample_content_bundle, mocker
    ):
        """Response with more segments than max should raise error."""
        over_limit = {
            "title": "Too Many Segments",
            "segments": [
                {
                    "speaker": "Host",
                    "text": f"Segment {i}",
                    "visual_prompt": f"Visual {i}",
                }
                if i % 2 == 0
                else {
                    "speaker": "Expert",
                    "text": f"Segment {i}",
                    "visual_prompt": f"Visual {i}",
                }
                for i in range(5)
            ],
        }
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(over_limit),
        )
        with pytest.raises(ScriptGenerationError, match="exceeding the maximum"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
                max_segments=2,
            )


# ---------------------------------------------------------------------------
# Speaker validation (Codex review finding #3)
# ---------------------------------------------------------------------------


class TestSpeakerValidation:
    def test_wrong_speaker_in_conversation_mode(
        self, generator, sample_content_bundle, mocker
    ):
        """Conversation mode should reject Narrator speaker."""
        bad_response = {
            "title": "Test",
            "segments": [
                {
                    "speaker": "Narrator",
                    "text": "Hello",
                    "visual_prompt": "Image",
                },
            ],
        }
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(bad_response),
        )
        with pytest.raises(ScriptGenerationError, match="Invalid speaker"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )

    def test_wrong_speaker_in_narration_mode(
        self, generator, sample_content_bundle, mocker
    ):
        """Narration mode should reject Host/Expert speakers."""
        bad_response = {
            "title": "Test",
            "segments": [
                {
                    "speaker": "Host",
                    "text": "Hello",
                    "visual_prompt": "Image",
                },
            ],
        }
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(bad_response),
        )
        with pytest.raises(ScriptGenerationError, match="Invalid speaker"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="narration",
                llm_backend="claude",
            )


# ---------------------------------------------------------------------------
# Claude error wrapper handling (Codex review finding #4)
# ---------------------------------------------------------------------------


class TestClaudeErrorWrapper:
    def test_claude_is_error_flag(self, generator, sample_content_bundle, mocker):
        """Claude wrapper with is_error=True should raise error."""
        error_wrapper = {
            "type": "result",
            "is_error": True,
            "result": "Rate limit exceeded",
        }
        mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(error_wrapper),
                stderr="",
            ),
        )
        with pytest.raises(ScriptGenerationError, match="Claude returned an error"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
            )


# ---------------------------------------------------------------------------
# Duration budget in prompt
# ---------------------------------------------------------------------------


class TestDurationBudgetInPrompt:
    """Tests that max_duration_minutes feeds a word/segment budget into the prompt."""

    def test_budget_word_count_in_prompt(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """max_duration_minutes should put a word budget in the prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_duration_minutes=5,
        )
        prompt = mock_run.call_args[0][0][2]
        # 5 minutes * 125 wpm = 625 words
        assert "625" in prompt
        assert "word" in prompt.lower()

    def test_budget_target_duration_in_prompt(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """The prompt should mention the target duration."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_duration_minutes=3,
        )
        prompt = mock_run.call_args[0][0][2]
        assert "3" in prompt
        assert "minute" in prompt.lower()

    def test_no_budget_when_none(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """When max_duration_minutes is None, no budget constraint in prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_duration_minutes=None,
        )
        prompt = mock_run.call_args[0][0][2]
        # Should NOT contain budget-specific language
        assert "DURATION BUDGET" not in prompt

    def test_budget_segments_in_prompt(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """When max_duration_minutes is given, segment budget appears in prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_duration_minutes=10,
        )
        prompt = mock_run.call_args[0][0][2]
        # Should have a budget section with segment count
        assert "DURATION BUDGET" in prompt

    def test_budget_overrides_max_segments_if_smaller(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """Budget-derived segment count should override max_segments if smaller."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        # 1 minute budget => very few segments, less than default max_segments=20
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_duration_minutes=1,
            max_segments=20,
        )
        prompt = mock_run.call_args[0][0][2]
        # The max_segments in the prompt should be capped by the budget
        # Budget for 1 min: 125 words, 2 segments
        # The prompt should NOT say "Maximum 20 segments"
        assert "Maximum 20 segments" not in prompt

    def test_10_minute_budget_word_count(
        self, generator, sample_content_bundle, valid_conversation_response, mocker
    ):
        """10 minutes => 1250 words in prompt."""
        mock_run = mocker.patch(
            "video_overview.script.generator.subprocess.run",
            return_value=_make_subprocess_result(valid_conversation_response),
        )
        generator.generate(
            content_bundle=sample_content_bundle,
            topic="My Project",
            mode="conversation",
            llm_backend="claude",
            max_duration_minutes=10,
        )
        prompt = mock_run.call_args[0][0][2]
        assert "1250" in prompt

    def test_invalid_max_duration_raises_script_error(
        self, generator, sample_content_bundle
    ):
        """Zero/negative max_duration_minutes raises ScriptGenerationError."""
        with pytest.raises(ScriptGenerationError, match="positive"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
                max_duration_minutes=0,
            )

    def test_negative_max_duration_raises_script_error(
        self, generator, sample_content_bundle
    ):
        """Negative max_duration_minutes raises ScriptGenerationError."""
        with pytest.raises(ScriptGenerationError, match="positive"):
            generator.generate(
                content_bundle=sample_content_bundle,
                topic="My Project",
                mode="conversation",
                llm_backend="claude",
                max_duration_minutes=-5,
            )
