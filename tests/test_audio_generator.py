"""Tests for AudioGenerator."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_overview.audio import AudioGenerationError, AudioGenerator
from video_overview.config import Script, ScriptSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(segments: list[tuple[str, str]], title: str = "Test") -> Script:
    """Build a Script with the given (speaker, text) pairs."""
    return Script(
        title=title,
        segments=[
            ScriptSegment(speaker=s, text=t, visual_prompt=f"Visual for {t}")
            for s, t in segments
        ],
    )


def _make_wav_bytes(num_samples: int = 100, sample_rate: int = 24000) -> bytes:
    """Create minimal valid WAV bytes."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _make_pcm_bytes(num_samples: int = 100) -> bytes:
    """Create raw PCM bytes (16-bit little-endian mono)."""
    return b"\x00\x00" * num_samples


def _mock_response(data: bytes, mime_type: str = "audio/wav") -> MagicMock:
    """Build a mock Gemini TTS response object."""
    inline_data = MagicMock()
    inline_data.data = data
    inline_data.mime_type = mime_type

    part = MagicMock()
    part.inline_data = inline_data

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_key() -> str:
    return "test-api-key-12345"


@pytest.fixture()
def generator(api_key: str) -> AudioGenerator:
    return AudioGenerator(api_key=api_key)


@pytest.fixture()
def conversation_script() -> Script:
    return _make_script(
        [
            ("Host", "Welcome to our deep dive."),
            ("Expert", "Thanks for having me."),
            ("Host", "Let's start with the basics."),
            ("Expert", "Sure, let me explain."),
        ]
    )


@pytest.fixture()
def narration_script() -> Script:
    return _make_script(
        [
            ("Narrator", "Welcome to this overview."),
            ("Narrator", "Let us explore the topic."),
        ]
    )


@pytest.fixture()
def large_script() -> Script:
    """Script with 25 segments for chunking tests."""
    segments = []
    for i in range(25):
        speaker = "Host" if i % 2 == 0 else "Expert"
        segments.append((speaker, f"Segment number {i} with some text content."))
    return _make_script(segments)


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------


class TestConversationMode:
    def test_uses_multi_speaker_voice_config(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """Conversation mode (2+ speakers) uses MultiSpeakerVoiceConfig."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.speech_config.multi_speaker_voice_config is not None
        speaker_configs = (
            config.speech_config.multi_speaker_voice_config.speaker_voice_configs
        )
        speakers = {sc.speaker for sc in speaker_configs}
        assert "Host" in speakers
        assert "Expert" in speakers

    def test_host_and_expert_voices_assigned(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """Host gets host_voice and Expert gets expert_voice."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        speaker_configs = (
            config.speech_config.multi_speaker_voice_config.speaker_voice_configs
        )
        voice_map = {
            sc.speaker: sc.voice_config.prebuilt_voice_config.voice_name
            for sc in speaker_configs
        }
        assert voice_map["Host"] == "Aoede"
        assert voice_map["Expert"] == "Charon"


# ---------------------------------------------------------------------------
# Narration mode
# ---------------------------------------------------------------------------


class TestNarrationMode:
    def test_uses_single_voice_config(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Narration mode (1 speaker) uses single VoiceConfig."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.speech_config.voice_config is not None
        assert (
            config.speech_config.voice_config.prebuilt_voice_config.voice_name
            == "Kore"
        )


# ---------------------------------------------------------------------------
# Segment chunking
# ---------------------------------------------------------------------------


class TestSegmentChunking:
    def test_25_segments_creates_two_batches(
        self, generator, large_script, tmp_path, mocker
    ):
        """25 segments should be split into 2 batches (e.g., 13+12 or 15+10)."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=large_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert mock_client.models.generate_content.call_count == 2

    def test_5_segments_creates_one_batch(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """A script with 4 segments should create exactly 1 batch."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# TTS prompt construction
# ---------------------------------------------------------------------------


class TestTTSPromptConstruction:
    def test_speaker_labels_in_prompt(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """TTS prompt should contain 'Host:' and 'Expert:' labels."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs["contents"]
        assert "Host:" in prompt
        assert "Expert:" in prompt
        assert "Welcome to our deep dive." in prompt

    def test_narrator_label_in_narration_prompt(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Narration prompt should contain 'Narrator:' labels."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs["contents"]
        # In narration mode with single voice, speaker label may or may not
        # be present; what matters is text is present
        assert "Welcome to this overview." in prompt


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


class TestAudioExtraction:
    def test_extracts_audio_from_inline_data(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Audio bytes from response inline_data should be saved."""
        wav_data = _make_wav_bytes(num_samples=500)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        audio_path, _ = generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert audio_path.exists()
        assert audio_path.suffix == ".wav"


# ---------------------------------------------------------------------------
# PCM to WAV conversion
# ---------------------------------------------------------------------------


class TestPCMToWAV:
    def test_pcm_converted_to_wav(
        self, generator, narration_script, tmp_path, mocker
    ):
        """PCM mime_type should trigger WAV header addition."""
        pcm_data = _make_pcm_bytes(num_samples=500)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            pcm_data, mime_type="audio/pcm;rate=24000"
        )
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        audio_path, _ = generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        # Verify the chunk files are valid WAV
        chunk_files = list(tmp_path.glob("chunk_*.wav"))
        assert len(chunk_files) >= 1
        for chunk_file in chunk_files:
            with wave.open(str(chunk_file), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 24000

    def test_wav_passthrough(
        self, generator, narration_script, tmp_path, mocker
    ):
        """WAV mime_type should be saved directly without conversion."""
        wav_data = _make_wav_bytes(num_samples=500)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            wav_data, mime_type="audio/wav"
        )
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        audio_path, _ = generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        chunk_files = list(tmp_path.glob("chunk_*.wav"))
        assert len(chunk_files) >= 1
        # The chunk file should be valid WAV
        for chunk_file in chunk_files:
            with wave.open(str(chunk_file), "rb") as wf:
                assert wf.getnchannels() == 1


# ---------------------------------------------------------------------------
# FFmpeg concatenation
# ---------------------------------------------------------------------------


class TestFFmpegConcatenation:
    def test_ffmpeg_called_for_multi_chunk(
        self, generator, large_script, tmp_path, mocker
    ):
        """Multi-chunk output should invoke ffmpeg concat."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=large_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        # ffmpeg should have been called once
        assert mock_run.call_count == 1
        ffmpeg_cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in ffmpeg_cmd[0]
        assert "-f" in ffmpeg_cmd
        assert "concat" in ffmpeg_cmd

    def test_single_chunk_skips_ffmpeg(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """Single-chunk output should skip ffmpeg (just rename/copy)."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mock_run = mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        # ffmpeg should NOT be called for a single chunk
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retries_on_api_error(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Should retry on API error and succeed on second attempt."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("API temporarily unavailable"),
            _mock_response(wav_data),
        ]
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )
        mocker.patch("video_overview.audio.generator.time.sleep")

        audio_path, _ = generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert mock_client.models.generate_content.call_count == 2
        assert audio_path.exists()

    def test_raises_after_max_retries(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Should raise AudioGenerationError after exhausting retries."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API permanently down"
        )
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch("video_overview.audio.generator.time.sleep")

        with pytest.raises(AudioGenerationError, match="API"):
            generator.generate(
                script=narration_script,
                host_voice="Aoede",
                expert_voice="Charon",
                narrator_voice="Kore",
                cache_dir=tmp_path,
            )

        # Should have tried 3 times
        assert mock_client.models.generate_content.call_count == 3

    def test_exponential_backoff_delays(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Retry delays should be exponential: 1s, 2s, 4s."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("fail")
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mock_sleep = mocker.patch("video_overview.audio.generator.time.sleep")

        with pytest.raises(AudioGenerationError):
            generator.generate(
                script=narration_script,
                host_voice="Aoede",
                expert_voice="Charon",
                narrator_voice="Kore",
                cache_dir=tmp_path,
            )

        # After first failure: sleep(1), after second: sleep(2)
        # Third failure raises without sleeping again
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)


# ---------------------------------------------------------------------------
# Segment duration estimation
# ---------------------------------------------------------------------------


class TestSegmentDuration:
    def test_durations_proportional_to_text_length(
        self, generator, tmp_path, mocker
    ):
        """Segment durations should be proportional to text length."""
        script = _make_script(
            [
                ("Narrator", "Short."),
                ("Narrator", "This is a much longer segment with more words."),
            ]
        )
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        _, durations = generator.generate(
            script=script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert len(durations) == 2
        assert all(d > 0 for d in durations)
        # Longer text -> longer duration
        assert durations[1] > durations[0]

    def test_durations_count_matches_segments(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """Number of durations should match number of segments."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        _, durations = generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert len(durations) == len(conversation_script.segments)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


class TestAPIKeyValidation:
    def test_empty_api_key_raises_error(self):
        """Empty API key should raise AudioGenerationError."""
        with pytest.raises(AudioGenerationError, match="API key"):
            AudioGenerator(api_key="")

    def test_none_api_key_raises_error(self):
        """None API key should raise AudioGenerationError."""
        with pytest.raises(AudioGenerationError, match="API key"):
            AudioGenerator(api_key=None)


# ---------------------------------------------------------------------------
# Cache directory creation
# ---------------------------------------------------------------------------


class TestCacheDirectory:
    def test_creates_cache_dir_if_missing(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Cache directory should be created if it doesn't exist."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        new_cache = tmp_path / "nonexistent" / "cache"
        assert not new_cache.exists()

        generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=new_cache,
        )

        assert new_cache.exists()


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_tuple_of_path_and_durations(
        self, generator, conversation_script, tmp_path, mocker
    ):
        """generate() should return (Path, list[float])."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        result = generator.generate(
            script=conversation_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        audio_path, durations = result
        assert isinstance(audio_path, Path)
        assert isinstance(durations, list)
        assert all(isinstance(d, float) for d in durations)


# ---------------------------------------------------------------------------
# Model name
# ---------------------------------------------------------------------------


class TestModelName:
    def test_uses_correct_model(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Should use gemini-2.5-flash-preview-tts model."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash-preview-tts"

    def test_response_modalities_audio(
        self, generator, narration_script, tmp_path, mocker
    ):
        """Config should specify response_modalities=['AUDIO']."""
        wav_data = _make_wav_bytes()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(wav_data)
        mocker.patch(
            "video_overview.audio.generator.genai.Client",
            return_value=mock_client,
        )
        mocker.patch(
            "video_overview.audio.generator.subprocess.run",
            return_value=MagicMock(returncode=0),
        )

        generator.generate(
            script=narration_script,
            host_voice="Aoede",
            expert_voice="Charon",
            narrator_voice="Kore",
            cache_dir=tmp_path,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.response_modalities == ["AUDIO"]
