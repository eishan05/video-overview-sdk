"""Tests for VideoAssembler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_overview.config import ScriptSegment
from video_overview.video import VideoAssembler, VideoAssemblyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segments(texts: list[str]) -> list[ScriptSegment]:
    """Build ScriptSegment list with given texts."""
    return [
        ScriptSegment(speaker="Host", text=t, visual_prompt=f"Visual for {t}")
        for t in texts
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def assembler(mocker) -> VideoAssembler:
    """VideoAssembler with mocked shutil.which returning ffmpeg."""
    mocker.patch(
        "video_overview.video.assembler.shutil.which",
        return_value="/usr/bin/ffmpeg",
    )
    return VideoAssembler()


@pytest.fixture()
def audio_path(tmp_path) -> Path:
    """Create a dummy audio file."""
    p = tmp_path / "input.wav"
    p.write_bytes(b"RIFF" + b"\x00" * 100)
    return p


@pytest.fixture()
def single_image(tmp_path) -> list[Path]:
    """Create a single dummy image file."""
    p = tmp_path / "img_000.png"
    p.write_bytes(b"\x89PNG" + b"\x00" * 100)
    return [p]


@pytest.fixture()
def three_images(tmp_path) -> list[Path]:
    """Create three dummy image files."""
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i:03d}.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 100)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# FFmpeg availability
# ---------------------------------------------------------------------------


class TestFFmpegValidation:
    def test_ffmpeg_not_available_raises_error(self, mocker):
        """Should raise VideoAssemblyError if ffmpeg is not on PATH."""
        mocker.patch(
            "video_overview.video.assembler.shutil.which",
            return_value=None,
        )
        with pytest.raises(VideoAssemblyError, match="ffmpeg"):
            VideoAssembler()

    def test_ffmpeg_available_succeeds(self, mocker):
        """Should succeed when ffmpeg is available."""
        mocker.patch(
            "video_overview.video.assembler.shutil.which",
            return_value="/usr/bin/ffmpeg",
        )
        assembler = VideoAssembler()
        assert assembler is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_image_list_raises_error(self, assembler, audio_path, tmp_path):
        """Should raise VideoAssemblyError for empty image list in video mode."""
        with pytest.raises(VideoAssemblyError, match="image"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=[],
                segment_durations=[],
                output_path=tmp_path / "out.mp4",
                format="video",
            )

    def test_mismatched_image_duration_counts_raises_error(
        self, assembler, audio_path, single_image, tmp_path
    ):
        """Should raise when image count doesn't match duration count."""
        with pytest.raises(VideoAssemblyError, match="mismatch"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=single_image,
                segment_durations=[5.0, 10.0],  # 2 durations, 1 image
                output_path=tmp_path / "out.mp4",
                format="video",
            )

    def test_non_positive_duration_raises_error(
        self, assembler, audio_path, single_image, tmp_path
    ):
        """Should raise when a segment duration is zero or negative."""
        with pytest.raises(VideoAssemblyError, match="non-positive"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=single_image,
                segment_durations=[0.0],
                output_path=tmp_path / "out.mp4",
                format="video",
            )

    def test_negative_duration_raises_error(
        self, assembler, audio_path, single_image, tmp_path
    ):
        """Should raise when a segment duration is negative."""
        with pytest.raises(VideoAssemblyError, match="non-positive"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=single_image,
                segment_durations=[-1.0],
                output_path=tmp_path / "out.mp4",
                format="video",
            )

    def test_duration_shorter_than_crossfade_raises_error(
        self, assembler, audio_path, tmp_path
    ):
        """Should raise when multi-image segment is shorter than crossfade."""
        images = []
        for i in range(2):
            p = tmp_path / f"img_{i:03d}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 100)
            images.append(p)

        with pytest.raises(VideoAssemblyError, match="crossfade"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=images,
                segment_durations=[0.3, 5.0],  # 0.3 < 0.5 crossfade
                output_path=tmp_path / "out.mp4",
                format="video",
            )


# ---------------------------------------------------------------------------
# Audio format
# ---------------------------------------------------------------------------


class TestAudioFormat:
    def test_audio_format_converts_wav_to_mp3(
        self, assembler, audio_path, tmp_path, mocker
    ):
        """Audio format should run ffmpeg to convert WAV to MP3."""
        output = tmp_path / "output.mp3"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=[],
            segment_durations=[],
            output_path=output,
            format="audio",
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "ffmpeg" in cmd_str
        assert "libmp3lame" in cmd_str
        assert str(audio_path) in cmd_str
        assert str(output) in cmd_str

    def test_audio_format_wav_output_copies_file(
        self, assembler, audio_path, tmp_path, mocker
    ):
        """Audio format with .wav output should copy file without ffmpeg."""
        output = tmp_path / "output.wav"
        mock_copy = mocker.patch(
            "video_overview.video.assembler.shutil.copy2",
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=[],
            segment_durations=[],
            output_path=output,
            format="audio",
        )

        mock_copy.assert_called_once_with(audio_path, output)

    def test_audio_format_output_path_is_correct(
        self, assembler, audio_path, tmp_path, mocker
    ):
        """Audio format should return the correct output path."""
        output = tmp_path / "output.mp3"
        mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        result = assembler.assemble(
            audio_path=audio_path,
            image_paths=[],
            segment_durations=[],
            output_path=output,
            format="audio",
        )

        assert result == output


# ---------------------------------------------------------------------------
# Video format - single image
# ---------------------------------------------------------------------------


class TestVideoSingleImage:
    def test_zoompan_filter_applied(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """Single image video should have zoompan filter applied."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=single_image,
            segment_durations=[5.0],
            output_path=output,
            format="video",
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "zoompan" in cmd_str


# ---------------------------------------------------------------------------
# Video format - multiple images
# ---------------------------------------------------------------------------


class TestVideoMultipleImages:
    def test_xfade_transitions_between_images(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Multiple images should have xfade transitions between them."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "xfade" in cmd_str

    def test_h264_aac_output_settings(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Video should use H.264 video and AAC audio codecs."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd
        assert "aac" in cmd

    def test_1920x1080_resolution_and_30fps(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Video should output at 1920x1080 resolution and 30fps."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "1920" in cmd_str
        assert "1080" in cmd_str
        # Check for 30 fps in the filter
        assert "fps=30" in cmd_str or "r 30" in cmd_str or ":30:" in cmd_str

    def test_audio_overlay(self, assembler, audio_path, three_images, tmp_path, mocker):
        """Video output should include the audio track."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        # Audio path should be an input
        assert str(audio_path) in [str(c) for c in cmd]


# ---------------------------------------------------------------------------
# Ken Burns effect
# ---------------------------------------------------------------------------


class TestKenBurnsEffect:
    def test_zoompan_filter_parameters(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """Zoompan should apply 5% zoom (1.05) over duration."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=single_image,
            segment_durations=[5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        # Should have zoompan with 1.05 max zoom
        assert "zoompan" in cmd_str
        assert "1.05" in cmd_str


# ---------------------------------------------------------------------------
# Crossfade transitions
# ---------------------------------------------------------------------------


class TestCrossfade:
    def test_xfade_filter_with_half_second_duration(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Xfade transitions should have 0.5s duration."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "xfade" in cmd_str
        assert "duration=0.5" in cmd_str


# ---------------------------------------------------------------------------
# Segment duration estimation
# ---------------------------------------------------------------------------


class TestSegmentDurationEstimation:
    def test_proportional_to_text_length(self, assembler):
        """Segment durations should be proportional to text length."""
        segments = _make_segments(
            [
                "Short",
                "This is a much longer segment with more text",
            ]
        )
        total_duration = 60.0

        durations = assembler._estimate_segment_durations(segments, total_duration)

        assert len(durations) == 2
        assert abs(sum(durations) - total_duration) < 0.01
        # Longer text should get longer duration
        assert durations[1] > durations[0]

    def test_equal_length_texts_get_equal_durations(self, assembler):
        """Segments with equal text lengths should get equal durations."""
        segments = _make_segments(["AAAA", "BBBB", "CCCC"])
        total_duration = 30.0

        durations = assembler._estimate_segment_durations(segments, total_duration)

        assert len(durations) == 3
        assert abs(sum(durations) - total_duration) < 0.01
        # Each should be ~10 seconds
        assert all(abs(d - 10.0) < 0.01 for d in durations)

    def test_empty_segments_raises_error(self, assembler):
        """Should raise VideoAssemblyError for empty segments list."""
        with pytest.raises(VideoAssemblyError, match="empty"):
            assembler._estimate_segment_durations([], 60.0)


# ---------------------------------------------------------------------------
# Video timeline correctness
# ---------------------------------------------------------------------------


class TestVideoTimeline:
    def test_effective_durations_compensate_for_crossfade(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Image input durations should be extended to compensate for xfade overlap."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        # Find -t values for image inputs (after the audio input)
        t_values = []
        for j, c in enumerate(cmd):
            if c == "-t" and j > 2:  # skip any -t before audio
                t_values.append(float(cmd[j + 1]))

        # First image: 5.0 + 0.5 = 5.5s (extended for right overlap)
        assert abs(t_values[0] - 5.5) < 0.01
        # Second image: 5.0 + 0.5 = 5.5s (extended for left+right, but only
        # interior clips get one additional overlap)
        assert abs(t_values[1] - 5.5) < 0.01
        # Third image: 5.0s (no extension, last clip)
        assert abs(t_values[2] - 5.0) < 0.01

    def test_single_image_no_duration_extension(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """Single image should not have its duration extended."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=single_image,
            segment_durations=[10.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        t_values = []
        for j, c in enumerate(cmd):
            if c == "-t" and j > 2:
                t_values.append(float(cmd[j + 1]))

        assert len(t_values) == 1
        assert abs(t_values[0] - 10.0) < 0.01

    def test_non_integer_duration_uses_ceil_for_frames(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """Non-integer durations should use math.ceil for frame counts."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        # 5.02s * 30fps = 150.6 frames -> ceil to 151
        assembler.assemble(
            audio_path=audio_path,
            image_paths=single_image,
            segment_durations=[5.02],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        # Should have d=151 (ceil of 150.6), not d=150
        assert "d=151" in cmd_str


# ---------------------------------------------------------------------------
# FFmpeg command failures
# ---------------------------------------------------------------------------


class TestFFmpegCommandFailures:
    def test_ffmpeg_nonzero_exit_raises_error(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """Non-zero ffmpeg exit code should raise VideoAssemblyError."""
        output = tmp_path / "output.mp4"
        mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="encoding error"),
        )

        with pytest.raises(VideoAssemblyError, match="ffmpeg"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=single_image,
                segment_durations=[5.0],
                output_path=output,
                format="video",
            )

    def test_ffmpeg_not_found_during_assembly_raises_error(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """FileNotFoundError during subprocess.run should raise VideoAssemblyError."""
        output = tmp_path / "output.mp4"
        mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            side_effect=FileNotFoundError("ffmpeg not found"),
        )

        with pytest.raises(VideoAssemblyError, match="ffmpeg"):
            assembler.assemble(
                audio_path=audio_path,
                image_paths=single_image,
                segment_durations=[5.0],
                output_path=output,
                format="video",
            )


# ---------------------------------------------------------------------------
# Filter complex construction
# ---------------------------------------------------------------------------


class TestFilterComplexConstruction:
    def test_single_image_no_xfade(
        self, assembler, audio_path, single_image, tmp_path, mocker
    ):
        """Single image should have zoompan but no xfade."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=single_image,
            segment_durations=[5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "zoompan" in cmd_str
        # Count actual xfade filter instances (xfade=), not label refs
        assert cmd_str.count("xfade=") == 0

    def test_two_images_one_xfade(self, assembler, audio_path, tmp_path, mocker):
        """Two images should produce exactly one xfade transition."""
        images = []
        for i in range(2):
            p = tmp_path / f"img_{i:03d}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 100)
            images.append(p)

        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=images,
            segment_durations=[5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        # Count actual xfade filter instances (xfade=), not label refs
        assert cmd_str.count("xfade=") == 1

    def test_three_images_two_xfades(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Three images should produce exactly two xfade transitions."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        # Count actual xfade filter instances (xfade=), not label refs
        assert cmd_str.count("xfade=") == 2

    def test_filter_complex_flag_present(
        self, assembler, audio_path, three_images, tmp_path, mocker
    ):
        """Video command should include -filter_complex flag."""
        output = tmp_path / "output.mp4"
        mock_run = mocker.patch(
            "video_overview.video.assembler.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        )

        assembler.assemble(
            audio_path=audio_path,
            image_paths=three_images,
            segment_durations=[5.0, 5.0, 5.0],
            output_path=output,
            format="video",
        )

        cmd = mock_run.call_args[0][0]
        assert "-filter_complex" in cmd
