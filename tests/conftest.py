"""Shared test fixtures for video-overview-sdk."""

import pytest


@pytest.fixture
def sample_video_url():
    """Provide a sample video URL for testing."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a temporary output directory for test artifacts."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
