"""Smoke tests to verify package structure and imports."""


def test_import_video_overview():
    """Test that the main package is importable."""
    import video_overview

    assert video_overview is not None


def test_version_exists():
    """Test that __version__ is defined."""
    import video_overview

    assert hasattr(video_overview, "__version__")
    assert isinstance(video_overview.__version__, str)
    assert len(video_overview.__version__) > 0


def test_import_content_subpackage():
    """Test that content subpackage is importable."""
    import video_overview.content

    assert video_overview.content is not None


def test_import_script_subpackage():
    """Test that script subpackage is importable."""
    import video_overview.script

    assert video_overview.script is not None


def test_import_audio_subpackage():
    """Test that audio subpackage is importable."""
    import video_overview.audio

    assert video_overview.audio is not None


def test_import_visuals_subpackage():
    """Test that visuals subpackage is importable."""
    import video_overview.visuals

    assert video_overview.visuals is not None


def test_import_video_subpackage():
    """Test that video subpackage is importable."""
    import video_overview.video

    assert video_overview.video is not None


def test_public_api_exports():
    """Test that public API names are exported from the main package."""
    import video_overview

    # __all__ should be defined
    assert hasattr(video_overview, "__all__")
    assert isinstance(video_overview.__all__, list)
