"""Smoke tests to verify package structure and imports."""

from click.testing import CliRunner


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


def test_all_exports_resolvable():
    """Test that every name in __all__ is actually accessible on the module."""
    import video_overview

    for name in video_overview.__all__:
        assert hasattr(video_overview, name), (
            f"{name!r} is listed in __all__ but is not an attribute of video_overview"
        )


def test_cli_version():
    """Test that the CLI entry point responds to --version."""
    from video_overview.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "video-overview" in result.output
    assert "0.1.0" in result.output
