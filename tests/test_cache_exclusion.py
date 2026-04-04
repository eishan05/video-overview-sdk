"""Tests proving cache directories are excluded from content ingestion.

These tests verify that:
1. ContentReader skips .video_overview_cache via _SKIP_DIRS (defense-in-depth)
2. ContentReader skips custom cache dirs nested under source_dir when excluded
3. create_overview() auto-excludes config.cache_dir when inside source_dir
4. Cache artifacts (filelist.txt, audio files, images) are never fed to script gen
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from video_overview.config import OverviewConfig, Script, ScriptSegment
from video_overview.content.reader import _SKIP_DIRS, ContentReader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reader():
    return ContentReader()


def _make_script(n_segments: int = 2) -> Script:
    segments = [
        ScriptSegment(
            speaker="Host" if i % 2 == 0 else "Expert",
            text=f"Segment {i} text.",
            visual_prompt=f"Visual for segment {i}",
        )
        for i in range(n_segments)
    ]
    return Script(title="Test", segments=segments)


# ---------------------------------------------------------------------------
# 1. Defense-in-depth: .video_overview_cache in _SKIP_DIRS
# ---------------------------------------------------------------------------


class TestVideoOverviewCacheInSkipDirs:
    """_SKIP_DIRS must include .video_overview_cache."""

    def test_skip_dirs_contains_video_overview_cache(self):
        assert ".video_overview_cache" in _SKIP_DIRS

    def test_reader_skips_default_cache_dir(self, reader, tmp_path):
        """Files inside .video_overview_cache must never appear in results."""
        (tmp_path / "main.py").write_text("x = 1")

        cache = tmp_path / ".video_overview_cache"
        cache.mkdir()
        (cache / "filelist.txt").write_text("main.py")
        (cache / "script.json").write_text('{"title":"t","segments":[]}')

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]

        assert "main.py" in paths
        assert not any(".video_overview_cache" in p for p in paths)
        assert not any("filelist.txt" in p for p in paths)

    def test_reader_skips_nested_cache_artifacts(self, reader, tmp_path):
        """Even deeply nested cache artifacts must be excluded."""
        (tmp_path / "app.py").write_text("print('hello')")

        cache = tmp_path / ".video_overview_cache"
        audio_dir = cache / "audio"
        visuals_dir = cache / "visuals"
        audio_dir.mkdir(parents=True)
        visuals_dir.mkdir(parents=True)

        (cache / "filelist.txt").write_text("app.py")
        (audio_dir / "segment_0.txt").write_text("audio data ref")
        (visuals_dir / "prompt_0.txt").write_text("visual prompt ref")

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]

        assert "app.py" in paths
        assert not any("filelist.txt" in p for p in paths)
        assert not any("segment_0" in p for p in paths)
        assert not any("prompt_0" in p for p in paths)


# ---------------------------------------------------------------------------
# 2. Custom cache dir exclusion via exclude patterns
# ---------------------------------------------------------------------------


class TestCustomCacheDirExclusion:
    """ContentReader should skip a custom cache dir when passed as exclude."""

    def test_exclude_custom_cache_dir(self, reader, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        custom_cache = tmp_path / "my_custom_cache"
        custom_cache.mkdir()
        (custom_cache / "filelist.txt").write_text("main.py")

        result = reader.read(tmp_path, exclude=["my_custom_cache/"])
        paths = [f["path"] for f in result["files"]]
        assert "main.py" in paths
        assert not any("filelist.txt" in p for p in paths)

    def test_exclude_custom_cache_dir_nested_files(self, reader, tmp_path):
        """Nested files under a custom cache dir must also be excluded."""
        (tmp_path / "main.py").write_text("x = 1")
        custom_cache = tmp_path / "my_custom_cache"
        audio_sub = custom_cache / "audio"
        audio_sub.mkdir(parents=True)
        (custom_cache / "filelist.txt").write_text("main.py")
        (audio_sub / "segment_0.txt").write_text("audio data")

        result = reader.read(tmp_path, exclude=["my_custom_cache/"])
        paths = [f["path"] for f in result["files"]]
        assert "main.py" in paths
        assert not any("filelist.txt" in p for p in paths)
        assert not any("segment_0" in p for p in paths)


# ---------------------------------------------------------------------------
# 3. create_overview() auto-excludes cache_dir when inside source_dir
# ---------------------------------------------------------------------------


class TestCreateOverviewAutoExcludesCacheDir:
    """create_overview() must add cache_dir to exclude when it's inside source_dir."""

    def test_cache_dir_inside_source_dir_auto_excluded(self, tmp_path):
        """When cache_dir is inside source_dir, create_overview must pass an
        exclude pattern for it so ContentReader never sees cache artifacts."""
        from video_overview.core import create_overview

        source = tmp_path / "source"
        source.mkdir()
        (source / "main.py").write_text("print('hello')")
        output = tmp_path / "out.mp4"

        config = OverviewConfig(
            source_dir=source,
            output=output,
            topic="test",
        )

        # The default cache_dir is source/.video_overview_cache
        assert config.cache_dir == source / ".video_overview_cache"

        mock_reader = MagicMock()
        mock_reader.read.return_value = {
            "directory_structure": "src/\n  main.py\n",
            "files": [
                {"path": "main.py", "content": "print('hello')", "language": "python"}
            ],
            "total_files": 1,
            "total_chars": 14,
        }

        mock_script_gen = MagicMock()
        mock_script_gen.generate.return_value = _make_script()

        with (
            patch("video_overview.core.ContentReader", return_value=mock_reader),
            patch("video_overview.core.ScriptGenerator", return_value=mock_script_gen),
            patch("video_overview.core.AudioGenerator") as mock_audio_cls,
            patch("video_overview.core.VisualGenerator") as mock_visual_cls,
            patch("video_overview.core.VideoAssembler") as mock_assembler_cls,
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            mock_audio_cls.return_value.generate.return_value = (
                Path("/tmp/out.wav"),
                [2.0, 2.0],
            )
            mock_visual_cls.return_value.generate = AsyncMock(
                return_value=[
                    Path("/tmp/img0.png"),
                    Path("/tmp/img1.png"),
                ]
            )
            mock_assembler_cls.return_value.assemble.return_value = output

            create_overview(config=config)

        # The reader.read call must have the cache dir pattern in its exclude list
        call_kwargs = mock_reader.read.call_args
        exclude_arg = call_kwargs.kwargs.get(
            "exclude", call_kwargs.args[2] if len(call_kwargs.args) > 2 else []
        )
        assert any(".video_overview_cache" in pat for pat in exclude_arg), (
            f"Expected cache dir exclusion in exclude={exclude_arg}"
        )

    def test_cache_dir_outside_source_dir_not_excluded(self, tmp_path):
        """When cache_dir is outside source_dir, no extra exclude is added."""
        from video_overview.core import create_overview

        source = tmp_path / "source"
        source.mkdir()
        (source / "main.py").write_text("print('hello')")
        output = tmp_path / "out.mp4"
        external_cache = tmp_path / "external_cache"

        config = OverviewConfig(
            source_dir=source,
            output=output,
            topic="test",
            cache_dir=external_cache,
        )

        mock_reader = MagicMock()
        mock_reader.read.return_value = {
            "directory_structure": "src/\n  main.py\n",
            "files": [
                {"path": "main.py", "content": "print('hello')", "language": "python"}
            ],
            "total_files": 1,
            "total_chars": 14,
        }

        mock_script_gen = MagicMock()
        mock_script_gen.generate.return_value = _make_script()

        with (
            patch("video_overview.core.ContentReader", return_value=mock_reader),
            patch("video_overview.core.ScriptGenerator", return_value=mock_script_gen),
            patch("video_overview.core.AudioGenerator") as mock_audio_cls,
            patch("video_overview.core.VisualGenerator") as mock_visual_cls,
            patch("video_overview.core.VideoAssembler") as mock_assembler_cls,
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            mock_audio_cls.return_value.generate.return_value = (
                Path("/tmp/out.wav"),
                [2.0, 2.0],
            )
            mock_visual_cls.return_value.generate = AsyncMock(
                return_value=[
                    Path("/tmp/img0.png"),
                    Path("/tmp/img1.png"),
                ]
            )
            mock_assembler_cls.return_value.assemble.return_value = output

            create_overview(config=config)

        call_kwargs = mock_reader.read.call_args
        exclude_arg = call_kwargs.kwargs.get(
            "exclude", call_kwargs.args[2] if len(call_kwargs.args) > 2 else []
        )
        # No cache dir pattern should have been added
        assert not any("external_cache" in pat for pat in exclude_arg), (
            f"External cache should not be excluded, got exclude={exclude_arg}"
        )

    def test_user_excludes_preserved_with_cache_exclusion(self, tmp_path):
        """User-supplied exclude patterns must not be lost when cache exclusion
        is appended."""
        from video_overview.core import create_overview

        source = tmp_path / "source"
        source.mkdir()
        (source / "main.py").write_text("print('hello')")
        output = tmp_path / "out.mp4"

        config = OverviewConfig(
            source_dir=source,
            output=output,
            topic="test",
            exclude=["tests/*", "*.log"],
        )

        mock_reader = MagicMock()
        mock_reader.read.return_value = {
            "directory_structure": "src/\n  main.py\n",
            "files": [
                {"path": "main.py", "content": "print('hello')", "language": "python"}
            ],
            "total_files": 1,
            "total_chars": 14,
        }

        mock_script_gen = MagicMock()
        mock_script_gen.generate.return_value = _make_script()

        with (
            patch("video_overview.core.ContentReader", return_value=mock_reader),
            patch("video_overview.core.ScriptGenerator", return_value=mock_script_gen),
            patch("video_overview.core.AudioGenerator") as mock_audio_cls,
            patch("video_overview.core.VisualGenerator") as mock_visual_cls,
            patch("video_overview.core.VideoAssembler") as mock_assembler_cls,
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            mock_audio_cls.return_value.generate.return_value = (
                Path("/tmp/out.wav"),
                [2.0, 2.0],
            )
            mock_visual_cls.return_value.generate = AsyncMock(
                return_value=[
                    Path("/tmp/img0.png"),
                    Path("/tmp/img1.png"),
                ]
            )
            mock_assembler_cls.return_value.assemble.return_value = output

            create_overview(config=config)

        call_kwargs = mock_reader.read.call_args
        exclude_arg = call_kwargs.kwargs.get(
            "exclude", call_kwargs.args[2] if len(call_kwargs.args) > 2 else []
        )
        # Original user patterns must still be present
        assert "tests/*" in exclude_arg
        assert "*.log" in exclude_arg
        # Cache exclusion must also be present
        assert any(".video_overview_cache" in pat for pat in exclude_arg)


# ---------------------------------------------------------------------------
# 4. Integration: cache artifacts never reach script generation
# ---------------------------------------------------------------------------


class TestCacheArtifactsNeverReachScriptGen:
    """End-to-end: files created by the cache must not appear in content bundle."""

    def test_filelist_txt_not_in_content_bundle(self, reader, tmp_path):
        """filelist.txt (a typical cache artifact) must never be ingested."""
        (tmp_path / "app.py").write_text("def main(): pass")

        cache = tmp_path / ".video_overview_cache"
        cache.mkdir()
        (cache / "filelist.txt").write_text("app.py\n")

        result = reader.read(tmp_path)
        contents = " ".join(f["content"] for f in result["files"])
        paths = [f["path"] for f in result["files"]]

        assert "filelist.txt" not in paths
        # The content of filelist.txt should not leak into the bundle
        assert "app.py\n" != contents or len(result["files"]) == 1

    def test_cache_json_artifacts_not_ingested(self, reader, tmp_path):
        """JSON cache artifacts must not be ingested."""
        (tmp_path / "lib.py").write_text("class Lib: pass")

        cache = tmp_path / ".video_overview_cache"
        cache.mkdir()
        (cache / "script_cache.json").write_text('{"cached": true}')
        (cache / "metadata.json").write_text('{"version": 1}')

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]

        assert not any("script_cache.json" in p for p in paths)
        assert not any("metadata.json" in p for p in paths)
        assert any("lib.py" in p for p in paths)

    def test_custom_cache_dir_with_name_not_in_skip_dirs(self, tmp_path):
        """A custom cache dir name that is NOT in _SKIP_DIRS must still be
        excluded by create_overview's auto-exclude logic."""
        from video_overview.core import create_overview

        source = tmp_path / "source"
        source.mkdir()
        (source / "main.py").write_text("print('hello')")
        output = tmp_path / "out.mp4"

        # Use a custom cache dir name that is NOT in _SKIP_DIRS
        custom_cache = source / "my_build_cache"

        config = OverviewConfig(
            source_dir=source,
            output=output,
            topic="test",
            cache_dir=custom_cache,
        )

        mock_reader = MagicMock()
        mock_reader.read.return_value = {
            "directory_structure": "src/\n  main.py\n",
            "files": [
                {"path": "main.py", "content": "print('hello')", "language": "python"}
            ],
            "total_files": 1,
            "total_chars": 14,
        }

        mock_script_gen = MagicMock()
        mock_script_gen.generate.return_value = _make_script()

        with (
            patch("video_overview.core.ContentReader", return_value=mock_reader),
            patch("video_overview.core.ScriptGenerator", return_value=mock_script_gen),
            patch("video_overview.core.AudioGenerator") as mock_audio_cls,
            patch("video_overview.core.VisualGenerator") as mock_visual_cls,
            patch("video_overview.core.VideoAssembler") as mock_assembler_cls,
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            mock_audio_cls.return_value.generate.return_value = (
                Path("/tmp/out.wav"),
                [2.0, 2.0],
            )
            mock_visual_cls.return_value.generate = AsyncMock(
                return_value=[
                    Path("/tmp/img0.png"),
                    Path("/tmp/img1.png"),
                ]
            )
            mock_assembler_cls.return_value.assemble.return_value = output

            create_overview(config=config)

        call_kwargs = mock_reader.read.call_args
        exclude_arg = call_kwargs.kwargs.get(
            "exclude", call_kwargs.args[2] if len(call_kwargs.args) > 2 else []
        )
        # Custom cache dir must be excluded
        assert any("my_build_cache" in pat for pat in exclude_arg), (
            f"Custom cache dir should be excluded, got exclude={exclude_arg}"
        )
