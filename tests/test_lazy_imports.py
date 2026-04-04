"""Tests for lazy imports -- verify non-Gemini modules work without google-genai."""

from __future__ import annotations

import subprocess
import sys
import types

# The subprocess-based tests use a proper ``find_spec``-based import
# blocker (``importlib.abc.MetaPathFinder``) to simulate google-genai
# not being installed.  This is more reliable than monkeypatching
# sys.modules in-process because Python caches already-imported
# modules and the google namespace package complicates teardown.

_BLOCKER_PREAMBLE = """\
import sys
from importlib.abc import MetaPathFinder

class _BlockGenAI(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "google.genai" or fullname.startswith("google.genai."):
            raise ImportError(
                f"No module named '{fullname}' (google-genai not installed)"
            )
        return None

sys.meta_path.insert(0, _BlockGenAI())

# Remove any pre-loaded google.genai from sys.modules
for _k in list(sys.modules):
    if _k == "google.genai" or _k.startswith("google.genai."):
        del sys.modules[_k]
"""


def _run_import_check(import_statement: str) -> subprocess.CompletedProcess:
    """Run *import_statement* in a subprocess with google.genai blocked."""
    code = _BLOCKER_PREAMBLE + "\n" + import_statement + "\nprint('OK')\n"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Subprocess-based tests: completely clean Python process
# ---------------------------------------------------------------------------


class TestImportWithoutGenAI:
    """Verify imports succeed in a subprocess where google-genai is absent.

    The key invariant: importing ``video_overview`` and its non-Gemini
    subpackages (``content``, ``video``, ``script``, ``config``) must
    not trigger ``google.genai`` imports.  Subpackages that *need*
    google-genai (``audio.generator``, ``visuals.generator``) may still
    fail when directly imported -- that is expected and tested here too.
    """

    def test_import_video_overview_without_genai(self):
        """``import video_overview`` must work without google-genai."""
        result = _run_import_check("import video_overview")
        assert result.returncode == 0, (
            f"import video_overview failed:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout

    def test_import_content_without_genai(self):
        """``import video_overview.content`` must work without google-genai."""
        result = _run_import_check("import video_overview.content")
        assert result.returncode == 0, (
            f"import video_overview.content failed:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout

    def test_import_video_without_genai(self):
        """``import video_overview.video`` must work without google-genai."""
        result = _run_import_check("import video_overview.video")
        assert result.returncode == 0, (
            f"import video_overview.video failed:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout

    def test_import_config_without_genai(self):
        """``import video_overview.config`` must work without google-genai."""
        result = _run_import_check("import video_overview.config")
        assert result.returncode == 0, (
            f"import video_overview.config failed:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout

    def test_import_script_without_genai(self):
        """``import video_overview.script`` must work without google-genai."""
        result = _run_import_check("import video_overview.script")
        assert result.returncode == 0, (
            f"import video_overview.script failed:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout

    def test_version_accessible_without_genai(self):
        """``video_overview.__version__`` must be accessible without google-genai."""
        result = _run_import_check(
            "import video_overview; assert video_overview.__version__"
        )
        assert result.returncode == 0, (
            f"__version__ not accessible:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_all_defined_without_genai(self):
        """``video_overview.__all__`` must be defined without google-genai."""
        result = _run_import_check(
            "import video_overview; assert isinstance(video_overview.__all__, list)"
        )
        assert result.returncode == 0, (
            f"__all__ not defined:\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_config_classes_accessible_without_genai(self):
        """Config classes must be accessible without google-genai."""
        result = _run_import_check(
            "from video_overview import OverviewConfig, OverviewResult\n"
            "from video_overview import Script, ScriptSegment\n"
            "assert OverviewConfig is not None"
        )
        assert result.returncode == 0, (
            f"Config classes not accessible:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_non_gemini_submodules_accessible_without_genai(self):
        """Non-Gemini submodules (content, video, script) accessible via attribute."""
        result = _run_import_check(
            "import video_overview\n"
            "assert video_overview.content is not None\n"
            "assert video_overview.video is not None\n"
            "assert video_overview.script is not None\n"
        )
        assert result.returncode == 0, (
            f"Non-Gemini submodules not accessible:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_audio_generator_import_fails_without_genai(self):
        """``import video_overview.audio.generator`` should fail without google-genai.

        This is expected: the audio generator *requires* google-genai at
        import time.  The point is that importing the top-level package
        does NOT trigger this failure.
        """
        result = _run_import_check("import video_overview.audio.generator")
        assert result.returncode != 0, (
            "audio.generator should fail to import without google-genai"
        )

    def test_visuals_generator_import_fails_without_genai(self):
        """Visuals generator should fail to import without genai."""
        result = _run_import_check("import video_overview.visuals.generator")
        assert result.returncode != 0, (
            "visuals.generator should fail to import without google-genai"
        )


class TestPublicAPIStableWithLazyImports:
    """Verify __all__ and public API names are accessible (with genai installed)."""

    def test_all_exports_defined(self):
        """Every name in __all__ must be accessible on the module."""
        import video_overview

        for name in video_overview.__all__:
            assert hasattr(video_overview, name), (
                f"{name!r} in __all__ but not accessible"
            )

    def test_all_contains_expected_names(self):
        """__all__ must contain the expected public API names."""
        import video_overview

        expected = {
            "__version__",
            "OverviewConfig",
            "OverviewResult",
            "Script",
            "ScriptSegment",
            "audio",
            "content",
            "create_overview",
            "script",
            "video",
            "visuals",
        }
        assert set(video_overview.__all__) == expected

    def test_submodule_attributes_are_modules(self):
        """Accessing audio, content, script, video, visuals returns modules."""
        import video_overview

        for name in ("audio", "content", "script", "video", "visuals"):
            obj = getattr(video_overview, name)
            assert isinstance(obj, types.ModuleType), (
                f"video_overview.{name} should be a module, got {type(obj)}"
            )

    def test_config_classes_accessible(self):
        """OverviewConfig, Script, etc. are directly importable."""
        from video_overview import (
            OverviewConfig,
            OverviewResult,
            Script,
            ScriptSegment,
        )

        assert OverviewConfig is not None
        assert OverviewResult is not None
        assert Script is not None
        assert ScriptSegment is not None

    def test_create_overview_accessible(self):
        """create_overview function is importable from top level."""
        from video_overview import create_overview

        assert callable(create_overview)
