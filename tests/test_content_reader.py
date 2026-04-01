"""Tests for ContentReader class."""

import pytest

from video_overview.content.reader import ContentReader


@pytest.fixture
def reader():
    """Create a ContentReader instance."""
    return ContentReader()


@pytest.fixture
def sample_dir(tmp_path):
    """Create a sample directory structure for testing."""
    # README at root
    (tmp_path / "README.md").write_text("# My Project\nThis is a sample project.")

    # Python source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main():\n    print('hello')\n")
    (src / "utils.py").write_text("def add(a, b):\n    return a + b\n")

    # A docs directory
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# User Guide\nSome documentation here.")

    # A nested directory
    nested = src / "sub"
    nested.mkdir()
    (nested / "helper.py").write_text("# helper module\n")

    # A config file
    (tmp_path / "config.yaml").write_text("key: value\n")

    return tmp_path


class TestReadSimpleDirectory:
    """Test reading a simple directory with a few files."""

    def test_reads_all_files(self, reader, sample_dir):
        result = reader.read(sample_dir)
        assert result["total_files"] > 0
        paths = [f["path"] for f in result["files"]]
        # Should contain all text files
        assert any("README.md" in p for p in paths)
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_result_has_required_keys(self, reader, sample_dir):
        result = reader.read(sample_dir)
        assert "directory_structure" in result
        assert "files" in result
        assert "total_files" in result
        assert "total_chars" in result

    def test_files_have_required_keys(self, reader, sample_dir):
        result = reader.read(sample_dir)
        for f in result["files"]:
            assert "path" in f
            assert "content" in f
            assert "language" in f

    def test_total_files_matches_files_list(self, reader, sample_dir):
        result = reader.read(sample_dir)
        assert result["total_files"] == len(result["files"])

    def test_total_chars_is_sum_of_content_lengths(self, reader, sample_dir):
        result = reader.read(sample_dir)
        total = sum(len(f["content"]) for f in result["files"])
        assert result["total_chars"] == total


class TestIncludeFilter:
    """Test include pattern filtering."""

    def test_include_only_python_files(self, reader, sample_dir):
        result = reader.read(sample_dir, include=["*.py"])
        for f in result["files"]:
            assert f["path"].endswith(".py"), f"Expected .py file, got {f['path']}"

    def test_include_only_markdown_files(self, reader, sample_dir):
        result = reader.read(sample_dir, include=["*.md"])
        for f in result["files"]:
            assert f["path"].endswith(".md"), f"Expected .md file, got {f['path']}"
        assert result["total_files"] >= 2  # README.md and guide.md

    def test_include_multiple_patterns(self, reader, sample_dir):
        result = reader.read(sample_dir, include=["*.py", "*.yaml"])
        for f in result["files"]:
            assert f["path"].endswith(".py") or f["path"].endswith(".yaml"), (
                f"Unexpected file: {f['path']}"
            )

    def test_include_path_based_pattern(self, reader, sample_dir):
        """Path-based include patterns like 'src/*.py' should work."""
        result = reader.read(sample_dir, include=["src/*.py"])
        assert result["total_files"] >= 1
        for f in result["files"]:
            assert f["path"].startswith("src/")
            assert f["path"].endswith(".py")

    def test_include_path_pattern_excludes_nested(self, reader, sample_dir):
        """'src/*.py' should NOT match 'src/sub/helper.py'."""
        result = reader.read(sample_dir, include=["src/*.py"])
        paths = [f["path"] for f in result["files"]]
        assert "src/sub/helper.py" not in paths

    def test_include_recursive_glob(self, reader, sample_dir):
        """'src/**/*.py' should match nested files."""
        result = reader.read(sample_dir, include=["src/**/*.py"])
        paths = [f["path"] for f in result["files"]]
        assert "src/sub/helper.py" in paths

    def test_include_pattern_anchored_to_root(self, reader, tmp_path):
        """'src/*.py' should NOT match 'foo/src/main.py'."""
        nested = tmp_path / "foo" / "src"
        nested.mkdir(parents=True)
        (nested / "main.py").write_text("x = 1")
        top = tmp_path / "src"
        top.mkdir()
        (top / "main.py").write_text("y = 2")

        result = reader.read(tmp_path, include=["src/*.py"])
        paths = [f["path"] for f in result["files"]]
        assert "src/main.py" in paths
        assert "foo/src/main.py" not in paths


class TestExcludeFilter:
    """Test exclude pattern filtering."""

    def test_exclude_yaml_files(self, reader, sample_dir):
        result = reader.read(sample_dir, exclude=["*.yaml"])
        for f in result["files"]:
            assert not f["path"].endswith(".yaml"), (
                f"Expected no .yaml files, got {f['path']}"
            )

    def test_exclude_directory_pattern(self, reader, sample_dir):
        result = reader.read(sample_dir, exclude=["docs/*"])
        paths = [f["path"] for f in result["files"]]
        for p in paths:
            assert not p.startswith("docs/"), (
                f"File inside docs/ should be excluded: {p}"
            )


class TestGitignoreRespect:
    """Test that .gitignore patterns are respected."""

    def test_ignores_gitignored_files(self, reader, tmp_path):
        # Create a .gitignore
        (tmp_path / ".gitignore").write_text("*.log\nsecret.txt\nbuild/\n")

        # Create files that should be ignored
        (tmp_path / "debug.log").write_text("log output")
        (tmp_path / "secret.txt").write_text("secret data")
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.js").write_text("compiled code")

        # Create files that should NOT be ignored
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Project")

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]

        # Ignored files should not appear
        assert not any("debug.log" in p for p in paths)
        assert not any("secret.txt" in p for p in paths)
        assert not any("output.js" in p for p in paths)

        # Non-ignored files should appear
        assert any("main.py" in p for p in paths)
        assert any("README.md" in p for p in paths)

    def test_works_without_gitignore(self, reader, tmp_path):
        """Should work fine when no .gitignore is present."""
        (tmp_path / "main.py").write_text("print('hello')")
        result = reader.read(tmp_path)
        assert result["total_files"] == 1


class TestBinaryFileSkipping:
    """Test that binary files are skipped."""

    def test_skips_binary_files(self, reader, tmp_path):
        # Create a binary file with null bytes
        (tmp_path / "image.bin").write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        (tmp_path / "main.py").write_text("print('hello')")

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]
        assert not any("image.bin" in p for p in paths)
        assert any("main.py" in p for p in paths)

    def test_skips_pyc_files(self, reader, tmp_path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.cpython-311.pyc").write_bytes(b"\x00\x01\x02")
        (tmp_path / "module.py").write_text("x = 1")

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]
        assert not any(".pyc" in p for p in paths)

    def test_skips_egg_info_directories(self, reader, tmp_path):
        """Directories like 'package.egg-info' should be skipped."""
        egg = tmp_path / "video_overview.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text("Metadata-Version: 2.1")
        (tmp_path / "main.py").write_text("x = 1")

        result = reader.read(tmp_path)
        paths = [f["path"] for f in result["files"]]
        assert not any("egg-info" in p for p in paths)
        assert any("main.py" in p for p in paths)


class TestLargeFileTruncation:
    """Test that large files are truncated."""

    def test_truncates_large_file(self, reader, tmp_path):
        large_content = "x" * 5000
        (tmp_path / "large.py").write_text(large_content)

        result = reader.read(tmp_path)
        file_entry = result["files"][0]
        # Should be truncated to 2000 chars plus a truncation note
        assert len(file_entry["content"]) < len(large_content)
        assert "truncated" in file_entry["content"].lower()

    def test_preserves_first_2000_chars(self, reader, tmp_path):
        content = "A" * 1000 + "B" * 1000 + "C" * 3000
        (tmp_path / "large.txt").write_text(content)

        result = reader.read(tmp_path)
        file_entry = result["files"][0]
        # The content should start with the first 2000 chars
        assert file_entry["content"].startswith("A" * 1000 + "B" * 1000)

    def test_small_files_not_truncated(self, reader, tmp_path):
        content = "small content"
        (tmp_path / "small.py").write_text(content)

        result = reader.read(tmp_path)
        file_entry = result["files"][0]
        assert file_entry["content"] == content


class TestMaxCharsBudget:
    """Test that total content respects max_chars budget."""

    def test_respects_max_chars(self, reader, tmp_path):
        # Create many files
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text("x" * 1000)

        result = reader.read(tmp_path, max_chars=5000)
        assert result["total_chars"] <= 5000

    def test_respects_max_chars_with_small_budget(self, reader, tmp_path):
        """Budget must hold even for very small values."""
        (tmp_path / "a.py").write_text("x" * 500)
        result = reader.read(tmp_path, max_chars=100)
        assert result["total_chars"] <= 100

    def test_very_small_budget_includes_content(self, reader, tmp_path):
        """Even a tiny budget should include some content."""
        (tmp_path / "a.py").write_text("x" * 500)
        result = reader.read(tmp_path, max_chars=10)
        assert result["total_chars"] <= 10
        assert result["total_files"] == 1
        assert len(result["files"][0]["content"]) == 10

    def test_includes_files_within_budget(self, reader, tmp_path):
        (tmp_path / "a.py").write_text("small")
        (tmp_path / "b.py").write_text("also small")

        result = reader.read(tmp_path, max_chars=100000)
        assert result["total_files"] == 2


class TestDirectoryStructure:
    """Test directory structure tree generation."""

    def test_generates_tree_string(self, reader, sample_dir):
        result = reader.read(sample_dir)
        tree = result["directory_structure"]
        assert isinstance(tree, str)
        assert len(tree) > 0

    def test_tree_contains_directory_names(self, reader, sample_dir):
        result = reader.read(sample_dir)
        tree = result["directory_structure"]
        assert "src" in tree
        assert "docs" in tree

    def test_tree_contains_file_names(self, reader, sample_dir):
        result = reader.read(sample_dir)
        tree = result["directory_structure"]
        assert "README.md" in tree
        assert "main.py" in tree


class TestFileSorting:
    """Test file sorting order: README first, then docs, then source."""

    def test_readme_comes_first(self, reader, sample_dir):
        result = reader.read(sample_dir)
        files = result["files"]
        if files:
            # README should be the first file
            assert "readme" in files[0]["path"].lower()

    def test_docs_before_source(self, reader, sample_dir):
        result = reader.read(sample_dir)
        files = result["files"]
        paths = [f["path"] for f in files]

        # Find the index positions
        doc_indices = [
            i
            for i, p in enumerate(paths)
            if p.endswith(".md") and "readme" not in p.lower()
        ]
        source_indices = [i for i, p in enumerate(paths) if p.endswith(".py")]

        if doc_indices and source_indices:
            # Average doc index should be less than average source index
            assert min(doc_indices) < max(source_indices)


class TestLanguageDetection:
    """Test language detection from file extension."""

    def test_python_detection(self, reader, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        result = reader.read(tmp_path)
        assert result["files"][0]["language"] == "python"

    def test_javascript_detection(self, reader, tmp_path):
        (tmp_path / "app.js").write_text("const x = 1;")
        result = reader.read(tmp_path)
        assert result["files"][0]["language"] == "javascript"

    def test_markdown_detection(self, reader, tmp_path):
        (tmp_path / "doc.md").write_text("# Title")
        result = reader.read(tmp_path)
        assert result["files"][0]["language"] == "markdown"

    def test_yaml_detection(self, reader, tmp_path):
        (tmp_path / "config.yaml").write_text("key: value")
        result = reader.read(tmp_path)
        assert result["files"][0]["language"] == "yaml"

    def test_unknown_extension(self, reader, tmp_path):
        (tmp_path / "data.xyz").write_text("data")
        result = reader.read(tmp_path)
        assert result["files"][0]["language"] == "text"


class TestEmptyDirectory:
    """Test handling of empty directories."""

    def test_empty_directory(self, reader, tmp_path):
        result = reader.read(tmp_path)
        assert result["total_files"] == 0
        assert result["total_chars"] == 0
        assert result["files"] == []
        assert isinstance(result["directory_structure"], str)

    def test_directory_with_only_subdirs(self, reader, tmp_path):
        (tmp_path / "empty_sub").mkdir()
        (tmp_path / "another_sub").mkdir()
        result = reader.read(tmp_path)
        assert result["total_files"] == 0


class TestNestedDirectoryWalking:
    """Test walking deeply nested directories."""

    def test_finds_deeply_nested_files(self, reader, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("# deep file")

        result = reader.read(tmp_path)
        assert result["total_files"] == 1
        assert any("deep.py" in f["path"] for f in result["files"])

    def test_relative_paths_in_result(self, reader, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "module.py").write_text("x = 1")

        result = reader.read(tmp_path)
        # Paths should be relative to source_dir
        path = result["files"][0]["path"]
        assert not path.startswith("/")
        assert path.startswith("src/") or path == "src/module.py"


class TestSourceDirValidation:
    """Test that invalid source_dir raises appropriate errors."""

    def test_nonexistent_dir_raises(self, reader, tmp_path):
        with pytest.raises(FileNotFoundError):
            reader.read(tmp_path / "nonexistent")

    def test_file_as_source_dir_raises(self, reader, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        with pytest.raises(NotADirectoryError):
            reader.read(f)


class TestDirectoryTreeFromFullCandidates:
    """Tree should reflect all filtered files, not just budget-kept."""

    def test_tree_includes_files_beyond_budget(self, reader, tmp_path):
        (tmp_path / "a.py").write_text("x" * 100)
        (tmp_path / "b.py").write_text("y" * 100)

        # Budget only allows one file
        result = reader.read(tmp_path, max_chars=120)
        tree = result["directory_structure"]
        # Both files should appear in the tree even if only one
        # is included in the content bundle.
        assert "a.py" in tree
        assert "b.py" in tree
