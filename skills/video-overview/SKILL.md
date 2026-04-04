---
name: video-overview
description: Use when generating NotebookLM-style audio or video overviews from a source code or docs directory with the video-overview CLI or Python SDK, including choosing conversation vs narration, video vs audio, include/exclude globs, Gemini voices, and fixing generation failures. Not for editing existing media, cutting footage, or general video production.
---

# Video Overview Generation

Generate educational audio/video overviews from codebases and documentation. The SDK reads source files, generates a script via an LLM, synthesizes speech with Gemini TTS, optionally generates images, and assembles the final output with ffmpeg.

**This skill is NOT for:** editing existing videos, cutting footage, transcoding media, or general video production. It only generates new overviews from source content.

## Preflight

Before generating anything, verify prerequisites. See [references/preflight.md](references/preflight.md) for exact checks.

Required: `GEMINI_API_KEY` or `GOOGLE_API_KEY` env var, `ffmpeg` on PATH (required for all video output and most audio runs; only a single-batch WAV output can succeed without it), and `claude` or `codex` CLI installed and authenticated.

## Decision Rules

**CLI vs Python API:** Default to the CLI for one-off generation. Use `create_overview()` only when embedding in a Python script, test, or pipeline. See [references/cli.md](references/cli.md) or [references/python-api.md](references/python-api.md).

**Format:**
- `video` (MP4) -- architecture walkthroughs, visual topics. Slower, requires image gen.
- `audio` (MP3/WAV) -- concise explainers, docs, background listening. Faster and cheaper.

**Mode:**
- `conversation` -- Host + Expert dialogue. Best for deep dives and complex topics.
- `narration` -- Single narrator. Best for summaries, docs, and quick overviews.

See [references/modes-formats-voices.md](references/modes-formats-voices.md) for voice selection.

## Workflow

1. **Preflight** -- confirm prerequisites are available
2. **Scope files** -- use `--include` / `--exclude` aggressively; quality drops when you feed too much content
3. **Pick a specific topic** -- "how auth flows from middleware to JWT issuance" not "authentication"
4. **Choose mode + format** -- see decision rules above
5. **Run generation:**
   ```bash
   video-overview ./src -t "your specific topic" -o output.mp4
   ```
6. **Verify output** -- check file exists and duration is reasonable

## Quick Examples

```bash
# Video conversation (default)
video-overview ./src -t "authentication system" -o overview.mp4

# Audio narration (faster)
video-overview ./docs -t "API reference" -f audio -m narration -o overview.mp3

# Scoped with filters
video-overview ./src -t "data pipeline" -o overview.mp4 -i "*.py" -e "tests/*"

# Skip image generation (faster video with static frame)
video-overview ./src -t "overview" -o output.mp4 --skip-visuals
```

```python
from video_overview import create_overview
result = create_overview(source_dir="./src", output="overview.mp4", topic="auth system")
```

## Common Mistakes

- Using `--skip-visuals` with audio format (only valid for video)
- Mismatching extension and format: `.mp4` requires `--format video`, `.mp3`/`.wav` requires `--format audio`
- Vague topics or unscoped repos produce generic, low-quality scripts
- Over-restrictive `--include` patterns that match no files (basename `"*.py"` vs path-based `"src/*.py"`)
- Forgetting `ffmpeg` is required for all video and most audio runs

## References

- [CLI flags and recipes](references/cli.md)
- [Python API details](references/python-api.md)
- [Preflight checks](references/preflight.md)
- [Modes, formats, and voices](references/modes-formats-voices.md)
- [Pipeline internals and caching](references/pipeline.md)
- [Troubleshooting](references/troubleshooting.md)
