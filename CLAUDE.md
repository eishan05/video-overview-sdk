# CLAUDE.md — Instructions for Coding Agents

## What This SDK Does

Generates NotebookLM-style video/audio overviews from source code directories.
Pipeline: read files → LLM script generation → Gemini TTS → Gemini image gen → ffmpeg assembly.

## Prerequisites

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` must be set
- `ffmpeg` on PATH
- `claude` CLI (default) or `codex` CLI installed and authenticated

## Usage

```bash
# Video overview (conversation between Host + Expert)
video-overview ./src -t "authentication system" -o overview.mp4

# Audio-only (single narrator, faster)
video-overview ./docs -t "API reference" -f audio -m narration -o overview.mp3

# With file filters and Codex backend
video-overview ./src -t "data pipeline" -o overview.mp4 -i "*.py" -i "*.yaml" -e "tests/*" --llm codex
```

```python
from video_overview import create_overview

result = create_overview(source_dir="./src", output="overview.mp4", topic="auth system")
```

## When to Use Video vs Audio

- **Video** (`-f video`, default): generates images + assembles MP4. Better for visual topics (architecture, flows). Slower.
- **Audio** (`-f audio`): TTS only, outputs MP3/WAV. Faster, cheaper, good for background listening.

## Topic Selection

Be specific: `"authentication and authorization flow"` not `"everything"`.
Combine with `--include` to focus on relevant files.

## Project Structure

```
src/video_overview/
    config.py            # OverviewConfig, Script, ScriptSegment, OverviewResult
    core.py              # create_overview() orchestrator
    cli.py               # Click CLI
    content/reader.py    # ContentReader — file walking, filtering, .gitignore
    script/generator.py  # ScriptGenerator — LLM subprocess (claude/codex)
    audio/generator.py   # AudioGenerator — Gemini TTS
    visuals/generator.py # VisualGenerator — Gemini image gen (async)
    video/assembler.py   # VideoAssembler — ffmpeg
```

## Dev Commands

```bash
pytest -v                    # run tests (275 total)
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
```
