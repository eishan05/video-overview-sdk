# CLAUDE.md -- Instructions for Coding Agents

This file provides guidance for Claude Code, Codex, and other coding agents
on how to use the Video Overview SDK.

## What This SDK Does

The Video Overview SDK generates NotebookLM-style audio and video overviews
from source code directories. It reads files, generates an educational script
with an LLM, synthesises speech via Gemini TTS, optionally creates visual
slides, and assembles everything into an MP4 or MP3.

## Prerequisites

- Python 3.11+
- `ffmpeg` on PATH
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable must be set
- Either `claude` (Claude Code CLI) or `codex` (Codex CLI) must be installed
  and authenticated. The default backend is `claude`; pass `--llm codex` to
  use Codex instead.

## Example Commands

### Video overview (conversation between Host and Expert)

```bash
video-overview ./src --topic "authentication system" --format video -o overview.mp4
```

### Audio-only overview (single narrator)

```bash
video-overview ./docs --topic "API reference" --format audio --mode narration -o overview.mp3
```

### With file filters

```bash
video-overview ./src \
  --topic "data pipeline" \
  -o overview.mp4 \
  --include "*.py" \
  --include "*.yaml" \
  --exclude "tests/*"
```

### Using Codex as the LLM backend

```bash
video-overview ./src --topic "project architecture" --llm codex -o overview.mp4
```

## When to Use Video vs Audio Mode

| Use Case | Recommended Mode |
|---|---|
| Presenting to stakeholders, demos, onboarding videos | `--format video` |
| Background listening, podcast-style consumption | `--format audio` |
| Quick turnaround (no image generation latency) | `--format audio` |
| Visual diagrams help explain the code (architecture, flows) | `--format video` |
| Limited API quota (image generation uses extra API calls) | `--format audio` |

**Video mode** (`--format video`, default) generates images for each script
segment and assembles them with Ken Burns zoom effects and crossfade
transitions. This takes longer and uses more API calls.

**Audio mode** (`--format audio`) skips image generation entirely and produces
an MP3 or WAV file. Faster and cheaper.

## How to Pick Good Topics

The `--topic` flag tells the LLM what aspect of the codebase to focus on.
Specific, scoped topics produce better results than vague ones.

**Good topics:**
- `"authentication and authorization flow"`
- `"database schema and migration strategy"`
- `"REST API endpoint design"`
- `"error handling patterns"`
- `"CI/CD pipeline configuration"`

**Less effective topics:**
- `"everything"` (too broad, script will be shallow)
- `"code"` (too vague)
- `"this project"` (no focus)

If the codebase is large, combine `--topic` with `--include` to focus on
relevant files:

```bash
video-overview ./src --topic "payment processing" --include "payments/*.py" --include "billing/*.py" -o payments.mp4
```

## Conversation vs Narration Mode

- `--mode conversation` (default): Two speakers (Host asks questions, Expert
  explains). Best for engaging, dialogue-style content.
- `--mode narration`: Single Narrator voice. Best for straightforward,
  documentary-style presentations.

## Python API Usage

```python
from video_overview import create_overview

result = create_overview(
    source_dir="./src",
    output="overview.mp4",
    topic="authentication system",
)
print(result.output_path, result.duration_seconds, result.segments_count)
```

## Project Structure

```
src/video_overview/
    __init__.py          # Public API exports
    config.py            # OverviewConfig, Script, ScriptSegment, OverviewResult
    core.py              # create_overview() orchestrator
    cli.py               # Click CLI entry point
    content/reader.py    # ContentReader (file walking, filtering)
    script/generator.py  # ScriptGenerator (LLM subprocess calls)
    audio/generator.py   # AudioGenerator (Gemini TTS)
    visuals/generator.py # VisualGenerator (Gemini image generation)
    video/assembler.py   # VideoAssembler (ffmpeg video/audio output)
```

## Running Tests

```bash
pytest -v
```

## Linting and Formatting

```bash
ruff check src/ tests/
ruff format src/ tests/
```
