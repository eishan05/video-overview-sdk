# Video Overview SDK - Design Spec

## Purpose

A Python SDK that creates informative video overviews (NotebookLM-style) from code and documentation directories. Designed to be used by coding agents (Claude Code, Codex) to generate educational video/audio content that helps developers learn about codebases.

## Architecture Overview

```
Source Files → Content Reader → Script Generator (Claude/Codex CLI)
                                       ↓
                              Conversational Script
                                    ↓         ↓
                            Audio Generator   Visual Generator
                            (Gemini TTS)      (Nano Banana)
                                    ↓         ↓
                              Video Assembler (ffmpeg)
                                       ↓
                                Output (MP4/MP3/WAV)
```

## Core Components

### 1. Content Reader (`video_overview.content.reader`)

Reads and filters source files from a directory to prepare context for script generation.

- Accept `source_dir`, `include_patterns` (globs), `exclude_patterns`
- Read files respecting `.gitignore`
- Truncate very large files, summarize directory structure
- Return structured content bundle with file paths, contents, and metadata
- Max context size configurable (default ~100K chars to fit in LLM context)

### 2. Script Generator (`video_overview.script.generator`)

Uses Claude CLI or Codex CLI to generate a conversational script from source content.

- Accept content bundle from Content Reader
- Generate a multi-speaker conversational script (2 speakers: "Host" and "Expert")
- Script format: JSON array of segments, each with `speaker`, `text`, and `visual_prompt`
- `visual_prompt` describes what image to generate for that segment
- Support single-speaker mode (narration) and multi-speaker mode (conversation)
- Uses `claude` CLI via subprocess (preferred) or `codex` CLI as fallback
- Prompt engineering for NotebookLM-style: engaging, educational, conversational

Script segment format:
```json
{
  "segments": [
    {
      "speaker": "Host",
      "text": "Welcome! Today we're diving into...",
      "visual_prompt": "A clean diagram showing the authentication flow with login, token generation, and session management steps"
    }
  ]
}
```

### 3. Audio Generator (`video_overview.audio.generator`)

Uses Gemini 2.5 Flash TTS API to generate audio from the script.

- Uses `google-genai` SDK with model `gemini-2.5-flash-preview-tts`
- Multi-speaker config: Host=Aoede, Expert=Charon (configurable)
- Chunks script into batches of ~10-15 segments to stay within API limits
- Each chunk generates a WAV file, then concatenates all chunks
- Output: single WAV/MP3 file with timing metadata (segment start/end times)
- Handles PCM→WAV conversion if needed
- Rate limiting and retry logic

### 4. Visual Generator (`video_overview.visuals.generator`)

Uses Gemini image generation (Nano Banana) to create visuals for each segment.

- Uses `google-genai` SDK with model `gemini-2.0-flash-exp` (or latest image-capable model)
- Generates one image per visual_prompt in the script
- Output: 16:9 aspect ratio images at 1920x1080 (or closest supported)
- Parallel generation with concurrency limit (default 3)
- Caches generated images to avoid re-generation
- Falls back to simple text-on-background slides if API fails

### 5. Video Assembler (`video_overview.video.assembler`)

Combines audio and visuals into a final video using ffmpeg.

- Uses ffmpeg via subprocess (no heavy Python video libs needed)
- Creates a slideshow-style video: each segment's image shown while its audio plays
- Smooth crossfade transitions between images (0.5s)
- Adds subtle Ken Burns effect (slow zoom/pan) on images to avoid static feel
- Output formats: MP4 (H.264 + AAC), configurable resolution
- For audio-only mode: just outputs the concatenated audio as MP3/WAV

### 6. Core Orchestrator (`video_overview.core`)

Main entry point that coordinates the pipeline.

```python
from video_overview import create_overview, OverviewConfig

# Full video
result = create_overview(
    source_dir="./my_project",
    output="overview.mp4",
    topic="the authentication system",
    include=["*.py", "*.md"],
    mode="conversation",  # or "narration"
)

# Audio only
result = create_overview(
    source_dir="./my_project", 
    output="overview.mp3",
    topic="API endpoints",
    format="audio",
)

# With config object for full control
config = OverviewConfig(
    source_dir="./my_project",
    output="overview.mp4",
    topic="database models",
    include=["*.py"],
    exclude=["*_test.py", "__pycache__"],
    mode="conversation",  # "conversation" or "narration"
    format="video",  # "video" or "audio"
    host_voice="Aoede",
    expert_voice="Charon",
    max_duration_minutes=10,
    llm_backend="claude",  # "claude" or "codex"
)
result = create_overview(config=config)
```

### 7. CLI (`video_overview.cli`)

Simple CLI wrapper for use from terminal or by coding agents.

```bash
# Basic usage
video-overview ./my_project --topic "auth system" -o overview.mp4

# Audio only
video-overview ./my_project --topic "API layer" --format audio -o overview.mp3

# With options
video-overview ./my_project \
  --topic "database models" \
  --include "*.py" "*.md" \
  --exclude "*_test.py" \
  --mode conversation \
  --llm claude \
  -o overview.mp4
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` — Required for TTS and image generation
- `VIDEO_OVERVIEW_LLM` — Default LLM backend ("claude" or "codex")
- `VIDEO_OVERVIEW_CACHE_DIR` — Cache directory for generated assets (default: `.video_overview_cache/`)

### API Key Requirements
- Single Google/Gemini API key for both TTS and image generation (same SDK)

## Project Setup

- Python 3.11+ (supports 3.11, 3.12, 3.13)
- Package manager: `uv` with venv
- Build system: `hatchling`
- Dependencies:
  - `google-genai>=1.0.0` — Gemini API (TTS + image generation)
  - `click>=8.0` — CLI framework
  - `pydantic>=2.0` — Config validation
- Dev dependencies:
  - `pytest`, `pytest-asyncio`, `pytest-mock`
  - `ruff` — Linting/formatting

## Error Handling

- Clear error messages when API keys are missing
- Graceful degradation: if image gen fails, use text slides; if TTS fails, report clearly
- Progress reporting via stderr (for coding agent consumption)
- All temp files cleaned up on completion (or on error)

## Testing Strategy

- Unit tests with mocked API calls for each component
- Integration test that runs the full pipeline with real API calls (marked as slow/requires-api)
- Test fixtures with sample content bundles and script segments

## File Structure

```
video-overview-sdk/
├── pyproject.toml
├── README.md
├── src/
│   └── video_overview/
│       ├── __init__.py          # Public API exports
│       ├── cli.py               # Click CLI
│       ├── core.py              # Orchestrator
│       ├── config.py            # OverviewConfig pydantic model
│       ├── content/
│       │   ├── __init__.py
│       │   └── reader.py        # Directory content reading
│       ├── script/
│       │   ├── __init__.py
│       │   └── generator.py     # LLM script generation
│       ├── audio/
│       │   ├── __init__.py
│       │   └── generator.py     # Gemini TTS
│       ├── visuals/
│       │   ├── __init__.py
│       │   └── generator.py     # Nano Banana image gen
│       └── video/
│           ├── __init__.py
│           └── assembler.py     # ffmpeg video assembly
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_content_reader.py
│   ├── test_script_generator.py
│   ├── test_audio_generator.py
│   ├── test_visual_generator.py
│   ├── test_video_assembler.py
│   └── test_core.py
└── docs/
```
