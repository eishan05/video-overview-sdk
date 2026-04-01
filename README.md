# Video Overview SDK

Generate NotebookLM-style audio and video overviews of a codebase using AI.

The Video Overview SDK reads source code from a directory, generates an
educational script with an LLM (Claude Code or Codex CLI), synthesises speech
with the Gemini TTS API, optionally creates visual assets with Gemini image
generation, and assembles everything into a polished MP4 video or MP3 audio
file.

## Requirements

| Dependency | Version / Notes |
|---|---|
| **Python** | 3.11 or later |
| **ffmpeg** | Must be on `PATH`. Used for audio concatenation, image fallback generation, and video assembly. |
| **Claude Code** or **Codex CLI** | Used as the LLM backend for script generation. At least one must be installed and authenticated. The default backend is Claude Code; pass `--llm codex` (CLI) or `llm_backend="codex"` (API) to use Codex instead. |
| **Gemini API key** | Required for TTS audio and image generation. Set via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). |

## Installation

```bash
# Clone the repository
git clone https://github.com/eishan05/video-overview-sdk.git
cd video-overview-sdk

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package (editable, with dev dependencies)
uv pip install -e ".[dev]"
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from video_overview import create_overview

result = create_overview(
    source_dir="./src",
    output="overview.mp4",
    topic="authentication system",
    format="video",
    mode="conversation",
)

print(f"Created: {result.output_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Segments: {result.segments_count}")
```

You can also construct an `OverviewConfig` object for full control:

```python
from video_overview import OverviewConfig, create_overview

config = OverviewConfig(
    source_dir="./src",
    output="overview.mp3",
    topic="API reference",
    format="audio",
    mode="narration",
    narrator_voice="Kore",
    llm_backend="claude",
    max_duration_minutes=15,
    include=["*.py", "*.md"],
    exclude=["tests/*"],
)

result = create_overview(config=config)
```

### CLI

```bash
# Video overview (conversation mode, default)
video-overview ./src --topic "authentication system" --output overview.mp4

# Audio-only overview (narration mode)
video-overview ./docs --topic "API reference" --format audio --mode narration --output overview.mp3

# With file filters
video-overview ./src \
  --topic "data pipeline" \
  --output overview.mp4 \
  --include "*.py" \
  --include "*.yaml" \
  --exclude "tests/*"

# Using Codex as the LLM backend
video-overview ./src --topic "project overview" --output overview.mp4 --llm codex
```

Run `video-overview --help` for the full list of options.

## Configuration Reference

### `OverviewConfig` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `source_dir` | `Path` | *required* | Directory containing the source code to analyse. |
| `output` | `Path` | *required* | Output file path (e.g. `overview.mp4` or `overview.mp3`). |
| `topic` | `str` | *required* | Topic or title for the generated overview. |
| `include` | `list[str]` | `["*"]` | Glob patterns for files to include. Supports path-based patterns like `src/**/*.py`. |
| `exclude` | `list[str]` | `[]` | Glob patterns for files to exclude. |
| `mode` | `"conversation"` \| `"narration"` | `"conversation"` | Script style. Conversation produces a Host/Expert dialogue; narration uses a single Narrator voice. |
| `format` | `"video"` \| `"audio"` | `"video"` | Output format. Video produces an MP4 with visual slides; audio produces an MP3/WAV. |
| `host_voice` | `str` | `"Aoede"` | Gemini TTS voice for the Host speaker (conversation mode). |
| `expert_voice` | `str` | `"Charon"` | Gemini TTS voice for the Expert speaker (conversation mode). |
| `narrator_voice` | `str` | `"Kore"` | Gemini TTS voice for the Narrator (narration mode). |
| `max_duration_minutes` | `int` | `10` | Maximum target duration in minutes (currently advisory). |
| `llm_backend` | `"claude"` \| `"codex"` | `"claude"` | Which CLI to use for script generation. |
| `cache_dir` | `Path` \| `None` | `<source_dir>/.video_overview_cache` | Directory for intermediate files (audio chunks, images). |

### Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Primary API key for Gemini TTS and image generation. |
| `GOOGLE_API_KEY` | Fallback API key (checked if `GEMINI_API_KEY` is not set). |

### CLI Options

| Option | Short | Description |
|---|---|---|
| `SOURCE_DIR` | | Positional argument: path to the source directory. |
| `--topic` | `-t` | Topic of the overview (required). |
| `--output` | `-o` | Output file path (required). |
| `--include` | `-i` | Glob patterns for files to include (repeatable). |
| `--exclude` | `-e` | Glob patterns for files to exclude (repeatable). |
| `--mode` | `-m` | `conversation` or `narration`. |
| `--format` | `-f` | `video` or `audio`. |
| `--host-voice` | | Voice for the Host speaker. |
| `--expert-voice` | | Voice for the Expert speaker. |
| `--narrator-voice` | | Voice for the Narrator. |
| `--llm` | | LLM backend: `claude` or `codex`. |
| `--max-duration` | | Maximum duration in minutes. |
| `--version` | | Show version and exit. |
| `--help` | | Show help and exit. |

## Architecture Overview

```
source_dir/
    |
    v
+----------------+     +------------------+     +------------------+
| ContentReader  | --> | ScriptGenerator  | --> |  AudioGenerator  |
| (read files,   |     | (LLM via Claude  |     |  (Gemini TTS,    |
|  filter, sort)  |     |  or Codex CLI)   |     |   chunking,      |
+----------------+     +------------------+     |   retry logic)   |
                                                 +------------------+
                                                         |
                                +------------------------+--------+
                                |                                  |
                                v (concurrent)                     v (concurrent)
                       +------------------+              +------------------+
                       | VisualGenerator  |              |  (audio output)  |
                       | (Gemini image    |              +------------------+
                       |  generation,     |                        |
                       |  fallback via    |                        v
                       |  ffmpeg)         |              +------------------+
                       +------------------+              | VideoAssembler   |
                                |                        | (ffmpeg: Ken     |
                                +----------------------> |  Burns effect,   |
                                                         |  xfade, H.264)  |
                                                         +------------------+
                                                                  |
                                                                  v
                                                           output.mp4 / .mp3
```

### Pipeline Steps

1. **ContentReader** -- Walks the source directory, respects `.gitignore`,
   applies include/exclude filters, detects languages, truncates large files,
   and enforces a character budget.

2. **ScriptGenerator** -- Sends the content bundle to an LLM (Claude Code or
   Codex CLI) with a structured prompt requesting a JSON script. Validates the
   response against a Pydantic schema.

3. **AudioGenerator** -- Converts the script into speech using the Gemini TTS
   API (`gemini-2.5-flash-preview-tts`). Supports multi-speaker conversation
   and single-speaker narration. Handles chunking, PCM-to-WAV conversion,
   retry with exponential backoff, and ffmpeg concatenation.

4. **VisualGenerator** -- Generates 16:9 images for each script segment using
   Gemini image generation (`gemini-2.0-flash-exp`). Includes caching,
   concurrent generation with a semaphore, and fallback to ffmpeg text slides.

5. **VideoAssembler** -- Combines audio and images into an MP4 video using
   ffmpeg. Applies Ken Burns zoom effect, crossfade transitions between
   segments, H.264/AAC encoding at 1920x1080 / 30fps. Also handles audio-only
   conversion (WAV to MP3).

## Audio-Only Mode

To generate an audio-only overview (no images or video assembly), set
`format="audio"`:

```bash
video-overview ./src --topic "project overview" --format audio --output overview.mp3
```

```python
result = create_overview(
    source_dir="./src",
    output="overview.mp3",
    topic="project overview",
    format="audio",
)
```

Audio mode skips visual generation and video assembly entirely. If the output
path ends in `.wav`, the raw audio is copied directly; otherwise ffmpeg
converts to the requested format (e.g. MP3).

## License

MIT -- see [LICENSE](LICENSE).
