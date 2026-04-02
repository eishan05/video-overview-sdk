# Video Overview SDK

Generate NotebookLM-style audio and video overviews from codebases and documentation using AI.

The SDK reads source files from a directory, generates an educational script via an LLM (Claude Code or Codex CLI), synthesizes speech with Gemini TTS, optionally generates visual slides with Gemini image generation (Nano Banana), and assembles everything into an MP4 video or MP3/WAV audio file.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python API Reference](#python-api-reference)
  - [create_overview()](#create_overview)
  - [OverviewConfig](#overviewconfig)
  - [OverviewResult](#overviewresult)
  - [Script and ScriptSegment](#script-and-scriptsegment)
- [CLI Reference](#cli-reference)
- [Component APIs](#component-apis)
  - [ContentReader](#contentreader)
  - [ScriptGenerator](#scriptgenerator)
  - [AudioGenerator](#audiogenerator)
  - [VisualGenerator](#visualgenerator)
  - [VideoAssembler](#videoassembler)
- [Environment Variables](#environment-variables)
- [Architecture](#architecture)
- [Available Voices](#available-voices)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Requirements

| Dependency | Notes |
|---|---|
| Python | 3.11+ |
| ffmpeg | Must be on `PATH`. Used for audio concatenation, fallback image generation, and video assembly. |
| Claude Code or Codex CLI | LLM backend for script generation. At least one must be installed and authenticated. |
| Gemini API key | Required for TTS and image generation. Set via `GEMINI_API_KEY` or `GOOGLE_API_KEY`. |

## Installation

```bash
git clone https://github.com/eishan05/video-overview-sdk.git
cd video-overview-sdk

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set your API key in your shell:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Or if you use `direnv` or similar tooling, copy the template:

```bash
cp .env.example .env
# Edit .env and add your key — note: the SDK reads from the process
# environment directly, so you need direnv, dotenv-cli, or similar
# to load .env files automatically.
```

---

## Quick Start

### CLI

```bash
# Video overview with two-speaker conversation
video-overview ./src -t "authentication system" -o overview.mp4

# Audio-only narration
video-overview ./docs -t "API reference" -f audio -m narration -o overview.mp3
```

### Python

```python
from video_overview import create_overview

result = create_overview(
    source_dir="./src",
    output="overview.mp4",
    topic="authentication system",
)
print(f"Created {result.output_path} — {result.duration_seconds:.1f}s, {result.segments_count} segments")
```

---

## Python API Reference

### `create_overview()`

```python
from video_overview import create_overview, OverviewConfig, OverviewResult

# Option 1: keyword arguments
result: OverviewResult = create_overview(
    source_dir="./src",
    output="overview.mp4",
    topic="authentication system",
    include=["*.py", "*.md"],
    exclude=["tests/*"],
    mode="conversation",       # "conversation" or "narration"
    format="video",            # "video" or "audio"
    host_voice="Aoede",        # conversation mode: Host voice
    expert_voice="Charon",     # conversation mode: Expert voice
    narrator_voice="Kore",     # narration mode: Narrator voice
    llm_backend="claude",      # "claude" or "codex"
    max_duration_minutes=10,
)

# Option 2: config object
config = OverviewConfig(
    source_dir="./src",
    output="overview.mp3",
    topic="API endpoints",
    format="audio",
    mode="narration",
)
result = create_overview(config=config)
```

**Pipeline:** The function runs these steps sequentially, printing progress to stderr:

1. Validate config and check for `GEMINI_API_KEY` / `GOOGLE_API_KEY`
2. Read source files via `ContentReader`
3. Generate script via `ScriptGenerator` (calls Claude or Codex CLI)
4. Generate audio (Gemini TTS) and visuals (Gemini image gen) concurrently
5. Assemble final output via `VideoAssembler` (ffmpeg)

In audio-only mode (`format="audio"`), visual generation is skipped. For `.wav` output, the final conversion step is skipped (audio copied directly). For `.mp3` output, `VideoAssembler` converts via ffmpeg. Note: ffmpeg is still required in all modes because `AudioGenerator` uses it internally to concatenate multi-chunk audio.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `OverviewConfig \| None` | `None` | Config object. If provided, keyword args are ignored. |
| `**kwargs` | | | Forwarded to `OverviewConfig()` if `config` is `None`. |

**Returns:** `OverviewResult`

**Raises:** `ValueError` if no API key is set. Sub-component errors (`ScriptGenerationError`, `AudioGenerationError`, `VisualGenerationError`, `VideoAssemblyError`) propagate unchanged.

---

### `OverviewConfig`

Pydantic model for all configuration. Extra fields are forbidden (`extra="forbid"`).

```python
from video_overview import OverviewConfig

config = OverviewConfig(
    source_dir="./my-project",      # must exist, must be a directory
    output="overview.mp4",          # parent dir must exist, must not be an existing directory
    topic="database models",
    include=["*.py"],               # glob patterns — supports path-based: "src/**/*.py"
    exclude=["*_test.py"],          # glob patterns
    mode="conversation",            # "conversation" (Host + Expert) or "narration" (Narrator)
    format="video",                 # "video" (MP4) or "audio" (MP3/WAV)
    host_voice="Aoede",             # Gemini TTS voice for Host
    expert_voice="Charon",          # Gemini TTS voice for Expert
    narrator_voice="Kore",          # Gemini TTS voice for Narrator
    max_duration_minutes=10,        # advisory, not yet enforced
    llm_backend="claude",           # "claude" or "codex"
    cache_dir=None,                 # defaults to <source_dir>/.video_overview_cache/
)
```

| Field | Type | Default | Validation |
|---|---|---|---|
| `source_dir` | `Path` | *required* | Must exist, must be a directory |
| `output` | `Path` | *required* | Parent must exist. Must be `.mp4` for video format, `.mp3` or `.wav` for audio format. |
| `topic` | `str` | *required* | |
| `include` | `list[str]` | `["*"]` | Glob patterns. `"*.py"` matches basename; `"src/*.py"` matches path. |
| `exclude` | `list[str]` | `[]` | Same pattern syntax as `include` |
| `mode` | `Literal` | `"conversation"` | `"conversation"` or `"narration"` |
| `format` | `Literal` | `"video"` | `"video"` or `"audio"` |
| `host_voice` | `str` | `"Aoede"` | Any Gemini TTS voice name |
| `expert_voice` | `str` | `"Charon"` | Any Gemini TTS voice name |
| `narrator_voice` | `str` | `"Kore"` | Any Gemini TTS voice name |
| `max_duration_minutes` | `PositiveInt` | `10` | Must be >= 1 |
| `llm_backend` | `Literal` | `"claude"` | `"claude"` or `"codex"` |
| `cache_dir` | `Path \| None` | `None` | Auto-set to `<source_dir>/.video_overview_cache/` |

**Property:**

```python
config.gemini_api_key  # -> str | None
# Reads GEMINI_API_KEY env var, falls back to GOOGLE_API_KEY
```

---

### `OverviewResult`

Returned by `create_overview()`.

```python
result.output_path      # Path — the generated file
result.duration_seconds  # float — total estimated duration
result.segments_count    # int — number of script segments
```

---

### Script and ScriptSegment

Data models for the generated script. Typically you don't construct these directly — they're produced by `ScriptGenerator` and consumed by the pipeline.

```python
from video_overview import Script, ScriptSegment

segment = ScriptSegment(
    speaker="Host",                    # "Host", "Expert", or "Narrator"
    text="Welcome to our overview!",   # what the speaker says
    visual_prompt="A welcome screen showing the project logo",  # image gen prompt
)

script = Script(
    title="Authentication Deep Dive",
    segments=[segment, ...],
)
```

---

## CLI Reference

```
video-overview SOURCE_DIR [OPTIONS]
```

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `SOURCE_DIR` | | path | *required* | Source directory (must exist) |
| `--topic` | `-t` | string | *required* | Topic for the overview |
| `--output` | `-o` | path | *required* | Output file path (`.mp4` for video, `.mp3`/`.wav` for audio) |
| `--include` | `-i` | string (repeatable) | `*` | File include patterns |
| `--exclude` | `-e` | string (repeatable) | | File exclude patterns |
| `--mode` | `-m` | choice | `conversation` | `conversation` or `narration` |
| `--format` | `-f` | choice | `video` | `video` or `audio` |
| `--host-voice` | | string | `Aoede` | Host speaker voice |
| `--expert-voice` | | string | `Charon` | Expert speaker voice |
| `--narrator-voice` | | string | `Kore` | Narrator voice |
| `--llm` | | choice | `claude` | `claude` or `codex` |
| `--max-duration` | | int (>= 1) | `10` | Max duration in minutes |
| `--version` | | | | Print version |
| `--help` | | | | Print help |

**Exit codes:** 0 on success, 1 on runtime error, 2 on invalid arguments/options (Click usage error), 130 on keyboard interrupt.

---

## Component APIs

Each pipeline stage is a standalone class that can be used independently.

### ContentReader

Reads and filters files from a directory into a structured content bundle.

```python
from video_overview.content import ContentReader

reader = ContentReader()
bundle = reader.read(
    source_dir="./src",
    include=["*.py", "*.md"],     # optional, default includes all
    exclude=["tests/*"],           # optional
    max_chars=100_000,             # total character budget (default 100K)
)

bundle["directory_structure"]  # str — indented tree of included files
bundle["files"]                # list[dict] — each has "path", "content", "language"
bundle["total_files"]          # int
bundle["total_chars"]          # int
```

**Behavior:**
- Respects `.gitignore` in the source directory (via `pathspec`)
- Skips binary files (null byte detection) and known binary extensions (.png, .pdf, .so, etc.)
- Skips common non-text directories (`__pycache__`, `.git`, `node_modules`, `.venv`, etc.)
- Truncates individual files at 2,000 chars with a `[truncated]` note
- Enforces total `max_chars` budget across all files
- Sorts by relevance: README first, then docs, config files, then source
- Detects language from 60+ file extensions

---

### ScriptGenerator

Generates a conversational or narration script by calling an LLM CLI.

```python
from video_overview.script import ScriptGenerator

gen = ScriptGenerator()
script = gen.generate(
    content_bundle=bundle,         # from ContentReader
    topic="authentication system",
    mode="conversation",           # "conversation" or "narration"
    llm_backend="claude",          # "claude" or "codex"
    max_segments=20,               # max segments in output
)

script.title      # str
script.segments   # list[ScriptSegment]
```

**LLM invocation:**
- Claude: `claude -p "<prompt>" --output-format json`
- Codex: `codex exec "<prompt>"`
- Timeout: 300 seconds

**Validation:** The response must be valid JSON matching the `Script` schema. Speaker names must match the mode (`Host`/`Expert` for conversation, `Narrator` for narration). Segment count must not exceed `max_segments`.

**Raises:** `ScriptGenerationError` on subprocess failure, timeout, JSON parse error, validation error, or invalid speakers.

---

### AudioGenerator

Generates speech audio from a script using Gemini 2.5 Flash TTS.

```python
from video_overview.audio import AudioGenerator

gen = AudioGenerator(api_key="your-gemini-key")
audio_path, segment_durations = gen.generate(
    script=script,
    host_voice="Aoede",
    expert_voice="Charon",
    narrator_voice="Kore",
    cache_dir=Path(".cache"),
)

audio_path          # Path — final WAV file
segment_durations   # list[float] — estimated seconds per segment
```

**Behavior:**
- Model: `gemini-2.5-flash-preview-tts`
- Conversation mode: uses `MultiSpeakerVoiceConfig` with Host + Expert voices
- Narration mode: uses single `VoiceConfig` with Narrator voice
- Chunks segments into batches of ~13 per API call
- Concatenates chunks via ffmpeg (`ffmpeg -f concat`)
- Handles PCM-to-WAV conversion when API returns `audio/pcm` mime type
- Retry: 3 attempts with exponential backoff (1s, 2s delays)
- Duration estimation: ~12.5 chars/second (min 0.5s per segment)
- Output: WAV at 24kHz

**Raises:** `AudioGenerationError`

---

### VisualGenerator

Generates images for each script segment using Gemini image generation.

```python
from video_overview.visuals import VisualGenerator

gen = VisualGenerator(api_key="your-gemini-key")
# Note: generate() is async
image_paths = await gen.generate(
    script=script,
    cache_dir=Path(".cache"),
)

image_paths  # list[Path] — one PNG per segment, ordered
```

**Behavior:**
- Model: `gemini-2.0-flash-exp` (configurable via `model` kwarg in constructor)
- Generates 16:9 images with `response_modalities=["IMAGE", "TEXT"]`
- Up to 3 concurrent API calls (semaphore-limited)
- Caches by MD5 hash of `visual_prompt` — identical prompts reuse cached images
- Per-prompt locking prevents duplicate API calls for the same prompt
- **Fallback:** on API failure, creates a 1920x1080 dark background (#1a1a2e) with white centered text via ffmpeg
- Failed prompts are tracked to skip redundant API calls

**Raises:** `VisualGenerationError`

---

### VideoAssembler

Combines audio and images into final video or converts audio format using ffmpeg.

```python
from video_overview.video import VideoAssembler

assembler = VideoAssembler()  # validates ffmpeg is on PATH

# Video mode
output = assembler.assemble(
    audio_path=Path("audio.wav"),
    image_paths=[Path("img1.png"), Path("img2.png")],
    segment_durations=[5.0, 8.0],
    output_path=Path("output.mp4"),
    format="video",
)

# Audio mode (WAV to MP3 conversion)
output = assembler.assemble(
    audio_path=Path("audio.wav"),
    image_paths=[],
    segment_durations=[],
    output_path=Path("output.mp3"),
    format="audio",
)
```

**Video output specs:**
- H.264 video + AAC audio in MP4 container
- 1920x1080 resolution at 30fps
- Ken Burns effect: 5% slow zoom on each image (via `zoompan` filter)
- 0.5s crossfade transitions between segments (via `xfade` filter)
- Images are scaled and padded to maintain aspect ratio

**Audio mode:** Converts WAV to MP3 via `libmp3lame` (quality 2). WAV-to-WAV copies directly.

**Duration estimation helper:**

```python
durations = assembler._estimate_segment_durations(
    segments=[seg1, seg2, seg3],   # list[ScriptSegment]
    total_audio_duration=30.0,      # total seconds to distribute
)
# Returns proportional durations based on text length, summing to total
```

**Raises:** `VideoAssemblyError`

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | API key for Gemini TTS and image generation (primary) |
| `GOOGLE_API_KEY` | Fallback API key if `GEMINI_API_KEY` is not set |

The SDK reads these via `OverviewConfig.gemini_api_key` at runtime. No `.env` auto-loading — set them in your shell or use a tool like `direnv`.

---

## Architecture

```
source_dir/
    │
    ▼
┌────────────────┐     ┌──────────────────┐
│ ContentReader   │────▶│ ScriptGenerator   │
│ read files,     │     │ LLM via Claude    │
│ filter, sort    │     │ or Codex CLI      │
└────────────────┘     └──────────────────┘
                               │
                       ┌───────┴────────┐
                       ▼                ▼
              ┌────────────────┐ ┌────────────────┐
              │ AudioGenerator │ │ VisualGenerator │
              │ Gemini TTS     │ │ Gemini image    │
              │ (concurrent)   │ │ (concurrent)    │
              └────────────────┘ └────────────────┘
                       │                │
                       └───────┬────────┘
                               ▼
                       ┌──────────────────┐
                       │ VideoAssembler    │
                       │ ffmpeg: Ken Burns │
                       │ xfade, H.264/AAC │
                       └──────────────────┘
                               │
                               ▼
                        output.mp4 / .mp3
```

Audio and visual generation run concurrently via `asyncio`. In audio-only mode, visual generation is skipped. For `.wav` output the final conversion is skipped; for `.mp3`, `VideoAssembler` converts via ffmpeg. Note that ffmpeg is always required because `AudioGenerator` uses it to concatenate multi-chunk TTS output. Only `.mp3` and `.wav` are valid audio output formats.

---

## Available Voices

Gemini TTS voices for `host_voice`, `expert_voice`, and `narrator_voice`:

| Voice | Character |
|---|---|
| **Aoede** | Breezy (default Host) |
| **Charon** | Informative (default Expert) |
| **Kore** | Firm (default Narrator) |
| **Puck** | Upbeat |
| **Zephyr** | Bright |
| **Fenrir** | Excitable |
| **Leda** | Youthful |
| **Orus** | Firm |

---

## Error Handling

Each component raises a specific exception:

| Exception | Source | Common Causes |
|---|---|---|
| `ScriptGenerationError` | `ScriptGenerator` | LLM CLI not found, timeout, invalid JSON response |
| `AudioGenerationError` | `AudioGenerator` | Missing API key, Gemini API error, ffmpeg not found |
| `VisualGenerationError` | `VisualGenerator` | Missing API key, Gemini API error, ffmpeg fallback failure |
| `VideoAssemblyError` | `VideoAssembler` | ffmpeg not found, invalid durations, encoding failure |
| `ValueError` | `create_overview` | Missing API key, invalid config values |

Import exceptions from their subpackages:

```python
from video_overview.script import ScriptGenerationError
from video_overview.audio import AudioGenerationError
from video_overview.visuals import VisualGenerationError
from video_overview.video import VideoAssemblyError
```

---

## Examples

### Video overview of a Python project

```bash
video-overview ./my-project \
  -t "how the REST API works" \
  -o api-overview.mp4 \
  -i "*.py" -i "*.md" \
  -e "tests/*" -e "migrations/*"
```

### Audio podcast from documentation

```bash
video-overview ./docs \
  -t "getting started guide" \
  -f audio \
  -m narration \
  --narrator-voice Puck \
  -o getting-started.mp3
```

### Programmatic usage with full config

```python
from pathlib import Path
from video_overview import create_overview, OverviewConfig

config = OverviewConfig(
    source_dir=Path("./backend"),
    output=Path("./output/backend-overview.mp4"),
    topic="payment processing pipeline",
    include=["payments/**/*.py", "billing/**/*.py"],
    exclude=["*_test.py", "__pycache__"],
    mode="conversation",
    format="video",
    host_voice="Zephyr",
    expert_voice="Charon",
    llm_backend="claude",
    max_duration_minutes=15,
)

result = create_overview(config=config)
print(f"Output: {result.output_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Segments: {result.segments_count}")
```

### Using individual components

```python
from pathlib import Path
from video_overview.content import ContentReader
from video_overview.script import ScriptGenerator

# Read content
reader = ContentReader()
bundle = reader.read("./src", include=["*.py"], max_chars=50_000)

print(f"Found {bundle['total_files']} files, {bundle['total_chars']} chars")
print(bundle["directory_structure"])

# Generate script only (no audio/video)
gen = ScriptGenerator()
script = gen.generate(
    content_bundle=bundle,
    topic="error handling",
    mode="narration",
    llm_backend="claude",
)

for seg in script.segments:
    print(f"[{seg.speaker}] {seg.text[:80]}...")
```

---

## License

MIT — see [LICENSE](LICENSE).
