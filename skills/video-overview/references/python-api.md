# Python API Reference

Use the Python API when embedding overview generation in scripts, tests, or pipelines. For one-off generation, prefer the CLI.

## create_overview()

```python
from video_overview import create_overview, OverviewConfig, OverviewResult

# Option 1: keyword arguments
result: OverviewResult = create_overview(
    source_dir="./src",
    output="overview.mp4",
    topic="authentication system",
    include=["*.py", "*.md"],
    exclude=["tests/*"],
    mode="conversation",
    format="video",
    host_voice="Aoede",
    expert_voice="Charon",
    max_duration_minutes=10,
    llm_backend="claude",
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

When `config` is provided, keyword arguments are ignored.

## OverviewConfig Fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `source_dir` | `Path` | *required* | Must exist, must be a directory |
| `output` | `Path` | *required* | `.mp4` for video, `.mp3`/`.wav` for audio. Parent must exist. |
| `topic` | `str` | *required* | Specific overview topic |
| `include` | `list[str]` | `["*"]` | Glob patterns. `"*.py"` matches basename; `"src/*.py"` matches path. |
| `exclude` | `list[str]` | `[]` | Same pattern syntax as include |
| `mode` | `Literal` | `"conversation"` | `"conversation"` or `"narration"` |
| `format` | `Literal` | `"video"` | `"video"` or `"audio"` |
| `host_voice` | `str` | `"Aoede"` | Conversation mode Host voice |
| `expert_voice` | `str` | `"Charon"` | Conversation mode Expert voice |
| `narrator_voice` | `str` | `"Kore"` | Narration mode voice |
| `max_duration_minutes` | `PositiveInt` | `10` | Advisory duration limit |
| `llm_backend` | `Literal` | `"claude"` | `"claude"` or `"codex"` |
| `cache_dir` | `Path \| None` | `None` | Auto-set to `<source_dir>/.video_overview_cache/`. Set a custom path to store cache elsewhere. |
| `skip_visuals` | `bool` | `False` | Video only: use static frame |
| `no_cache` | `bool` | `False` | Always regenerate |

### Advanced Fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_tokens_per_batch` | `PositiveInt` | `8000` | Token budget per audio TTS batch |
| `max_segments_per_batch` | `PositiveInt` | `13` | Max segments per audio TTS batch |
| `audio_max_attempts` | `PositiveInt` | `3` | Total API attempts per batch (initial + retries) |
| `video_width` | `PositiveInt` | `1920` | Must be even (libx264 requirement) |
| `video_height` | `PositiveInt` | `1080` | Must be even (libx264 requirement) |
| `video_fps` | `PositiveInt` | `30` | Frames per second |
| `crossfade_seconds` | `NonNegativeFloat` | `0.5` | Crossfade duration between images (0-60s) |
| `ken_burns_zoom_percent` | `NonNegativeFloat` | `5.0` | Zoom intensity for Ken Burns effect (%) |

Extra fields are forbidden (`extra="forbid"` on the Pydantic model).

## OverviewResult

```python
result.output_path      # Path -- the generated file
result.duration_seconds  # float -- total estimated duration in seconds
result.segments_count    # int -- number of script segments
```

## Script and ScriptSegment

Data models produced by `ScriptGenerator`. You typically don't construct these directly.

```python
from video_overview import Script, ScriptSegment

segment = ScriptSegment(
    speaker="Host",
    text="Welcome to our overview!",
    visual_prompt="A welcome screen showing the project logo",
)

script = Script(
    title="Authentication Deep Dive",
    segments=[segment],
)
```

Valid speakers: `"Host"` and `"Expert"` in conversation mode, `"Narrator"` in narration mode.

## Error Handling

```python
from video_overview.script.generator import ScriptGenerationError
from video_overview.audio.generator import AudioGenerationError
from video_overview.visuals.generator import VisualGenerationError
from video_overview.video.assembler import VideoAssemblyError

try:
    result = create_overview(source_dir="./src", output="out.mp4", topic="auth")
except ValueError as e:
    # Missing API key, invalid config, no readable files
    print(f"Config error: {e}")
except ScriptGenerationError as e:
    # LLM subprocess failed, timeout, invalid response
    print(f"Script error: {e}")
except AudioGenerationError as e:
    # Gemini TTS failed after retries
    print(f"Audio error: {e}")
except VisualGenerationError as e:
    # Image generation failed (fallback also failed)
    print(f"Visual error: {e}")
except VideoAssemblyError as e:
    # ffmpeg assembly failed
    print(f"Assembly error: {e}")
```

## Accessing the API Key

```python
config = OverviewConfig(source_dir="./src", output="out.mp4", topic="test")
key = config.gemini_api_key  # reads GEMINI_API_KEY, falls back to GOOGLE_API_KEY
```
