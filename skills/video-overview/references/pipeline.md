# Pipeline Internals and Caching

## Pipeline Stages

```
source_dir -> ContentReader -> ScriptGenerator -> [AudioGenerator + VisualGenerator] -> VideoAssembler -> output
```

### 1. Content Reading (ContentReader)

Walks the source directory and produces a content bundle:
- Respects `.gitignore` patterns automatically
- Skips binary files, `__pycache__`, `.git`, `node_modules`, `.venv`, etc.
- Applies `include`/`exclude` glob patterns
- Sorts files by relevance: README > docs > markdown > config > source
- Truncates individual files at 2,000 characters (appends `[truncated — N total chars]`)
- Enforces a 100,000-character overall bundle budget (truncates the last file to fit)
- Produces `directory_structure` (tree) and `files` (path, content, language)

**Pattern matching:** `"*.py"` matches basename only. `"src/*.py"` uses path-based gitignore-style matching.

### 2. Script Generation (ScriptGenerator)

Calls an LLM CLI subprocess to produce a structured script:
- **Claude:** `claude -p "<prompt>" --output-format json`
- **Codex:** `codex exec "<prompt>"`
- Timeout: 300 seconds
- Output: `Script` model with `title` and `segments` (each with `speaker`, `text`, `visual_prompt`)
- Validates speaker names match the mode (Host/Expert for conversation, Narrator for narration)
- Prompt includes a duration budget (word count, segment count) when `max_duration_minutes` is set
- `ScriptGenerator` raises `ScriptGenerationError` if the LLM returns more than `max_segments` (default 20)
- After generation, `create_overview()` separately truncates the segment list to fit `max_duration_minutes` by estimated duration

### 3. Audio Generation (AudioGenerator)

Gemini TTS via `gemini-2.5-flash-preview-tts` model:
- **Conversation mode:** `MultiSpeakerVoiceConfig` with Host + Expert voices
- **Narration mode:** Single `VoiceConfig` with Narrator voice
- Sample rate: 24kHz WAV
- Segments are batched by token budget (8000 tokens, ~32K chars) and segment count (13 per batch)
- Each batch is cached independently using content-addressed MD5 hashing
- Retry with exponential backoff: 3 total attempts by default (delays of 1s, 2s between attempts)
- Single-chunk output is copied directly (no ffmpeg); multi-chunk output is concatenated with ffmpeg

### 4. Visual Generation (VisualGenerator)

Gemini image generation via `gemini-2.0-flash-exp`:
- One image per segment, derived from the `visual_prompt` field
- Aspect ratio: 16:9
- Max 3 concurrent API calls (semaphore-limited)
- Per-prompt deduplication (same prompt across segments only calls API once)
- **Fallback:** On API failure, generates a dark frame (#1a1a2e) with segment text using ffmpeg
- Runs concurrently with audio generation

### 5. Video Assembly (VideoAssembler)

ffmpeg-based final assembly:
- **Video mode:** Ken Burns zoom effect on images, crossfade transitions between segments, libx264 + aac encoding
- **Audio mode (MP3):** Converts WAV to MP3 via ffmpeg (libmp3lame, quality=2)
- **Audio mode (WAV):** Direct copy (no conversion)
- Timeout: 600 seconds
- Video dimensions: 1920x1080 (must be even for libx264)

## Caching

Default cache location: `<source_dir>/.video_overview_cache/` (auto-created, auto-excluded from content reading). Use `cache_dir` in `OverviewConfig` to choose a custom generation cache location. Use `video-overview cache ... --cache-dir` to inspect or clear a specific cache directory.

### What Is Cached

| Asset | Cache Key | Location |
|---|---|---|
| Audio chunks | MD5(model + schema_version + segments + voices + multi_speaker) | `audio_<hash>.wav` |
| Visual images | MD5(visual_prompt) | `visuals/<hash>.png` |
| Static frame | Dimensions | `static_frame_{width}x{height}.png` |

### Cache Behavior

- By default, cached assets are reused when the cache key matches
- `--no-cache` skips reading from cache but still writes new assets
- Cache is content-addressed: changing segment text, voices, or speaker assignment invalidates the relevant chunks
- Audio chunks use atomic writes (temp file then rename) to prevent corrupt cache entries

### Managing the Cache

```bash
# See what's cached
video-overview cache list --source-dir ./src

# Clear everything
video-overview cache clear --source-dir ./src --yes
```

The cache directory is automatically added to `.gitignore`-style exclusion during content reading, so it never appears in the content bundle fed to the LLM.
