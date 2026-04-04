# CLI Reference

## Usage

```
video-overview SOURCE_DIR [OPTIONS]
```

## Required Options

| Option | Short | Description |
|---|---|---|
| `SOURCE_DIR` | | Source directory to read (must exist) |
| `--topic` | `-t` | Specific topic for the overview |
| `--output` | `-o` | Output file path |

## Optional Flags

| Option | Short | Default | Description |
|---|---|---|---|
| `--include` | `-i` | `*` | Glob patterns for files to include (repeatable) |
| `--exclude` | `-e` | | Glob patterns for files to exclude (repeatable) |
| `--mode` | `-m` | `conversation` | `conversation` (Host+Expert) or `narration` (Narrator) |
| `--format` | `-f` | `video` | `video` (MP4) or `audio` (MP3/WAV) |
| `--host-voice` | | `Aoede` | Host voice (conversation mode) |
| `--expert-voice` | | `Charon` | Expert voice (conversation mode) |
| `--narrator-voice` | | `Kore` | Narrator voice (narration mode) |
| `--llm` | | `claude` | LLM backend: `claude` or `codex` |
| `--max-duration` | | `10` | Max duration in minutes (advisory) |
| `--skip-visuals` | | `false` | Use static dark frame instead of generated images (video only) |
| `--no-cache` | | `false` | Skip cache, always regenerate |
| `--verbose` | `-v` | `false` | Enable INFO-level logging |

## Extension Rules

- `--format video` requires `.mp4` extension
- `--format audio` requires `.mp3` or `.wav` extension
- `--skip-visuals` is only valid with `--format video`

## Command Recipes

### Video conversation (default)
```bash
video-overview ./src -t "authentication system" -o overview.mp4
```

### Audio narration
```bash
video-overview ./docs -t "API reference" -f audio -m narration -o overview.mp3
```

### Scoped Python files only
```bash
video-overview ./src -t "data pipeline" -o overview.mp4 -i "*.py" -i "*.yaml" -e "tests/*"
```

### Fast video (skip image generation)
```bash
video-overview ./src -t "system overview" -o overview.mp4 --skip-visuals
```

### With Codex backend
```bash
video-overview ./src -t "architecture" -o overview.mp4 --llm codex
```

### Custom voices
```bash
video-overview ./src -t "API design" -o overview.mp4 \
  --host-voice Zephyr --expert-voice Orus
```

### Short duration, verbose
```bash
video-overview ./src -t "quick summary" -o overview.mp3 \
  -f audio -m narration --max-duration 3 -v
```

### WAV output (lossless)
```bash
video-overview ./src -t "deep dive" -f audio -m narration -o overview.wav
```

## Cache Management

```bash
# List cached assets (by source directory)
video-overview cache list --source-dir ./src

# List cached assets (by explicit cache path)
video-overview cache list --cache-dir /path/to/custom/cache

# Clear cache (with confirmation)
video-overview cache clear --source-dir ./src

# Clear cache (no prompt)
video-overview cache clear --source-dir ./src --yes
```

Default cache location: `<source_dir>/.video_overview_cache/`. Use `--cache-dir` to point at a custom location. Contains audio chunks (WAV) and generated images (PNG).

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Runtime error (API failure, validation, ffmpeg error) |
| 2 | Invalid arguments (missing required options, bad values) |
| 130 | Interrupted (Ctrl+C) |
