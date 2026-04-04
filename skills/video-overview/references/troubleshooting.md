# Troubleshooting

## Missing API Key

**Error:** `ValueError: Gemini API key is required. Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.`

**Fix:** Export one of the required keys:
```bash
export GEMINI_API_KEY="your-key"
```

The SDK does not load `.env` files automatically. Use `direnv` or `dotenv-cli` if you need that.

## ffmpeg Not Found

**Error:** `ffmpeg is not installed or not on PATH` (raised by AudioGenerator, VideoAssembler, or VisualGenerator).

**Fix:** Install ffmpeg and ensure it's on PATH:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

Required for all video output and most audio runs. Only a single-batch `.wav` output can succeed without it.

## LLM CLI Not Found

**Error:** `ScriptGenerationError: Failed to execute LLM command: [Errno 2] No such file or directory`

**Fix:** Install and authenticate the LLM backend:
```bash
# For Claude (default)
claude --version

# For Codex
codex --version
```

Switch backend with `--llm codex` if only Codex is available.

## Script Generation Timeout

**Error:** `ScriptGenerationError: LLM subprocess timed out after 300 seconds`

**Likely cause:** Very large content bundle or slow LLM response.

**Fixes:**
- Reduce input size with `--include` / `--exclude` patterns
- Use a more specific `--topic` to focus the script
- Reduce `--max-duration` to limit script length

## Format/Extension Mismatch

**Error:** Pydantic validation error on `OverviewConfig` (output format does not match extension).

**Fix:** Match the output extension to the format:
- `--format video` requires `.mp4`
- `--format audio` requires `.mp3` or `.wav`

## skip-visuals with Audio

**Error:** Pydantic validation error on `OverviewConfig` (skip_visuals only valid for video format).

**Fix:** `--skip-visuals` is only valid with `--format video`. Remove the flag for audio output.

## Include Patterns Match No Files

**Error:** `ValueError: No files matched include patterns: *.xyz`

**Likely cause:** Pattern syntax mismatch. Basename patterns like `"*.py"` match filename only. Path-based patterns like `"src/*.py"` use gitignore-style matching against the relative path.

**Fixes:**
- Use `"*.py"` to match all Python files anywhere in the tree
- Use `"src/*.py"` to match only Python files directly in `src/`
- Use `"src/**/*.py"` for recursive matching within `src/`
- Check that the source directory actually contains files matching the pattern

## Audio API Failures

**Error:** `AudioGenerationError: API call failed after 3 attempts: <details>`

**Likely causes:**
- Invalid API key or quota exceeded
- Network issues
- Invalid voice name

**Fixes:**
- Verify the API key is valid and has TTS quota
- Check network connectivity
- Use a valid Gemini TTS voice name (see modes-formats-voices.md)
- Retry with `--no-cache` to force fresh API calls

## Visual Generation Failures

**Error:** `VisualGenerationError: ...`

**Note:** The SDK automatically falls back to dark frames with text when image generation fails. This error only appears if both image generation AND the fallback frame creation fail.

**Likely cause:** ffmpeg issue (the fallback uses ffmpeg to create text frames).

**Fix:** Ensure ffmpeg is working. Or use `--skip-visuals` to bypass image generation entirely.

## Video Assembly Failures

**Error:** `VideoAssemblyError: ffmpeg failed (exit code N): ...`

**Likely causes:**
- Corrupt audio or image files in cache
- Disk space issues
- ffmpeg version incompatibility

**Fixes:**
- Clear cache: `video-overview cache clear --source-dir ./src --yes`
- Check disk space
- Update ffmpeg to a recent version
- Run with `--verbose` to see detailed ffmpeg output

## No Readable Files Found

**Error:** `ValueError: No readable files found in <path>`

**Likely causes:**
- Wrong source directory
- `--include` patterns don't match any files (see "Include Patterns Match No Files" above)
- All files are binary or in excluded directories

**Fix:** Verify the source directory contains text files matching your include patterns:
```bash
ls ./src  # Check directory contents
```

## Output Parent Directory Does Not Exist

**Error:** `output parent directory does not exist: <path>`

**Fix:** Create the parent directory before running:
```bash
mkdir -p /path/to/output/dir
```
