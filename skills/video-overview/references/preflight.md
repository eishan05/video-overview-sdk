# Preflight Checks

Run these checks before generating any overview. The API key and LLM CLI are always required. ffmpeg is required for all video output and most audio runs.

## 1. Gemini API Key

```bash
# Must have one of these set
echo "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-NOT SET}}"
```

If neither is set, export one:
```bash
export GEMINI_API_KEY="your-key"
```

The SDK reads from the process environment directly. It does not load `.env` files -- use `direnv`, `dotenv-cli`, or similar if you need `.env` support.

## 2. ffmpeg

```bash
ffmpeg -version
```

Required for all video output and most audio runs. `AudioGenerator` uses ffmpeg to concatenate multi-chunk audio; only a single-batch `.wav` run (short scripts) can succeed without it. MP3 output always requires ffmpeg for format conversion.

Install if missing:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: `choco install ffmpeg` or download from ffmpeg.org

## 3. LLM CLI

At least one must be installed and authenticated:

```bash
# Check Claude CLI
claude --version

# Check Codex CLI
codex --version
```

The default backend is `claude`. Use `--llm codex` to switch.

## Quick Validation Script

```bash
# Run all checks at once
echo "=== Preflight ==="
echo "API key: ${GEMINI_API_KEY:+set}${GEMINI_API_KEY:-${GOOGLE_API_KEY:+set (GOOGLE_API_KEY)}${GOOGLE_API_KEY:-NOT SET}}"
echo "ffmpeg: $(command -v ffmpeg >/dev/null 2>&1 && echo 'found' || echo 'MISSING')"
echo "claude: $(command -v claude >/dev/null 2>&1 && echo 'found' || echo 'not found')"
echo "codex: $(command -v codex >/dev/null 2>&1 && echo 'found' || echo 'not found')"
```

API key must be set. At least one backend CLI (claude or codex) must be available. ffmpeg must be available for video output and most audio runs.
