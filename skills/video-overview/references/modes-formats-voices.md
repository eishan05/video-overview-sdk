# Modes, Formats, and Voices

## Choosing a Mode

| Mode | Speakers | Best For |
|---|---|---|
| `conversation` | Host (asks questions) + Expert (explains) | Deep dives, architecture walkthroughs, complex topics |
| `narration` | Single Narrator | Summaries, documentation, quick overviews, background listening |

Default is `conversation`. Use `narration` when the content is straightforward or you want a more concise output.

## Choosing a Format

| Format | Output | Best For |
|---|---|---|
| `video` | MP4 with generated images + audio | Visual topics (architecture, data flows, UI). Slower due to image generation. |
| `audio` | MP3 or WAV, audio only | Background listening, documentation, faster generation. No image gen cost. |

Default is `video`. Use `audio` when:
- Speed matters more than visuals
- Content is conceptual rather than visual
- You want a podcast-style output

Use `--skip-visuals` with video format for a middle ground: MP4 output with a static dark frame instead of generated images. Faster than full video but still produces an MP4 file.

## Voices

The SDK accepts any valid Gemini TTS prebuilt voice name (it does not validate against a fixed list). Common defaults and recommendations:

| Voice | Character | Recommended For |
|---|---|---|
| **Aoede** | Breezy | Host (default) |
| **Charon** | Informative | Expert (default) |
| **Kore** | Firm | Narrator (default) |
| Puck | Upbeat | Host or Narrator for energetic content |
| Zephyr | Bright | Host for lighter topics |
| Fenrir | Excitable | Host for enthusiastic delivery |
| Leda | Youthful | Host for approachable tone |
| Orus | Firm | Expert or Narrator for authoritative tone |

For the full list of 30+ available voices, see the Gemini TTS documentation.

### Voice Assignment by Mode

**Conversation mode** uses two voices:
- `--host-voice` (default: Aoede) -- the questioner who drives the conversation
- `--expert-voice` (default: Charon) -- the explainer who provides technical depth

**Narration mode** uses one voice:
- `--narrator-voice` (default: Kore) -- single speaker for the entire overview

### Voice Pairing Tips

- Match contrasting tones for conversation: a breezy Host (Aoede, Puck) with an authoritative Expert (Charon, Orus)
- For technical content, use Charon or Orus as Expert
- For approachable content, use Puck or Leda as Host
- Kore is the best general-purpose Narrator voice

## Common Combinations

```bash
# Default: casual conversation
video-overview ./src -t "topic" -o out.mp4
# (Aoede + Charon)

# Authoritative conversation
video-overview ./src -t "topic" -o out.mp4 --host-voice Zephyr --expert-voice Orus

# Energetic narration
video-overview ./docs -t "topic" -f audio -m narration -o out.mp3 --narrator-voice Puck

# Quick audio summary
video-overview ./docs -t "topic" -f audio -m narration -o out.mp3
# (Kore)
```
