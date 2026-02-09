# Qwen3-TTS Voice Clone WebUI

A WebUI application that clones a voice from a YouTube URL, microphone recording, or uploaded audio file, and synthesizes any text with the cloned voice.

## How to Youtube 
<a href="https://www.youtube.com/watch?v=QbgzrLisIhQ">
  <img src="https://img.youtube.com/vi/QbgzrLisIhQ/0.jpg" width="640" alt="動画タイトル">
</a>

## Features

- Automatically download a 3-second audio clip from a YouTube URL at a specified timestamp
- Record audio from a microphone
- Upload your own audio file (WAV, MP3, FLAC, OGG, OPUS, M4A, AAC, WMA, WebM)
- Automatic transcription via [pywhispercpp](https://github.com/absadiki/pywhispercpp) (large-v3-turbo)
- Voice cloning powered by [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- Save and load clone profiles (LoRA-style voice reproduction)
- WebUI built with [Gradio 6.x](https://www.gradio.app/)

## Related Links

| Component | URL |
|---|---|
| Qwen3-TTS Official Blog | <https://qwen.ai/blog?id=qwen3tts-0115> |
| Qwen3-TTS GitHub | <https://github.com/QwenLM/Qwen3-TTS> |
| Qwen3-TTS-12Hz-1.7B-Base (HuggingFace) | <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base> |
| Qwen3-TTS-Tokenizer-12Hz (HuggingFace) | <https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz> |
| pywhispercpp (GitHub) | <https://github.com/absadiki/pywhispercpp> |
| whisper.cpp (GitHub) | <https://github.com/ggml-org/whisper.cpp> |
| yt-dlp (GitHub) | <https://github.com/yt-dlp/yt-dlp> |
| Gradio (Official Site) | <https://www.gradio.app/> |
| uv Package Manager (Docs) | <https://docs.astral.sh/uv/> |

## Requirements

- Mac mini M4 (Apple Silicon, 24 GB unified memory)
- macOS 15.7.2+
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

### 1. System Dependencies

```bash
brew install sox portaudio ffmpeg
```

### 2. Project Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project directory
cd qwen-voice-clone-webui

# Install dependencies
uv sync
```

### 3. Model Download (Optional: Pre-download)

```bash
# Models are automatically downloaded on first run, but you can pre-download them:
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./model/Qwen3-TTS-12Hz-1.7B-Base
uv run hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./model/Qwen3-TTS-Tokenizer-12Hz
```

If you use local models, update `QWEN_TTS_MODEL_ID` in `src/voice_clone_webui/config.py` to point to the local path.

### 4. Launch

```bash
uv run python -m voice_clone_webui.app
```

Open `http://localhost:7860` in your browser.

## Usage

1. **Fetch Audio**: Choose one of three methods to provide reference audio:
   - **YouTube URL**: Paste a URL with a time parameter and click "Fetch Audio"
   - **Microphone**: Record approximately 3 seconds of audio directly
   - **Audio File Upload**: Upload your own audio file and click "Process File"
2. **URL with Time Parameter**: On YouTube, play the video to the "desired start time you want to share" and pause it. Click "Share" below the video. Check the "Start At" option in the share dialog. The URL with time parameter will be copied to your clipboard.
3. **Review Transcription**: Check the automatic transcription result and edit if necessary
4. **Save Profile**: Enter a name and click "Save Profile" — you can load it from the dropdown next time
5. **Generate Speech**: Enter text and click "Generate Speech"

## Directory Structure

```
qwen-voice-clone-webui/
├── pyproject.toml
├── src/voice_clone_webui/   # Application source code
├── voice_profiles/          # Saved clone profiles (.pt)
└── tmp/                     # Temporary files
```

---

## Quick Setup Summary

```bash
# 1. Create project directory
mkdir qwen-voice-clone-webui
cd qwen-voice-clone-webui

# 2. Place all files above, then install dependencies
uv sync

# 3. Launch
uv run python -m voice_clone_webui.app
```

## Technical Notes

**Apple Silicon MPS Support**: The default configuration uses `dtype=torch.float32` with `attn_implementation="sdpa"`. If `qwen-tts-demo` already runs with `--dtype bfloat16` in your environment, you can change `TORCH_DTYPE_STR` in `config.py` to `"bfloat16"`.

**Profile Storage**: The return value of `create_voice_clone_prompt()` is saved as a `.pt` file via `torch.save`, along with the reference audio WAV data and transcription text. This allows instant voice clone generation simply by loading a profile.

**Audio File Upload**: Uploaded audio files are automatically converted to 16kHz mono WAV using ffmpeg before transcription and voice cloning. Supported formats include WAV, MP3, FLAC, OGG, OPUS, M4A, AAC, WMA, and WebM. There is no duration limit on uploaded files, but shorter clips (around 3–10 seconds of clear speech) tend to produce the best voice cloning results.

**Gradio 6.x Compatibility**: The `gr.Blocks()` constructor is used without `theme`/`css` arguments; these are passed to `launch()` if needed. Components such as `gr.Audio` use `sources=["microphone"]` or `sources=["upload"]` in accordance with the latest API specification.

**Lazy Initialization**: Both the Whisper model and the Qwen TTS model are loaded on first use to reduce startup time.

## Disclaimer

- This tool uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to fetch audio from YouTube. YouTube's Terms of Service restrict downloading via third-party tools. Use at your own risk.
- Voice cloning should only be used with the consent of the voice owner, or for personal experimentation and research purposes.
