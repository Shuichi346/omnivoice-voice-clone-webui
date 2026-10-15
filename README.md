<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README-Japanse.md">日本語</a></th>
      <th style="text-align:center"><a href="README.md">English</a></th>
    </tr>
  </thead>
</table>

# OmniVoice Voice Clone WebUI

Turn a short voice sample into speech from your own text. This Gradio app runs on Apple Silicon Macs and combines **OmniVoice** for voice cloning with **Whisper large-v3-turbo** for reference transcription. No model fine-tuning is required.

Use a YouTube timestamp, a microphone recording, or an audio file as the reference. Trim it in the browser, check its transcript, then generate speech in Japanese or another language from OmniVoice's language list. Save a voice profile to reuse the same reference later.

[<img src="githubreadme/reference-audio.png" alt="OmniVoice WebUI showing YouTube audio retrieval, the reference waveform, and its editable transcript" width="480">](githubreadme/reference-audio.png)

*Reference audio and transcription in the WebUI. Click any screenshot to view it at full size.*

## Requirements

- **Apple Silicon Mac running macOS 26.x or later.** The project's verified environment is a Mac mini M4 with 24 GB unified memory; this is not a measured minimum memory requirement.
- **Python 3.13** (`>=3.13,<3.14`), selected by `.python-version`.
- **Git**, **[Homebrew](https://brew.sh/)**, **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, and **FFmpeg** for setup and audio processing.
- Internet access for initial model downloads and when fetching YouTube audio.

The default TTS configuration uses MPS, float16 weights, and SDPA attention. Other operating systems and hardware are outside the project's verified setup.

## Install and launch

Run these commands in Terminal:

```bash
brew install uv ffmpeg

git clone https://github.com/Shuichi346/omnivoice-voice-clone-webui.git
cd omnivoice-voice-clone-webui

uv sync --locked
uv run --locked voice-clone-webui
```

Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser. Stop the server with `Ctrl+C` in Terminal. To start it again, run `uv run --locked voice-clone-webui` from the repository directory.

`uv sync --locked` creates the project-local `.venv` using the committed dependency lockfile. No separate environment activation is needed. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/) for how uv manages the project environment and Python.

Models load on first use: Whisper when transcribing reference audio, and OmniVoice when saving a profile or generating speech. Missing weights are downloaded automatically, so those first operations take longer.

## Use the WebUI

### 1. Get reference audio

Choose **Transcription Language (Whisper)** before processing the reference. Leave it on **Auto**, or select the language spoken in the sample.

| Source tab | What to do |
| --- | --- |
| **YouTube URL** | Paste a YouTube URL with a `t` timestamp and click **Fetch Audio**. |
| **Microphone Recording** | Allow microphone access, record a short sample, and stop recording. Transcription starts automatically. |
| **Audio File Upload** | Upload a file and click **Process File**. |

Uploads accept WAV, MP3, FLAC, OGG, OPUS, M4A, AAC, WMA, and WebM. The app converts uploaded files to 16 kHz mono WAV for processing. Start with a clear, single-speaker sample of around 3–10 seconds, and trim away silence or unrelated speech.

For YouTube, the timestamp is the **center** of the requested clip. For example, a URL ending in `?t=1018` requests **16:50–17:06**: eight seconds before and after 16:58. `?t=16m58s` also works.

At the start or end of a video, only the available audio is used. A URL without a timestamp requests the first eight seconds. The **Status** field reports the actual downloaded duration.

### 2. Trim and check the transcript

In **Review Reference Audio & Transcription**:

1. Play **Reference Audio** to check the sample.
2. Open the scissors tool, drag the selection edges, and click **Trim** to keep the selected range.
3. Wait for the selected audio to be transcribed again.
4. Correct **Transcription Result (editable)** so that it matches the spoken words. A nonempty reference transcript is required for cloning.

Use the waveform's reset control to restore the previous audio selection. Changing the reference audio or editing its transcript invalidates the cached clone prompt; subsequent saving or generation uses the current reference.

[<img src="githubreadme/trim-reference.png" alt="Reference waveform with an orange selection and Trim and Cancel controls" width="480">](githubreadme/trim-reference.png)

*Select only the speech you want to use as the voice reference.*

### 3. Save a voice profile (optional)

Enter a **Profile Name**, then click **Save Profile**. The app saves the reference audio, transcript, and reusable OmniVoice prompt in `voice_profiles/`.

Next time, select the profile under **Saved Profiles** and click **Load** to restore it. **Refresh** reloads the list of saved profiles. Saving under the same sanitized filename replaces that profile.

You can generate speech directly from the current reference without saving a profile.

[<img src="githubreadme/voice-profiles.png" alt="Profile Management with the saved woman1 profile selected and a successful save status" width="480">](githubreadme/voice-profiles.png)

*A saved profile lets you reuse the reference without fetching or recording it again.*

### 4. Generate speech

Enter your text in **Text to Synthesize**, choose **TTS Language (OmniVoice)**, and click **Generate Speech**. Play the result in **Generated Audio**, or use its download control to save it.

The two language controls serve different purposes: **Transcription Language** describes the reference recording, while **TTS Language** describes the speech you want to generate. For TTS, **Auto** leaves the language unspecified for OmniVoice. Japanese is available in the supported-language dropdown.

[<img src="githubreadme/speech-generation.png" alt="Speech Generation with Japanese input text, Auto language selection, and generated audio playback" width="480">](githubreadme/speech-generation.png)

*Japanese text and generated audio in the speech-generation section.*

## Settings and stored files

Edit [src/voice_clone_webui/config.py](src/voice_clone_webui/config.py) and restart the server to change settings. The app does not read these settings from a `.env` file.

| Setting | Default | Purpose |
| --- | --- | --- |
| `OMNIVOICE_MODEL_ID` | `"k2-fsa/OmniVoice"` | Hugging Face model ID, or a complete local model directory including its audio tokenizer. |
| `WHISPER_MODEL_NAME` | `"large-v3-turbo"` | Model used by pywhispercpp for transcription. |
| `GENERATION_STEPS` | `32` | OmniVoice generation steps. |
| `GENERATION_SPEED` | `1.0` | Speech generation speed parameter. |
| `CLIP_CONTEXT_SEC` | `8` | Seconds requested on each side of a YouTube timestamp. |
| `SERVER_NAME` / `SERVER_PORT` | `"127.0.0.1"` / `7860` | Local server address and port. |

Keep the default MPS, float16, and SDPA settings for the verified Apple Silicon path. The app disables asynchronous Transformers weight loading on MPS and retains OmniVoice's audio tokenizer on CPU to avoid unsupported operations.

| Location | Contents |
| --- | --- |
| `voice_profiles/*.pt` | Saved profiles, including the reference recording and transcript. Excluded from Git. |
| `tmp/session_<uuid>/` | Temporary reference files separated by browser session and request. Excluded from Git. |
| `githubreadme/` | Screenshots used by the English and Japanese READMEs. |

Gradio removes session reference files after the closed browser session expires. A forced shutdown can leave temporary directories behind; stop the app before manually removing leftovers. Model weights use the model libraries' caches and are not included in this repository.

Current profiles use format 2 with the `omnivoice` engine identifier and restricted tensor/container loading. Older format-1 profiles can reuse their stored audio and transcript, but their old prompt is rebuilt for OmniVoice. Save them again to cache the new prompt. Unsupported legacy pickle profiles must be recreated from the original recording.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ffmpeg` is missing or audio conversion fails | Run `ffmpeg -version`; install it with `brew install ffmpeg` if missing. Confirm that the source audio is readable. |
| The first transcription, save, or generation takes longer | Check Terminal for model download/loading progress and confirm internet access. Models load on demand. |
| Saving or generation reports a missing reference | Fetch, record, upload, or load a profile first. Wait for transcription and check that its text is not empty. |
| A trimmed clip has the wrong transcript | Wait for retranscription to finish, then edit the text to match the selected audio before saving or generating. |
| A YouTube download fails | Read **Status** and the Terminal error; check the URL, timestamp, and whether the video is accessible. You can also use a local audio file. |
| Port 7860 is already in use | Stop the other server, or change `SERVER_PORT` in `config.py` and restart. |
| An older profile cannot be loaded | Recreate it from its original audio if it uses an unsupported pickle format. |

## Development

```bash
uv run --locked ruff check src tests
uv run --locked python -m unittest discover -s tests -v
```

The regression tests require FFmpeg and cover audio conversion, reference handling, profile storage, and UI event wiring without downloading model weights. They do not establish transcription accuracy or generated-speech quality. Use `uv sync` deliberately when updating dependencies, and include the resulting `uv.lock` changes.

## Components and license

- [OmniVoice](https://github.com/k2-fsa/OmniVoice) · [model weights](https://huggingface.co/k2-fsa/OmniVoice)
- [pywhispercpp](https://github.com/absadiki/pywhispercpp) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Gradio](https://www.gradio.app/) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [FFmpeg](https://ffmpeg.org/)

This repository is licensed under the [MIT License](LICENSE). Model weights and dependencies have their own licenses. Use reference recordings with the speaker's permission and respect the source service's terms.
