# Changelog

## 2026-10-15

### Fixed

- Prevented an MPS native crash during OmniVoice model loading by disabling asynchronous Transformers weight conversion.
- Prevented cached voice prompts from being reused after reference audio or transcript changes, including overlapping save callbacks.
- Isolated temporary audio by request and browser session, and added cleanup when Gradio expired the session state.
- Removed duplicate microphone transcription events and preserved quiet recordings' original amplitude.
- Converted unsupported Whisper inputs to 16 kHz mono PCM16 WAV even when their original sample rate was already 16 kHz.
- Serialized access to shared Whisper and Qwen models across concurrent UI callbacks.
- Blocked profile path traversal and symlink access, used restricted profile deserialization, and made profile writes atomic.
- Selected sanitized profile filenames after saving and preserved encoded YouTube query parameters while rejecting non-YouTube inputs.

### Added

- Added OmniVoice prompt storage with an engine identifier and migration of saved reference audio/transcripts from previous profiles.
- Exposed OmniVoice's complete supported-language list in the speech-generation UI.
- Enabled reference waveform trimming with automatic retranscription, prompt invalidation, and restoration of the previous selection.
- Added a dependency lockfile and automated regression checks for audio conversion, profile safety, prompt reuse, and session cleanup.
- Added four repository-local screenshots illustrating reference retrieval, waveform trimming, profile management, and speech generation.

### Changed

- Replaced the TTS/voice-cloning backend with OmniVoice 0.2.1 and its native generation API; upgraded the locked Transformers dependency to 5.19.0.
- Renamed the application/package to OmniVoice Clone WebUI / `omnivoice-clone-webui` while retaining the source module and launch command.
- Changed YouTube reference downloads from three seconds after a timestamp to eight seconds before and after it, with shorter clips at video boundaries and actual-duration reporting.
- Switched the default Qwen model weights from float32 to float16 on MPS and documented the restart requirement and BF16 alternative.
- Migrated the project environment from Python 3.11 to Python 3.13 and replaced the incompatible NumPy 1.x constraint with NumPy 2.1 or newer.
- Limited Gradio to the supported 6.x API and removed the obsolete Hugging Face CLI extra.
- Changed saved profiles to a versioned tensor/container format; legacy pickle profiles now require recreation from the original reference audio.
- Updated English and Japanese setup instructions for macOS 26.x, locked dependency installation, and development checks.
- Rewrote the English README and its Japanese translation around one installation path, illustrated usage, configuration, stored files, and troubleshooting.
- Updated repository URLs, clone-directory instructions, and agent guidance for the renamed `omnivoice-voice-clone-webui` repository.

### Removed

- Removed the `qwen-tts` dependency, model loader, prompt reconstruction, generation parameters, and Qwen-specific UI/download instructions.
