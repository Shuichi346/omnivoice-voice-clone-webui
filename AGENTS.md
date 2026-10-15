# AGENTS.md

Project instructions for coding agents working in `Shuichi346/qwen-voice-clone-webui`.

## Environment and verification

- Target Python 3.13 (`>=3.13,<3.14`) and native Apple Silicon on macOS 26.x or later. Keep `.python-version`, `pyproject.toml`, and `uv.lock` consistent.
- Use `uv sync --locked` for normal setup and `uv run --locked` for checks. Resolve dependency updates deliberately with `uv sync`; commit `uv.lock`.
- Run `uv run --locked ruff check src tests` and `uv run --locked python -m unittest discover -s tests -v` for affected implementation changes. Tests require ffmpeg and do not download model weights.

## Runtime contracts

- Use OmniVoice 0.2.x with MPS, `float16` weights, and SDPA attention. Keep its upstream audio tokenizer on CPU and disable asynchronous Transformers weight loading on MPS (`HF_DEACTIVATE_ASYNC_LOAD=1`). Audio NumPy arrays remain float32. Validate backend/precision changes with a bounded real GPU inference smoke check, including finite nonempty output.
- YouTube reference windows span `timestamp ± CLIP_CONTEXT_SEC` (8 seconds per side). Clamp the start to zero, allow shorter boundary clips, and report actual duration. Keep audio-only ranged downloads with accurate cutting enabled.
- Gradio waveform trimming emits `change`, not `input`. Clear reference/prompt state before retranscription, skip unchanged backend/cache copies and already-cleared backend state, and do not output back to the waveform from its change callback. Keep reference-source and edit callbacks in the same concurrency group.
- Bind cached clone prompts to their reference audio path and transcript; validate both at generation time because Gradio callbacks may overlap.
- Keep reference files separate by request inside the browser session directory, and preserve the session state's cleanup callback. Keep Whisper and OmniVoice singleton inference serialized within each model. Always supply reference transcripts to OmniVoice so it does not load a second ASR model.
- Save profiles as format 2 with the `omnivoice` engine identifier, plain containers, and CPU tensors; load with `weights_only=True`. Reuse only audio/text from restricted format-1 profiles and rebuild their prompt. Reject unsafe pickle profiles with recovery guidance; do not add an unrestricted pickle fallback.
- Restrict profile load/delete paths to the profile directory. Keep atomic profile writes and select the sanitized saved filename in the UI.

## Documentation maintenance

- Keep reader-facing commands and behavior consistent between `README.md` and `README-Japanse.md`, including their language-switch links. Both READMEs use shared repository-local screenshots in `githubreadme/`; preserve their relative image links when editing either version.
- After completed implementation work, update `CHANGELOG.md`, `NOTES.md`, and `AGENTS.md` in English using the current task's evidence, including already committed or pushed work.
- Record what changed in the changelog in past tense under `Fixed`, `Added`, `Changed`, and `Removed`. Omit empty categories and routine cleanup or test-only changes unless their impact matters.
- Use `## YYYY-MM-DD` for new work, newest first, and append to existing same-day categories. Use the completion date in the user's timezone; do not guess dates or release versions. Preserve existing dated/versioned history and classify undated historical entries under `## DATE UNKNOWN` with `### Fixed`, `### Added`, `### Changed`, and `### Removed`. Do not create new `Unreleased` sections.
- Keep causes, solution rationale, behavior to preserve, known useful paths/tests, and unresolved limitations in `NOTES.md`, following its existing conventions. Keep durable, verified repository instructions in `AGENTS.md`; avoid duplicate summaries, session logs, and one-time rules.
- Read only relevant sections needed to avoid duplicates or contradictions. Reuse task evidence; inspect the smallest relevant diff or source excerpt only for missing material facts, without routinely rebuilding history.
- For a full update, consider all three root files and create missing files with minimal headings. Leave existing files unchanged when there is nothing material to add.
- Review the documentation diff and unresolved references, and run required documentation checks. Documentation-only changes do not require rerunning implementation tests solely to record the work.
