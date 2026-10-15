"""OmniVoice Voice Clone WebUI main application.

WebUI using Gradio 6.x.
Retrieves reference audio from YouTube URLs, microphone recordings,
or uploaded audio files, and creates/manages voice clone profiles
and generates speech.
"""

import filecmp
import shutil
import subprocess
import traceback
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import soundfile as sf

from voice_clone_webui.config import (
    SERVER_NAME,
    SERVER_PORT,
    WHISPER_LANGUAGE_CHOICES,
    WHISPER_LANGUAGE_DEFAULT,
    CLIP_CONTEXT_SEC,
    CLIP_DURATION_SEC,
    TMP_DIR,
    VOICE_PROFILES_DIR,
)
from voice_clone_webui.audio_downloader import download_youtube_audio
from voice_clone_webui.transcriber import transcribe_audio
from voice_clone_webui.voice_cloner import (
    create_clone_prompt,
    generate_speech,
    get_supported_languages,
    _normalize_audio,
)
from voice_clone_webui.profile_manager import (
    save_profile,
    load_profile,
    list_profiles,
)


# Supported audio file extensions for upload
_SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".webm",
}


def _ensure_dirs():
    """Create required directories."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def _reference_directory(session_id: str | None) -> Path:
    """Keep a browser session's audio together for deferred Gradio cleanup."""
    directory = TMP_DIR if session_id is None else TMP_DIR / f"session_{session_id}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cleanup_session(session_id: str):
    """Remove only application-owned audio when Gradio expires session state."""
    if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
        return
    directory = TMP_DIR / f"session_{session_id}"
    if not directory.is_symlink() and directory.resolve().parent == TMP_DIR.resolve():
        shutil.rmtree(directory, ignore_errors=True)


def _cache_prompt(prompt, audio_path: str | None, ref_text: str) -> dict:
    """Bind a prompt to its inputs to reject stale overlapping callback results."""
    return {"prompt": prompt, "audio_path": audio_path, "ref_text": ref_text.strip()}


def _convert_to_16khz_mono_wav(input_path: Path, output_path: Path) -> Path:
    """Convert an audio file to 16kHz mono WAV using ffmpeg.

    Args:
        input_path: Input audio file path.
        output_path: Output WAV file path.

    Returns:
        Output file path.

    Raises:
        RuntimeError: If ffmpeg conversion fails.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        error_detail = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg conversion error: {error_detail}")
    return output_path


# ============================================================
# Callback Functions
# ============================================================

def on_fetch_youtube(url: str, whisper_lang: str, session_id: str | None = None):
    """Fetch audio from a YouTube URL and transcribe it.

    Returns:
        (reference audio path, transcription text, status, audio path, cleared prompt)
    """
    if not url or not url.strip():
        return None, "", "Error: Please enter a URL.", None, None

    try:
        audio_path, dl_status = download_youtube_audio(
            url.strip(), output_dir=_reference_directory(session_id),
        )
        ref_text = transcribe_audio(audio_path, language=whisper_lang)

        status = f"{dl_status}\nTranscription complete: {ref_text[:80]}..."
        return str(audio_path), ref_text, status, str(audio_path), None

    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {e}"
        traceback.print_exc()
        return None, "", error_msg, None, None


def on_mic_recording_stop(audio_data, whisper_lang: str, session_id: str | None = None):
    """Handle microphone recording stop event.

    Args:
        audio_data: Input from Gradio Audio component.
            type="numpy": (sample_rate, numpy_array)
            type="filepath": file path string
        whisper_lang: Whisper language setting.

    Returns:
        (reference audio path, transcription text, status, audio path, cleared prompt)
    """
    if audio_data is None:
        return None, "", "No recording data available.", None, None

    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_data, str):
            audio_path = Path(audio_data)
        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, wav = audio_data
            wav = _normalize_audio(wav)
            if wav.size == 0 or not np.isfinite(wav).all() or sr <= 0:
                raise ValueError("Recording must contain finite audio samples.")
            audio_path = _reference_directory(session_id) / f"mic_{uuid4().hex}.wav"
            sf.write(str(audio_path), wav, sr)
        else:
            return None, "", "Unsupported audio format.", None, None

        ref_text = transcribe_audio(audio_path, language=whisper_lang)

        status = f"Recording captured.\nTranscription complete: {ref_text[:80]}..."
        return str(audio_path), ref_text, status, str(audio_path), None

    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {e}"
        traceback.print_exc()
        return None, "", error_msg, None, None


def on_audio_file_upload(
    file_path: str | None, whisper_lang: str, session_id: str | None = None,
):
    """Handle uploaded audio file.

    Converts the uploaded file to 16kHz mono WAV, then transcribes it.

    Args:
        file_path: File path string from Gradio Audio component
            (type="filepath").
        whisper_lang: Whisper language setting.

    Returns:
        (reference audio path, transcription text, status, audio path, cleared prompt)
    """
    if file_path is None:
        return None, "", "No file uploaded.", None, None

    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        source_path = Path(file_path)

        if not source_path.exists():
            return None, "", "Error: Uploaded file not found.", None, None

        # Validate file extension
        suffix = source_path.suffix.lower()
        if suffix not in _SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(_SUPPORTED_AUDIO_EXTENSIONS))
            return (
                None,
                "",
                f"Error: Unsupported file format '{suffix}'. "
                f"Supported formats: {supported}",
                None,
                None,
            )

        # Convert to 16kHz mono WAV
        converted_path = _reference_directory(session_id) / f"uploaded_{uuid4().hex}.wav"
        _convert_to_16khz_mono_wav(source_path, converted_path)

        # Verify the converted file
        info = sf.info(str(converted_path))
        if info.frames == 0:
            raise ValueError("Reference audio must not be empty.")
        duration_sec = info.duration

        # Transcribe
        ref_text = transcribe_audio(converted_path, language=whisper_lang)

        status = (
            f"File loaded: {source_path.name} "
            f"(duration: {duration_sec:.1f}s, converted to 16kHz mono WAV)\n"
            f"Transcription complete: {ref_text[:80]}..."
        )
        return str(converted_path), ref_text, status, str(converted_path), None

    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {e}"
        traceback.print_exc()
        return None, "", error_msg, None, None


def on_reference_audio_change(
    audio_path: str | None,
    current_audio_path: str | None,
    whisper_lang: str,
    session_id: str | None = None,
):
    """Apply waveform edits without retriggering edits from backend display updates."""
    if audio_path is None and current_audio_path is None:
        # A failed source callback has already cleared both and reported its error.
        yield tuple(gr.skip() for _ in range(4))
        return
    # Gradio may copy backend output into its file cache before displaying it.
    # Compare bytes too so that fetching/loading audio does not transcribe twice.
    if audio_path and current_audio_path:
        displayed = Path(audio_path)
        current = Path(current_audio_path)
        if displayed.is_file() and current.is_file() and (
            displayed.resolve() == current.resolve()
            or filecmp.cmp(displayed, current, shallow=False)
        ):
            yield tuple(gr.skip() for _ in range(4))
            return

    # Invalidate the old reference before the potentially slow transcription.
    yield "", "Updating reference audio..." if audio_path else "Reference audio cleared.", None, None
    if not audio_path:
        return

    _, ref_text, status, path, prompt = on_audio_file_upload(
        audio_path, whisper_lang, session_id,
    )
    yield ref_text, status, path, prompt


def on_save_profile(
    profile_name: str,
    ref_text: str,
    ref_audio_path_state: str | None,
):
    """Save a profile.

    Returns:
        (status message, profile list update, prompt State)
    """
    if not profile_name or not profile_name.strip():
        return "Error: Please enter a profile name.", gr.update(), None

    if not ref_audio_path_state:
        return (
            "Error: No reference audio available. "
            "Please fetch audio first.",
            gr.update(),
            None,
        )

    if not ref_text or not ref_text.strip():
        return "Error: No transcription text available.", gr.update(), None

    try:
        prompt = create_clone_prompt(ref_audio_path_state, ref_text.strip())

        save_path = save_profile(
            name=profile_name.strip(),
            voice_clone_prompt=prompt,
            ref_text=ref_text.strip(),
            ref_audio_path=ref_audio_path_state,
        )

        profiles = list_profiles()
        status = (
            f"Profile \"{profile_name.strip()}\" saved: {save_path.name}"
        )
        return (
            status, gr.update(choices=profiles, value=save_path.stem),
            _cache_prompt(prompt, ref_audio_path_state, ref_text),
        )

    except Exception as e:
        error_msg = f"Save error: {type(e).__name__}: {e}"
        traceback.print_exc()
        return error_msg, gr.update(), None


def on_load_profile(selected_profile: str | None, session_id: str | None = None):
    """Load a saved profile.

    Returns:
        (status message, transcription text, reference audio,
         prompt State, audio path State)
    """
    if not selected_profile:
        return "Error: Please select a profile.", "", None, None, None

    try:
        profile = load_profile(selected_profile)
        ref_text = profile.get("ref_text", "")
        prompt = profile.get("voice_clone_prompt")

        ref_audio_display = None
        ref_audio_path_state = None
        audio_wav = profile.get("ref_audio_wav")
        audio_sr = profile.get("ref_audio_sr")
        if audio_wav is not None and audio_sr is not None:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            restore_path = _reference_directory(session_id) / f"profile_{uuid4().hex}.wav"
            sf.write(str(restore_path), audio_wav, audio_sr)
            ref_audio_display = str(restore_path)
            ref_audio_path_state = str(restore_path)

        status = f"Profile \"{selected_profile}\" loaded."
        if profile.get("needs_prompt_regeneration"):
            status += " OmniVoice will use the saved audio; save again to cache its new prompt."
        cached = _cache_prompt(prompt, ref_audio_path_state, ref_text) if prompt is not None else None
        return status, ref_text, ref_audio_display, cached, ref_audio_path_state

    except Exception as e:
        error_msg = f"Load error: {type(e).__name__}: {e}"
        traceback.print_exc()
        return error_msg, "", None, None, None


def on_refresh_profiles():
    """Refresh the profile list."""
    profiles = list_profiles()
    return gr.update(choices=profiles)


def on_generate_speech(
    text: str,
    language: str,
    prompt_state,
    ref_audio_path_state: str | None,
    ref_text: str,
):
    """Generate speech.

    Returns:
        (generated audio, status message)
    """
    if not text or not text.strip():
        return None, "Error: Please enter text to synthesize."

    try:
        if (
            prompt_state is not None
            and prompt_state["audio_path"] == ref_audio_path_state
            and prompt_state["ref_text"] == (ref_text or "").strip()
        ):
            wav, sr = generate_speech(
                text=text.strip(),
                language=language,
                voice_clone_prompt=prompt_state["prompt"],
            )
        elif ref_audio_path_state and ref_text and ref_text.strip():
            from voice_clone_webui.voice_cloner import generate_speech_direct
            wav, sr = generate_speech_direct(
                text=text.strip(),
                language=language,
                ref_audio_path=ref_audio_path_state,
                ref_text=ref_text.strip(),
            )
        else:
            return (
                None,
                "Error: No reference audio available. "
                "Please fetch audio or load a profile.",
            )

        return (sr, wav), "Speech generation complete!"

    except Exception as e:
        error_msg = f"Generation error: {type(e).__name__}: {e}"
        traceback.print_exc()
        return None, error_msg


# ============================================================
# Gradio UI Construction
# ============================================================

def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""

    _ensure_dirs()

    with gr.Blocks(title="OmniVoice Clone WebUI") as demo:

        # === Hidden State Components ===
        prompt_state = gr.State(value=None)
        ref_audio_path_state = gr.State(value=None)
        session_state = gr.State(value=lambda: uuid4().hex, delete_callback=_cleanup_session)

        # === Header ===
        gr.Markdown(
            "# OmniVoice Voice Clone WebUI\n"
            "Clone a voice from a YouTube URL, microphone recording, "
            "or uploaded audio file, then synthesize any text."
        )

        # === Section 1: Audio Source ===
        gr.Markdown("## 1. Get Reference Audio")

        whisper_lang_dropdown = gr.Dropdown(
            label="Transcription Language (Whisper)",
            choices=WHISPER_LANGUAGE_CHOICES,
            value=WHISPER_LANGUAGE_DEFAULT,
            interactive=True,
        )

        with gr.Tabs():
            # --- Tab 1: YouTube URL ---
            with gr.TabItem("YouTube URL"):
                gr.Markdown(
                    "Enter a YouTube URL with a time parameter. "
                    f"Fetch {CLIP_CONTEXT_SEC} seconds before and after it "
                    f"({CLIP_DURATION_SEC} seconds total), then trim below.  \n"
                    "Example: `https://youtu.be/xxxxx?t=1018` "
                    "(fetches 1010s–1026s). "
                    "Near the start or end of a video, only available audio is fetched."
                )
                with gr.Row():
                    url_input = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://youtu.be/xxxxx?t=1018",
                        scale=4,
                    )
                    fetch_btn = gr.Button(
                        "Fetch Audio", variant="primary", scale=1,
                    )

            # --- Tab 2: Microphone Recording ---
            with gr.TabItem("Microphone Recording"):
                gr.Markdown(
                    "Record approximately 3 seconds of audio "
                    "using your microphone."
                )
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Microphone Recording",
                    format="wav",
                )

            # --- Tab 3: Audio File Upload ---
            with gr.TabItem("Audio File Upload"):
                gr.Markdown(
                    "Upload your own audio file as reference. "
                    "Supported formats: WAV, MP3, FLAC, OGG, OPUS, "
                    "M4A, AAC, WMA, WebM.  \n"
                    "The file will be converted to 16kHz mono WAV "
                    "and transcribed automatically."
                )
                with gr.Row():
                    file_upload_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Upload Audio File",
                        scale=4,
                    )
                    upload_btn = gr.Button(
                        "Process File", variant="primary", scale=1,
                    )

        # === Section 2: Reference Confirmation ===
        gr.Markdown("## 2. Review Reference Audio & Transcription")
        gr.Markdown(
            "Use the reference audio's trim tool, drag the selection edges, "
            "then confirm to keep the selected range. "
            "The transcription updates automatically; wait for it before saving or generating. "
            "Use the reset control to restore the previous audio."
        )

        with gr.Row():
            ref_audio_display = gr.Audio(
                label="Reference Audio",
                type="filepath",
                sources=["upload"],
                format="wav",
                interactive=True,
                editable=True,
                waveform_options=gr.WaveformOptions(sample_rate=16000),
                scale=1,
            )
            ref_text_box = gr.Textbox(
                label="Transcription Result (editable)",
                lines=3,
                placeholder=(
                    "The transcription result will appear here. "
                    "Edit as needed."
                ),
                scale=2,
            )

        status_box = gr.Textbox(
            label="Status",
            lines=2,
            interactive=False,
        )

        # === Section 3: Profile Management ===
        gr.Markdown("## 3. Profile Management")
        gr.Markdown(
            "Save clone prompts with a name, and load them from "
            "the dropdown next time."
        )

        with gr.Row():
            profile_name_input = gr.Textbox(
                label="Profile Name",
                placeholder="e.g. tanaka_san_voice",
                scale=2,
            )
            save_profile_btn = gr.Button(
                "Save Profile", variant="primary", scale=1,
            )

        with gr.Row():
            profile_dropdown = gr.Dropdown(
                label="Saved Profiles",
                choices=list_profiles(),
                interactive=True,
                scale=2,
            )
            load_profile_btn = gr.Button("Load", scale=1)
            refresh_profiles_btn = gr.Button("Refresh", scale=1)

        profile_status_box = gr.Textbox(
            label="Profile Operation Status",
            lines=1,
            interactive=False,
        )

        # === Section 4: Speech Generation ===
        gr.Markdown("## 4. Speech Generation")

        with gr.Row():
            tts_text_input = gr.Textbox(
                label="Text to Synthesize",
                lines=5,
                placeholder="Enter the text you want to synthesize here.",
                scale=3,
            )
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    label="TTS Language (OmniVoice)",
                    choices=get_supported_languages(),
                    value="Auto",
                    interactive=True,
                )
                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg",
                )

        output_audio = gr.Audio(
            label="Generated Audio",
            type="numpy",
            interactive=False,
        )

        generate_status_box = gr.Textbox(
            label="Generation Status",
            lines=1,
            interactive=False,
        )

        # ============================================================
        # Event Bindings
        # ============================================================

        # YouTube audio fetch
        fetch_btn.click(
            fn=on_fetch_youtube,
            inputs=[url_input, whisper_lang_dropdown, session_state],
            outputs=[
                ref_audio_display,
                ref_text_box,
                status_box,
                ref_audio_path_state,
                prompt_state,
            ],
            concurrency_id="reference_audio",
        )

        # One event handles completed recordings and clearing the input.
        mic_input.change(
            fn=on_mic_recording_stop,
            inputs=[mic_input, whisper_lang_dropdown, session_state],
            outputs=[
                ref_audio_display,
                ref_text_box,
                status_box,
                ref_audio_path_state,
                prompt_state,
            ],
            concurrency_id="reference_audio",
        )

        # Audio file upload - process button click
        upload_btn.click(
            fn=on_audio_file_upload,
            inputs=[file_upload_input, whisper_lang_dropdown, session_state],
            outputs=[
                ref_audio_display,
                ref_text_box,
                status_box,
                ref_audio_path_state,
                prompt_state,
            ],
            concurrency_id="reference_audio",
        )

        # Edits emit change rather than input in Gradio's waveform editor.
        # Do not output to the audio component here, which would form a loop.
        ref_audio_display.change(
            fn=on_reference_audio_change,
            inputs=[
                ref_audio_display, ref_audio_path_state,
                whisper_lang_dropdown, session_state,
            ],
            outputs=[ref_text_box, status_box, ref_audio_path_state, prompt_state],
            concurrency_id="reference_audio",
            trigger_mode="always_last",
        )

        # A cached prompt includes the transcript, so manual edits invalidate it.
        ref_text_box.input(fn=lambda: None, inputs=[], outputs=[prompt_state])

        # Save profile
        save_profile_btn.click(
            fn=on_save_profile,
            inputs=[profile_name_input, ref_text_box, ref_audio_path_state],
            outputs=[profile_status_box, profile_dropdown, prompt_state],
        )

        # Load profile
        load_profile_btn.click(
            fn=on_load_profile,
            inputs=[profile_dropdown, session_state],
            outputs=[
                profile_status_box,
                ref_text_box,
                ref_audio_display,
                prompt_state,
                ref_audio_path_state,
            ],
            concurrency_id="reference_audio",
        )

        # Refresh profile list
        refresh_profiles_btn.click(
            fn=on_refresh_profiles,
            inputs=[],
            outputs=[profile_dropdown],
        )

        # Speech generation
        generate_btn.click(
            fn=on_generate_speech,
            inputs=[
                tts_text_input,
                language_dropdown,
                prompt_state,
                ref_audio_path_state,
                ref_text_box,
            ],
            outputs=[output_audio, generate_status_box],
        )

    return demo


# ============================================================
# Entry Point
# ============================================================

def main():
    """Application main entry point."""
    demo = build_ui()
    demo.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
    )


if __name__ == "__main__":
    main()
