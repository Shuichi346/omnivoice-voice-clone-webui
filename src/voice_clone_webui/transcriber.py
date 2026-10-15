"""Audio transcription module using pywhispercpp."""

import subprocess
from pathlib import Path
from threading import RLock
from uuid import uuid4

from voice_clone_webui.config import (
    WHISPER_MODEL_NAME,
    WHISPER_LANGUAGES,
    WHISPER_LANGUAGE_DEFAULT,
    TMP_DIR,
)

# Sample rate required by Whisper
_WHISPER_SAMPLE_RATE = 16000

# Module-level model holder (lazy initialization)
_whisper_model = None
_whisper_lock = RLock()


def _get_whisper_model():
    """Get the singleton instance of the Whisper model."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from pywhispercpp.model import Model
            _whisper_model = Model(
                WHISPER_MODEL_NAME,
                print_realtime=False,
                print_progress=False,
            )
    return _whisper_model


def _ensure_16khz(audio_path: Path) -> Path:
    """Resample audio file to 16kHz using ffmpeg if needed.

    Args:
        audio_path: Input audio file path

    Returns:
        Path to 16kHz WAV file (returns as-is if no conversion needed)
    """
    import soundfile as sf

    info = sf.info(str(audio_path))
    if (
        info.samplerate == _WHISPER_SAMPLE_RATE
        and info.channels == 1
        and info.format == "WAV"
        and info.subtype == "PCM_16"
    ):
        return audio_path

    # Resampling required
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    resampled_path = TMP_DIR / f"whisper_{uuid4().hex}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-ar", str(_WHISPER_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(resampled_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg resampling error: {result.stderr.strip()}")
    except Exception:
        resampled_path.unlink(missing_ok=True)
        raise

    return resampled_path


def transcribe_audio(
    audio_path: str | Path,
    language: str = WHISPER_LANGUAGE_DEFAULT,
) -> str:
    """Transcribe an audio file.

    Automatically resamples to 16kHz if the input is not already at that rate.

    Args:
        audio_path: Path to the audio file (WAV recommended)
        language: Language specification (key from config.WHISPER_LANGUAGES).
                  "Auto" for auto-detection.

    Returns:
        Transcribed text
    """
    audio_path = Path(audio_path)

    # Ensure 16kHz for Whisper
    whisper_input = _ensure_16khz(audio_path)

    # Resolve language code
    try:
        lang_code = WHISPER_LANGUAGES[language]
        # whisper.cpp uses mutable decode parameters and a shared context.
        with _whisper_lock:
            model = _get_whisper_model()
            segments = model.transcribe(str(whisper_input), language=lang_code)
            return " ".join(seg.text for seg in segments).strip()
    finally:
        if whisper_input != audio_path:
            whisper_input.unlink(missing_ok=True)
