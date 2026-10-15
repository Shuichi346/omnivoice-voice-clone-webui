"""Voice cloning and synthesis with OmniVoice on MPS or CPU."""

from functools import wraps
import os
from pathlib import Path
from threading import RLock

import numpy as np
import torch

from voice_clone_webui.config import (
    OMNIVOICE_MODEL_ID,
    DEVICE,
    TORCH_DTYPE_STR,
    ATTN_IMPLEMENTATION,
    GENERATION_STEPS,
    GENERATION_SPEED,
)

# Module-level model holder (lazy initialization)
_tts_model = None
_tts_lock = RLock()


def _serialized_inference(function):
    """Protect lazy loading and inference on the shared model across UI events."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _tts_lock, torch.inference_mode():
            return function(*args, **kwargs)
    return wrapped

# dtype mapping
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _get_tts_model():
    """Load OmniVoice once, retaining its upstream CPU audio tokenizer on MPS."""
    global _tts_model
    if _tts_model is None:
        if DEVICE == "mps":
            # Parallel dtype conversion during weight loading can crash MPS.
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        from omnivoice import OmniVoice

        if DEVICE not in {"mps", "cpu"}:
            raise ValueError("The supported devices are mps and cpu.")
        dtype = _DTYPE_MAP[TORCH_DTYPE_STR]

        _tts_model = OmniVoice.from_pretrained(
            OMNIVOICE_MODEL_ID,
            device_map=DEVICE,
            dtype=dtype,
            attn_implementation=ATTN_IMPLEMENTATION,
            load_asr=False,
        )
    return _tts_model


def get_supported_languages() -> list[str]:
    """List OmniVoice languages without downloading or loading model weights."""
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

    return ["Auto", *sorted(lang_display_name(name) for name in LANG_NAMES)]


def _reference_text(ref_text: str) -> str:
    """Keep transcription local to pywhispercpp instead of loading another ASR."""
    if not ref_text or not ref_text.strip():
        raise ValueError("A reference transcript is required for voice cloning.")
    return ref_text.strip()


def _speech_result(wavs, model) -> tuple[np.ndarray, int]:
    """Validate model output before returning it to Gradio."""
    if not wavs:
        raise RuntimeError("OmniVoice returned no audio.")
    wav = np.asarray(wavs[0], dtype=np.float32)
    if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
        raise RuntimeError("OmniVoice returned empty or invalid audio.")
    return wav, int(model.sampling_rate)


def _normalize_audio(wav, eps=1e-12):
    """Normalize audio data to float32 [-1, 1]."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    y = np.clip(y, -1.0, 1.0)

    # Stereo → Mono conversion
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


@_serialized_inference
def create_clone_prompt(
    ref_audio_path: str | Path,
    ref_text: str,
) -> object:
    """Create a clone prompt from reference audio and transcription.

    Args:
        ref_audio_path: Path to the reference audio file
        ref_text: Transcription text of the reference audio

    Returns:
        voice_clone_prompt object (reusable)
    """
    ref_text = _reference_text(ref_text)
    model = _get_tts_model()
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )
    return prompt


@_serialized_inference
def create_clone_prompt_from_numpy(
    audio_data: np.ndarray,
    sample_rate: int,
    ref_text: str,
) -> object:
    """Create a clone prompt from a NumPy array of reference audio.

    Args:
        audio_data: Audio data (numpy array)
        sample_rate: Sample rate
        ref_text: Transcription text of the reference audio

    Returns:
        voice_clone_prompt object (reusable)
    """
    ref_text = _reference_text(ref_text)
    wav = _normalize_audio(audio_data)
    if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all() or sample_rate <= 0:
        raise ValueError("Reference audio must contain finite mono samples at a positive rate.")
    model = _get_tts_model()
    prompt = model.create_voice_clone_prompt(
        ref_audio=(torch.from_numpy(wav), sample_rate),
        ref_text=ref_text,
    )
    return prompt


@_serialized_inference
def generate_speech(
    text: str,
    language: str,
    voice_clone_prompt,
) -> tuple[np.ndarray, int]:
    """Generate speech using a clone prompt.

    Args:
        text: Text to synthesize
        language: Language specification ("Auto" for auto-detection)
        voice_clone_prompt: Prompt created by create_clone_prompt

    Returns:
        Tuple of (audio data numpy array, sample rate)
    """
    model = _get_tts_model()

    # MPS device synchronization
    if DEVICE == "mps":
        torch.mps.synchronize()

    wavs = model.generate(
        text=text,
        language=None if language == "Auto" else language,
        voice_clone_prompt=voice_clone_prompt,
        num_step=GENERATION_STEPS,
        speed=GENERATION_SPEED,
    )

    if DEVICE == "mps":
        torch.mps.synchronize()

    return _speech_result(wavs, model)


@_serialized_inference
def generate_speech_direct(
    text: str,
    language: str,
    ref_audio_path: str | Path,
    ref_text: str,
) -> tuple[np.ndarray, int]:
    """Generate speech directly from reference audio (without saving prompt).

    Args:
        text: Text to synthesize
        language: Language specification
        ref_audio_path: Path to the reference audio file
        ref_text: Transcription of the reference audio

    Returns:
        Tuple of (audio data numpy array, sample rate)
    """
    ref_text = _reference_text(ref_text)
    model = _get_tts_model()

    if DEVICE == "mps":
        torch.mps.synchronize()

    wavs = model.generate(
        text=text,
        language=None if language == "Auto" else language,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        num_step=GENERATION_STEPS,
        speed=GENERATION_SPEED,
    )

    if DEVICE == "mps":
        torch.mps.synchronize()

    return _speech_result(wavs, model)
