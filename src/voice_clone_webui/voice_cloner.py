"""Voice cloning module using Qwen3-TTS."""

from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from voice_clone_webui.config import (
    QWEN_TTS_MODEL_ID,
    DEVICE,
    TORCH_DTYPE_STR,
    ATTN_IMPLEMENTATION,
)

# Module-level model holder (lazy initialization)
_tts_model = None

# dtype mapping
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _get_tts_model():
    """Get the singleton instance of the Qwen3-TTS model."""
    global _tts_model
    if _tts_model is None:
        from qwen_tts import Qwen3TTSModel

        dtype = _DTYPE_MAP.get(TORCH_DTYPE_STR, torch.float32)

        _tts_model = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL_ID,
            device_map=DEVICE,
            dtype=dtype,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
    return _tts_model


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


def create_clone_prompt(
    ref_audio_path: str | Path,
    ref_text: str,
) -> dict:
    """Create a clone prompt from reference audio and transcription.

    Args:
        ref_audio_path: Path to the reference audio file
        ref_text: Transcription text of the reference audio

    Returns:
        voice_clone_prompt object (reusable)
    """
    model = _get_tts_model()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    return prompt_items


def create_clone_prompt_from_numpy(
    audio_data: np.ndarray,
    sample_rate: int,
    ref_text: str,
) -> dict:
    """Create a clone prompt from a NumPy array of reference audio.

    Args:
        audio_data: Audio data (numpy array)
        sample_rate: Sample rate
        ref_text: Transcription text of the reference audio

    Returns:
        voice_clone_prompt object (reusable)
    """
    model = _get_tts_model()
    wav = _normalize_audio(audio_data)
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=(wav, sample_rate),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    return prompt_items


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

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=2048,
    )

    if DEVICE == "mps":
        torch.mps.synchronize()

    return wavs[0], sr


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
    model = _get_tts_model()

    if DEVICE == "mps":
        torch.mps.synchronize()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        max_new_tokens=2048,
    )

    if DEVICE == "mps":
        torch.mps.synchronize()

    return wavs[0], sr