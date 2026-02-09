"""Voice clone profile save/load management module.

Profiles store the following:
  - voice_clone_prompt: Reusable clone prompt for Qwen TTS
  - ref_text: Transcription text of the reference audio
  - ref_audio_wav: Binary data of the reference audio (backup for regeneration)
  - ref_audio_sr: Sample rate of the reference audio
"""

import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voice_clone_webui.config import VOICE_PROFILES_DIR


def _ensure_dir():
    """Ensure the profile directory exists."""
    VOICE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def save_profile(
    name: str,
    voice_clone_prompt,
    ref_text: str,
    ref_audio_path: str | Path | None = None,
    ref_audio_numpy: np.ndarray | None = None,
    ref_audio_sr: int | None = None,
) -> Path:
    """Save a profile.

    Args:
        name: Profile name (used as filename)
        voice_clone_prompt: Return value of create_voice_clone_prompt
        ref_text: Transcription of the reference audio
        ref_audio_path: Reference audio file path (for WAV reading)
        ref_audio_numpy: Reference audio numpy array (when path is not available)
        ref_audio_sr: Sample rate (required when specifying numpy)

    Returns:
        Path to the saved file
    """
    _ensure_dir()

    # Retrieve reference audio
    if ref_audio_numpy is not None and ref_audio_sr is not None:
        audio_data = ref_audio_numpy
        sr = ref_audio_sr
    elif ref_audio_path is not None:
        audio_data, sr = sf.read(str(ref_audio_path))
    else:
        audio_data = None
        sr = None

    profile = {
        "name": name,
        "voice_clone_prompt": voice_clone_prompt,
        "ref_text": ref_text,
        "ref_audio_wav": audio_data,
        "ref_audio_sr": sr,
    }

    # Sanitize filename (basic safety measure)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    safe_name = safe_name.strip()
    if not safe_name:
        safe_name = "unnamed_profile"

    save_path = VOICE_PROFILES_DIR / f"{safe_name}.pt"
    torch.save(profile, save_path)

    return save_path


def load_profile(name: str) -> dict:
    """Load a profile.

    Args:
        name: Profile name (without extension)

    Returns:
        Profile dictionary:
          - voice_clone_prompt
          - ref_text
          - ref_audio_wav (numpy array or None)
          - ref_audio_sr (int or None)
          - name (str)
    """
    load_path = VOICE_PROFILES_DIR / f"{name}.pt"
    if not load_path.exists():
        raise FileNotFoundError(f"Profile not found: {name}")

    profile = torch.load(load_path, map_location="cpu", weights_only=False)
    return profile


def list_profiles() -> list[str]:
    """Return a list of saved profile names.

    Returns:
        List of profile names (without extension)
    """
    _ensure_dir()
    profiles = sorted(p.stem for p in VOICE_PROFILES_DIR.glob("*.pt"))
    return profiles


def delete_profile(name: str) -> bool:
    """Delete a profile.

    Args:
        name: Profile name

    Returns:
        Whether the deletion was successful
    """
    path = VOICE_PROFILES_DIR / f"{name}.pt"
    if path.exists():
        path.unlink()
        return True
    return False
