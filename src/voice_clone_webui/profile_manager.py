"""Voice clone profile save/load management module.

Profiles store the following:
  - voice_clone_prompt: Reusable OmniVoice prompt
  - ref_text: Transcription text of the reference audio
  - ref_audio_wav: Binary data of the reference audio (backup for regeneration)
  - ref_audio_sr: Sample rate of the reference audio
"""

import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch

from voice_clone_webui.config import OMNIVOICE_MODEL_ID, VOICE_PROFILES_DIR


def _profile_path(name: str) -> Path:
    """Accept only a single local filename, including when loading or deleting."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Invalid profile name.")
    path = VOICE_PROFILES_DIR / f"{name}.pt"
    if path.resolve().parent != VOICE_PROFILES_DIR.resolve() or path.is_symlink():
        raise ValueError("Profile must be stored in the profile directory.")
    return path


def _cpu_value(value):
    """Store only tensors and plain containers, independent of the GPU device."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value) and not isinstance(value, type):
        return _cpu_value(asdict(value))
    if isinstance(value, dict):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported profile value: {type(value).__name__}")


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
        "format_version": 2,
        "engine": "omnivoice",
        "model_id": OMNIVOICE_MODEL_ID,
        "name": name,
        "voice_clone_prompt": _cpu_value(voice_clone_prompt),
        "ref_text": ref_text,
        "ref_audio_wav": (
            torch.from_numpy(np.asarray(audio_data).copy())
            if audio_data is not None else None
        ),
        "ref_audio_sr": sr,
    }

    # Sanitize filename (basic safety measure)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    safe_name = safe_name.strip()
    if not safe_name:
        safe_name = "unnamed_profile"

    save_path = _profile_path(safe_name)
    temporary_path = VOICE_PROFILES_DIR / f".{uuid4().hex}.tmp"
    try:
        torch.save(profile, temporary_path)
        temporary_path.replace(save_path)
    finally:
        temporary_path.unlink(missing_ok=True)

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
    load_path = _profile_path(name)
    if not load_path.exists():
        raise FileNotFoundError(f"Profile not found: {name}")

    try:
        profile = torch.load(load_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "This profile uses an unsupported or legacy pickle format. "
            "Recreate it from its original reference audio."
        ) from exc
    if not isinstance(profile, dict) or profile.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported profile format; recreate the profile.")
    if profile["format_version"] == 1:
        # Reuse only the reference data; cached tokens belong to a different engine.
        if profile.get("ref_audio_wav") is None or not profile.get("ref_text"):
            raise ValueError("This older profile needs its original reference audio and transcript.")
        profile["voice_clone_prompt"] = None
        profile["needs_prompt_regeneration"] = True
    elif profile.get("engine") != "omnivoice":
        raise ValueError("This profile belongs to an unsupported voice engine.")
    else:
        prompt = profile.get("voice_clone_prompt")
        if prompt is not None:
            from omnivoice import VoiceClonePrompt
            profile["voice_clone_prompt"] = VoiceClonePrompt(**prompt)
    audio = profile.get("ref_audio_wav")
    if audio is not None:
        profile["ref_audio_wav"] = audio.numpy()
    return profile


def list_profiles() -> list[str]:
    """Return a list of saved profile names.

    Returns:
        List of profile names (without extension)
    """
    _ensure_dir()
    profiles = sorted(
        p.stem for p in VOICE_PROFILES_DIR.glob("*.pt")
        if p.is_file() and not p.is_symlink()
    )
    return profiles


def delete_profile(name: str) -> bool:
    """Delete a profile.

    Args:
        name: Profile name

    Returns:
        Whether the deletion was successful
    """
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        return True
    return False
