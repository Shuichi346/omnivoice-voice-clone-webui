"""Application-wide configuration constants."""

from pathlib import Path

# === Project Root ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# === Model Settings ===
# Specify path if model is downloaded locally. Otherwise use HuggingFace ID.
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
WHISPER_MODEL_NAME = "large-v3-turbo"
GENERATION_STEPS = 32
GENERATION_SPEED = 1.0

# === Device Settings (Mac M4 MPS) ===
DEVICE = "mps"
# Use half-precision model weights to reduce MPS memory use.
TORCH_DTYPE_STR = "float16"
# FlashAttention2 is not supported on MPS. Use SDPA instead.
ATTN_IMPLEMENTATION = "sdpa"

# === Directories ===
VOICE_PROFILES_DIR = PROJECT_ROOT / "voice_profiles"
TMP_DIR = PROJECT_ROOT / "tmp"

# === Audio Clip Settings ===
CLIP_CONTEXT_SEC = 8
CLIP_DURATION_SEC = CLIP_CONTEXT_SEC * 2

# === Server Settings ===
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860

# === Whisper Language Settings ===
# Value passed to pywhispercpp's language parameter
# Empty string "" = auto-detect, otherwise ISO 639-1 language code
WHISPER_LANGUAGES = {
    "Auto": "auto",
    "Japanese": "ja",
    "English": "en",
    "Chinese": "zh",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Russian": "ru",
    "Italian": "it",
}

# Whisper language display name list (for dropdown)
WHISPER_LANGUAGE_CHOICES = list(WHISPER_LANGUAGES.keys())

# Whisper language default value
WHISPER_LANGUAGE_DEFAULT = "Auto"
