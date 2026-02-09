"""Application-wide configuration constants."""

from pathlib import Path

# === Project Root ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# === Model Settings ===
# Specify path if model is downloaded locally. Otherwise use HuggingFace ID.
QWEN_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WHISPER_MODEL_NAME = "large-v3-turbo"

# === Device Settings (Mac M4 MPS) ===
DEVICE = "mps"
# bfloat16/float16 can be unstable on MPS. float32 is safer.
TORCH_DTYPE_STR = "float32"
# FlashAttention2 is not supported on MPS. Use SDPA instead.
ATTN_IMPLEMENTATION = "sdpa"

# === Directories ===
VOICE_PROFILES_DIR = PROJECT_ROOT / "voice_profiles"
TMP_DIR = PROJECT_ROOT / "tmp"

# === Audio Clip Settings ===
CLIP_DURATION_SEC = 3

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

# === Qwen TTS Supported Languages ===
SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]
