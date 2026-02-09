"""Module for downloading audio from YouTube URLs within a specified time range."""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from voice_clone_webui.config import CLIP_DURATION_SEC, TMP_DIR


def _parse_start_seconds(url: str) -> int:
    """Extract the start time in seconds from a YouTube URL.

    Supported formats:
      - ?t=1018 / &t=1018 (seconds only)
      - ?t=16m58s / &t=16m58s (minutes and seconds)
      - ?t=1h2m3s (hours, minutes, and seconds)

    Returns 0 if no parameter is found.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    # Also check fragment (#t=...)
    if "t" not in params:
        fragment_params = parse_qs(parsed.fragment)
        if "t" in fragment_params:
            params["t"] = fragment_params["t"]

    if "t" not in params:
        return 0

    t_value = params["t"][0]

    # Pure numeric (seconds)
    if t_value.isdigit():
        return int(t_value)

    # Hours/minutes/seconds pattern: 1h2m3s, 16m58s, 30s, etc.
    pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.match(pattern, t_value)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    return 0


def _seconds_to_mmss(total_seconds: int) -> str:
    """Convert seconds to MM:SS format."""
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def _strip_time_param(url: str) -> str:
    """Remove the t parameter from the URL (for passing to yt-dlp)."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("t", None)

    # Rebuild query string
    new_query_parts = []
    for key, values in params.items():
        for value in values:
            new_query_parts.append(f"{key}={value}")
    new_query = "&".join(new_query_parts)

    return parsed._replace(query=new_query).geturl()


def _resample_to_16khz(input_path: Path, output_path: Path) -> Path:
    """Resample WAV to 16kHz mono using ffmpeg.

    Whisper requires 16000Hz mono WAV, so this converts audio
    downloaded by yt-dlp from any sample rate.

    Args:
        input_path: Input audio file path
        output_path: Output WAV file path

    Returns:
        Output file path
    """
    cmd = [
        "ffmpeg",
        "-y",                   # Allow overwrite
        "-i", str(input_path),
        "-ar", "16000",         # 16kHz
        "-ac", "1",             # Mono
        "-sample_fmt", "s16",   # 16bit PCM
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        error_detail = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg resampling error: {error_detail}")
    return output_path


def _trim_audio(input_path: Path, output_path: Path, duration_sec: int) -> Path:
    """Precisely trim audio from the beginning to the specified duration using ffmpeg.

    yt-dlp's --download-sections cuts at keyframe boundaries,
    so the downloaded audio may be longer than specified.
    ffmpeg is used for precise trimming.

    Args:
        input_path: Input audio file path
        output_path: Output WAV file path
        duration_sec: Duration after trimming (seconds)

    Returns:
        Output file path
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-t", str(duration_sec),    # Extract specified seconds from the beginning
        "-ar", "16000",             # 16kHz (resampling performed simultaneously)
        "-ac", "1",                 # Mono
        "-sample_fmt", "s16",       # 16bit PCM
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        error_detail = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg trimming error: {error_detail}")
    return output_path


def download_youtube_audio(url: str) -> tuple[Path, str]:
    """Download audio from a YouTube URL.

    yt-dlp is installed as a Python package via pyproject.toml.
    It is invoked via sys.executable -m yt_dlp to ensure the virtual
    environment's yt-dlp is used.

    After downloading, ffmpeg precisely trims to the specified duration
    and converts to 16kHz mono WAV.

    Args:
        url: YouTube URL (recommended with t parameter)

    Returns:
        Tuple of (WAV file path, status message)

    Raises:
        RuntimeError: If yt-dlp or ffmpeg execution fails
    """
    # Create output directory
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate time range
    start_sec = _parse_start_seconds(url)
    end_sec = start_sec + CLIP_DURATION_SEC
    start_mmss = _seconds_to_mmss(start_sec)
    end_mmss = _seconds_to_mmss(end_sec)
    section_spec = f"*{start_mmss}-{end_mmss}"

    # URL with t parameter removed
    clean_url = _strip_time_param(url)

    # Output file paths (yt-dlp download destination → final output after trimming)
    raw_output_path = TMP_DIR / "ref_audio_raw"
    final_output_path = TMP_DIR / "ref_audio.wav"

    # Remove existing files
    for pattern in ["ref_audio_raw.*", "ref_audio.wav"]:
        for existing in TMP_DIR.glob(pattern):
            existing.unlink()

    # Build yt-dlp command (executed as Python module)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--download-sections", section_spec,
        "-x",
        "--audio-format", "wav",
        "-o", str(raw_output_path.with_suffix(".%(ext)s")),
        "--no-playlist",
        "--quiet",
        clean_url,
    ]

    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        error_detail = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"yt-dlp error: {error_detail}")

    # Verify yt-dlp output file
    raw_wav_path = raw_output_path.with_suffix(".wav")
    if not raw_wav_path.exists():
        candidates = list(TMP_DIR.glob("ref_audio_raw.*"))
        if candidates:
            raw_wav_path = candidates[0]
        else:
            raise RuntimeError("Failed to download audio file.")

    # Precisely trim to 3 seconds + convert to 16kHz mono with ffmpeg (single step)
    _trim_audio(raw_wav_path, final_output_path, CLIP_DURATION_SEC)

    # Remove intermediate file
    if raw_wav_path.exists() and raw_wav_path != final_output_path:
        raw_wav_path.unlink()

    status = f"Download complete: {start_mmss} ~ {end_mmss} ({CLIP_DURATION_SEC} seconds)"
    return final_output_path, status
