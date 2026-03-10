"""Speech Transcription Pipeline — Local Python version."""

from .audio_compat import ensure_huggingface_hub_compat, ensure_torchaudio_compat

ensure_torchaudio_compat()
ensure_huggingface_hub_compat()
