"""
Runtime compatibility shims for audio dependencies.
"""

from __future__ import annotations


def ensure_torchaudio_compat() -> None:
    """
    Patch torchaudio API differences for libraries expecting legacy backend APIs.
    """
    try:
        import torchaudio  # noqa: F401
    except Exception:
        return

    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends() -> list[str]:
            return ["soundfile"]

        torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "set_audio_backend"):
        def _set_audio_backend(_backend: str) -> None:
            return None

        torchaudio.set_audio_backend = _set_audio_backend  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "get_audio_backend"):
        def _get_audio_backend() -> str:
            return "soundfile"

        torchaudio.get_audio_backend = _get_audio_backend  # type: ignore[attr-defined]


def ensure_huggingface_hub_compat() -> None:
    """
    Patch huggingface_hub API differences for older libraries (e.g., SpeechBrain).
    """
    try:
        import inspect
        import huggingface_hub as hfh
    except Exception:
        return

    try:
        sig = inspect.signature(hfh.hf_hub_download)
    except Exception:
        return

    if "use_auth_token" in sig.parameters:
        return

    _orig = hfh.hf_hub_download
    RemoteEntryNotFoundError = getattr(hfh.errors, "RemoteEntryNotFoundError", None)

    def _hf_hub_download_compat(*args, use_auth_token=None, token=None, **kwargs):
        if token is None and use_auth_token is not None:
            token = use_auth_token
        try:
            return _orig(*args, token=token, **kwargs)
        except Exception as exc:
            # SpeechBrain expects missing optional custom.py fetches to surface
            # as ValueError and then gracefully continues.
            filename = kwargs.get("filename")
            if (
                RemoteEntryNotFoundError is not None
                and isinstance(exc, RemoteEntryNotFoundError)
                and filename == "custom.py"
            ):
                raise ValueError("File not found on HF hub: custom.py") from exc
            raise

    hfh.hf_hub_download = _hf_hub_download_compat  # type: ignore[assignment]
