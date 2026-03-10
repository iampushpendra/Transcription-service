"""
Transcription module — dual-engine support for Hinglish and standard Whisper.

Engines:
    - "hinglish": Oriserve/Whisper-Hindi2Hinglish-Prime via HuggingFace Transformers
    - "whisper":  OpenAI Whisper (original)
"""

import math
from typing import Any

import torch
import torchaudio

from .config import PipelineConfig


# ── Model Loading ────────────────────────────────────────────────────


def load_model(cfg: PipelineConfig, device: str) -> tuple[Any, str]:
    """
    Load the ASR model based on the configured engine.

    Returns (model_or_pipe, engine_name).
    """
    engine = cfg.asr_engine

    if engine == "hinglish":
        return _load_hinglish(cfg.hinglish_model, device), "hinglish"
    else:
        return _load_whisper(cfg.whisper_model, device), "whisper"


def _load_hinglish(model_id: str, device: str):
    """Load Oriserve Hinglish Whisper model via HuggingFace Transformers."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    # MPS doesn't reliably support fp16 for Whisper-class models
    use_cuda = (device == "cuda")
    compute_dtype = torch.float16 if use_cuda else torch.float32
    # Ensure HuggingFace executes natively on MPS using f32 to avoid fp16 Apple bugs
    pipe_device = device

    print(f"🧠 Loading Hinglish model: {model_id} on {pipe_device}...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(pipe_device)

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"   ✅ Hinglish model & processor ready ({pipe_device})")
    
    # Return a tuple so we can unpack it later
    return (model, processor, pipe_device, compute_dtype)


def _load_whisper(model_name: str, device: str):
    """Load OpenAI Whisper model."""
    import whisper

    if device == "mps":
        print("   ⚠️  Falling back to CPU for Whisper to avoid MPS float64 bug")
        device = "cpu"

    print(f"🧠 Loading Whisper: {model_name} on {device}")
    model = whisper.load_model(model_name, device=device)
    print(f"   ✅ Whisper ready ({device})")
    return model


# ── Chunking ─────────────────────────────────────────────────────────


def make_chunks(
    speaker_segs: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """
    Split speaker segments into ASR-friendly chunks.

    Long segments are split into overlapping sub-chunks of
    cfg.chunk_seconds with cfg.overlap_seconds overlap.
    """
    chunk_s = cfg.chunk_seconds
    overlap = cfg.overlap_seconds
    min_tail = cfg.min_tail_seconds
    chunks = []

    for seg in speaker_segs:
        t0 = seg.get("t0", seg.get("start"))
        t1 = seg.get("t1", seg.get("end"))
        dur = t1 - t0
        speaker = seg.get("speaker", seg.get("role", "unknown"))

        if dur <= chunk_s:
            chunks.append({"t0": t0, "t1": t1, "speaker": speaker})
        else:
            n = math.ceil((dur - overlap) / (chunk_s - overlap))
            step = (dur - overlap) / max(n, 1)
            for i in range(n):
                c_start = t0 + i * step
                c_end = min(c_start + chunk_s, t1)
                if t1 - c_end < min_tail and i == n - 1:
                    c_end = t1
                chunks.append({"t0": c_start, "t1": c_end, "speaker": speaker})

    print(f"🔪 {len(speaker_segs)} segments → {len(chunks)} ASR chunks")
    return chunks


# ── Transcription ────────────────────────────────────────────────────


def transcribe_chunks(
    chunks: list[dict],
    audio_path: str,
    model: Any,
    cfg: PipelineConfig,
    device: str,
    progress_callback: callable = None
) -> list[dict]:
    """
    Transcribe a list of audio chunks.

    Routes to the correct engine based on cfg.asr_engine.
    """
    if cfg.asr_engine == "hinglish":
        return _transcribe_hinglish(chunks, audio_path, model, cfg, progress_callback)
    else:
        return _transcribe_whisper(chunks, audio_path, model, cfg, device, progress_callback)


def _transcribe_hinglish(
    chunks: list[dict],
    audio_path: str,
    hinglish_bundle: tuple,
    cfg: PipelineConfig,
    progress_callback: callable = None
) -> list[dict]:
    """
    Transcribe using raw HuggingFace model.generate() (Hinglish model).
    """
    from tqdm import tqdm
    import numpy as np

    print(f"📝 Hinglish ASR on {len(chunks)} chunks...")
    
    model, processor, device, dtype = hinglish_bundle

    # Load full audio
    audio, sr = torchaudio.load(audio_path)
    if sr != cfg.sample_rate:
        audio = torchaudio.transforms.Resample(sr, cfg.sample_rate)(audio)
        sr = cfg.sample_rate
    audio_np = audio.squeeze().numpy()

    # Prepare forced decoder ID for Hinglish
    # The Hinglish model often maps to "hi" or "en", we let it auto-detect or force if needed.
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    results = []
    
    total_chunks = len(chunks)
    for i, c in enumerate(tqdm(chunks, desc="   ASR (seq)")):
        if progress_callback:
            progress_callback(i, total_chunks)
            
        start_s = int(c["t0"] * sr)
        end_s = int(c["t1"] * sr)
        seg_audio = audio_np[start_s:end_s].astype(np.float32)

        if len(seg_audio) < int(sr * 0.2):
            continue

        try:
            # 1. Feature extraction
            input_features = processor(
                seg_audio, sampling_rate=sr, return_tensors="pt"
            ).input_features.to(device, dtype=dtype)

            # 2. Generate
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256,
                    # Fallback to greedy decoding for max speed, or uncomment below for stability
                    # num_beams=3,                 
                    # repetition_penalty=1.1,      
                    # temperature=(0.0, 0.2, 0.4)  
                )
            
            # 3. Decode
            text = processor.decode(predicted_ids[0], skip_special_tokens=True).strip()
            
            if text:
                results.append({
                    "speaker": c["speaker"],
                    "t0": c["t0"],
                    "t1": c["t1"],
                    "text": text,
                })
        except Exception as e:
            print(f"   ⚠️  Chunk error: {e}")

    print(f"   ✅ {len(results)} segments transcribed")
    return results


def _transcribe_whisper(
    chunks: list[dict],
    audio_path: str,
    model: Any,
    cfg: PipelineConfig,
    device: str,
    progress_callback: callable = None
) -> list[dict]:
    """
    Transcribe using OpenAI Whisper (original engine).
    """
    import numpy as np
    from tqdm import tqdm

    print(f"📝 Whisper ASR ({cfg.whisper_model}) on {len(chunks)} chunks...")

    # Load full audio
    audio, sr = torchaudio.load(audio_path)
    if sr != cfg.sample_rate:
        audio = torchaudio.transforms.Resample(sr, cfg.sample_rate)(audio)
        sr = cfg.sample_rate
    audio_np = audio.squeeze().numpy()

    results = []
    total_chunks = len(chunks)
    for i, c in enumerate(tqdm(chunks, desc="   ASR")):
        if progress_callback:
            progress_callback(i, total_chunks)
            
        start_s = int(c["t0"] * sr)
        end_s = int(c["t1"] * sr)
        seg_audio = audio_np[start_s:end_s].astype(np.float32)

        if len(seg_audio) < int(sr * 0.2):
            continue

        try:
            out = model.transcribe(
                seg_audio,
                language=cfg.language,
                fp16=(device == "cuda"),
                no_speech_threshold=0.5,
                condition_on_previous_text=False,
                initial_prompt=cfg.initial_prompt,
                temperature=(0.0, 0.2, 0.4, 0.8),
                word_timestamps=True,
            )
            text = out["text"].strip()
            if text:
                results.append({
                    "speaker": c["speaker"],
                    "t0": c["t0"],
                    "t1": c["t1"],
                    "text": text,
                })
        except Exception as e:
            print(f"   ⚠️  Chunk error: {e}")

    print(f"   ✅ {len(results)} segments transcribed")
    return results
