"""
Speaker diarization — pyannote.audio primary, NeMo secondary, MFCC fallback.
Also includes VAD∩diarization intersection and role inference.
"""

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .utils import compute_mfcc, load_audio

# Global caching for pyannote pipeline to support fast API restarts
_pyannote_pipeline = None


# ─────────────────────────────────────────────────────────────
# Tier 1: pyannote.audio (best quality)
# ─────────────────────────────────────────────────────────────


def diarize_pyannote(path: str, cfg: PipelineConfig) -> list[dict]:
    """
    Speaker diarization using pyannote.audio 4.x.
    Requires HF_TOKEN with model access.
    """
    import torch
    from pyannote.audio import Pipeline

    print("🗣️  pyannote.audio diarization...")

    global _pyannote_pipeline
    pipeline = _pyannote_pipeline

    if pipeline is None:
        # Models to try in order (3.1 is best, community-1 is fallback)
        models = [
            "pyannote/speaker-diarization-3.1",
        ]

        for model_id in models:
            try:
                print(f"   Trying model: {model_id}")
                # pyannote 4.x uses 'token' (not 'use_auth_token')
                pipeline = Pipeline.from_pretrained(
                    model_id,
                    token=cfg.hf_token,
                )
                print(f"   ✅ Loaded {model_id}")
                break
            except Exception as e:
                err_str = str(e)
                if "403" in err_str or "gated" in err_str.lower() or "restricted" in err_str.lower():
                    print(f"   ❌ Access denied for {model_id}")
                    print(f"   👉 Go to https://huggingface.co/{model_id}")
                    print(f"      and click 'Agree and access repository'")
                    # Also check for sub-model access
                    if "speaker-diarization" in model_id:
                        print(f"   👉 Also accept: https://huggingface.co/pyannote/segmentation-3.0")
                        print(f"   👉 Also accept: https://huggingface.co/pyannote/speaker-diarization-community-1")
                else:
                    print(f"   ⚠️  {model_id}: {e}")

        if pipeline is None:
            raise RuntimeError(
                "Could not load any pyannote model. "
                "Accept model licenses at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  https://huggingface.co/pyannote/segmentation-3.0\n"
                "  https://huggingface.co/pyannote/speaker-diarization-community-1"
            )

        # Use GPU if available (CUDA or MPS)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("   Running on GPU (CUDA)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("   Running on GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("   Running on CPU (this may take a while for long audio)")
        
        pipeline.to(device)
        _pyannote_pipeline = pipeline

    # Use GPU if available (CUDA or MPS)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("   Running on GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("   Running on GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("   Running on CPU (this may take a while for long audio)")
    
    pipeline.to(device)

    diarization = pipeline(path, num_speakers=cfg.num_speakers)

    # pyannote.audio 4.0 returns DiarizeOutput, which contains the annotation
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    segs = [
        {
            "speaker": spk,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        }
        for turn, _, spk in diarization.itertracks(yield_label=True)
    ]

    n_speakers = len(set(s["speaker"] for s in segs))
    total_dur = {spk: 0.0 for spk in set(s["speaker"] for s in segs)}
    for s in segs:
        total_dur[s["speaker"]] += s["end"] - s["start"]

    print(f"   {len(segs)} segments, {n_speakers} speakers detected")
    for spk, dur in sorted(total_dur.items()):
        print(f"   {spk}: {dur:.1f}s")
    print("✅ pyannote diarization done")
    return segs


# ─────────────────────────────────────────────────────────────
# Tier 2: NVIDIA NeMo (high quality, no HF token needed)
# ─────────────────────────────────────────────────────────────


def diarize_nemo(path: str, cfg: PipelineConfig) -> list[dict]:
    """
    Speaker diarization using NVIDIA NeMo MSDD.
    Falls back gracefully if NeMo is not installed.
    """
    from nemo.collections.asr.models import ClusteringDiarizer

    print("🗣️  NeMo MSDD diarization...")
    import os, yaml, tempfile

    # NeMo expects a manifest file
    manifest_path = os.path.join(tempfile.gettempdir(), "nemo_diar_manifest.json")
    import json

    manifest = {
        "audio_filepath": os.path.abspath(path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": cfg.num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
        f.write("\n")

    output_dir = os.path.join(tempfile.gettempdir(), "nemo_diar_output")
    os.makedirs(output_dir, exist_ok=True)

    # Use ClusteringDiarizer with default config
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    from omegaconf import OmegaConf

    cfg_nemo = OmegaConf.load(config_url) if False else OmegaConf.create()
    cfg_nemo.diarizer = OmegaConf.create(
        {
            "manifest_filepath": manifest_path,
            "out_dir": output_dir,
            "oracle_vad": False,
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": True,
                    "max_num_speakers": cfg.num_speakers,
                }
            },
        }
    )

    diarizer = ClusteringDiarizer(cfg=cfg_nemo)
    diarizer.diarize()

    # Parse RTTM output
    rttm_files = [f for f in os.listdir(output_dir) if f.endswith(".rttm")]
    if not rttm_files:
        raise RuntimeError("NeMo produced no RTTM output")

    segs = []
    rttm_path = os.path.join(output_dir, rttm_files[0])
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                segs.append(
                    {
                        "speaker": spk,
                        "start": round(start, 3),
                        "end": round(start + dur, 3),
                    }
                )

    segs.sort(key=lambda x: x["start"])
    n_speakers = len(set(s["speaker"] for s in segs))
    print(f"   {len(segs)} segments, {n_speakers} speakers detected")
    print("✅ NeMo diarization done")
    return segs


# ─────────────────────────────────────────────────────────────
# Tier 3: MFCC + Agglomerative Clustering (no extra deps)
# ─────────────────────────────────────────────────────────────


def diarize_fallback(
    path: str, vad_segs: list[dict], cfg: PipelineConfig
) -> list[dict]:
    """Fallback: MFCC embeddings + agglomerative clustering."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    print("🗣️  Fallback diarization (MFCC + clustering)...")

    sr = cfg.sample_rate
    audio, _ = load_audio(path, sr=sr)
    embeddings, valid_segs = [], []

    for seg in tqdm(vad_segs, desc="   Extracting MFCC"):
        chunk = audio[int(seg["start"] * sr) : int(seg["end"] * sr)]
        if len(chunk) < int(sr * 0.3):  # skip < 0.3s
            continue
        mfcc = compute_mfcc(chunk, sr=sr)
        # Use mean, std, min, max and delta of MFCCs for richer embedding
        feat_mean = mfcc.mean(axis=1)
        feat_std = mfcc.std(axis=1)
        feat_min = mfcc.min(axis=1)
        feat_max = mfcc.max(axis=1)
        # Delta (first derivative approximation)
        if mfcc.shape[1] > 2:
            delta = np.diff(mfcc, axis=1)
            feat_delta_mean = delta.mean(axis=1)
            feat_delta_std = delta.std(axis=1)
        else:
            feat_delta_mean = np.zeros_like(feat_mean)
            feat_delta_std = np.zeros_like(feat_std)
        embedding = np.concatenate(
            [feat_mean, feat_std, feat_min, feat_max, feat_delta_mean, feat_delta_std]
        )
        embeddings.append(embedding)
        valid_segs.append(seg)

    if not embeddings:
        print("   ⚠️  No valid segments")
        return []

    X = np.array(embeddings)
    # Normalize features for better clustering
    X = StandardScaler().fit_transform(X)

    n_clust = min(cfg.num_speakers, len(X))
    labels = AgglomerativeClustering(
        n_clusters=n_clust, metric="cosine", linkage="average"
    ).fit_predict(X)

    segs = [
        {
            "speaker": f"SPEAKER_{lbl:02d}",
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
        }
        for seg, lbl in zip(valid_segs, labels)
    ]

    # Print per-speaker stats
    total_dur = defaultdict(float)
    for s in segs:
        total_dur[s["speaker"]] += s["end"] - s["start"]

    print(f"   {len(segs)} segments, {len(set(labels))} speakers")
    for spk, dur in sorted(total_dur.items()):
        print(f"   {spk}: {dur:.1f}s")
    print("✅ Fallback diarization done")
    return segs


# ─────────────────────────────────────────────────────────────
# Run diarization (advanced-first with safe fallback)
# ─────────────────────────────────────────────────────────────


def run_legacy_diarization(
    path: str,
    vad_segs: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], str]:
    """
    Legacy diarization with automatic method selection.

    Priority:
      1. pyannote.audio (if HF_TOKEN set)
      2. NeMo (if installed)
      3. MFCC + clustering (always available)

    Returns:
        (segments, method_name)
    """
    # ── Tier 1: pyannote ──
    if cfg.hf_token and cfg.hf_token.strip():
        try:
            segs = diarize_pyannote(path, cfg)
            return segs, "pyannote"
        except Exception as e:
            print(f"⚠️  pyannote failed: {e}")

    # ── Tier 2: NeMo ──
    try:
        import nemo  # noqa: F401

        print("ℹ️  Trying NeMo diarization...")
        segs = diarize_nemo(path, cfg)
        return segs, "nemo_msdd"
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  NeMo failed: {e}")

    # ── Tier 3: MFCC fallback ──
    if not cfg.hf_token:
        print("ℹ️  No HF_TOKEN → using MFCC fallback diarization")
    else:
        print("   Falling back to MFCC+clustering...")
    segs = diarize_fallback(path, vad_segs, cfg)
    return segs, "mfcc_clustering"


def run_diarization(
    path: str,
    vad_segs: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], str]:
    """
    Production diarization wrapper.

    Behavior:
      1) Attempt advanced diarization module (additive, safe)
      2) If any advanced stage fails, automatically fallback to legacy stack
    """
    if cfg.enable_advanced_diarization and not cfg.advanced_force_legacy:
        try:
            from .advanced_diarization import run_advanced_diarization

            segs, method = run_advanced_diarization(path, vad_segs, cfg)
            if segs:
                print(f"✅ Advanced diarization succeeded ({method})")
                return segs, method
            print("⚠️  Advanced diarization returned no segments; using legacy fallback.")
        except Exception as e:
            print(f"⚠️  Advanced diarization failed: {e}")
            print("   Falling back to legacy diarization stack...")

    return run_legacy_diarization(path, vad_segs, cfg)


# ─────────────────────────────────────────────────────────────
# VAD ∩ Diarization
# ─────────────────────────────────────────────────────────────


def intersect_vad_diar(
    vad_segs: list[dict], diar_segs: list[dict]
) -> list[dict]:
    """Intersect VAD and diarization: keep only voiced + attributed segments.
    Applies aggressive turn smoothing for mono-audio telephonic calls."""
    out = []
    for d in diar_segs:
        for v in vad_segs:
            # Drop very short VAD voices that are likely just breathing/clicks
            if (v["end"] - v["start"]) < 0.4:
                continue
                
            start = max(d["start"], v["start"])
            end = min(d["end"], v["end"])
            
            # Require minimum overlapping speech duration
            if start < end and (end - start) > 0.3:
                out.append(
                    {
                        "speaker": d["speaker"],
                        "start": round(start, 3),
                        "end": round(end, 3),
                    }
                )

    out.sort(key=lambda x: x["start"])

    # Phase 2: Aggressive Turn Smoothing (Mono-Audio Hardening)
    # 1. Merge adjacent same-speaker segments with larger gaps allowed
    # 2. Absorb tiny interruptions (short bursts < 1.0s by another speaker sandwiched between the primary speaker)
    
    if not out:
        return []

    merged = [out[0].copy()]
    for seg in out[1:]:
        prev = merged[-1]
        
        # Merge if it's the same speaker and very close
        if seg["speaker"] == prev["speaker"]:
            if seg["start"] - prev["end"] < 1.0: # Increased gap tolerance from 0.3 to 1.0s
                prev["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        else:
            merged.append(seg.copy())

    # Phase 3: Sandwich Removal (e.g. Spk1 -> [short Spk2] -> Spk1)
    # We must explicitly PRESERVE short backchannels (haan, hmm, etc) which are usually 0.4s to 1.0s.
    # We only absorb truly microscopic non-speech static (< 0.4s).
    smoothed = []
    for i in range(len(merged)):
        seg = merged[i]
        
        # Check if this segment is microscopic static masquerading as a speaker
        if (seg["end"] - seg["start"]) < 0.4:
            if 0 < i < len(merged) - 1:
                prev_seg = merged[i-1]
                next_seg = merged[i+1]
                # If surrounded by the SAME speaker, absorb the static to prevent a 3-way split
                if prev_seg["speaker"] == next_seg["speaker"] and prev_seg["speaker"] != seg["speaker"]:
                    continue
        
        smoothed.append(seg)
        
    # Final merge after swallowing sandwiches
    final_segs = []
    for seg in smoothed:
        if final_segs and final_segs[-1]["speaker"] == seg["speaker"] and seg["start"] - final_segs[-1]["end"] < 1.5:
            final_segs[-1]["end"] = max(final_segs[-1]["end"], seg["end"])
        else:
            final_segs.append(seg.copy())

    print(f"🔗 VAD ∩ Diar: {len(diar_segs)} raw → {len(final_segs)} smoothed segments")
    return final_segs


# ─────────────────────────────────────────────────────────────
# Speaker Role Inference
# ─────────────────────────────────────────────────────────────

def infer_roles_linguistic(
    transcript: list[dict], cfg: PipelineConfig
) -> tuple[list[dict], dict[str, str]]:
    """
    LLM-based role inference using the first few turns of the transcript.
    """
    import json
    import openai

    print("🏷️  Inferring speaker roles (Linguistic)...")
    
    # Grab first 20 valid turns or up to 1000 chars
    sample_texts = []
    speakers_found = set()
    for s in transcript:
        if s["text"].strip():
            sample_texts.append(f"Speaker {s['speaker']}: {s['text']}")
            speakers_found.add(s["speaker"])
        if len(sample_texts) >= 20:
            break

    if len(speakers_found) < 2:
        print("   ⚠️  Not enough speakers for distinct roles")
        role_map = {spk: "agent" for spk in speakers_found}
    else:
        sample_text = "\n".join(sample_texts)
        
        client = openai.OpenAI(
            api_key=cfg.openai_api_key,
            timeout=cfg.openai_timeout_s,
            max_retries=cfg.openai_max_retries,
        )
        
        system_prompt = (
            "You are a helpful analyst. I will provide a short excerpt of a transcript "
            "from a debt/financial advisory call. Your job is to identify which Speaker ID "
            "is the 'agent' (the FREED advisor) and which is the 'customer'.\n"
            "Return a strictly valid JSON object mapping each Speaker ID to either 'agent' or 'customer'."
        )
        
        try:
            response = client.chat.completions.create(
                model=cfg.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript:\n{sample_text}"}
                ],
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            # Normalize to lower-case roles and remove "Speaker " if the LLM included it
            role_map = {}
            for k, v in result.items():
                clean_k = k.replace("Speaker ", "").strip()
                role_map[clean_k] = v.lower()
                
            # Fallback for any missed speakers
            for spk in speakers_found:
                if spk not in role_map:
                    role_map[spk] = "unknown"
                    
        except Exception as e:
            print(f"   ⚠️  LLM role inference failed: {e}")
            # Fallback arbitrary mapping
            spk_list = list(speakers_found)
            role_map = {spk_list[0]: "agent"}
            for spk in spk_list[1:]:
                role_map[spk] = "customer"

    # Print summary
    print(f"   Role Mapping: {role_map}")

    # Apply roles
    for seg in transcript:
        seg["role"] = role_map.get(seg["speaker"], "unknown")
        # Ensure 'speaker' is updated if needed downstream, but downstream 
        # normally uses "speaker" = "agent" or "customer". Let's overwrite "speaker"
        seg["speaker"] = seg["role"]

    print("✅ Roles assigned")
    return transcript, role_map
