"""
Pipeline configuration — all tunable parameters in one place.
"""

import os
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


@dataclass
class PipelineConfig:
    """All pipeline parameters."""

    # Audio
    sample_rate: int = 16000
    bandpass_lo: int = 300
    bandpass_hi: int = 3400
    target_dbfs: float = -20.0

    # VAD (Tuned for telephonic sales calls)
    vad_silence_ms: int = 500  # tighter silence to avoid trailing noise
    vad_threshold: float = 0.6 # slightly higher threshold to ignore background
    vad_speech_pad_ms: int = 200 # extra padding to not cut off words

    # Diarization
    num_speakers: int = 2
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    enable_advanced_diarization: bool = True
    advanced_force_legacy: bool = False
    advanced_min_segment_s: float = 0.35
    advanced_short_segment_s: float = 0.5
    advanced_change_point_threshold: float = 0.25
    advanced_merge_similarity: float = 0.75
    advanced_relabel_margin: float = 0.05
    advanced_max_speakers: int = 4
    advanced_asr_frame_s: float = 0.02
    advanced_overlap_threshold: float = 0.55

    # Chunking
    chunk_seconds: int = 25
    overlap_seconds: float = 1.5
    min_tail_seconds: float = 2.0

    # ASR
    asr_engine: str = "hinglish"  # "hinglish" or "whisper"
    hinglish_model: str = "Oriserve/Whisper-Hindi2Hinglish-Prime"
    whisper_model: str = "large-v3"
    language: str = "hi"
    
    # Expanded prompt with specific loan settlement terminology
    initial_prompt: str = (
        "This is a Hindi-Hinglish financial advisory sales call for FREED. "
        "Terms: Debt Consolidation, single EMI, Debt Resolution, loan settlement 60 percent, "
        "outstanding amount, processing fees, one-time fees, approval, credit card, home loan, "
        "SBI, ICICI, Experian, CIBIL, credit report, recovery agent, harassment calls, "
        "DRA certificate, authorization letter, Special Purpose Account, SPA, Debt Management Plan, "
        "DMP, Self Saving Program, Virtual Account, One-Time Settlement, OTS, Non-Performing Asset, "
        "NPA, NACH, e-Mandate, No Dues Certificate, NDC, Legal Notice, SARFAESI, Loan Maafi, "
        "Write-off, Bounce Charges, Penal Charges, Overdue, Default, Foreclosure."
    )
    
    # Hallucination Filters
    hallucinations: list[str] = field(default_factory=lambda: [
        "subtitles by", "amara.org", "thanks for watching", "nan", "subscribe"
    ])

    # Terminology Normalization Map (Aggressive Contextual Filtering)
    term_replacements: dict[str, str] = field(default_factory=lambda: {
        # Strict Terminology
        "civil score": "CIBIL score",
        "civil": "CIBIL",
        "knocks": "NOC",
        "noc": "NOC",
        "no objection certificate": "NOC",
        "settlement later": "settlement letter",
        "settlement amount": "settlement amount",
        "principle amount": "principal amount",
        "weaver": "waiver",
        "bounce charge": "bounce charges",
        "harassment": "harassment",
        "emi": "EMI",
        "emis": "EMIs",
        "nbfc": "NBFC",
        "nach": "NACH",
        "debt consolidation": "Debt Consolidation",
        "resolution": "Resolution",
        "experian": "Experian",
        "dra": "DRA",
        "authorization letter": "authorization letter",
        "mandate": "mandate",
        "e-mandate": "e-Mandate",
        "spa": "SPA",
        "special purpose account": "Special Purpose Account",
        "dmp": "DMP",
        "debt management plan": "Debt Management Plan",
        "self saving program": "Self Saving Program",
        "virtual account": "Virtual Account",
        "ots": "OTS",
        "one time settlement": "One-Time Settlement",
        "npa": "NPA",
        "non performing asset": "Non-Performing Asset",
        "ndc": "NDC",
        "no dues certificate": "No Dues Certificate",
        "legal notice": "legal notice",
        "sarfaesi": "SARFAESI",
        "loan maafi": "Loan Maafi",
        "write off": "write-off",
        "penal charges": "penal charges",
        "overdue": "overdue",
        "default": "default",
        "foreclosure": "foreclosure",

        # Common Homophones & Context-Specific Mis-transcriptions
        "views": "dues",
        "due's": "dues",
        "news": "dues",
        "bound": "bounce",
        "bone charge": "bounce charge",
        "loan account": "loan account"
    })

    # Reconstruction
    merge_gap_seconds: float = 2.5

    # Output
    output_path: str = "transcript_output.json"

    # Summarization & Correction
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_timeout_s: float = 90.0
    openai_summary_timeout_s: float = 180.0
    openai_max_retries: int = 1
    summary_model: str = "gpt-5-mini"
    enable_rephrase: bool = True
    rephrase_model: str = "gpt-4o-mini"
    enable_emotion_analysis: bool = True

    def __post_init__(self):
        if self.hf_token == "your_token_here":
            self.hf_token = ""


def detect_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"🚀 GPU: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 Apple Silicon (MPS)")
        return "mps"  # Note: whisper may not fully support MPS yet
    else:
        print("💻 CPU mode")
        return "cpu"
