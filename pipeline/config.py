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
    
    # Expanded prompt with specific loan settlement terminology from FREED knowledge base
    initial_prompt: str = (
        "This is a Hindi-Hinglish financial advisory sales call for FREED — India's first debt relief platform. "
        "FREED offers: Debt Resolution Program (DRP) — settlement at ~45% outstanding with 15%+GST service fee; "
        "Debt Consolidation Program (DCP) — multiple loans merged into one EMI via third-party lender; "
        "Debt Elimination Program (DEP) — structured accelerated repayment. "
        "Key terms: SPA (Special Purpose Account / escrow account), CIBIL score, Experian credit report, "
        "CHPP (Creditor Harassment Protection Programme), FREED Shield, i-FREED app, "
        "DRA certificate, authorization letter, NACH, e-Mandate, "
        "OTS (One-Time Settlement), NPA (Non-Performing Asset), NDC (No Dues Certificate), "
        "evaluation fee, platform fee, service fee, settlement letter, principal amount, "
        "delinquency, waiver, bounce charges, penal charges, foreclosure, write-off, "
        "SARFAESI, Legal Notice, recovery agent, home visit, borrower rights, "
        "Loan Maafi, overdue, default, NBFC, SBI, ICICI, HDFC, Axis Bank."
    )
    
    # Hallucination Filters
    hallucinations: list[str] = field(default_factory=lambda: [
        "subtitles by", "amara.org", "thanks for watching", "nan", "subscribe"
    ])

    # Terminology Normalization Map — enriched from FREED knowledge base
    term_replacements: dict[str, str] = field(default_factory=lambda: {
        # FREED Programs
        "debt resolution program": "Debt Resolution Program",
        "drp": "DRP",
        "debt consolidation program": "Debt Consolidation Program",
        "dcp": "DCP",
        "debt elimination program": "Debt Elimination Program",
        "dep": "DEP",

        # Core Product Terminology
        "civil score": "CIBIL score",
        "civil": "CIBIL",
        "cibil": "CIBIL",
        "experian": "Experian",
        "special purpose account": "Special Purpose Account",
        "spa": "SPA",
        "chpp": "CHPP",
        "creditor harassment protection programme": "CHPP",
        "freed shield": "FREED Shield",
        "i freed": "i-FREED",
        "i-freed": "i-FREED",
        "ifreed": "i-FREED",

        # Settlement & Fees
        "settlement later": "settlement letter",
        "principle amount": "principal amount",
        "evaluation fees": "evaluation fee",
        "platform fees": "platform fee",
        "service fees": "service fee",
        "one time settlement": "One-Time Settlement",
        "ots": "OTS",

        # Debt & Account Terms
        "nach": "NACH",
        "e-mandate": "e-Mandate",
        "e mandate": "e-Mandate",
        "dmp": "DMP",
        "debt management plan": "Debt Management Plan",
        "npa": "NPA",
        "non performing asset": "Non-Performing Asset",
        "ndc": "NDC",
        "no dues certificate": "No Dues Certificate",
        "noc": "NOC",
        "knocks": "NOC",
        "no objection certificate": "NOC",
        "nbfc": "NBFC",
        "emi": "EMI",
        "emis": "EMIs",
        "waiver": "waiver",
        "weaver": "waiver",
        "bounce charge": "bounce charges",
        "dra": "DRA",
        "sarfaesi": "SARFAESI",
        "loan maafi": "Loan Maafi",
        "write off": "write-off",
        "debt consolidation": "Debt Consolidation",
        "foreclosure": "foreclosure",
        "delinquency": "delinquency",
        "authorization letter": "authorization letter",
        "virtual account": "Virtual Account",
        "self saving program": "Self Saving Program",

        # Common Homophones & Mis-transcriptions
        "views": "dues",
        "due's": "dues",
        "news": "dues",
        "bound": "bounce",
        "bone charge": "bounce charge",
        "freed": "FREED",
        "free d": "FREED",
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
    summary_model: str = "gpt-4o-mini"
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
