#!/usr/bin/env python3
"""
Speech Transcription Pipeline — CLI Entry Point.

Usage:
    python run.py <audio_file> [options]

Examples:
    python run.py call.wav
    python run.py call.mp3 --speakers 3 --language hi --model large-v3
    python run.py call.wav --output result.json
"""

import argparse
import json
import os
import shutil
import sys
import time

from pipeline.config import PipelineConfig, detect_device
from pipeline.preprocess import preprocess
from pipeline.vad import run_vad
from pipeline.diarize import run_diarization, intersect_vad_diar, infer_roles_linguistic
from pipeline.transcribe import load_model, make_chunks, transcribe_chunks
from pipeline.reconstruct import (
    reconstruct,
    correct_transcript_llm,
    summarize_customer,
    summarize_call_structured,
    format_transcript_preview,
    format_structured_summary,
)
from pipeline.utils import fmt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe Indian sales calls with speaker diarization, or summarize an existing transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py call.wav\n"
            "  python run.py call.mp3 --speakers 3 --model medium\n"
            "  python run.py call.wav --output my_result.json\n"
            "  python run.py transcript_output.json --output text_with_summary.json\n"
        ),
    )
    parser.add_argument("input", help="Path to audio file (wav, mp3) or existing transcript JSON (.json)")
    parser.add_argument(
        "--speakers", "-s", type=int, default=2, help="Expected number of speakers (default: 2)"
    )
    parser.add_argument(
        "--engine", "-e", default="hinglish",
        choices=["hinglish", "whisper"],
        help="ASR engine: 'hinglish' (Oriserve, default) or 'whisper' (OpenAI)",
    )
    parser.add_argument(
        "--language", "-l", default="hi", help="Language code for Whisper engine (default: hi)"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Whisper model size (tiny, base, small, medium, large-v3). "
             "Only used with --engine whisper. Default: large-v3 (GPU) or medium (CPU)",
    )
    parser.add_argument(
        "--output", "-o", default="transcript_output.json",
        help="Output JSON path (default: transcript_output.json)",
    )
    parser.add_argument(
        "--summary-output", default=None,
        help="Output TXT path for the summary. Default: same as --output but with .txt extension.",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip customer summary"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    t_start = time.time()
    
    # ── Handle JSON Input (Summarize Only) ──
    if args.input.lower().endswith(".json"):
        print(f"\n{'=' * 60}")
        print(f"  📝  Summarizing Existing Transcript")
        print(f"{'=' * 60}")
        print(f"  Input    : {args.input}")
        
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "segments" not in data:
            print("❌ Invalid JSON transcript: missing 'segments' key.")
            sys.exit(1)
            
        cfg = PipelineConfig(
            output_path=args.output,
        )
        
        # Determine duration from segments if not in metadata
        duration = data.get("metadata", {}).get("duration_s", 0.0)
        if not duration and data["segments"]:
            duration = float(data["segments"][-1]["end"].split()[-1]) # Hacky but usually timestamps aren't raw here if loaded like this.
            
        transcript = data["segments"]
        
        # Generate summary 
        summary_txt_path = args.summary_output
        if not summary_txt_path:
            base, ext = os.path.splitext(cfg.output_path)
            summary_txt_path = base + "_summary.txt"

        if not args.no_summary:
            extractive = summarize_customer(transcript, duration)
            structured = summarize_call_structured(transcript, cfg)
            
            with open(summary_txt_path, "w", encoding="utf-8") as fs:
                fs.write(f"Customer Extraction Stats:\n- Words: {extractive.get('customer_words', 0)}\n- Segments: {extractive.get('customer_segments', 0)}\n\n")
                fs.write("EXTRACTIVE SUMMARY:\n")
                fs.write(extractive.get("text", "None") + "\n\n")
                fs.write("="*40 + "\n\n")
                fs.write(format_structured_summary(structured))
            print(f"  Summary saved to {summary_txt_path}")
            
        with open(cfg.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"  ✅ DONE summarizing")
        print(f"{'=' * 60}")
        print(f"  Time         : {elapsed:.1f}s")
        print(f"  Output       : {cfg.output_path} ({os.path.getsize(cfg.output_path) / 1024:.1f} KB)")
        print(f"{'=' * 60}")
        return data

    # Check ffmpeg for audio
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg not found! It is required for audio processing.")
        print("   Install it with one of:")
        print("     brew install ffmpeg")
        print("     conda install ffmpeg")
        print("     Download from: https://ffmpeg.org/download.html")
        sys.exit(1)

    t_start = time.time()

    # ── Config ──
    device = detect_device()
    cfg = PipelineConfig(
        num_speakers=args.speakers,
        language=args.language,
        asr_engine=args.engine,
        whisper_model=args.model or ("large-v3" if device == "cuda" else "medium"),
        output_path=args.output,
    )

    # Determine ASR model display name
    if cfg.asr_engine == "hinglish":
        model_display = cfg.hinglish_model
    else:
        model_display = f"whisper-{cfg.whisper_model}"

    print(f"\n{'=' * 60}")
    print(f"  🎙️  Speech Transcription Pipeline")
    print(f"{'=' * 60}")
    print(f"  Input    : {args.input}")
    print(f"  Device   : {device}")
    print(f"  Engine   : {cfg.asr_engine}")
    print(f"  Model    : {model_display}")
    print(f"  Language : {cfg.language}")
    print(f"  Speakers : {cfg.num_speakers}")
    print(f"  HF Token : {'✅ set' if cfg.hf_token else '❌ not set (fallback diarization)'}")
    print(f"{'=' * 60}\n")

    # ── Stage 1: Preprocess ──
    print("\n[1/7] Preprocess")
    prep_path, duration = preprocess(args.input, cfg=cfg)

    # ── Stage 2: VAD ──
    print("\n[2/7] Voice Activity Detection")
    vad_segs = run_vad(prep_path, cfg=cfg)

    # ── Stage 3: Diarization ──
    print("\n[3/7] Speaker Diarization")
    diar_segs, diar_method = run_diarization(prep_path, vad_segs, cfg)

    # ── Stage 4: Intersect ──
    print("\n[4/7] VAD ∩ Diarization")
    speaker_segs = intersect_vad_diar(vad_segs, diar_segs)

    # ── Stage 5: Chunk + Transcribe ──
    print("\n[5/7] Chunking + ASR")
    chunks = make_chunks(speaker_segs, cfg)
    asr_model, engine_used = load_model(cfg, device)
    raw_transcript = transcribe_chunks(chunks, prep_path, asr_model, cfg, device)
    del asr_model  # free memory

    # ── Stage 6: Role Inference ──
    print("\n[6/7] Speaker Role Inference")
    raw_transcript, role_map = infer_roles_linguistic(raw_transcript, cfg)

    # ── Stage 7: Reconstruct + Output ──
    print("\n[7/7] Reconstruct & Save")
    transcript = reconstruct(raw_transcript, cfg)
    
    # ── Stage 7A: LLM Contextual Transcription Correction ──
    transcript = correct_transcript_llm(transcript, cfg)

    # Build output JSON
    agent_dur = float(sum(s["t1"] - s["t0"] for s in transcript if s["speaker"] == "agent"))
    cust_dur = float(sum(s["t1"] - s["t0"] for s in transcript if s["speaker"] == "customer"))

    output = {
        "metadata": {
            "duration_s": round(float(duration), 2),
            "duration": fmt(duration),
            "speakers": len(set(s["speaker"] for s in transcript)),
            "segments": len(transcript),
            "agent_s": round(agent_dur, 2),
            "customer_s": round(cust_dur, 2),
            "model": model_display,
            "device": device,
            "diarization": diar_method,
        },
        "segments": [
            {
                "id": i + 1,
                "speaker": s["speaker"],
                "start": fmt(s["t0"]),
                "end": fmt(s["t1"]),
                "dur_s": round(s["t1"] - s["t0"], 2),
                "text": s["text"],
                "dialogue": f"[{fmt(s['t0'])}] {s['speaker'].upper()}: {s['text']}"
            }
            for i, s in enumerate(transcript)
        ],
    }

    # Save formatted human-readable text transcript
    txt_path = cfg.output_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(format_transcript_preview(transcript, len(transcript)))

    # Optional summary
    summary_txt_path = args.summary_output
    if not summary_txt_path:
        base, ext = os.path.splitext(cfg.output_path)
        summary_txt_path = base + "_summary.txt"

    if not args.no_summary:
        extractive = summarize_customer(transcript, duration)
        structured = summarize_call_structured(transcript, cfg)
        with open(summary_txt_path, "w", encoding="utf-8") as fs:
            fs.write(f"Customer Extraction Stats:\n- Words: {extractive.get('customer_words', 0)}\n- Segments: {extractive.get('customer_segments', 0)}\n\n")
            fs.write("EXTRACTIVE SUMMARY:\n")
            fs.write(extractive.get("text", "None") + "\n\n")
            fs.write("="*40 + "\n\n")
            fs.write(format_structured_summary(structured))
        print(f"  Summary saved to {summary_txt_path}")

    # Save
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start

    # ── Results ──
    print(f"\n{'=' * 60}")
    print(f"  ✅ DONE")
    print(f"{'=' * 60}")
    print(f"  Time         : {elapsed:.1f}s (RTF {elapsed / float(duration):.2f}x)")
    print(f"  Segments     : {len(transcript)} (Agent {sum(1 for s in transcript if s['speaker'] == 'agent')}, Customer {sum(1 for s in transcript if s['speaker'] == 'customer')})")
    print(f"  Diarization  : {diar_method}")
    print(f"  Output       : {cfg.output_path} ({os.path.getsize(cfg.output_path) / 1024:.1f} KB)")
    print(f"{'=' * 60}")

    # Preview
    print("\n📜 Transcript Preview (first 10):")
    print("-" * 60)
    print(format_transcript_preview(transcript, 10))
    print("-" * 60)

    # Cleanup
    if os.path.exists("preprocessed.wav"):
        os.remove("preprocessed.wav")

    return output


if __name__ == "__main__":
    main()
