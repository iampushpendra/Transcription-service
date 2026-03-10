import os
import sys
import json
import logging

# Ensure pipeline modules can be natively imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
from pipeline.config import PipelineConfig
from pipeline.reconstruct import (
    summarize_call_structured,
    verify_and_inject_inline_citations,
    format_structured_summary,
    rephrase_transcript_llm,
)
from pipeline.emotion import analyze_emotion, extract_acoustic_features
from pipeline.triggers import analyze_triggers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    cfg = PipelineConfig()
    if not os.path.exists(OUTPUTS_DIR):
        logger.error(f"Outputs directory {OUTPUTS_DIR} does not exist.")
        return

    logger.info("Scanning for transcripts to rephrase and regenerate summaries...")
    
    dirs = sorted([d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))])
    
    for dir_name in dirs:
        dir_path = os.path.join(OUTPUTS_DIR, dir_name)
        json_path = os.path.join(dir_path, "transcript.json")
        
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            segments = data.get("segments", [])
            
            if not segments:
                logger.warning(f"Skipping {dir_name}: No dialogue segments found.")
                continue

            # Check if already rephrased (has original_text fields)
            already_rephrased = any(seg.get("original_text") for seg in segments)
            
            if already_rephrased:
                logger.info(f"Skipping rephrase for {dir_name}: already has original_text fields.")
            else:
                logger.info(f"Rephrasing transcript for: {dir_name}...")
                segments = rephrase_transcript_llm(segments, cfg)
                data["segments"] = segments

            # Emotion Analysis + Trigger Phrases (requires preprocessed audio)
            if cfg.enable_emotion_analysis and "emotion_analysis" not in data:
                prep_audio = os.path.join(dir_path, "preprocessed.wav")
                if not os.path.exists(prep_audio):
                    prep_audio = "preprocessed.wav"  # fallback to cwd
                
                if os.path.exists(prep_audio):
                    logger.info(f"Running emotion analysis for: {dir_name}...")
                    speaker_segs = [
                        {"speaker": s.get("speaker", "unknown"), "start": s["t0"], "end": s["t1"]}
                        for s in segments
                    ]
                    emotion = analyze_emotion(prep_audio, speaker_segs, segments)
                    all_heated = (
                        emotion.get("agent_heated_segments", []) +
                        emotion.get("customer_heated_segments", [])
                    )
                    features = extract_acoustic_features(prep_audio, speaker_segs)
                    triggers = analyze_triggers(all_heated, features, segments)
                    data["emotion_analysis"] = emotion
                    data["trigger_phrases"] = triggers
                else:
                    logger.info(f"Skipping emotion for {dir_name}: no audio file available.")

            # Regenerate summary with updated prompt (dedup + bullet formatting)
            logger.info(f"Regenerating summary for: {dir_name}...")
            summary = summarize_call_structured(segments, cfg)
            
            if isinstance(summary, dict) and "error" in summary:
                logger.error(f"OpenAI error on {dir_name}: {summary['error']}")
                continue

            # Verify and snap inline citations
            summary = verify_and_inject_inline_citations(summary, segments)
            data["summary"] = summary

            # Save updated transcript.json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Also regenerate summary.txt
            summary_text = format_structured_summary(summary)
            summary_txt_path = os.path.join(dir_path, "summary.txt")
            with open(summary_txt_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            logger.info(f"✅ Successfully processed {dir_name}")

        except Exception as e:
            logger.error(f"Fatal error processing {dir_name}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
