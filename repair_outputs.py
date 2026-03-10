import os
import json
from pipeline.config import PipelineConfig
from pipeline.reconstruct import summarize_call_structured, format_structured_summary

OUTPUTS_DIR = "outputs"

def repair():
    cfg = PipelineConfig()
    
    if not cfg.openai_api_key:
        print("ERROR: No OPENAI_API_KEY available. Cannot run repair.")
        return

    print("Checking for legacy outputs needing schema updates...")
    
    if not os.path.exists(OUTPUTS_DIR):
        print("No outputs directory found.")
        return

    for folder in os.listdir(OUTPUTS_DIR):
        path = os.path.join(OUTPUTS_DIR, folder)
        if os.path.isdir(path):
            json_path = os.path.join(path, "transcript.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check for legacy summary schema
                summary = data.get("summary")
                needs_update = False
                
                if not summary:
                    needs_update = True
                elif isinstance(summary, dict):
                    if "call_categories" not in summary:
                        needs_update = True
                    elif "major_keywords" not in summary.get("call_categories", {}):
                        needs_update = True
                
                if needs_update:
                    print(f"Executing LLM schema correction for: {folder}")
                    segments = data.get("segments", [])
                    try:
                        new_summary = summarize_call_structured(segments, cfg)
                        data["summary"] = new_summary
                        
                        # Overwrite JSON
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4)
                            
                        # Overwrite summary.txt
                        txt_path = os.path.join(path, "summary.txt")
                        summary_txt = format_structured_summary(new_summary)
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(summary_txt)
                        
                        print(f"Successfully repaired {folder}")
                    except Exception as e:
                        print(f"Failed to repair {folder}. Error: {e}")
                else:
                    print(f"Skipping {folder}: Already on latest schema.")

if __name__ == "__main__":
    repair()
