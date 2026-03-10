import os
import json
import shutil
import time

OUTPUTS_DIR = "outputs"
OLD_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "old outputs")

def migrate():
    if not os.path.exists(OLD_OUTPUTS_DIR):
        print("No old outputs directory found.")
        return

    print("Migrating flat JSON transcripts from old outputs to standardized schema directories...")

    for f in os.listdir(OLD_OUTPUTS_DIR):
        if f.endswith(".json"):
            json_path = os.path.join(OLD_OUTPUTS_DIR, f)
            try:
                with open(json_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                
                # Check if this is actually a transcript (needs segments)
                if "segments" not in data:
                    print(f"Skipping {f} - no segments.")
                    continue
                
                # Create a standardized folder name for it
                # We'll generate a dummy ID loosely matching the current format: date_time_hash
                ts = int(time.time())
                folder_name = f"legacy_migrated_{f.replace('.json', '')}_{ts}"
                folder_path = os.path.join(OUTPUTS_DIR, folder_name)
                
                os.makedirs(folder_path, exist_ok=True)
                
                # Move/Copy JSON
                new_json_path = os.path.join(folder_path, "transcript.json")
                with open(new_json_path, "w", encoding="utf-8") as new_file:
                    json.dump(data, new_file, indent=4)
                
                print(f"Migrated {f} to {folder_name}")
            except Exception as e:
                print(f"Failed migrating {f}: {e}")

if __name__ == "__main__":
    migrate()
