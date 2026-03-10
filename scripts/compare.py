import json
import sys
import os

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_comparison(old_path, new_path):
    print(f"🔄 Comparing Transcripts...\n   OLD: {old_path}\n   NEW: {new_path}\n")
    
    old_data = load_json(old_path)
    new_data = load_json(new_path)
    
    old_segs = old_data.get("segments", [])
    new_segs = new_data.get("segments", [])
    
    old_total_segs = len(old_segs)
    new_total_segs = len(new_segs)
    
    def count_short_segments(segs, threshold=1.0):
        return sum(1 for s in segs if s.get("dur_s", 0) < threshold)
        
    def get_avg_dur(segs):
        if not segs: return 0.0
        return sum(s.get("dur_s", 0) for s in segs) / len(segs)
        
    def count_nans(segs):
        return sum(1 for s in segs if "nan" in s.get("text", "").lower() or text_is_mostly_nan(s.get("text", "")))

    def text_is_mostly_nan(t):
        words = t.lower().replace(".", "").split()
        if not words: return False
        nan_count = words.count("nan")
        return nan_count / len(words) > 0.5
        
    def count_terms(segs, terms):
        count = 0
        full_text = " ".join(s.get("text", "") for s in segs).lower()
        for t in terms:
            count += full_text.count(t.lower())
        return count
        
    loan_terms = ["cibil", "noc", "settlement letter", "principal amount", "waiver", "bounce charges", "nbfc"]
    
    old_short = count_short_segments(old_segs)
    new_short = count_short_segments(new_segs)
    
    old_nan = count_nans(old_segs)
    new_nan = count_nans(new_segs)
    
    old_terms = count_terms(old_segs, loan_terms)
    new_terms = count_terms(new_segs, loan_terms)
    
    old_dur_m = old_data.get("metadata", {}).get("duration_s", 1) / 60.0
    new_dur_m = new_data.get("metadata", {}).get("duration_s", 1) / 60.0
    
    print(f"Old Duration: {old_dur_m:.1f}m | New Duration: {new_dur_m:.1f}m\n")
    print(f"{'Metric':<30} | {'Old/Min':<15} | {'New/Min':<15} | {'Change'}")
    print("-" * 75)
    
    def print_metric(name, old_val, new_val, is_lower_better=False, is_float=False):
        old_norm = old_val / old_dur_m
        new_norm = new_val / new_dur_m
        
        fmt_str = "{:.2f}"
        ov = fmt_str.format(old_norm)
        nv = fmt_str.format(new_norm)
        
        diff = new_norm - old_norm
        diff_str = f"{diff:+.2f}"
        
        if abs(diff) < 0.01:
            trend = "=="
        elif (diff < 0 and is_lower_better) or (diff > 0 and not is_lower_better):
            trend = "✅"
        else:
            trend = "❌"
            
        print(f"{name:<30} | {ov:<15} | {nv:<15} | {diff_str} {trend}")
        
    print_metric("Total Speaker Turns", old_total_segs, new_total_segs, is_lower_better=True)
    print_metric("Micro-Turns (<1.0s)", old_short, new_short, is_lower_better=True)
    print_metric("Hallucinated 'nan' turns", old_nan, new_nan, is_lower_better=True)
    print_metric("Specific Loan Term Hits", old_terms, new_terms, is_lower_better=False)
    
    print("\n💡 Summary: Metrics are normalized per minute.")
    if new_total_segs < old_total_segs:
        reduction = (old_total_segs - new_total_segs) / old_total_segs * 100
        print(f"  - Turn fragmentation reduced by {reduction:.1f}%. The transcript is much more readable.")
    if new_nan < old_nan:
        print(f"  - Suppressed {old_nan - new_nan} hallucinated 'nan' segments.")
    if new_terms > old_terms:
        print(f"  - Captured {new_terms - old_terms} more specific loan terms due to targeted prompting and normalization.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <old_json> <new_json>")
        sys.exit(1)
    run_comparison(sys.argv[1], sys.argv[2])
