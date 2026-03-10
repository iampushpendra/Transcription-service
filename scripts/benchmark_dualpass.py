"""
Dual-Pass Numeric Micro-Pipeline — Benchmark Script

NON-DESTRUCTIVE: This script does NOT modify any existing code, configs, or output schemas.
It reads existing transcript.json files and produces annotated outputs in a new directory.

Usage:
    python benchmark_dualpass.py --mode dualpass_numeric --input-dir outputs/ --output-dir results/dualpass/

Modes:
    baseline        - Copy existing outputs as-is (for comparison baseline)
    dualpass_numeric - Run numeric candidate detection + Hindi number parser on transcripts
"""

import argparse
import json
import os
import re
import copy
from datetime import datetime


# ─── Hindi Number Word Parser (Deterministic, CPU-only) ───────────────────────

HINDI_DIGITS = {
    'ek': 1, 'do': 2, 'teen': 3, 'chaar': 4, 'paanch': 5,
    'chhah': 6, 'saat': 7, 'aath': 8, 'nau': 9, 'das': 10,
    'gyarah': 11, 'baarah': 12, 'terah': 13, 'chaudah': 14, 'pandrah': 15,
    'solah': 16, 'satrah': 17, 'atharah': 18, 'unnis': 19, 'bees': 20,
    'ikkis': 21, 'bais': 22, 'teis': 23, 'chaubis': 24, 'pachchis': 25,
    'chhabbis': 26, 'sattais': 27, 'atthaais': 28, 'untees': 29, 'tees': 30,
    'ikatees': 31, 'battis': 32, 'taintees': 33, 'chautis': 34, 'paintees': 35,
    'chhattis': 36, 'saintees': 37, 'adtees': 38, 'untalis': 39, 'chalis': 40,
    'iktalis': 41, 'bayalis': 42, 'taintalis': 43, 'chawalis': 44, 'paintalis': 45,
    'chhiyalis': 46, 'saintalis': 47, 'adtalis': 48, 'unchas': 49, 'pachaas': 50,
    'ikyavan': 51, 'bavan': 52, 'tirpan': 53, 'chauvan': 54, 'pachpan': 55,
    'chhappan': 56, 'sattavan': 57, 'atthavan': 58, 'unsath': 59, 'saath': 60,
    'iksath': 61, 'basath': 62, 'tirsath': 63, 'chausath': 64, 'painsath': 65,
    'chhiyasath': 66, 'sarsath': 67, 'adsath': 68, 'unhattar': 69, 'sattar': 70,
    'ikhattar': 71, 'bahattar': 72, 'tihattar': 73, 'chauhattar': 74, 'pachattar': 75,
    'chhihattar': 76, 'sathattar': 77, 'athhattar': 78, 'unasi': 79, 'assi': 80,
    'ikyasi': 81, 'bayasi': 82, 'tirasi': 83, 'chaurasi': 84, 'pachaasi': 85,
    'chhiyasi': 86, 'sattasi': 87, 'athasi': 88, 'nawasi': 89, 'nabbe': 90,
    'ikyaanbe': 91, 'baanbe': 92, 'tiraanbe': 93, 'chauraanbe': 94, 'panchaanbe': 95,
    'chhiyaanbe': 96, 'sattaanbe': 97, 'athaanbe': 98, 'ninaanbe': 99,
}

HINDI_MULTIPLIERS = {
    'sau': 100, 'hazaar': 1000, 'hazar': 1000, 'hajaar': 1000,
    'lakh': 100000, 'lac': 100000, 'crore': 10000000, 'karod': 10000000,
}


def parse_hindi_number(text: str) -> int | None:
    """
    Parse a Hindi/Hinglish number expression into an integer.
    E.g., 'unnis hazaar' -> 19000, 'tees lakh' -> 3000000
    """
    words = text.lower().strip().split()
    if not words:
        return None
    
    total = 0
    current = 0
    found_any = False
    
    for word in words:
        word = word.strip(',').strip('.')
        if word in HINDI_DIGITS:
            current += HINDI_DIGITS[word]
            found_any = True
        elif word in HINDI_MULTIPLIERS:
            mult = HINDI_MULTIPLIERS[word]
            if current == 0:
                current = 1
            current *= mult
            total += current
            current = 0
            found_any = True
        else:
            # Not a number word — skip
            pass
    
    total += current
    return total if found_any else None


def format_indian_number(n: int) -> str:
    """Format number with Indian comma notation (e.g., 1,00,000)."""
    s = str(n)
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while rest:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups.reverse()
    return ','.join(groups) + ',' + last3


# ─── Numeric Candidate Detection ─────────────────────────────────────────────

NUMERIC_REGEX = re.compile(
    r'\b\d[\d,\.]*\b|'                     # Digit sequences
    r'\b(?:hazaar|hazar|hajaar|lakh|lac|crore|karod|sau)\b|'  # Hindi multipliers
    r'\b(?:rupee|rupaye|rupay|paisa)\b|'    # Currency words
    r'[₹]|Rs\.?|INR|'                       # Currency symbols
    r'\b(?:EMI|emi|percent|%)\b|'           # Domain keywords
    r'\b(?:' + '|'.join(HINDI_DIGITS.keys()) + r')\b',  # Hindi number words
    re.IGNORECASE
)


def detect_numeric_candidates(segments: list[dict]) -> list[dict]:
    """
    Flag segments that contain numeric content.
    Returns a list of segments with 'has_numeric' flag and 'numeric_tokens' count.
    """
    flagged = []
    for seg in segments:
        text = seg.get('text', '')
        matches = NUMERIC_REGEX.findall(text)
        seg_copy = copy.deepcopy(seg)
        seg_copy['has_numeric'] = len(matches) > 0
        seg_copy['numeric_token_count'] = len(matches)
        seg_copy['numeric_matches'] = matches
        flagged.append(seg_copy)
    return flagged


def apply_hindi_number_corrections(segments: list[dict]) -> list[dict]:
    """
    For each flagged segment, attempt to parse Hindi number expressions
    and annotate with original_text and corrected_text.
    """
    corrected = []
    for seg in segments:
        seg_out = copy.deepcopy(seg)
        text = seg.get('text', '')
        
        if seg.get('has_numeric', False):
            original_text = text
            
            # Find Hindi number word sequences and replace with digits
            # Pattern: sequence of Hindi number words possibly followed by multiplier
            hindi_num_pattern = re.compile(
                r'\b(' + '|'.join(sorted(list(HINDI_DIGITS.keys()) + list(HINDI_MULTIPLIERS.keys()), key=len, reverse=True)) + r')(?:\s+(' + '|'.join(sorted(list(HINDI_DIGITS.keys()) + list(HINDI_MULTIPLIERS.keys()), key=len, reverse=True)) + r'))*\b',
                re.IGNORECASE
            )
            
            def replace_hindi_nums(match):
                parsed = parse_hindi_number(match.group(0))
                if parsed is not None and parsed > 0:
                    return format_indian_number(parsed)
                return match.group(0)
            
            corrected_text = hindi_num_pattern.sub(replace_hindi_nums, text)
            
            if corrected_text != original_text:
                seg_out['original_text'] = original_text
                seg_out['corrected_text'] = corrected_text
                seg_out['text'] = corrected_text
                seg_out['correction_applied'] = True
            else:
                seg_out['correction_applied'] = False
        else:
            seg_out['correction_applied'] = False
        
        corrected.append(seg_out)
    return corrected


# ─── Main Benchmark Runner ───────────────────────────────────────────────────

def run_benchmark(input_dir: str, output_dir: str, mode: str):
    """Run the benchmark in the specified mode."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for dir_name in sorted(os.listdir(input_dir)):
        dir_path = os.path.join(input_dir, dir_name)
        json_path = os.path.join(dir_path, 'transcript.json')
        
        if not os.path.isdir(dir_path) or not os.path.exists(json_path):
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        
        segments = data.get('segments', [])
        if not segments:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {dir_name}")
        print(f"{'='*60}")
        
        # Step 1: Detect numeric candidates
        flagged = detect_numeric_candidates(segments)
        numeric_segs = sum(1 for s in flagged if s['has_numeric'])
        total_numeric_tokens = sum(s['numeric_token_count'] for s in flagged)
        
        print(f"  Segments: {len(segments)}")
        print(f"  Numeric segments: {numeric_segs}/{len(segments)}")
        print(f"  Total numeric tokens: {total_numeric_tokens}")
        
        if mode == 'dualpass_numeric':
            # Step 2: Apply Hindi number corrections
            corrected = apply_hindi_number_corrections(flagged)
            corrections_made = sum(1 for s in corrected if s.get('correction_applied', False))
            print(f"  Corrections applied: {corrections_made}")
            
            # Step 3: Save annotated output
            output_data = copy.deepcopy(data)
            output_data['segments'] = corrected
            output_data['benchmark'] = {
                'mode': 'dualpass_numeric',
                'numeric_segments': numeric_segs,
                'total_numeric_tokens': total_numeric_tokens,
                'corrections_applied': corrections_made,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Baseline — just copy with detection metadata
            output_data = copy.deepcopy(data)
            output_data['benchmark'] = {
                'mode': 'baseline',
                'numeric_segments': numeric_segs,
                'total_numeric_tokens': total_numeric_tokens,
                'timestamp': datetime.now().isoformat()
            }
        
        # Save output
        out_dir_path = os.path.join(output_dir, dir_name)
        os.makedirs(out_dir_path, exist_ok=True)
        out_json_path = os.path.join(out_dir_path, 'transcript.json')
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        results.append({
            'call': dir_name,
            'segments': len(segments),
            'numeric_segments': numeric_segs,
            'numeric_tokens': total_numeric_tokens,
            'corrections': sum(1 for s in (corrected if mode == 'dualpass_numeric' else flagged) if s.get('correction_applied', False)),
            'mode': mode
        })
        
        print(f"  ✅ Saved to {out_dir_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY — Mode: {mode}")
    print(f"{'='*60}")
    print(f"{'Call':<50} {'Segs':>5} {'NumSegs':>7} {'Tokens':>7} {'Fixed':>5}")
    print('-' * 80)
    for r in results:
        short_name = r['call'][:48]
        print(f"{short_name:<50} {r['segments']:>5} {r['numeric_segments']:>7} {r['numeric_tokens']:>7} {r['corrections']:>5}")
    
    total_tokens = sum(r['numeric_tokens'] for r in results)
    total_fixes = sum(r['corrections'] for r in results)
    print('-' * 80)
    print(f"{'TOTAL':<50} {sum(r['segments'] for r in results):>5} {sum(r['numeric_segments'] for r in results):>7} {total_tokens:>7} {total_fixes:>5}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Pass Numeric Micro-Pipeline Benchmark')
    parser.add_argument('--mode', choices=['baseline', 'dualpass_numeric'], default='dualpass_numeric',
                        help='Benchmark mode: baseline (copy as-is) or dualpass_numeric (apply corrections)')
    parser.add_argument('--input-dir', default='outputs/',
                        help='Input directory containing transcript.json files')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: results/<mode>/)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/benchmark_{timestamp}/{args.mode}/'
    
    print(f"Dual-Pass Numeric Benchmark")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    run_benchmark(args.input_dir, args.output_dir, args.mode)
