#!/bin/bash
# ============================================================
# Speech Transcription Pipeline — Setup Script
# Creates virtual environment, installs dependencies, validates
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  🎙️  Speech Transcription Pipeline — Setup"
echo "============================================================"

# ── 1. Check Python version ──
echo ""
echo "[1/5] Checking Python..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
            PYTHON="$candidate"
            echo "   ✅ Found $candidate (Python $ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "   ⚠️  No Python 3.10-3.12 found. Trying python3..."
    PYTHON="python3"
    echo "   Using: $($PYTHON --version 2>&1)"
    echo "   Note: If torch/whisper fail to install, install Python 3.11 or 3.12"
fi

# ── 2. Check ffmpeg ──
echo ""
echo "[2/5] Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "   ✅ ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "   ❌ ffmpeg not found."
    echo "   Install it with one of:"
    echo "     brew install ffmpeg"
    echo "     conda install ffmpeg"
    echo "   Or download from: https://ffmpeg.org/download.html"
    echo ""
    read -p "   Continue without ffmpeg? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ── 3. Create virtual environment ──
echo ""
echo "[3/5] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "   ⚠️  .venv already exists. Removing..."
    rm -rf .venv
fi
$PYTHON -m venv .venv
source .venv/bin/activate
echo "   ✅ Created .venv ($(python --version))"

# ── 4. Install dependencies ──
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"

# ── 5. Validate imports ──
echo ""
echo "[5/5] Validating imports..."
python -c "
import sys
all_ok = True
checks = [
    ('numpy',       'numpy'),
    ('torch',       'torch'),
    ('torchaudio',  'torchaudio'),
    ('whisper',     'whisper'),
    ('pyannote',    'pyannote.audio'),
    ('soundfile',   'soundfile'),
    ('pydub',       'pydub'),
    ('dotenv',      'dotenv'),
]
for name, mod in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '✓')
        print(f'  ✅ {name:14s} {ver}')
    except Exception as e:
        print(f'  ❌ {name:14s} {e}')
        all_ok = False

if all_ok:
    print()
    print('✅ All imports OK!')
else:
    print()
    print('❌ Some imports failed. Check errors above.')
    sys.exit(1)
"

# ── Done ──
echo ""
echo "============================================================"
echo "  ✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit .env and add your HF_TOKEN"
echo "    2. Activate: source .venv/bin/activate"
echo "    3. Run:      python run.py <audio_file>"
echo "============================================================"
