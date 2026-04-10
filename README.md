# FREED AI — Call Analyser

AI-powered speech transcription and call analysis pipeline for Hindi-Hinglish sales calls. Built for debt relief sales teams to understand what closes deals.

## What It Does

Processes a raw audio recording of a sales call and produces:

- **Full transcript** with speaker diarization (agent vs. customer)
- **Emotion analysis** — raised voice detection, escalation events, emotion timeline
- **Trigger phrases** — hesitation markers, buying signals, objection moments
- **Business insights** — LLM-extracted actionable signals from the conversation
- **Call summary** — structured QA brief with agent checklist
- **Winning Patterns** — cross-call analysis that identifies what converts customers

## Tech Stack

| Layer | Tools |
|---|---|
| ASR | Oriserve Whisper-Hindi2Hinglish-Prime (fine-tuned Whisper) |
| Diarization | pyannote.audio 3.1 / NeMo MSDD / MFCC fallback |
| VAD | Silero VAD |
| Emotion | openSMILE eGeMAPS acoustic features |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Backend | FastAPI + async job queue |
| Frontend | Vanilla HTML/CSS/JS dashboard |

## Requirements

- Python 3.10–3.11
- ffmpeg
- ~6GB disk space (ASR model cache)
- OpenAI API key (for summaries, insights, correction)
- HuggingFace token with access to pyannote models (for best diarization)

## Setup

### 1. Install system dependencies

```bash
brew install python@3.11 ffmpeg   # macOS
# or: sudo apt install python3.11 ffmpeg  # Ubuntu
```

### 2. Clone and create virtual environment

```bash
git clone https://github.com/iampushpendra/Transcription-service.git
cd Transcription-service
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
HF_TOKEN=hf_your_token_here
OPENAI_API_KEY=sk-your-key-here
```

**HuggingFace token** — get one at huggingface.co/settings/tokens, then accept terms at:
- huggingface.co/pyannote/speaker-diarization-3.1
- huggingface.co/pyannote/segmentation-3.0

**OpenAI key** — get one at platform.openai.com

### 4. Run

```bash
python server.py
```

First run downloads the ASR model (~4.7GB). Open **http://localhost:8000** in your browser.

## Dashboard

| Tab | What you see |
|---|---|
| Overview | Aggregate stats across all processed calls |
| New Process | Upload an audio file (MP3, WAV, M4A, OGG) |
| Job History | Past transcriptions, click to open full analysis |
| Winning Patterns | Cross-call LLM analysis of what converts customers |

Each processed call shows:
- Full transcript with speaker coloring
- Emotion timeline and heated moment detection
- Trigger phrases (hesitation, buying signals, objections)
- Structured call summary with agent checklist

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/transcribe` | Upload audio file, returns `job_id` |
| GET | `/api/status/{job_id}` | Poll job progress (0–100%) |
| GET | `/api/queue` | Active jobs |
| DELETE | `/api/queue/{job_id}` | Cancel a queued job |
| GET | `/api/history` | All past transcriptions |
| DELETE | `/api/history/{job_id}` | Delete a transcription |
| GET | `/api/insights` | Winning patterns (cached) |
| POST | `/api/insights/refresh` | Force-regenerate winning patterns |

## Pipeline Stages

```
Audio → Preprocess → VAD → Diarize → Acoustic Features
     → ASR → Role Inference → Reconstruct → LLM Correction
     → Emotion Analysis → Trigger Extraction → Sarcasm Detection
     → LLM Summary → Output JSON
```

## Output Structure

Each processed call saves to `outputs/{filename}_{timestamp}/`:

- `transcript.json` — full structured output (segments, emotion, triggers, summary)
- `transcript.txt` — plain dialogue
- `summary.txt` — human-readable QA brief

## Supported Audio Formats

MP3, WAV, M4A, OGG

## Cost Notes

- **ASR, diarization, VAD, emotion** — run fully locally, no API cost
- **LLM steps** (summary, correction, insights) — billed to your OpenAI account (~$0.02–$0.10 per call depending on duration)
- LLM steps are optional: if `OPENAI_API_KEY` is not set, the pipeline still runs and produces a transcript with emotion analysis

## Versioning

| Version | Description |
|---|---|
| v1.0.0 | Stable baseline — full pipeline working |
| v1.1.0 | Winning Patterns cross-call analysis |

## License

MIT
