import os
import tempfile
import logging
import uuid
import json
import asyncio
import shutil
from datetime import datetime
from contextlib import asynccontextmanager

import secrets

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import torch
import uvicorn

from pipeline.config import PipelineConfig
from pipeline.preprocess import preprocess
from pipeline.vad import run_vad
from pipeline.diarize import run_diarization, intersect_vad_diar, infer_roles_linguistic
from pipeline.transcribe import make_chunks, transcribe_chunks, load_model
from pipeline.reconstruct import reconstruct, summarize_call_structured, format_structured_summary, correct_transcript_llm, rephrase_transcript_llm, verify_and_inject_inline_citations
from pipeline.emotion import extract_acoustic_features, analyze_emotion
from pipeline.triggers import analyze_triggers
from pipeline.sarcasm import analyze_sarcasm
from pipeline.insights import refresh_insights
from pipeline import chain as chain_mod
from pipeline import auth as auth_mod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state memory
global_state = {
    "asr_model": None,
    "engine": None,
    "device": None,
    "cfg": None
}

# In-memory job tracking
jobs = {}

# Background Queue for strict sequential processing
audio_queue: asyncio.Queue = None

async def queue_worker():
    """Continuously processes jobs from the queue strictly one by one."""
    logger.info("Background sequential queue worker started.")
    while True:
        try:
            item = await audio_queue.get()
            # Queue items may carry an optional chain assignment.
            # Tuple shapes supported: (job_id, temp_path, filename) or (job_id, temp_path, filename, chain_ctx)
            if len(item) == 4:
                job_id, temp_path, original_filename, chain_ctx = item
            else:
                job_id, temp_path, original_filename = item
                chain_ctx = None

            # Check if job was cancelled while waiting in queue
            if jobs.get(job_id, {}).get("status") == "cancelled":
                logger.info(f"Worker skipped cancelled job {job_id} ({original_filename})")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                audio_queue.task_done()
                continue

            logger.info(f"Worker picked up queued job {job_id} ({original_filename})")

            # Run the heavy blocking ML inference in a threadpool so we don't block the ASGI loop
            try:
                await asyncio.to_thread(process_audio_task, job_id, temp_path, original_filename)
                # After successful processing, attach to its chain if one was requested.
                if chain_ctx and jobs.get(job_id, {}).get("status") == "completed":
                    output_dir = jobs[job_id].get("output_dir")
                    if output_dir:
                        call_dir_name = os.path.basename(output_dir)
                        try:
                            chain_mod.append_call_to_chain(
                                chain_ctx["chain_id"], call_dir_name, chain_ctx.get("index"),
                            )
                            logger.info(f"Attached job {job_id} to chain {chain_ctx['chain_id']}")
                        except Exception as e:
                            logger.warning(f"Failed to attach job {job_id} to chain: {e}")
            except Exception as e:
                logger.error(f"Worker failed on job {job_id}: {e}", exc_info=True)
            finally:
                audio_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Queue worker shutting down.")
            break

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    global audio_queue
    audio_queue = asyncio.Queue()
    
    # Start the sequential background worker
    worker_task = asyncio.create_task(queue_worker())
    
    logger.info("Initializing Transcription Service...")
    
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using GPU (MPS)")
    else:
        device = "cpu"
        logger.info("Using CPU")
        
    cfg = PipelineConfig(
        asr_engine="hinglish",
        language="hi",
        num_speakers=2
    )
    
    global_state["cfg"] = cfg
    global_state["device"] = device
    
    logger.info("Loading ASR model into global memory...")
    asr_model, engine_used = load_model(cfg, device)
    global_state["asr_model"] = asr_model
    global_state["engine"] = engine_used
            
    logger.info("Service is ready to accept requests.")
    yield
    logger.info("Shutting down Transcription Service...")
    worker_task.cancel()
    global_state.clear()


app = FastAPI(
    title="Transcription Service API",
    description="Production backend for audio transcription with Background Tasks",
    version="1.1.0",
    lifespan=lifespan
)

# --- Session / auth setup ---------------------------------------------------
_SESSION_SECRET = os.environ.get("SESSION_SECRET_KEY")
if not _SESSION_SECRET:
    _SESSION_SECRET = secrets.token_urlsafe(32)
    logger.warning(
        "SESSION_SECRET_KEY not set — using an ephemeral random key. "
        "Sessions will reset on server restart. Set SESSION_SECRET_KEY in your environment for persistence."
    )

# Endpoints that don't require auth.
_PUBLIC_PATHS = {
    "/api/auth/signup",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/me",
    "/api/auth/teams",
}


def get_current_user(request: Request) -> dict:
    """FastAPI dependency — returns the current user or raises 401."""
    uid = request.session.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    u = auth_mod.get_user_by_id(uid)
    if u is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session expired")
    return u


def get_optional_user(request: Request) -> dict | None:
    """Like get_current_user but returns None instead of raising."""
    try:
        uid = request.session.get("user_id")
    except Exception:
        return None
    if not uid:
        return None
    return auth_mod.get_user_by_id(uid)


# IMPORTANT: middlewares execute in reverse-of-add order. We add the auth gate
# FIRST so it runs AFTER SessionMiddleware (which is added below). This makes
# request.session available when the gate runs.
@app.middleware("http")
async def _api_auth_gate(request: Request, call_next):
    """Gate all /api/* routes behind auth once at least one user exists."""
    path = request.url.path
    if path.startswith("/api/") and path not in _PUBLIC_PATHS:
        if auth_mod.user_exists():
            try:
                uid = request.session.get("user_id")
            except Exception:
                uid = None
            if not uid or auth_mod.get_user_by_id(uid) is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Not authenticated"},
                )
    return await call_next(request)


app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET,
    session_cookie="ts_session",
    max_age=14 * 24 * 3600,  # 14 days
    same_site="lax",
    https_only=False,
)


def process_audio_task(job_id: str, temp_path: str, original_filename: str):
    """Background worker for transcribing the audio so the client doesn't time out."""
    cfg = global_state["cfg"]
    device = global_state["device"]
    asr_model = global_state["asr_model"]
    
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = "Starting transcription pipeline..."
    jobs[job_id]["progress_percent"] = 5
    
    try:
        # 1. Preprocess
        jobs[job_id]["progress"] = "Preprocessing audio..."
        jobs[job_id]["progress_percent"] = 10
        prep_path, duration = preprocess(temp_path, cfg=cfg)
        
        # 2. VAD
        jobs[job_id]["progress"] = "Running Voice Activity Detection..."
        jobs[job_id]["progress_percent"] = 15
        vad_segs = run_vad(prep_path, cfg=cfg)
        
        # 3. Diarization
        jobs[job_id]["progress"] = "Running Speaker Diarization..."
        jobs[job_id]["progress_percent"] = 20
        diar_segs, diar_method = run_diarization(prep_path, vad_segs, cfg)
        
        # 4. Intersect
        jobs[job_id]["progress"] = "Fusing VAD and Diarization..."
        jobs[job_id]["progress_percent"] = 35
        speaker_segs = intersect_vad_diar(vad_segs, diar_segs)
        
        if not speaker_segs:
            raise ValueError("No speech segments detected in audio.")
        
        # 4a. Acoustic Feature Extraction (NEW — runs on preprocessed audio)
        acoustic_features = []
        if cfg.enable_emotion_analysis:
            try:
                jobs[job_id]["progress"] = "Extracting Acoustic Features..."
                jobs[job_id]["progress_percent"] = 37
                acoustic_features = extract_acoustic_features(prep_path, speaker_segs)
            except Exception as e:
                logger.warning(f"Acoustic feature extraction failed (non-fatal): {e}")
                acoustic_features = []
            
        # 5. Chunking + ASR
        jobs[job_id]["progress"] = "Generating Transcriptions..."
        jobs[job_id]["progress_percent"] = 40
        asr_chunks = make_chunks(speaker_segs, cfg)
        
        def _update_asr_progress(current, total):
            # Scale ASR progress from 40% to 80%
            if total > 0:
                fraction = current / total
                jobs[job_id]["progress_percent"] = 40 + int(fraction * 40)
        
        raw_transcript = transcribe_chunks(
            chunks=asr_chunks,
            audio_path=prep_path,
            model=asr_model,
            cfg=cfg,
            device=device,
            progress_callback=_update_asr_progress
        )
        
        # 6. Roles
        jobs[job_id]["progress"] = "Inferring Speaker Roles..."
        jobs[job_id]["progress_percent"] = 85
        raw_transcript, role_map = infer_roles_linguistic(raw_transcript, cfg)
        
        # 7. Reconstruction
        jobs[job_id]["progress"] = "Reconstructing Timeline & Formatting Output..."
        jobs[job_id]["progress_percent"] = 90
        transcript = reconstruct(raw_transcript, cfg)
        
        # 7a. LLM Contextual Transcription Correction
        if cfg.openai_api_key:
            jobs[job_id]["progress"] = "Running LLM Contextual Correction..."
            jobs[job_id]["progress_percent"] = 92
            transcript = correct_transcript_llm(transcript, cfg)
        
        # 7b. LLM Transcript Rephrasing
        if cfg.openai_api_key and cfg.enable_rephrase:
            jobs[job_id]["progress"] = "Rephrasing Transcript for Readability..."
            jobs[job_id]["progress_percent"] = 93
            transcript = rephrase_transcript_llm(transcript, cfg)
        
        # 8a. Emotion Analysis (NEW — uses acoustic features + transcript)
        emotion_analysis = None
        trigger_phrases = None
        if cfg.enable_emotion_analysis:
            try:
                jobs[job_id]["progress"] = "Analyzing Emotion & Escalation..."
                jobs[job_id]["progress_percent"] = 94
                emotion_analysis = analyze_emotion(prep_path, speaker_segs, transcript)
                
                # 8b. Trigger Phrase Extraction
                all_heated = (
                    emotion_analysis.get("agent_heated_segments", []) +
                    emotion_analysis.get("customer_heated_segments", [])
                )
                trigger_phrases = analyze_triggers(all_heated, acoustic_features, transcript, model=cfg.summary_model, api_key=cfg.openai_api_key)
            except Exception as e:
                logger.warning(f"Emotion/trigger analysis failed (non-fatal): {e}")
                emotion_analysis = None
                trigger_phrases = None
        
        # 8c. Sarcasm Detection (additive, after emotion analysis)
        sarcasm_analysis = None
        if emotion_analysis and cfg.enable_emotion_analysis:
            try:
                sarcasm_analysis = analyze_sarcasm(
                    transcript, acoustic_features,
                    emotion_analysis.get("emotion_timeline", [])
                )
            except Exception as e:
                logger.warning(f"Sarcasm detection failed (non-fatal): {e}")
                sarcasm_analysis = None
        
        # 8. Summarization
        summary_dict = None
        summary_text = None
        if cfg.openai_api_key:
            jobs[job_id]["progress"] = "Generating Call Summary..."
            jobs[job_id]["progress_percent"] = 95
            try:
                summary_dict = summarize_call_structured(
                    transcript, cfg,
                    emotion_analysis=emotion_analysis,
                    trigger_phrases=trigger_phrases,
                )
                # Resolve `[S<id>]` / legacy `[MM:SS]` citations to clickable
                # transcript links with exact `t0` values.
                summary_dict = verify_and_inject_inline_citations(summary_dict, transcript)
                summary_text = format_structured_summary(summary_dict)
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                
        # Build output structure
        dialogue = []
        for s in transcript:
            speaker_label = s.get("role", s.get("speaker", "unknown")).upper()
            dialogue.append(f"{speaker_label} [{s['t0']:.1f}s - {s['t1']:.1f}s]: {s['text']}")

        owner_user_id = jobs[job_id].get("owner_user_id")
        output_data = {
            "metadata": {
                "duration_seconds": duration,
                "asr_engine": cfg.asr_engine,
                "diarization_method": diar_method,
                "role_map": role_map,
                "original_filename": original_filename
            },
            "summary": summary_dict,  # Store as structured dict for frontend rendering
            "segments": transcript,
            "dialogue": dialogue,
        }
        if owner_user_id:
            output_data["owner_user_id"] = owner_user_id
        
        # Append NEW keys only — backward compatible
        if emotion_analysis is not None:
            output_data["emotion_analysis"] = emotion_analysis
        if trigger_phrases is not None:
            output_data["trigger_phrases"] = trigger_phrases
        if sarcasm_analysis is not None:
            output_data["sarcasm_analysis"] = sarcasm_analysis
        
        # Save to outputs directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_fname = os.path.splitext(os.path.basename(original_filename))[0].replace(" ", "_")
        job_dir = os.path.join(OUTPUTS_DIR, f"{safe_fname}_{timestamp}_{job_id[:6]}")
        os.makedirs(job_dir, exist_ok=True)
        
        json_path = os.path.join(job_dir, "transcript.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        if summary_text:
            summary_path = os.path.join(job_dir, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
                
        txt_path = os.path.join(job_dir, "transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dialogue))
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Done"
        jobs[job_id]["progress_percent"] = 100
        jobs[job_id]["result"] = output_data
        jobs[job_id]["output_dir"] = job_dir
        logger.info(f"Job {job_id} completed successfully. Saved to {job_dir}")

        # Refresh winning patterns in background (non-blocking, best-effort)
        try:
            refresh_insights(cfg.openai_api_key)
        except Exception as insights_err:
            logger.warning(f"Insights refresh failed (non-fatal): {insights_err}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress_percent"] = 100
        jobs[job_id]["error"] = str(e)
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/transcribe")
async def transcribe_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Ingest audio and add it to the sequential task queue.
    """
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file extension format.")

    if not global_state.get("asr_model"):
        raise HTTPException(status_code=503, detail="Models are not initialized yet.")

    current_user = get_optional_user(request)
    job_id = str(uuid.uuid4())

    # Save audio temporarily
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode="wb") as tmp:
        try:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            temp_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write uploaded file: {e}")

    # Initialize job tracking
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": "Queued in position",
        "original_filename": file.filename,
        "result": None,
        "error": None,
        "owner_user_id": current_user["id"] if current_user else None,
    }

    # Add to sequential asyncio queue
    await audio_queue.put((job_id, temp_path, file.filename))
    
    logger.info(f"Queued job {job_id} for {file.filename}. Queue size: {audio_queue.qsize()}")
    
    return JSONResponse(content={
        "job_id": job_id, 
        "status": "queued", 
        "queue_position": audio_queue.qsize(),
        "message": "File added to sequential processing queue."
    })


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll for the status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=jobs[job_id])


@app.get("/api/queue")
async def get_active_queue():
    """Returns all jobs currently queued, uploading, or processing across active sessions."""
    active_jobs = {
        jid: job for jid, job in jobs.items() 
        if job["status"] in ["queued", "processing"]
    }
    return JSONResponse(content={"active": active_jobs})


@app.delete("/api/queue/{job_id}")
async def cancel_job(job_id: str):
    """Cancels a job if it is still queued. If processing, merely marks it failed UI-side."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job["status"] == "completed":
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
        
    if job["status"] == "queued":
        job["status"] = "cancelled"
        job["progress"] = "Cancelled by User"
        logger.info(f"User cancelled queued job {job_id}")
        return JSONResponse(content={"status": "cancelled", "message": "Job removed from queue."})
        
    if job["status"] == "processing":
        job["status"] = "cancelled"
        job["progress"] = "Cancelled by User (Warning: ML inferencing may finish invisibly)"
        logger.info(f"User cancelled processing job {job_id} (inference thread detached)")
        return JSONResponse(content={"status": "cancelled", "message": "Active job UI cancelled."})
    
    return JSONResponse(content={"status": job["status"]})


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/signup")
async def auth_signup_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    try:
        user = auth_mod.signup(
            username=body.get("username", ""),
            name=body.get("name", ""),
            team=body.get("team", ""),
            password=body.get("password", ""),
        )
    except auth_mod.SignupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    request.session["user_id"] = user["id"]
    return JSONResponse(content={"user": user, "message": "Account created."})


@app.post("/api/auth/login")
async def auth_login_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    user = auth_mod.verify_credentials(body.get("username", ""), body.get("password", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    request.session["user_id"] = user["id"]
    return JSONResponse(content={"user": user})


@app.post("/api/auth/logout")
async def auth_logout_endpoint(request: Request):
    request.session.clear()
    return JSONResponse(content={"message": "Logged out."})


@app.get("/api/auth/me")
async def auth_me_endpoint(request: Request):
    uid = request.session.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth_mod.get_user_by_id(uid)
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session expired")
    return JSONResponse(content={"user": user})


@app.get("/api/auth/teams")
async def auth_teams_endpoint():
    return JSONResponse(content={"teams": auth_mod.list_teams()})


@app.get("/api/history")
async def get_history(request: Request):
    """Scan the outputs directory and return a list of past transcriptions."""
    history = []
    if os.path.exists(OUTPUTS_DIR):
        for dir_name in os.listdir(OUTPUTS_DIR):
            dir_path = os.path.join(OUTPUTS_DIR, dir_name)
            if os.path.isdir(dir_path):
                json_path = os.path.join(dir_path, "transcript.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                            # Fallback for old legacy JSONs missing original_filename metadata
                            filename = data.get("metadata", {}).get("original_filename")
                            if not filename or filename == "Unknown":
                                # Try to parse a cleaner name from the ugly migration folder
                                # e.g. "legacy_migrated_transcript_short_new_1771822898" -> "short_new"
                                if dir_name.startswith("legacy_migrated_transcript_"):
                                    extracted = dir_name.replace("legacy_migrated_transcript_", "")
                                    # Strip epoch suffix e.g _1771822898
                                    extracted = "_".join(extracted.split("_")[:-1])
                                    filename = f"{extracted}.mp3" if extracted else dir_name
                                else:
                                    filename = dir_name

                            # Resolve citations at read time so historical outputs (written before
                            # the resolver was wired in) still get exact transcript links.
                            segments = data.get("segments", [])
                            summary = data.get("summary")
                            if summary and isinstance(summary, dict) and "error" not in summary:
                                summary = verify_and_inject_inline_citations(summary, segments)

                            history.append({
                                "id": dir_name,
                                "filename": filename,
                                "duration": data.get("metadata", {}).get("duration_seconds", 0),
                                "summary": summary,
                                "metadata": data.get("metadata", {}),
                                "segments": segments,
                                "emotion_analysis": data.get("emotion_analysis", None),
                                "trigger_phrases": data.get("trigger_phrases", None),
                                "owner_user_id": data.get("owner_user_id"),
                            })
                    except Exception as e:
                        logger.warning(f"Failed to read history {dir_name}: {e}")
    # Sort descending assuming directory names start with or contain timestamps (they do)
    history.sort(key=lambda x: x["id"], reverse=True)

    # Group chain members under their manifest; everything else is a "single" call.
    chain_manifests = chain_mod.list_chains()
    manifest_by_id = {m["id"]: m for m in chain_manifests}
    call_to_chain_id: dict[str, str] = {}
    for m in chain_manifests:
        for c in m["calls"]:
            call_to_chain_id[c["dir"]] = m["id"]

    chains_out: dict[str, dict] = {}
    singles_out: list[dict] = []
    for item in history:
        cid = call_to_chain_id.get(item["id"])
        if cid:
            if cid not in chains_out:
                m = manifest_by_id[cid]
                chains_out[cid] = {
                    "id": cid,
                    "slug": m["slug"],
                    "label": m["label"],
                    "customer_identifier": m.get("customer_identifier"),
                    "owner_user_id": m.get("owner_user_id"),
                    "created_at": m.get("created_at"),
                    "updated_at": m.get("updated_at"),
                    "closed": m.get("closed", False),
                    "total_calls_planned": len(m["calls"]),
                    "member_calls": [],
                }
            chains_out[cid]["member_calls"].append(item)
        else:
            singles_out.append(item)

    # Preserve chain-member order by index, not by timestamp
    for cid, entry in chains_out.items():
        index_by_dir = {c["dir"]: c["index"] for c in manifest_by_id[cid]["calls"]}
        entry["member_calls"].sort(key=lambda x: index_by_dir.get(x["id"], 999))

    # Partition by ownership for the current user. Unowned items (legacy) surface
    # as a shared archive visible to everyone.
    current_user = get_optional_user(request)
    uid = current_user["id"] if current_user else None

    my_chains, legacy_chains = [], []
    for c in chains_out.values():
        (my_chains if uid and c.get("owner_user_id") == uid
         else legacy_chains if not c.get("owner_user_id")
         else []).append(c)

    my_singles, legacy_singles = [], []
    for s in singles_out:
        (my_singles if uid and s.get("owner_user_id") == uid
         else legacy_singles if not s.get("owner_user_id")
         else []).append(s)

    # Flat "history" field retained for backwards compatibility with existing dashboard code
    return JSONResponse(content={
        "history": history,                         # legacy flat list (all calls)
        "chains": list(chains_out.values()),        # legacy (all chains)
        "singles": singles_out,                     # legacy (all singles)
        "my_chains": my_chains,
        "my_singles": my_singles,
        "legacy_chains": legacy_chains,
        "legacy_singles": legacy_singles,
    })

@app.delete("/api/history/{job_id}")
async def delete_history_item(job_id: str):
    """Securely deletes a specific transcription job's output folder."""
    if not job_id or ".." in job_id or "/" in job_id:
        raise HTTPException(status_code=400, detail="Invalid job ID")

    target_path = os.path.join(OUTPUTS_DIR, job_id)
    if not os.path.exists(target_path) or not os.path.isdir(target_path):
        raise HTTPException(status_code=404, detail="Job history not found")

    try:
        shutil.rmtree(target_path)
        logger.info(f"Deleted historic output folder: {job_id}")
        return JSONResponse(content={"status": "success", "message": "History item deleted."})
    except Exception as e:
        logger.error(f"Failed to delete history item {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete directory: {str(e)}")

@app.get("/api/insights")
async def get_insights():
    """Return cached winning patterns. Returns 202 if not yet generated."""
    cfg = global_state.get("cfg")
    api_key = cfg.openai_api_key if cfg else os.getenv("OPENAI_API_KEY", "")

    if os.path.exists(os.path.join(OUTPUTS_DIR, "winning_patterns.json")):
        with open(os.path.join(OUTPUTS_DIR, "winning_patterns.json"), "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    # Not cached yet — generate synchronously (first time only)
    result = await asyncio.to_thread(refresh_insights, api_key)
    if result is None:
        return JSONResponse(status_code=202, content={"message": "Not enough calls to generate insights yet (need at least 2)."})
    return JSONResponse(content=result)


@app.post("/api/insights/refresh")
async def force_refresh_insights():
    """Force-regenerate winning patterns from all calls."""
    cfg = global_state.get("cfg")
    api_key = cfg.openai_api_key if cfg else os.getenv("OPENAI_API_KEY", "")
    result = await asyncio.to_thread(refresh_insights, api_key, True)
    if result is None:
        return JSONResponse(status_code=202, content={"message": "Not enough calls to generate insights yet (need at least 2)."})
    return JSONResponse(content=result)



# ---------------------------------------------------------------------------
# Chain endpoints (Slice 1 — call chaining infrastructure)
# ---------------------------------------------------------------------------

def _safe_chain_id(chain_id: str) -> str:
    if not chain_id or ".." in chain_id or "/" in chain_id or "\\" in chain_id:
        raise HTTPException(status_code=400, detail="Invalid chain ID")
    return chain_id


async def _enqueue_file_for_chain(
    file: UploadFile, chain_id: str, index: int, owner_user_id: str | None = None,
) -> str:
    """Persist upload to temp, register job, enqueue with chain context. Returns job_id."""
    if not file.filename or not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {file.filename}")
    if not global_state.get("asr_model"):
        raise HTTPException(status_code=503, detail="Models are not initialized yet.")

    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode="wb") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        temp_path = tmp.name

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": f"Queued (chain call {index})",
        "original_filename": file.filename,
        "result": None,
        "error": None,
        "chain_id": chain_id,
        "chain_index": index,
        "owner_user_id": owner_user_id,
    }
    await audio_queue.put((job_id, temp_path, file.filename, {"chain_id": chain_id, "index": index}))
    return job_id


@app.post("/api/chain")
async def create_chain_endpoint(
    request: Request,
    label: str | None = Form(None),
    customer_identifier: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    """Create a chain and enqueue all supplied files in the order received."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    current_user = get_optional_user(request)
    owner_user_id = current_user["id"] if current_user else None

    manifest = chain_mod.create_chain(
        label=label,
        customer_identifier=customer_identifier,
        owner_user_id=owner_user_id,
    )

    job_ids: list[str] = []
    for i, f in enumerate(files, start=1):
        job_ids.append(await _enqueue_file_for_chain(f, manifest["id"], i, owner_user_id))

    logger.info(f"Created chain {manifest['id']} ({manifest['slug']}) with {len(files)} queued calls.")
    return JSONResponse(content={
        "chain_id": manifest["id"],
        "slug": manifest["slug"],
        "label": manifest["label"],
        "call_job_ids": job_ids,
        "message": f"Chain created with {len(files)} calls queued.",
    })


@app.get("/api/chain/{chain_id}")
async def get_chain_endpoint(chain_id: str):
    chain_id = _safe_chain_id(chain_id)
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Chain not found")

    # Attach per-call job statuses for any jobs still tracked in memory
    in_flight = [
        j for j in jobs.values()
        if j.get("chain_id") == chain_id and j.get("status") in ("queued", "processing", "failed")
    ]
    return JSONResponse(content={**manifest, "in_flight_jobs": in_flight})


@app.post("/api/chain/{chain_id}/append")
async def append_to_chain_endpoint(chain_id: str, file: UploadFile = File(...)):
    chain_id = _safe_chain_id(chain_id)
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Chain not found")

    next_index = len(manifest["calls"]) + 1
    job_id = await _enqueue_file_for_chain(file, chain_id, next_index)
    return JSONResponse(content={
        "chain_id": chain_id,
        "job_id": job_id,
        "index": next_index,
    })


def _kickoff_chain_summary_task(chain_id: str) -> dict:
    """
    Fire-and-forget chain summarization. Returns a dict describing what
    happened so the caller can surface the right UI state. Skips when:
      - chain has no members or missing,
      - OPENAI_API_KEY is unset,
      - a generation is already in-flight,
      - cache is already fresh (manifest sig matches).
    Never raises.
    """
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None or not manifest.get("calls"):
        return {"triggered": False, "reason": "empty_chain"}

    cfg = global_state.get("cfg") or PipelineConfig()
    if not cfg.openai_api_key:
        return {"triggered": False, "reason": "no_api_key"}

    if chain_id in _chain_summary_in_flight:
        return {"triggered": False, "reason": "already_generating"}

    cached = chain_mod.get_chain_summary(chain_id)
    if not cached.get("stale"):
        return {"triggered": False, "reason": "cache_fresh"}

    async def _run():
        _chain_summary_in_flight.add(chain_id)
        try:
            logger.info(f"Auto-summary starting for chain {chain_id} (triggered by close)")
            await asyncio.to_thread(chain_mod.summarize_chain, chain_id, cfg)
            logger.info(f"Auto-summary complete for chain {chain_id}")
        except Exception as e:
            logger.error(f"Auto-summary failed for chain {chain_id}: {e}", exc_info=True)
        finally:
            _chain_summary_in_flight.discard(chain_id)

    asyncio.create_task(_run())
    return {"triggered": True, "reason": "started"}


@app.post("/api/chain/{chain_id}/close")
async def close_chain_endpoint(chain_id: str):
    chain_id = _safe_chain_id(chain_id)
    try:
        manifest = chain_mod.close_chain(chain_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Chain not found")

    auto = _kickoff_chain_summary_task(chain_id)
    return JSONResponse(content={**manifest, "auto_summary": auto})


@app.delete("/api/chain/{chain_id}")
async def delete_chain_endpoint(chain_id: str):
    chain_id = _safe_chain_id(chain_id)
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Chain not found")
    chain_mod.delete_chain(chain_id)
    logger.info(f"Deleted chain {chain_id}")
    return JSONResponse(content={"status": "success", "message": "Chain deleted; member calls preserved."})


@app.get("/api/chain/{chain_id}/transcript")
async def get_chain_transcript_endpoint(chain_id: str):
    """Return the chain's combined transcript (rebuilt on demand for freshness)."""
    chain_id = _safe_chain_id(chain_id)
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Chain not found")
    return JSONResponse(content=chain_mod.rebuild_chain_transcript(chain_id))


# --- Chain summary (Slice 2) ------------------------------------------------

# Tracks chains that are currently being summarized so concurrent requests
# return 409 instead of redundantly calling the LLM.
_chain_summary_in_flight: set[str] = set()


@app.get("/api/chain/{chain_id}/summary")
async def get_chain_summary_endpoint(chain_id: str):
    """Return the cached chain summary plus a staleness indicator. Never
    triggers generation — use POST /summarize for that."""
    chain_id = _safe_chain_id(chain_id)
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Chain not found")
    result = chain_mod.get_chain_summary(chain_id)
    result["generating"] = chain_id in _chain_summary_in_flight
    return JSONResponse(content=result)


@app.post("/api/chain/{chain_id}/summarize")
async def summarize_chain_endpoint(chain_id: str):
    """Run chain-level LLM summarization. Blocking (~30–120s)."""
    chain_id = _safe_chain_id(chain_id)
    manifest = chain_mod.get_chain(chain_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Chain not found")
    if chain_id in _chain_summary_in_flight:
        raise HTTPException(status_code=409, detail="Chain summary is already generating.")

    cfg = global_state.get("cfg") or PipelineConfig()
    if not cfg.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required for chain summarization.")

    _chain_summary_in_flight.add(chain_id)
    try:
        logger.info(f"Starting chain summary for {chain_id}")
        # Run the blocking LLM call in a threadpool so we don't block the ASGI loop.
        out = await asyncio.to_thread(chain_mod.summarize_chain, chain_id, cfg)
        if "error" in out:
            raise HTTPException(status_code=500, detail=out["error"])
        logger.info(f"Chain summary complete for {chain_id}")
        return JSONResponse(content=out)
    finally:
        _chain_summary_in_flight.discard(chain_id)

# Mount the static directory to serve the GUI Dashboard at root
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcription Service API with Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    
    print(f"\n🌐 Dashboard will be available at: http://localhost:{args.port}/\n")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
