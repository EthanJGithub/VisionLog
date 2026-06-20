"""VisionLog API routes.

Two detection paths:
  • Upload (server-side): YOLO26 n/s/m/x via onnxruntime, or YOLOE-26 open-vocabulary.
  • Webcam (client-side WebGPU): the browser runs inference and POSTs sampled detections
    here for logging — see /client-sessions.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile, File

from src import config, store, video
from src.api import schemas
from src.agent import chat_graph
from src.detect import openvocab, registry, ultra

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=schemas.HealthOut)
def health() -> schemas.HealthOut:
    # n/s: lean onnx (by file). m/x: ultralytics "full" image (by availability).
    server_models = dict(registry.available_models())
    heavy = ultra.available()
    server_models["yolo26m"] = heavy
    server_models["yolo26x"] = heavy
    return schemas.HealthOut(
        status="ok" if any(server_models.values()) else "degraded",
        model_version=config.MODEL_VERSION,
        model_loaded=any(server_models.values()),
        sample_fps=config.SAMPLE_FPS,
        max_upload_mb=config.MAX_UPLOAD_MB,
        max_duration_seconds=config.MAX_DURATION_SECONDS,
        server_models=server_models,
        open_vocab_available=openvocab.available(),
        chat_available=chat_graph.available(),
    )


@router.post("/sources", response_model=schemas.UploadResult)
async def upload_and_detect(
    file: UploadFile = File(...),
    model: str = Form("yolo26n"),
    prompts: str = Form(""),
) -> schemas.UploadResult:
    """Upload a video, run the selected model over sampled frames, and log to the DB.

    Data minimization (COMPLIANCE.md): the raw upload is deleted once processing ends.
    """
    data = await file.read()
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    try:
        tmp = video.save_upload(data, suffix=suffix)
    except video.VideoTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    # Resolve the detector for the requested model (no silent fallback).
    is_openvocab = model == "yoloe26"
    is_heavy = ultra.supports(model)  # yolo26m / yolo26x
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
    engine = None
    if is_openvocab:
        if not openvocab.available():
            tmp.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail="Open-vocabulary (YOLOE-26) needs the full server image with "
                "`ultralytics` installed. Not available on this deployment.",
            )
        if not prompt_list:
            tmp.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail="Open-vocabulary needs `prompts` (comma-separated class names).",
            )
        model_version = openvocab.model_version()
        conf_threshold = config.CONF_THRESHOLD
    elif is_heavy:
        if not ultra.available():
            tmp.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail=f"{model} needs the full server image with `ultralytics` installed. "
                "Not available on this deployment — use n/s (they run in your browser).",
            )
        model_version = f"{model}-ultralytics"
        conf_threshold = config.CONF_THRESHOLD
    else:
        try:
            engine = registry.get_onnx_engine(model)
        except (KeyError, FileNotFoundError) as exc:
            tmp.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        model_version = engine.model_version
        conf_threshold = engine.conf_threshold

    try:
        source_id = store.create_source(
            "upload",
            filename=file.filename,
            fps=config.SAMPLE_FPS,
            model_version=model_version,
            conf_threshold=conf_threshold,
        )
        frames_analysed = 0
        all_dets: list[schemas.DetectionOut] = []
        total_logged = 0
        try:
            for frame_no, ts, frame in video.iter_sampled_frames(tmp):
                frames_analysed += 1
                if is_openvocab:
                    dets = openvocab.detect_frame(frame, prompt_list, conf_threshold)
                elif is_heavy:
                    dets = ultra.detect_frame(model, frame, conf_threshold)
                else:
                    dets = [d.as_dict() for d in engine.detect(frame)]
                total_logged += store.add_detections(source_id, frame_no, ts, dets)
                for d in dets:
                    all_dets.append(
                        schemas.DetectionOut(frame_number=frame_no, ts_seconds=ts, **d)
                    )
        except video.VideoTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows may briefly hold the handle; best-effort cleanup

    src_row = next((s for s in store.list_sources() if s["id"] == source_id), None)
    return schemas.UploadResult(
        source=schemas.SourceOut(**src_row),
        frames_analysed=frames_analysed,
        detections_logged=total_logged,
        detections=all_dets,
    )


@router.get("/sources", response_model=list[schemas.SourceOut])
def get_sources() -> list[schemas.SourceOut]:
    return [schemas.SourceOut(**s) for s in store.list_sources()]


@router.get("/sources/{source_id}/detections", response_model=list[schemas.DetectionOut])
def source_detections(source_id: int) -> list[schemas.DetectionOut]:
    return [schemas.DetectionOut(**d) for d in store.get_detections(source_id)]


@router.delete("/sources/{source_id}")
def delete_source(source_id: int) -> dict:
    """Delete one run (and its detections). 404 if it doesn't exist."""
    if not store.delete_source(source_id):
        raise HTTPException(status_code=404, detail=f"source {source_id} not found")
    return {"deleted": source_id}


@router.delete("/sources")
def clear_all_sources() -> dict:
    """Reset: delete ALL runs + detections (user-initiated)."""
    return store.clear_all()


@router.get("/stats", response_model=schemas.StatsOut)
def stats(source_id: int | None = None) -> schemas.StatsOut:
    # source_id scopes the totals + per-class counts to one run; omitted = all runs.
    return schemas.StatsOut(
        totals=store.totals(source_id=source_id),
        class_counts=[schemas.ClassCount(**c) for c in store.class_counts(source_id=source_id)],
    )


# --- Client-side (WebGPU) webcam logging --------------------------------------------
@router.post("/client-sessions", response_model=schemas.SourceOut)
def create_client_session(body: schemas.ClientSessionIn) -> schemas.SourceOut:
    """Open a logging session for in-browser detection (one source row)."""
    source_id = store.create_source(
        body.kind,
        filename=None,
        fps=None,
        model_version=body.model_version,
        conf_threshold=body.conf_threshold,
    )
    src_row = next((s for s in store.list_sources() if s["id"] == source_id), None)
    return schemas.SourceOut(**src_row)


@router.post("/client-sessions/{source_id}/detections", response_model=schemas.ClientLogResult)
def log_client_detections(source_id: int, body: schemas.ClientLogIn) -> schemas.ClientLogResult:
    """Bulk-log detections produced by the browser (sampled, ~1x/sec)."""
    logged = 0
    for frame in body.items:
        dets = [d.model_dump() for d in frame.detections]
        logged += store.add_detections(source_id, frame.frame_number, frame.ts_seconds, dets)
        for d in dets:  # a thumb is sent once per new confirmed track → store the object crop
            if d.get("thumb") and d.get("track_id") is not None:
                store.upsert_object_crop(
                    source_id, d["track_id"], d["class_label"], d["confidence"], d["thumb"]
                )
    return schemas.ClientLogResult(logged=logged)


@router.get("/objects", response_model=list[schemas.ObjectCropOut])
def get_objects(source_id: int | None = None, class_label: str | None = None) -> list[schemas.ObjectCropOut]:
    return [
        schemas.ObjectCropOut(**c)
        for c in store.get_object_crops(source_id=source_id, class_label=class_label)
    ]


# --- Chatbot: LangGraph multi-agent text-to-SQL over the detections DB ----------------
@router.post("/chat", response_model=schemas.ChatOut)
def chat(body: schemas.ChatIn) -> schemas.ChatOut:
    if not chat_graph.available():
        raise HTTPException(
            status_code=503,
            detail="Chatbot needs the full server image (langgraph/langchain-groq) and a "
            "GROQ_API_KEY. Not available on this deployment.",
        )
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Ask a question about the detections.")
    try:
        return schemas.ChatOut(**chat_graph.ask(body.question.strip()))
    except Exception as exc:  # pragma: no cover - LLM/runtime errors
        raise HTTPException(status_code=500, detail=f"chat failed: {str(exc)[:200]}") from exc
