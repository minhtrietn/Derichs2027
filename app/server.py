"""FastAPI + WebSocket backend server for DERICH YOLO Auto-Training Studio."""

from __future__ import annotations

import asyncio
import json
import platform
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.auto_labeler import auto_labeler
from app.config import (
    EXPORT_FORMATS,
    HYPERPARAM_DOCS,
    MODEL_CATALOG,
    MODEL_FAMILY_INFO,
    PLATFORM_PROFILES,
    RUNS_DIR,
    STATIC_DIR,
    TARGET_CUDA_VERSION,
    TARGET_PYTHON_VERSION,
    TRAINING_PRESETS,
    build_cu118_install_command,
    build_setup_command,
    configure_cuda_environment,
    ensure_directories,
    get_os_type,
)
from app.dataset_manager import dataset_manager
from app.exporter import model_exporter
from app.trainer import training_engine
from app.ws_manager import collect_system_telemetry, ws_manager


def _parse_torch_cuda_tag(torch_version: str) -> str | None:
    """Parse CUDA version string from PyTorch wheel version tag (e.g. '2.7.1+cu118' -> '11.8')."""
    m = re.search(r"\+cu(\d+)(\d)\b", torch_version)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return None


def _probe_ml_stack() -> dict[str, Any]:
    """Fast, deadlock-free version probe using package metadata (avoids importing torch in worker threads)."""
    import importlib.metadata

    cuda_env = configure_cuda_environment()
    info: dict[str, Any] = {
        "os_type": get_os_type(),
        "os_name": platform.system(),
        "torch_version": "not installed",
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_toolkit_version": cuda_env.get("cuda_toolkit_version"),
        "cuda_toolkit_path": cuda_env.get("cuda_toolkit_path"),
        "cuda_version_mismatch": False,
        "target_cuda": TARGET_CUDA_VERSION,
        "target_python": TARGET_PYTHON_VERSION,
        "python_version": platform.python_version(),
        "python_compatible": sys.version_info[:2] in {(3, 11), (3, 12)},
        "cuda_install_cmd": build_cu118_install_command(sys.executable),
        "setup_cmd": build_setup_command(),
        "ultralytics_version": "not installed",
    }
    try:
        tv = importlib.metadata.version("torch")
        info["torch_version"] = tv
        parsed_cuda = _parse_torch_cuda_tag(tv)
        info["torch_cuda_version"] = parsed_cuda

        # +cuXXX wheels enable CUDA when an NVIDIA GPU is present; +cpu wheels never have CUDA
        if "+cu" in tv or ("+cpu" not in tv and "+rocm" not in tv):
            if "torch" in sys.modules:
                mod = sys.modules["torch"]
                info["cuda_available"] = bool(mod.cuda.is_available())
                if getattr(mod.version, "cuda", None):
                    info["torch_cuda_version"] = str(mod.version.cuda)
            else:
                from app.ws_manager import _query_nvidia_smi

                info["cuda_available"] = _query_nvidia_smi() is not None and bool(parsed_cuda)

        if info["torch_cuda_version"] and info["torch_cuda_version"] != TARGET_CUDA_VERSION:
            info["cuda_version_mismatch"] = True
    except Exception:
        pass
    try:
        info["ultralytics_version"] = importlib.metadata.version("ultralytics")
    except Exception:
        pass
    return info


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_directories()
    dataset_manager.ensure_default_project()
    ws_manager.bind_loop(asyncio.get_running_loop())
    ws_manager.torch_info = _probe_ml_stack()
    info = ws_manager.torch_info
    print(
        f"[DERICH Runtime] Python {info['python_version']} (target {info['target_python']}) | "
        f"PyTorch {info['torch_version']} | CUDA Toolkit {info.get('cuda_toolkit_version') or 'N/A'} | "
        f"CUDA Ready: {info['cuda_available']}"
    )
    if not info.get("python_compatible", True):
        print(f"[DERICH Runtime] WARNING: Expected Python 3.11.x (target {info['target_python']}).")
    if info.get("cuda_version_mismatch"):
        print(
            f"[DERICH Runtime] INFO: PyTorch CUDA is {info.get('torch_cuda_version')} "
            f"(target {info['target_cuda']}). To switch to CUDA {info['target_cuda']}, run:\n"
            f"  {info['cuda_install_cmd']}"
        )
    telemetry_task = asyncio.create_task(ws_manager.telemetry_loop())
    yield
    telemetry_task.cancel()
    try:
        await telemetry_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="DERICH YOLO Auto-Training Studio",
    description="WebSocket-connected YOLOv8 / YOLO11 / YOLO12 / YOLO26 automatic training, auto-labeling & multi-format export platform.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


# ----------------------------------------------------------------------
# Request Schemas
# ----------------------------------------------------------------------


class CreateProjectReq(BaseModel):
    name: str
    classes: list[str] = ["cube", "cylinder", "sphere"]
    notes: str = ""


class UpdateClassesReq(BaseModel):
    classes: list[str]


class SaveLabelsReq(BaseModel):
    boxes: list[dict[str, Any]]


class BatchDeleteImagesReq(BaseModel):
    filenames: list[str]


class LocalFolderReq(BaseModel):
    folder_path: str


class WebcamFrameReq(BaseModel):
    data_url: str
    label_hint: str = "cam"


class DemoShapesReq(BaseModel):
    count: int = 45
    include_labels: bool = True
    img_size: int = 640


class PreprocessReq(BaseModel):
    remove_duplicates: bool = True
    remove_blurry: bool = False
    blur_threshold: float = 35.0
    max_side: int = 1280
    apply_clahe: bool = False
    clean_labels: bool = True


class AutoLabelReq(BaseModel):
    engine: str = "cv_smart"  # cv_smart | open_vocab | yolo_model
    overwrite_existing: bool = False
    conf: float = 0.25
    class_mode: str = "auto_geometry"  # auto_geometry | single_class
    target_class_id: int = 0
    model_name: str = "yolo26n.pt"
    sensitivity: float = 0.5


class StartTrainReq(BaseModel):
    project_id: str
    params: dict[str, Any] = {}
    auto_export_formats: list[str] = []
    export_precision: str = "fp32"


class ExportReq(BaseModel):
    weights_path: str
    formats: list[str]
    imgsz: int = 640
    precision: str = "fp32"
    dynamic: bool = False
    simplify: bool = True
    nms: bool | None = None
    opset: int | None = None
    data_yaml: str | None = None
    run_id: str | None = None


# ----------------------------------------------------------------------
# WebSocket Real-Time Hub
# ----------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        # Immediately send current telemetry & training snapshot
        stats = await asyncio.to_thread(collect_system_telemetry, ws_manager.torch_info)
        await ws_manager.send_personal(websocket, {"type": "telemetry", "data": stats})
        await ws_manager.send_personal(websocket, {"type": "train_status", "data": training_engine.get_state()})

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            action = msg.get("action")
            payload = msg.get("payload") or {}

            if action == "ping":
                await ws_manager.send_personal(websocket, {"type": "pong", "data": {"ok": True}})
            elif action == "request_state":
                await ws_manager.send_personal(websocket, {"type": "train_status", "data": training_engine.get_state()})
            elif action == "start_training":
                try:
                    state = training_engine.start_training(payload)
                    await ws_manager.broadcast("train_status", state)
                except Exception as exc:
                    await ws_manager.send_personal(websocket, {"type": "error", "data": {"message": str(exc)}})
            elif action == "stop_training":
                res = training_engine.request_stop()
                await ws_manager.send_personal(websocket, {"type": "info", "data": res})
            elif action == "run_autolabel":
                pid = payload.get("project_id")
                if pid:
                    asyncio.create_task(_bg_autolabel(pid, payload))
            elif action == "run_preprocess":
                pid = payload.get("project_id")
                if pid:
                    asyncio.create_task(_bg_preprocess(pid, payload))
            elif action == "run_export":
                asyncio.create_task(_bg_export(payload))
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


async def _bg_autolabel(project_id: str, opts: dict[str, Any]) -> None:
    try:
        res = await asyncio.to_thread(
            auto_labeler.run,
            project_id=project_id,
            engine=opts.get("engine", "cv_smart"),
            overwrite_existing=bool(opts.get("overwrite_existing", False)),
            conf=float(opts.get("conf", 0.25)),
            class_mode=opts.get("class_mode", "auto_geometry"),
            target_class_id=int(opts.get("target_class_id", 0)),
            model_name=opts.get("model_name", "yolo26n.pt"),
            sensitivity=float(opts.get("sensitivity", 0.5)),
        )
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id, "result": res})
    except Exception as exc:
        ws_manager.log_sync(f"❌ Lỗi Auto-Label: {exc}", level="error", source="autolabel")
        await ws_manager.broadcast("error", {"message": str(exc)})


async def _bg_preprocess(project_id: str, opts: dict[str, Any]) -> None:
    try:
        res = await asyncio.to_thread(
            dataset_manager.preprocess_project,
            project_id=project_id,
            remove_duplicates=bool(opts.get("remove_duplicates", True)),
            remove_blurry=bool(opts.get("remove_blurry", False)),
            blur_threshold=float(opts.get("blur_threshold", 35.0)),
            max_side=int(opts.get("max_side", 1280)),
            apply_clahe=bool(opts.get("apply_clahe", False)),
            clean_labels=bool(opts.get("clean_labels", True)),
        )
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id, "result": res})
    except Exception as exc:
        ws_manager.log_sync(f"❌ Lỗi Tiền xử lý: {exc}", level="error", source="dataset")
        await ws_manager.broadcast("error", {"message": str(exc)})


async def _bg_export(opts: dict[str, Any]) -> None:
    try:
        await asyncio.to_thread(
            model_exporter.export_batch,
            weights_path=opts["weights_path"],
            formats=opts.get("formats") or ["onnx"],
            imgsz=int(opts.get("imgsz", 640)),
            precision=opts.get("precision", "fp32"),
            dynamic=bool(opts.get("dynamic", False)),
            simplify=bool(opts.get("simplify", True)),
            nms=opts.get("nms"),
            opset=opts.get("opset"),
            data_yaml=opts.get("data_yaml"),
            run_id=opts.get("run_id"),
        )
    except Exception as exc:
        ws_manager.log_sync(f"❌ Lỗi Export: {exc}", level="error", source="export")
        await ws_manager.broadcast("error", {"message": str(exc)})


# ----------------------------------------------------------------------
# Metadata & System API
# ----------------------------------------------------------------------


@app.get("/api/bootstrap")
async def get_bootstrap() -> dict[str, Any]:
    """Return full static catalogs, system info, projects, training state, runs, and exports in one call."""
    telemetry = await asyncio.to_thread(collect_system_telemetry, ws_manager.torch_info)
    return {
        "telemetry": telemetry,
        "model_families": MODEL_FAMILY_INFO,
        "model_catalog": MODEL_CATALOG,
        "custom_weights": training_engine.list_custom_weights(),
        "platform_profiles": PLATFORM_PROFILES,
        "training_presets": TRAINING_PRESETS,
        "hyperparam_docs": HYPERPARAM_DOCS,
        "export_formats": EXPORT_FORMATS,
        "projects": dataset_manager.list_projects(),
        "training_state": training_engine.get_state(),
        "runs": training_engine.list_runs(),
        "artifacts": model_exporter.list_artifacts(),
    }


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {"status": "ok", "ws_clients": ws_manager.client_count}


# ----------------------------------------------------------------------
# Projects & Multi-Source Dataset Ingestion API
# ----------------------------------------------------------------------


@app.get("/api/projects")
async def list_projects() -> list[dict[str, Any]]:
    return dataset_manager.list_projects()


@app.post("/api/projects")
async def create_project(req: CreateProjectReq) -> dict[str, Any]:
    proj = dataset_manager.create_project(req.name, req.classes, req.notes)
    await ws_manager.broadcast("dataset_updated", {"project_id": proj["id"]})
    return proj


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str) -> dict[str, Any]:
    dataset_manager.delete_project(project_id)
    dataset_manager.ensure_default_project()
    await ws_manager.broadcast("dataset_updated", {"project_id": None})
    return {"deleted": True}


@app.put("/api/projects/{project_id}/classes")
async def update_classes(project_id: str, req: UpdateClassesReq) -> dict[str, Any]:
    try:
        proj = dataset_manager.update_classes(project_id, req.classes)
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
        return proj
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/images")
async def get_project_images(project_id: str) -> dict[str, Any]:
    try:
        return {
            "project": dataset_manager.get_project(project_id),
            "images": dataset_manager.list_images(project_id),
            "health": dataset_manager.analyze_health(project_id),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/images/{filename}")
async def serve_project_image(project_id: str, filename: str):
    try:
        path = dataset_manager.get_image_path(project_id, filename)
        return FileResponse(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/projects/{project_id}/images/{filename}")
async def delete_project_image(project_id: str, filename: str) -> dict[str, Any]:
    deleted = dataset_manager.delete_image(project_id, filename)
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
    return {"deleted": deleted, "project": dataset_manager.get_project(project_id)}


@app.post("/api/projects/{project_id}/images/batch-delete")
async def batch_delete_project_images(project_id: str, req: BatchDeleteImagesReq) -> dict[str, Any]:
    deleted = dataset_manager.delete_images(project_id, req.filenames)
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
    return {"deleted": deleted, "project": dataset_manager.get_project(project_id)}


@app.delete("/api/projects/{project_id}/images")
async def clear_project_images(project_id: str) -> dict[str, Any]:
    dataset_manager.clear_images(project_id)
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
    return {"cleared": True, "project": dataset_manager.get_project(project_id)}


@app.put("/api/projects/{project_id}/labels/{filename}")
async def save_image_labels(project_id: str, filename: str, req: SaveLabelsReq) -> dict[str, Any]:
    boxes = dataset_manager.save_labels(project_id, filename, req.boxes)
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id, "filename": filename})
    return {"boxes": boxes, "project": dataset_manager.get_project(project_id)}


@app.post("/api/projects/{project_id}/upload/images")
async def upload_images(project_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    pairs: list[tuple[str, bytes]] = []
    for f in files:
        pairs.append((f.filename or "image.jpg", await f.read()))
    res = await asyncio.to_thread(dataset_manager.ingest_image_files, project_id, pairs)
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
    return res


@app.post("/api/projects/{project_id}/upload/zip")
async def upload_zip(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    try:
        res = await asyncio.to_thread(dataset_manager.ingest_zip_archive, project_id, data)
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể nhập ZIP: {exc}") from exc


@app.post("/api/projects/{project_id}/upload/video")
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    fps_extract: float = Form(2.0),
    max_frames: int = Form(75),
    skip_blurry: bool = Form(True),
    blur_threshold: float = Form(40.0),
    dedup: bool = Form(True),
) -> dict[str, Any]:
    data = await file.read()
    try:
        res = await asyncio.to_thread(
            dataset_manager.ingest_video,
            project_id,
            data,
            file.filename or "video.mp4",
            fps_extract,
            max_frames,
            skip_blurry,
            blur_threshold,
            dedup,
        )
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/ingest/folder")
async def ingest_local_folder(project_id: str, req: LocalFolderReq) -> dict[str, Any]:
    try:
        res = await asyncio.to_thread(dataset_manager.ingest_local_folder, project_id, req.folder_path)
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/ingest/webcam")
async def ingest_webcam_frame(project_id: str, req: WebcamFrameReq) -> dict[str, Any]:
    try:
        res = await asyncio.to_thread(dataset_manager.ingest_webcam_frame, project_id, req.data_url, req.label_hint)
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/generate-demo")
async def generate_demo(project_id: str, req: DemoShapesReq) -> dict[str, Any]:
    res = await asyncio.to_thread(
        dataset_manager.generate_demo_shapes,
        project_id,
        req.count,
        req.include_labels,
        req.img_size,
    )
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
    return res


@app.post("/api/projects/{project_id}/preprocess")
async def preprocess_project(project_id: str, req: PreprocessReq) -> dict[str, Any]:
    res = await asyncio.to_thread(
        dataset_manager.preprocess_project,
        project_id,
        req.remove_duplicates,
        req.remove_blurry,
        req.blur_threshold,
        req.max_side,
        req.apply_clahe,
        req.clean_labels,
    )
    await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
    return res


@app.post("/api/projects/{project_id}/autolabel")
async def autolabel_project(project_id: str, req: AutoLabelReq) -> dict[str, Any]:
    try:
        res = await asyncio.to_thread(
            auto_labeler.run,
            project_id,
            req.engine,
            req.overwrite_existing,
            req.conf,
            req.class_mode,
            req.target_class_id,
            req.model_name,
            req.sensitivity,
        )
        await ws_manager.broadcast("dataset_updated", {"project_id": project_id})
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ----------------------------------------------------------------------
# Training, Runs & Live Test API
# ----------------------------------------------------------------------


@app.post("/api/weights/upload")
async def upload_custom_weight(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    return training_engine.save_custom_weight(file.filename or "custom.pt", data)


@app.post("/api/train/start")
async def start_training(req: StartTrainReq) -> dict[str, Any]:
    try:
        return training_engine.start_training(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train/stop")
async def stop_training() -> dict[str, Any]:
    return training_engine.request_stop()


@app.get("/api/train/status")
async def train_status() -> dict[str, Any]:
    return training_engine.get_state()


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    return training_engine.list_runs()


@app.get("/api/runs/{run_id}/files/{filename}")
async def serve_run_file(run_id: str, filename: str):
    safe_run = Path(run_id).name
    safe_file = Path(filename).name
    path = RUNS_DIR / safe_run / safe_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file biểu đồ.")
    return FileResponse(path)


@app.post("/api/predict")
async def predict_live(
    file: UploadFile = File(...),
    weights_path: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.5),
    imgsz: int = Form(640),
) -> dict[str, Any]:
    data = await file.read()
    try:
        return await asyncio.to_thread(
            training_engine.predict_image,
            weights_path,
            data,
            conf,
            iou,
            imgsz,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ----------------------------------------------------------------------
# Multi-Format Export API
# ----------------------------------------------------------------------


@app.post("/api/export")
async def export_model(req: ExportReq) -> dict[str, Any]:
    results = await asyncio.to_thread(
        model_exporter.export_batch,
        req.weights_path,
        req.formats,
        req.imgsz,
        req.precision,
        req.dynamic,
        req.simplify,
        req.nms,
        req.opset,
        req.data_yaml,
        req.run_id,
    )
    return {"results": results, "artifacts": model_exporter.list_artifacts()}


@app.get("/api/exports")
async def list_exports() -> list[dict[str, Any]]:
    return model_exporter.list_artifacts()


@app.get("/api/exports/download/{filename}")
async def download_export(filename: str):
    try:
        path = model_exporter.get_artifact_path(filename)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/exports/{filename}")
async def delete_export(filename: str) -> dict[str, Any]:
    model_exporter.delete_artifact(filename)
    return {"deleted": True}


# ----------------------------------------------------------------------
# Static Web UI
# ----------------------------------------------------------------------


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "Static UI not found. Build static/index.html."})
    return FileResponse(index_file)
