"""Automatic Ultralytics YOLO trainer (YOLOv8 / YOLO11 / YOLO12 / YOLO26 / Custom .pt) with live WebSocket telemetry."""

from __future__ import annotations

import base64
import json
import platform
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import (
    PRETRAINED_DIR,
    RUNS_DIR,
    TRAINING_PRESETS,
    build_cu118_install_command,
    configure_cuda_environment,
    ensure_directories,
)
from app.dataset_manager import dataset_manager
from app.ws_manager import ws_manager


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return round(f, 5)
    except Exception:
        return default


def _resolve_device(torch_mod: Any, req_device: str = "auto") -> tuple[Any, bool]:
    """Resolve training/inference device cleanly for CUDA 11.8 or CPU fallback."""
    cuda_ok = bool(torch_mod.cuda.is_available())
    req = str(req_device or "auto").strip().lower()
    if req == "cpu":
        return "cpu", False
    if req in {"auto", "cuda", "cuda:0"}:
        return (0 if cuda_ok else "cpu"), cuda_ok
    if req.startswith("cuda:"):
        idx = req.split(":", 1)[1]
        return (idx if cuda_ok else "cpu"), cuda_ok
    return (req if cuda_ok else "cpu"), cuda_ok


def _configure_tf32(torch_mod: Any, allow_tf32: bool) -> bool:
    """Safely configure TF32 on NVIDIA Ampere/Ada+ GPUs (compute capability >= 8.0) under CUDA 11.8."""
    if not torch_mod.cuda.is_available():
        return False
    try:
        major, _minor = torch_mod.cuda.get_device_capability(0)
        effective = bool(allow_tf32) and major >= 8
    except Exception:
        effective = bool(allow_tf32)
    torch_mod.backends.cuda.matmul.allow_tf32 = effective
    torch_mod.backends.cudnn.allow_tf32 = effective
    if hasattr(torch_mod, "set_float32_matmul_precision"):
        torch_mod.set_float32_matmul_precision("high" if effective else "highest")
    return effective


class TrainingEngine:
    """Manages background YOLO training, live WebSocket callbacks, run registry, and inference testing."""

    def __init__(self) -> None:
        ensure_directories()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._active_trainer: Any = None
        self.state: dict[str, Any] = {
            "status": "idle",  # idle | preparing | training | exporting | completed | stopped | error
            "run_id": None,
            "project_id": None,
            "model": None,
            "epoch": 0,
            "total_epochs": 0,
            "batch": 0,
            "total_batches": 0,
            "progress_pct": 0.0,
            "elapsed_s": 0.0,
            "eta_s": 0.0,
            "best_map50": 0.0,
            "best_map50_95": 0.0,
            "metrics_history": [],
            "dataset_summary": None,
            "best_weights": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }

    def is_running(self) -> bool:
        return self.state["status"] in {"preparing", "training", "exporting"}

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False))

    def list_custom_weights(self) -> list[dict[str, Any]]:
        ensure_directories()
        items: list[dict[str, Any]] = []
        for p in sorted(PRETRAINED_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            items.append(
                {
                    "id": str(p.resolve()),
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                }
            )
        return items

    def save_custom_weight(self, filename: str, data: bytes) -> dict[str, Any]:
        ensure_directories()
        safe = Path(filename).name
        if not safe.lower().endswith(".pt"):
            safe += ".pt"
        dest = PRETRAINED_DIR / safe
        dest.write_bytes(data)
        ws_manager.log_sync(f"Đã tải lên trọng số tùy chỉnh: {safe} ({round(len(data)/(1024*1024), 2)} MB).", source="trainer")
        return {"id": str(dest.resolve()), "name": safe, "size_mb": round(len(data) / (1024 * 1024), 2)}

    def list_runs(self) -> list[dict[str, Any]]:
        ensure_directories()
        runs: list[dict[str, Any]] = []
        for rdir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta_file = rdir / "run_meta.json"
            if rdir.is_dir() and meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    best_pt = rdir / "weights" / "best.pt"
                    last_pt = rdir / "weights" / "last.pt"
                    meta["has_best"] = best_pt.exists()
                    meta["best_path"] = str(best_pt) if best_pt.exists() else (str(last_pt) if last_pt.exists() else None)
                    plots = []
                    for pname in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg", "PR_curve.png", "F1_curve.png"):
                        if (rdir / pname).exists():
                            plots.append({"name": pname, "url": f"/api/runs/{rdir.name}/files/{pname}"})
                    meta["plots"] = plots
                    runs.append(meta)
                except Exception:
                    continue
        return runs

    def get_run(self, run_id: str) -> dict[str, Any]:
        for r in self.list_runs():
            if r.get("run_id") == run_id:
                return r
        raise FileNotFoundError(f"Không tìm thấy run '{run_id}'")

    def request_stop(self) -> dict[str, Any]:
        if not self.is_running():
            return {"stopped": False, "message": "Hiện không có tiến trình huấn luyện nào đang chạy."}
        self._stop_requested = True
        if self._active_trainer is not None:
            try:
                self._active_trainer.stop = True
            except Exception:
                pass
        ws_manager.log_sync("Đã nhận lệnh DỪNG AN TOÀN — hệ thống sẽ lưu best.pt ở cuối batch/epoch hiện tại...", level="warning", source="trainer")
        ws_manager.broadcast_sync("train_status", self.get_state())
        return {"stopped": True, "message": "Đang dừng huấn luyện an toàn và lưu trọng số tốt nhất..."}

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.is_running():
                raise RuntimeError("Một tiến trình huấn luyện đang chạy. Vui lòng đợi hoàn tất hoặc bấm Dừng.")

            project_id = str(payload.get("project_id") or "").strip()
            if not project_id:
                raise ValueError("Thiếu project_id.")
            project = dataset_manager.get_project(project_id)

            # Merge defaults from small_shapes_default preset
            base_params = dict(TRAINING_PRESETS["small_shapes_default"]["params"])
            user_params = payload.get("params") or {}
            base_params.update(user_params)

            auto_export_formats = list(payload.get("auto_export_formats") or [])
            export_precision = str(payload.get("export_precision") or "fp32")

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_stem = Path(str(base_params.get("model", "yolo26n.pt"))).stem
            run_id = f"{model_stem}_{project_id[:16]}_{stamp}"

            self._stop_requested = False
            self._active_trainer = None
            self.state = {
                "status": "preparing",
                "run_id": run_id,
                "project_id": project_id,
                "project_name": project.get("name"),
                "model": str(base_params.get("model", "yolo26n.pt")),
                "params": base_params,
                "auto_export_formats": auto_export_formats,
                "export_precision": export_precision,
                "epoch": 0,
                "total_epochs": int(base_params.get("epochs", 120)),
                "batch": 0,
                "total_batches": 0,
                "progress_pct": 0.0,
                "elapsed_s": 0.0,
                "eta_s": 0.0,
                "best_map50": 0.0,
                "best_map50_95": 0.0,
                "metrics_history": [],
                "dataset_summary": None,
                "best_weights": None,
                "error": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
            }

        ws_manager.broadcast_sync("train_status", self.get_state())
        self._thread = threading.Thread(
            target=self._run_worker,
            args=(run_id, project_id, base_params, auto_export_formats, export_precision),
            daemon=True,
        )
        self._thread.start()
        return self.get_state()

    def _save_run_meta(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_meta.json").write_text(
            json.dumps(self.get_state(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _run_worker(
        self,
        run_id: str,
        project_id: str,
        params: dict[str, Any],
        auto_export_formats: list[str],
        export_precision: str,
    ) -> None:
        configure_cuda_environment()
        import torch
        from ultralytics import YOLO

        run_dir = RUNS_DIR / run_id
        t_start = time.time()

        try:
            ws_manager.log_sync(
                f"🚀 Khởi tạo huấn luyện [{run_id}] với model {params.get('model')} | "
                f"Python {platform.python_version()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda or 'CPU'}...",
                source="trainer",
            )

            # 1. Stratified split & leak-free offline expansion
            prep_info = dataset_manager.prepare_yolo_dataset(
                project_id=project_id,
                val_split=float(params.get("val_split", 0.2)),
                offline_aug_multiplier=int(params.get("offline_aug_multiplier", 3)),
                seed=int(params.get("seed", 42)),
            )
            with self._lock:
                self.state["dataset_summary"] = prep_info
                self.state["status"] = "training"
            ws_manager.log_sync(
                f"📦 Dữ liệu sẵn sàng: Train gốc {prep_info['train_original']} ảnh → mở rộng thành "
                f"{prep_info['train_effective']} ảnh (x{prep_info['offline_multiplier']}) | Val giữ nguyên {prep_info['val_count']} ảnh | "
                f"Classes: {prep_info['classes']}.",
                source="trainer",
            )
            ws_manager.broadcast_sync("train_status", self.get_state())

            # 2. Hardware & TF32 setup (CUDA 11.8 + NVIDIA Ampere/Ada+ safe)
            req_device = str(params.get("device", "auto")).lower()
            device, cuda_ok = _resolve_device(torch, req_device)
            if req_device != "cpu" and not cuda_ok:
                ws_manager.log_sync(
                    "⚠️ PyTorch hiện chưa bật CUDA nên tự động chuyển sang CPU. "
                    f"Cài PyTorch CUDA 11.8 bằng lệnh: {build_cu118_install_command()}",
                    level="warning",
                    source="trainer",
                )

            allow_tf32 = bool(params.get("allow_tf32", True))
            if cuda_ok:
                tf32_enabled = _configure_tf32(torch, allow_tf32)
                cudnn_ver = torch.backends.cudnn.version()
                ws_manager.log_sync(
                    f"⚡ GPU CUDA {torch.version.cuda or '?'} kích hoạt ({torch.cuda.get_device_name(0)}, "
                    f"cuDNN {cudnn_ver}) | TF32={'BẬT' if tf32_enabled else 'TẮT'}.",
                    source="trainer",
                )
            else:
                ws_manager.log_sync("🖥️ Đang huấn luyện trên CPU (khuyên dùng imgsz=416–640 và Nano model).", source="trainer")

            # 3. Load YOLO model (YOLOv8 / YOLO11 / YOLO12 / YOLO26 / Custom .pt)
            model_spec = str(params.get("model", "yolo26n.pt")).strip()
            pretrained = bool(params.get("pretrained", True))
            try:
                model = YOLO(model_spec)
            except Exception as load_err:
                if not pretrained and model_spec.endswith(".pt"):
                    yaml_fallback = model_spec[:-3] + ".yaml"
                    ws_manager.log_sync(
                        f"Không tải được {model_spec} ({load_err}), chuyển sang kiến trúc rỗng {yaml_fallback}.",
                        level="warning",
                        source="trainer",
                    )
                    model = YOLO(yaml_fallback)
                else:
                    raise RuntimeError(
                        f"Không thể nạp trọng số '{model_spec}': {load_err}. "
                        "Hãy kiểm tra kết nối mạng (lần đầu Ultralytics tải file .pt ~5-20MB) hoặc tải lên file .pt cục bộ."
                    ) from load_err

            # 4. Register live WebSocket callbacks
            total_epochs = max(1, int(params.get("epochs", 120)))

            def on_train_start(trainer: Any) -> None:
                self._active_trainer = trainer
                ws_manager.log_sync("🔥 Bắt đầu vòng lặp tối ưu hóa trọng số YOLO...", source="trainer")

            def on_train_epoch_start(trainer: Any) -> None:
                if self._stop_requested:
                    trainer.stop = True
                with self._lock:
                    self.state["epoch"] = int(trainer.epoch) + 1
                    self.state["total_epochs"] = int(trainer.epochs)
                    self.state["batch"] = 0
                ws_manager.broadcast_sync("train_status", self.get_state())

            def on_train_batch_end(trainer: Any) -> None:
                if self._stop_requested:
                    trainer.stop = True
                try:
                    nb = len(trainer.train_loader)
                except Exception:
                    nb = 1
                with self._lock:
                    self.state["batch"] = min(nb, self.state["batch"] + 1)
                    self.state["total_batches"] = nb
                    ep_done = max(0, int(trainer.epoch))
                    frac = (ep_done + self.state["batch"] / max(1, nb)) / max(1, int(trainer.epochs))
                    self.state["progress_pct"] = round(min(99.5, frac * 100.0), 2)
                    elapsed = time.time() - t_start
                    self.state["elapsed_s"] = round(elapsed, 1)
                    if frac > 0.01:
                        self.state["eta_s"] = round(elapsed * (1.0 - frac) / frac, 1)

            def on_fit_epoch_end(trainer: Any) -> None:
                tot = max(1, int(trainer.epochs))
                ep = min(tot, int(trainer.epoch) + 1)
                metrics = getattr(trainer, "metrics", {}) or {}
                loss_items = getattr(trainer, "tloss", None)
                box_l, cls_l, dfl_l = 0.0, 0.0, 0.0
                if isinstance(loss_items, dict):
                    box_l = _safe_float(loss_items.get("box_loss", 0.0))
                    cls_l = _safe_float(loss_items.get("cls_loss", 0.0))
                    dfl_l = _safe_float(loss_items.get("dfl_loss", loss_items.get("l1_loss", 0.0)))
                elif loss_items is not None:
                    try:
                        vals = [float(x) for x in loss_items]
                        if len(vals) >= 1:
                            box_l = round(vals[0], 5)
                        if len(vals) >= 2:
                            cls_l = round(vals[1], 5)
                        if len(vals) >= 3:
                            dfl_l = round(vals[2], 5)
                    except Exception:
                        pass

                map50 = _safe_float(metrics.get("metrics/mAP50(B)", 0.0))
                map50_95 = _safe_float(metrics.get("metrics/mAP50-95(B)", 0.0))
                prec = _safe_float(metrics.get("metrics/precision(B)", 0.0))
                rec = _safe_float(metrics.get("metrics/recall(B)", 0.0))
                lr_val = 0.0
                try:
                    lr_val = _safe_float(trainer.optimizer.param_groups[0]["lr"], 0.0)
                except Exception:
                    pass

                point = {
                    "epoch": ep,
                    "box_loss": box_l,
                    "cls_loss": cls_l,
                    "dfl_loss": dfl_l,
                    "map50": map50,
                    "map50_95": map50_95,
                    "precision": prec,
                    "recall": rec,
                    "lr": lr_val,
                }
                elapsed = time.time() - t_start
                eta = (elapsed / max(1, ep)) * max(0, tot - ep)

                with self._lock:
                    self.state["epoch"] = ep
                    self.state["total_epochs"] = tot
                    self.state["progress_pct"] = round(min(99.5, (ep / max(1, tot)) * 100.0), 2)
                    self.state["elapsed_s"] = round(elapsed, 1)
                    self.state["eta_s"] = round(eta, 1)
                    self.state["best_map50"] = max(self.state["best_map50"], map50)
                    self.state["best_map50_95"] = max(self.state["best_map50_95"], map50_95)
                    hist = self.state["metrics_history"]
                    if hist and hist[-1]["epoch"] == ep:
                        # Final validation callback fires after last epoch; preserve train losses if cleared
                        if point["box_loss"] == 0.0 and hist[-1]["box_loss"] > 0.0:
                            point["box_loss"] = hist[-1]["box_loss"]
                            point["cls_loss"] = hist[-1]["cls_loss"]
                            point["dfl_loss"] = hist[-1]["dfl_loss"]
                        hist[-1] = point
                    else:
                        hist.append(point)

                ws_manager.broadcast_sync("train_metrics", {"point": point, "state": self.get_state()})
                ws_manager.log_sync(
                    f"Epoch {ep}/{tot} | box={box_l:.4f} cls={cls_l:.4f} | "
                    f"mAP50={map50*100:.1f}% mAP50-95={map50_95*100:.1f}% P={prec*100:.1f}% R={rec*100:.1f}% | ETA {int(eta)}s",
                    source="trainer",
                )
                self._save_run_meta(run_dir)

            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            # 5. Safe hyperparameter sanitization for small datasets
            eff_train = max(2, int(prep_info["train_effective"]))
            req_batch = int(params.get("batch", 8))
            if req_batch != -1:
                req_batch = max(2, min(req_batch, eff_train))
            elif device == "cpu":
                req_batch = min(8, eff_train)

            close_mosaic = int(params.get("close_mosaic", 20))
            if total_epochs <= close_mosaic:
                close_mosaic = max(0, total_epochs // 4)

            cache_opt = params.get("cache", "ram")
            if isinstance(cache_opt, str):
                cl = cache_opt.lower()
                if cl == "false":
                    cache_val: Any = False
                elif cl in {"ram", "disk"}:
                    cache_val = cl
                else:
                    cache_val = "ram"
            else:
                cache_val = bool(cache_opt)

            freeze_val = int(params.get("freeze", 10))
            train_kwargs: dict[str, Any] = {
                "data": prep_info["yaml_path"],
                "epochs": total_epochs,
                "patience": int(params.get("patience", 35)),
                "imgsz": int(params.get("imgsz", 640)),
                "batch": req_batch,
                "device": device,
                "workers": int(params.get("workers", 0)),
                "project": str(RUNS_DIR.resolve()),
                "name": run_id,
                "exist_ok": True,
                "pretrained": pretrained,
                "optimizer": str(params.get("optimizer", "AdamW")),
                "lr0": float(params.get("lr0", 0.001)),
                "lrf": float(params.get("lrf", 0.01)),
                "cos_lr": bool(params.get("cos_lr", True)),
                "weight_decay": float(params.get("weight_decay", 0.001)),
                "warmup_epochs": min(float(params.get("warmup_epochs", 5.0)), max(1.0, total_epochs * 0.25)),
                "mosaic": float(params.get("mosaic", 1.0)),
                "close_mosaic": close_mosaic,
                "mixup": float(params.get("mixup", 0.15)),
                "degrees": float(params.get("degrees", 15.0)),
                "translate": float(params.get("translate", 0.15)),
                "scale": float(params.get("scale", 0.5)),
                "shear": float(params.get("shear", 3.0)),
                "perspective": float(params.get("perspective", 0.0005)),
                "fliplr": float(params.get("fliplr", 0.5)),
                "flipud": float(params.get("flipud", 0.15)),
                "hsv_h": float(params.get("hsv_h", 0.015)),
                "hsv_s": float(params.get("hsv_s", 0.6)),
                "hsv_v": float(params.get("hsv_v", 0.45)),
                "erasing": float(params.get("erasing", 0.2)),
                "cache": cache_val,
                "amp": bool(params.get("amp", True)) if device != "cpu" else False,
                "seed": int(params.get("seed", 42)),
                "plots": True,
                "verbose": False,
            }
            # Pass cls_pw only when supported by the installed Ultralytics version
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT

                if "cls_pw" in DEFAULT_CFG_DICT:
                    train_kwargs["cls_pw"] = float(params.get("cls_pw", 0.5))
            except Exception:
                train_kwargs["cls_pw"] = float(params.get("cls_pw", 0.5))
            if freeze_val > 0:
                train_kwargs["freeze"] = freeze_val

            model.train(**train_kwargs)

            best_pt = run_dir / "weights" / "best.pt"
            last_pt = run_dir / "weights" / "last.pt"
            final_weight = str(best_pt) if best_pt.exists() else (str(last_pt) if last_pt.exists() else None)

            # 6. Auto-export if requested
            exported_artifacts: list[dict[str, Any]] = []
            if auto_export_formats and final_weight and not self._stop_requested:
                with self._lock:
                    self.state["status"] = "exporting"
                    self.state["best_weights"] = final_weight
                ws_manager.broadcast_sync("train_status", self.get_state())
                from app.exporter import model_exporter

                exported_artifacts = model_exporter.export_batch(
                    weights_path=final_weight,
                    formats=auto_export_formats,
                    imgsz=int(params.get("imgsz", 640)),
                    precision=export_precision,
                    data_yaml=prep_info["yaml_path"],
                    run_id=run_id,
                )

            with self._lock:
                self.state["status"] = "stopped" if self._stop_requested else "completed"
                self.state["progress_pct"] = 100.0 if not self._stop_requested else self.state["progress_pct"]
                self.state["elapsed_s"] = round(time.time() - t_start, 1)
                self.state["eta_s"] = 0.0
                self.state["best_weights"] = final_weight
                self.state["exported_artifacts"] = exported_artifacts
                self.state["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._save_run_meta(run_dir)
            ws_manager.log_sync(
                f"✅ Huấn luyện [{run_id}] {'đã dừng an toàn' if self._stop_requested else 'HOÀN TẤT'}! "
                f"mAP50 cao nhất: {self.state['best_map50']*100:.1f}% | Trọng số: {final_weight}",
                level="success",
                source="trainer",
            )
            ws_manager.broadcast_sync("train_done", self.get_state())

        except Exception as exc:
            err_msg = f"{exc}"
            tb = traceback.format_exc()
            with self._lock:
                self.state["status"] = "error"
                self.state["error"] = err_msg
                self.state["elapsed_s"] = round(time.time() - t_start, 1)
                self.state["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._save_run_meta(run_dir)
            ws_manager.log_sync(f"❌ Lỗi huấn luyện: {err_msg}", level="error", source="trainer")
            ws_manager.log_sync(tb[-700:], level="error", source="trainer")
            ws_manager.broadcast_sync("train_done", self.get_state())
        finally:
            self._active_trainer = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def predict_image(
        self,
        weights_path: str,
        image_bytes: bytes,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ) -> dict[str, Any]:
        """Run live inference test on an uploaded image or webcam frame and return annotated base64 JPEG."""
        configure_cuda_environment()
        import torch
        from ultralytics import YOLO

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể giải mã ảnh kiểm thử.")

        device, cuda_ok = _resolve_device(torch, "auto")
        if cuda_ok:
            _configure_tf32(torch, True)

        model = YOLO(weights_path)
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)

        detections: list[dict[str, Any]] = []
        annotated = img
        if results:
            res = results[0]
            annotated = res.plot()
            names = res.names or {}
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, sc, cid in zip(xyxy, confs, clss):
                    detections.append(
                        {
                            "class_id": int(cid),
                            "class_name": str(names.get(int(cid), str(cid))),
                            "confidence": round(float(sc), 4),
                            "xyxy": [round(float(v), 1) for v in box],
                        }
                    )

        ok, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Không thể mã hóa ảnh kết quả.")
        b64 = base64.b64encode(enc.tobytes()).decode("ascii")
        return {
            "inference_ms": ms,
            "count": len(detections),
            "detections": detections,
            "annotated_data_url": f"data:image/jpeg;base64,{b64}",
        }


training_engine = TrainingEngine()
