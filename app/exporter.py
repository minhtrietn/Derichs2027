"""Multi-format YOLO model exporter (PT, TF32 Bundle, ONNX, LiteRT/TFLite, NCNN, TensorRT, OpenVINO, SavedModel, PB, TorchScript, CoreML, MNN)."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import (
    EXPORT_FORMATS,
    EXPORTS_DIR,
    configure_cuda_environment,
    ensure_directories,
)
from app.ws_manager import ws_manager

FORMAT_MAP = {item["id"]: item for item in EXPORT_FORMATS}

INSTALL_HINTS: dict[str, str] = {
    "onnx": "pip install onnx onnxslim onnxruntime (hoặc onnxruntime-gpu==1.18.1 cho CUDA 11.8)",
    "litert": "pip install ai-edge-litert tensorflow onnx2tf sng4onnx onnx_graphsurgeon",
    "ncnn": "pip install ncnn pnnx",
    "openvino": "pip install openvino",
    "engine": "Cài PyTorch CUDA 11.8 (torch==2.7.1+cu118) + tensorrt tương thích CUDA 11.8 trên máy có GPU NVIDIA",
    "saved_model": "pip install tensorflow onnx2tf",
    "pb": "pip install tensorflow onnx2tf",
    "coreml": "pip install coremltools (khuyên chạy trên macOS)",
    "mnn": "pip install MNN",
}

TF32_INFER_SCRIPT = '''"""DERICH YOLO Studio — NVIDIA TensorFloat-32 (TF32) Accelerated Inference Script (Python 3.11.6 + CUDA 11.8)."""
import sys
import torch
from ultralytics import YOLO

# Enable NVIDIA TF32 Tensor Cores (RTX 30xx / 40xx / Ampere / Ada / Hopper)
device = "cpu"
if torch.cuda.is_available():
    device = 0
    major, _ = torch.cuda.get_device_capability(0)
    use_tf32 = major >= 8
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    print(f"[CUDA {torch.version.cuda}] GPU: {torch.cuda.get_device_name(0)} | TF32={'ON' if use_tf32 else 'OFF'}")
else:
    print("[CPU MODE] CUDA not available; falling back to FP32 CPU.")

model = YOLO("best.pt")
source = sys.argv[1] if len(sys.argv) > 1 else 0  # Image path, video, or webcam 0
with torch.inference_mode():
    results = model.predict(source=source, imgsz=640, conf=0.25, device=device, show=True)
print(f"Processed {len(results)} frame(s).")
'''


def _zip_directory(folder: Path, zip_dest: Path) -> Path:
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(folder.parent))
    return zip_dest


def _precision_to_quantize(fmt: str, precision: str) -> Any:
    """Translate UI precision selection ('fp32', 'tf32', 'fp16', 'int8') to Ultralytics 8.4 quantize arg."""
    p = (precision or "fp32").lower()
    if p == "int8":
        return 8
    if p == "fp16":
        return 16
    # fp32 or tf32 -> standard 32-bit export graph (TensorRT/PyTorch runs TF32 on FP32 tensors automatically on RTX GPUs)
    return None


class ModelExporter:
    """Exports trained weights into multiple deployment formats and manages downloadable artifacts."""

    def __init__(self) -> None:
        ensure_directories()

    def _registry_path(self) -> Path:
        return EXPORTS_DIR / "artifacts.json"

    def _load_registry(self) -> list[dict[str, Any]]:
        rp = self._registry_path()
        if not rp.exists():
            return []
        try:
            items = json.loads(rp.read_text(encoding="utf-8"))
            # Filter out deleted files, resolving cross-OS paths via EXPORTS_DIR / filename
            valid: list[dict[str, Any]] = []
            for it in items:
                fname = Path(str(it.get("filename", "")).replace("\\", "/")).name
                local_path = EXPORTS_DIR / fname if fname else None
                raw_path = Path(str(it.get("file_path", ""))) if it.get("file_path") else None
                if local_path and local_path.exists():
                    it["file_path"] = str(local_path)
                    valid.append(it)
                elif raw_path and raw_path.exists():
                    valid.append(it)
            return valid
        except Exception:
            return []

    def _save_registry(self, items: list[dict[str, Any]]) -> None:
        self._registry_path().write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_artifacts(self) -> list[dict[str, Any]]:
        items = self._load_registry()
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_artifact_path(self, filename: str) -> Path:
        safe = Path(filename).name
        path = EXPORTS_DIR / safe
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp xuất '{safe}'.")
        return path

    def delete_artifact(self, filename: str) -> None:
        safe = Path(filename).name
        p = EXPORTS_DIR / safe
        p.unlink(missing_ok=True)
        items = [it for it in self._load_registry() if it.get("filename") != safe]
        self._save_registry(items)

    def _register_artifact(self, record: dict[str, Any]) -> None:
        items = self._load_registry()
        items = [it for it in items if it.get("filename") != record["filename"]]
        items.insert(0, record)
        self._save_registry(items)

    def export_single(
        self,
        weights_path: str,
        fmt: str,
        imgsz: int = 640,
        precision: str = "fp32",
        dynamic: bool = False,
        simplify: bool = True,
        nms: bool | None = None,
        opset: int | None = None,
        data_yaml: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        configure_cuda_environment()
        import torch
        from ultralytics import YOLO

        wpath = Path(weights_path)
        if not wpath.exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")

        fmt_id = fmt.strip().lower()
        if fmt_id == "tflite":
            fmt_id = "litert"
        if fmt_id not in FORMAT_MAP:
            raise ValueError(f"Định dạng xuất chưa được hỗ trợ: {fmt}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_tag = f"{run_id or wpath.stem}_{fmt_id}_{precision}_{stamp}"
        ws_manager.log_sync(f"📤 Đang xuất định dạng [{FORMAT_MAP[fmt_id]['name']}] ({precision.upper()}, imgsz={imgsz})...", source="export")

        # 1. Native PyTorch .pt copy
        if fmt_id == "pt":
            dest = EXPORTS_DIR / f"{base_tag}.pt"
            shutil.copy2(wpath, dest)
            rec = self._build_record(dest, fmt_id, precision, imgsz, run_id, str(wpath))
            self._register_artifact(rec)
            return rec

        # 2. PyTorch TF32 Accelerated Package (.zip)
        if fmt_id == "tf32_bundle":
            dest = EXPORTS_DIR / f"{base_tag}.zip"
            model = YOLO(str(wpath))
            meta_json = json.dumps(
                {
                    "format": "pytorch_tf32",
                    "precision": "TF32 (TensorFloat-32)",
                    "target_runtime": "Python 3.11.6 + CUDA 11.8 (PyTorch 2.7.1+cu118)",
                    "imgsz": imgsz,
                    "names": model.names,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(wpath, arcname="best.pt")
                zf.writestr("infer_tf32.py", TF32_INFER_SCRIPT)
                zf.writestr("metadata.json", meta_json)
            rec = self._build_record(dest, fmt_id, "tf32", imgsz, run_id, str(wpath))
            self._register_artifact(rec)
            return rec

        cuda_ok = bool(torch.cuda.is_available())

        # 3. Hardware guard for TensorRT engine
        if fmt_id == "engine" and not cuda_ok:
            raise RuntimeError(
                "Xuất TensorRT (.engine) yêu cầu PyTorch CUDA 11.8 (torch==2.7.1+cu118) + GPU NVIDIA (hiện PyTorch đang ở chế độ CPU). "
                "Giải pháp ngay: hãy xuất ONNX (.onnx) rồi chuyển sang .engine trên máy/Jetson có TensorRT, "
                "hoặc cài PyTorch bản CUDA 11.8 cho GPU NVIDIA."
            )

        # Enable TF32 during export if requested on CUDA Ampere/Ada+
        if cuda_ok and precision.lower() == "tf32":
            try:
                major, _ = torch.cuda.get_device_capability(0)
                use_tf32 = major >= 8
            except Exception:
                use_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

        # Work on a temporary copy so Ultralytics export artifacts don't collide inside run folder
        work_dir = EXPORTS_DIR / f"_tmp_{base_tag}"
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_pt = work_dir / "model.pt"
        shutil.copy2(wpath, tmp_pt)

        try:
            model = YOLO(str(tmp_pt))
            ultralytics_fmt = FORMAT_MAP[fmt_id]["ultralytics_fmt"]
            export_kwargs: dict[str, Any] = {
                "format": ultralytics_fmt,
                "imgsz": int(imgsz),
            }

            # Detect Ultralytics 8.4 quantize API vs legacy half/int8 flags
            use_quantize_api = True
            fp16_supported_formats: set[str] | None = None
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT

                use_quantize_api = "quantize" in DEFAULT_CFG_DICT
                if use_quantize_api:
                    from ultralytics.engine.exporter import FP16_FORMATS

                    fp16_supported_formats = set(FP16_FORMATS)
            except Exception:
                pass

            q_val = _precision_to_quantize(fmt_id, precision)
            if q_val == 16:
                if fp16_supported_formats is not None and ultralytics_fmt not in fp16_supported_formats:
                    ws_manager.log_sync(
                        f"ℹ️ Định dạng {fmt_id.upper()} trên Ultralytics hiện tại dùng FP32 thay cho FP16.",
                        level="warning",
                        source="export",
                    )
                    precision = "fp32"
                    q_val = None
                elif fmt_id in {"onnx", "torchscript"} and not cuda_ok:
                    ws_manager.log_sync(
                        f"ℹ️ Chuyển {fmt_id.upper()} sang FP32 vì môi trường export hiện tại chạy trên CPU.",
                        level="warning",
                        source="export",
                    )
                    precision = "fp32"
                    q_val = None

            if q_val is not None:
                if use_quantize_api:
                    export_kwargs["quantize"] = q_val
                elif q_val == 16:
                    export_kwargs["half"] = True
                elif q_val == 8:
                    export_kwargs["int8"] = True

            # Pass GPU device=0 explicitly for CUDA-accelerated exports (Ultralytics defaults to CPU if omitted)
            if cuda_ok and (fmt_id == "engine" or (q_val == 16 and fmt_id in {"onnx", "torchscript"})):
                export_kwargs["device"] = 0

            if fmt_id in {"onnx", "openvino", "engine", "coreml", "mnn", "torchscript"} and dynamic:
                export_kwargs["dynamic"] = True
            if fmt_id in {"onnx", "engine", "mnn"}:
                export_kwargs["simplify"] = bool(simplify)
            if fmt_id == "onnx" and opset:
                export_kwargs["opset"] = int(opset)
            if nms is not None and fmt_id in {"onnx", "openvino", "engine", "coreml", "torchscript", "saved_model", "mnn"}:
                export_kwargs["nms"] = bool(nms)
            if q_val == 8 and data_yaml and Path(data_yaml).exists():
                export_kwargs["data"] = data_yaml

            exported_raw = Path(str(model.export(**export_kwargs)))
            if not exported_raw.exists():
                raise RuntimeError("Ultralytics không trả về tệp xuất hợp lệ.")

            if exported_raw.is_dir():
                dest = EXPORTS_DIR / f"{base_tag}.zip"
                _zip_directory(exported_raw, dest)
            else:
                dest = EXPORTS_DIR / f"{base_tag}{exported_raw.suffix}"
                shutil.copy2(exported_raw, dest)

            rec = self._build_record(dest, fmt_id, precision, imgsz, run_id, str(wpath))
            self._register_artifact(rec)
            ws_manager.log_sync(
                f"✅ Xuất thành công {dest.name} ({rec['size_mb']} MB).",
                level="success",
                source="export",
            )
            return rec
        except Exception as exc:
            hint = INSTALL_HINTS.get(fmt_id, "")
            msg = f"Xuất {fmt_id.upper()} thất bại: {exc}"
            if hint:
                msg += f" | Gợi ý cài thư viện: {hint}"
            ws_manager.log_sync(f"❌ {msg}", level="error", source="export")
            raise RuntimeError(msg) from exc
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            try:
                if cuda_ok:
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _build_record(
        self,
        dest: Path,
        fmt_id: str,
        precision: str,
        imgsz: int,
        run_id: str | None,
        source_weights: str,
    ) -> dict[str, Any]:
        info = FORMAT_MAP[fmt_id]
        return {
            "filename": dest.name,
            "file_path": str(dest.resolve()),
            "download_url": f"/api/exports/download/{dest.name}",
            "format": fmt_id,
            "format_name": info["name"],
            "precision": precision.upper(),
            "imgsz": imgsz,
            "platforms": info["platforms"],
            "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
            "run_id": run_id,
            "source_weights": source_weights,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "snippet": self.get_integration_snippet(fmt_id, dest.name, imgsz),
        }

    def export_batch(
        self,
        weights_path: str,
        formats: list[str],
        imgsz: int = 640,
        precision: str = "fp32",
        dynamic: bool = False,
        simplify: bool = True,
        nms: bool | None = None,
        opset: int | None = None,
        data_yaml: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        total = max(1, len(formats))
        for idx, fmt in enumerate(formats):
            ws_manager.broadcast_sync(
                "export_status",
                {
                    "progress": int((idx / total) * 100),
                    "current_format": fmt,
                    "message": f"Đang xuất ({idx + 1}/{total}): {fmt.upper()}...",
                },
            )
            try:
                rec = self.export_single(
                    weights_path=weights_path,
                    fmt=fmt,
                    imgsz=imgsz,
                    precision=precision,
                    dynamic=dynamic,
                    simplify=simplify,
                    nms=nms,
                    opset=opset,
                    data_yaml=data_yaml,
                    run_id=run_id,
                )
                results.append({"ok": True, "format": fmt, "artifact": rec})
            except Exception as exc:
                results.append({"ok": False, "format": fmt, "error": str(exc)})

        ws_manager.broadcast_sync(
            "export_done",
            {
                "progress": 100,
                "results": results,
                "artifacts": self.list_artifacts(),
            },
        )
        return results

    def get_integration_snippet(self, fmt_id: str, filename: str, imgsz: int = 640) -> str:
        if fmt_id == "onnx":
            return (
                f"# Windows (DirectML / CUDA / CPU) & Linux ONNX Runtime\n"
                f"from ultralytics import YOLO\n"
                f"model = YOLO('{filename}')\n"
                f"results = model.predict('test.jpg', imgsz={imgsz}, conf=0.25)\n"
                f"# Hoặc trực tiếp với onnxruntime (DirectML trên mọi GPU Windows):\n"
                f"# import onnxruntime as ort\n"
                f"# sess = ort.InferenceSession('{filename}', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])"
            )
        if fmt_id == "tf32_bundle":
            return (
                f"# Giải nén {filename} và chạy với NVIDIA TF32 Tensor Cores:\n"
                f"import torch\n"
                f"from ultralytics import YOLO\n"
                f"torch.backends.cuda.matmul.allow_tf32 = True\n"
                f"torch.backends.cudnn.allow_tf32 = True\n"
                f"model = YOLO('best.pt')\n"
                f"results = model.predict(0, imgsz={imgsz})"
            )
        if fmt_id == "litert":
            return (
                f"// Android (Kotlin + LiteRT / TensorFlow Lite)\n"
                f"// Đặt file {filename} vào app/src/main/assets/\n"
                f"val options = Interpreter.Options().apply {{\n"
                f"    setNumThreads(4)\n"
                f"    // addDelegate(GpuDelegate()) // Bật GPU Android\n"
                f"}}\n"
                f"val interpreter = Interpreter(loadModelFile(assets, \"{filename}\"), options)"
            )
        if fmt_id == "ncnn":
            return (
                f"# Raspberry Pi / Linux ARM / Android NCNN\n"
                f"# Giải nén {filename} thành thư mục *_ncnn_model\n"
                f"from ultralytics import YOLO\n"
                f"model = YOLO('model_ncnn_model')\n"
                f"results = model.predict('test.jpg', imgsz={imgsz})"
            )
        if fmt_id == "openvino":
            return (
                f"# Windows / Linux Intel CPU & iGPU (OpenVINO — nhanh 2-4x)\n"
                f"from ultralytics import YOLO\n"
                f"model = YOLO('model_openvino_model')\n"
                f"results = model.predict('test.jpg', imgsz={imgsz})"
            )
        if fmt_id == "engine":
            return (
                f"# NVIDIA GPU / Jetson TensorRT Engine\n"
                f"from ultralytics import YOLO\n"
                f"model = YOLO('{filename}')\n"
                f"results = model.predict(0, imgsz={imgsz}, half=True)"
            )
        return (
            f"from ultralytics import YOLO\n"
            f"model = YOLO('{filename}')\n"
            f"results = model.predict('test.jpg', imgsz={imgsz}, conf=0.25)"
        )


model_exporter = ModelExporter()
