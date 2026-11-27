"""Configuration, presets, platform advisor, and hyperparameter impact catalog."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_ROOT = BASE_DIR / "studio_data"
PROJECTS_DIR = DATA_ROOT / "projects"
RUNS_DIR = DATA_ROOT / "runs"
EXPORTS_DIR = DATA_ROOT / "exports"
PRETRAINED_DIR = DATA_ROOT / "pretrained"

# Target runtime profile
TARGET_PYTHON_VERSION = "3.11.6"
TARGET_CUDA_VERSION = "11.8"
PYTORCH_CU118_PACKAGES = "torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
PYTORCH_CU118_INDEX_URL = "https://download.pytorch.org/whl/cu118"

_DLL_DIR_HANDLES: list[Any] = []
_CUDA_ENV_INFO: dict[str, Any] | None = None


def get_os_type() -> str:
    """Return normalized OS identifier: 'windows', 'linux', or 'macos'."""
    sys_name = platform.system().lower()
    if sys_name.startswith("win") or os.name == "nt":
        return "windows"
    if sys_name == "darwin":
        return "macos"
    return "linux"


def build_setup_command() -> str:
    """Return the unified cross-platform setup command for the current OS."""
    if get_os_type() == "windows":
        return "python run.py --setup"
    return "python3 run.py --setup"


def build_cu118_install_command(python_executable: str | None = None) -> str:
    """Return the exact pip install command for PyTorch CUDA 11.8 on Python 3.11.6."""
    py = python_executable or sys.executable or ("python" if os.name == "nt" else "python3")
    py_quoted = f'"{py}"' if " " in py else py
    return (
        f"{py_quoted} -m pip install --upgrade --force-reinstall "
        f"{PYTORCH_CU118_PACKAGES} --index-url {PYTORCH_CU118_INDEX_URL}"
    )


def _detect_cuda_toolkit() -> tuple[Path | None, str | None]:
    """Locate local CUDA Toolkit (preferring CUDA 11.8) and infer its version."""
    candidates: list[Path] = []
    if os.name == "nt":
        for env_key in ("CUDA_PATH_V11_8", "CUDA_PATH"):
            val = os.environ.get(env_key)
            if val:
                candidates.append(Path(val))
        win_cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        candidates.append(win_cuda_base / "v11.8")
        if win_cuda_base.is_dir():
            for child in sorted(win_cuda_base.iterdir(), reverse=True):
                if child.is_dir():
                    candidates.append(child)
    else:
        candidates.append(Path("/usr/local/cuda-11.8"))
        for env_key in ("CUDA_HOME", "CUDA_PATH"):
            val = os.environ.get(env_key)
            if val:
                candidates.append(Path(val))
        candidates.append(Path("/usr/local/cuda"))
        usr_local = Path("/usr/local")
        if usr_local.is_dir():
            for child in sorted(usr_local.glob("cuda-*"), reverse=True):
                if child.is_dir():
                    candidates.append(child)

    seen: set[str] = set()
    for root in candidates:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        if not root.exists():
            continue

        version: str | None = None
        version_json = root / "version.json"
        if version_json.exists():
            try:
                data = json.loads(version_json.read_text(encoding="utf-8", errors="ignore"))
                raw_ver = (
                    (data.get("cuda_cudart") or {}).get("version")
                    or (data.get("cuda") or {}).get("version")
                    or ""
                )
                m = re.match(r"(\d+\.\d+)", str(raw_ver))
                if m:
                    version = m.group(1)
            except Exception:
                pass
        if version is None:
            m = re.search(r"v?(\d+\.\d+)", root.name)
            if m:
                version = m.group(1)
        return root, version
    return None, None


def configure_cuda_environment() -> dict[str, Any]:
    """Prepare Windows/Linux DLL paths and allocator settings for Python 3.11.6 + CUDA 11.8."""
    global _CUDA_ENV_INFO
    if _CUDA_ENV_INFO is not None:
        return dict(_CUDA_ENV_INFO)

    cuda_root, cuda_ver = _detect_cuda_toolkit()
    added_dirs: list[str] = []

    if cuda_root is not None:
        os.environ.setdefault("CUDA_PATH", str(cuda_root))
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        if os.name == "nt":
            dll_subdirs = (
                cuda_root / "bin",
                cuda_root / "libnvvp",
                cuda_root / "extras" / "CUPTI" / "lib64",
            )
            cur_path = os.environ.get("PATH", "")
            path_entries = cur_path.split(os.pathsep) if cur_path else []
            path_lower = {p.lower() for p in path_entries}

            for d in dll_subdirs:
                if not d.is_dir():
                    continue
                d_str = str(d)
                if d_str.lower() not in path_lower:
                    path_entries.insert(0, d_str)
                    path_lower.add(d_str.lower())
                if hasattr(os, "add_dll_directory"):
                    try:
                        _DLL_DIR_HANDLES.append(os.add_dll_directory(d_str))
                        added_dirs.append(d_str)
                    except OSError:
                        pass
            os.environ["PATH"] = os.pathsep.join(path_entries)
        else:
            # Linux / POSIX: prepend CUDA bin to PATH and lib64 to LD_LIBRARY_PATH
            bin_dir = cuda_root / "bin"
            if bin_dir.is_dir():
                cur_path = os.environ.get("PATH", "")
                path_entries = cur_path.split(os.pathsep) if cur_path else []
                if str(bin_dir) not in path_entries:
                    path_entries.insert(0, str(bin_dir))
                    os.environ["PATH"] = os.pathsep.join(path_entries)

            lib_subdirs = (
                cuda_root / "lib64",
                cuda_root / "extras" / "CUPTI" / "lib64",
            )
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            ld_entries = [p for p in cur_ld.split(os.pathsep) if p] if cur_ld else []
            for d in lib_subdirs:
                if d.is_dir():
                    d_str = str(d)
                    if d_str not in ld_entries:
                        ld_entries.insert(0, d_str)
                    added_dirs.append(d_str)
            if ld_entries:
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_entries)

    # CUDA 11.8 compatible allocator setting (works cleanly on both Windows and Linux)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    _CUDA_ENV_INFO = {
        "os_type": get_os_type(),
        "cuda_toolkit_path": str(cuda_root) if cuda_root else None,
        "cuda_toolkit_version": cuda_ver,
        "dll_dirs_registered": added_dirs,
    }
    return dict(_CUDA_ENV_INFO)


def ensure_directories() -> None:
    """Create workspace data directories, configure CUDA 11.8 DLL paths, and set safe Ultralytics config path."""
    configure_cuda_environment()
    for d in (DATA_ROOT, PROJECTS_DIR, RUNS_DIR, EXPORTS_DIR, PRETRAINED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if "YOLO_CONFIG_DIR" not in os.environ:
        candidate = DATA_ROOT / ".ultralytics"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(candidate)
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "derich_ultralytics_cfg"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(fallback)


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Vietnamese <-> English prompt helpers for open-vocabulary auto-labeling
VI_EN_LABEL_HINTS: dict[str, str] = {
    "khối lập phương": "cube",
    "khoi lap phuong": "cube",
    "hình lập phương": "cube",
    "lập phương": "cube",
    "khối vuông": "cube",
    "hình hộp": "box",
    "hình trụ": "cylinder",
    "hinh tru": "cylinder",
    "hình trụ tròn": "cylinder",
    "hinh tru tron": "cylinder",
    "khối trụ": "cylinder",
    "ống trụ": "cylinder",
    "hình cầu": "sphere",
    "hinh cau": "sphere",
    "quả cầu": "sphere",
    "khối cầu": "sphere",
    "quả bóng": "ball",
    "hình nón": "cone",
    "hinh non": "cone",
    "khối nón": "cone",
    "hình chóp": "pyramid",
    "hinh chop": "pyramid",
    "kim tự tháp": "pyramid",
    "lăng trụ": "prism",
    "hình chữ nhật": "rectangular prism",
    "khối hộp": "rectangular box",
    "chai": "bottle",
    "cốc": "cup",
    "lon": "can",
    "người": "person",
    "xe hơi": "car",
    "xe máy": "motorcycle",
}

MODEL_FAMILY_INFO: dict[str, dict[str, Any]] = {
    "yolo26": {
        "name": "YOLO26 (Flagship 2026 — Mới nhất)",
        "badge": "Khuyên dùng",
        "release": "Ultralytics 2026",
        "highlights": [
            "Kiến trúc NMS-Free End-to-End: không cần bước hậu xử lý NMS, độ trễ ổn định.",
            "Loại bỏ Distribution Focal Loss (DFL): xuất ONNX / NCNN / LiteRT / TensorRT mượt và nhẹ hơn.",
            "Nhanh hơn tới ~43% trên CPU/ARM so với YOLO11n — cực mạnh cho Android & Raspberry Pi.",
            "Hỗ trợ MuSGD & ProgLoss + STAL giúp nhận diện vật thể nhỏ và dataset ít ảnh tốt hơn.",
        ],
        "note": "Thế hệ mới nhất của Ultralytics (nếu bạn từng nghe 'YOLO23' thường là nhầm số với YOLO26 hoặc YOLOv8 ra mắt năm 2023; bạn vẫn có thể upload file .pt tùy chỉnh bất kỳ ở mục Custom).",
    },
    "yolo11": {
        "name": "YOLO11 (Ổn định Production)",
        "badge": "Ổn định cao",
        "release": "Ultralytics 2024–2025",
        "highlights": [
            "Khối C3k2 và cơ chế chú ý C2PSA giúp ít tham số hơn ~22% so với YOLOv8 mà mAP cao hơn.",
            "Rất ổn định trên tập dữ liệu ít ảnh khi bật transfer learning + freeze backbone.",
            "Hệ sinh thái triển khai rộng rãi trên Android, Linux Jetson, Windows OpenVINO/ONNX.",
        ],
        "note": "Lựa chọn cân bằng nhất nếu cần tương thích với code triển khai đã có sẵn cho YOLO11.",
    },
    "yolov8": {
        "name": "YOLOv8 (Nền tảng phổ biến nhất)",
        "badge": "Tương thích tối đa",
        "release": "Ultralytics 2023",
        "highlights": [
            "Kiến trúc anchor-free kinh điển với khối C2f, dễ tinh chỉnh và rất dễ deploy trên mọi thư viện cũ/mới.",
            "Có hàng nghìn mẫu code tích hợp sẵn cho Android Java/Kotlin, C# WinForms, C++ TensorRT, ESP/Edge.",
            "Hội tụ nhanh và rất ổn định với các hình khối cơ bản (lập phương, hình trụ, hình cầu).",
        ],
        "note": "Phù hợp nhất khi dự án mục tiêu dùng SDK/app Android hoặc C++ viết sẵn theo chuẩn output YOLOv8.",
    },
    "yolo12": {
        "name": "YOLO12 (Attention-Centric)",
        "badge": "Chuyên cảnh phức tạp",
        "release": "Ultralytics 2025",
        "highlights": [
            "Tích hợp Area Attention cho khả năng tách nền phức tạp và vật thể chồng lấn tốt.",
            "Phù hợp khi chạy trên GPU CUDA (RTX / Jetson), tốc độ CPU sẽ chậm hơn YOLO26.",
        ],
        "note": "Ưu tiên dùng khi có GPU rời và khung cảnh nhiều nhiễu.",
    },
}

MODEL_CATALOG: list[dict[str, Any]] = [
    # YOLO26
    {"id": "yolo26n.pt", "family": "yolo26", "size": "n", "label": "YOLO26n (Nano)", "params_m": 2.4, "pt_mb": 5.3, "overfit_risk": "Thấp (Tốt cho <100 ảnh)", "speed": "Siêu nhanh", "best_for": ["Android", "Linux ARM / RPi", "Windows CPU", "Data <100 ảnh"]},
    {"id": "yolo26s.pt", "family": "yolo26", "size": "s", "label": "YOLO26s (Small)", "params_m": 9.5, "pt_mb": 19.5, "overfit_risk": "Thấp–Vừa", "speed": "Rất nhanh", "best_for": ["Windows GPU/CPU", "Linux Jetson", "Android Flagship", "Data 50–300 ảnh"]},
    {"id": "yolo26m.pt", "family": "yolo26", "size": "m", "label": "YOLO26m (Medium)", "params_m": 20.4, "pt_mb": 42.0, "overfit_risk": "Trung bình", "speed": "Nhanh (GPU)", "best_for": ["Windows RTX GPU", "Linux Server GPU"]},
    {"id": "yolo26l.pt", "family": "yolo26", "size": "l", "label": "YOLO26l (Large)", "params_m": 24.8, "pt_mb": 51.0, "overfit_risk": "Cao nếu <100 ảnh", "speed": "Trung bình", "best_for": ["Linux GPU Server", "Dataset lớn"]},
    {"id": "yolo26x.pt", "family": "yolo26", "size": "x", "label": "YOLO26x (XLarge)", "params_m": 55.7, "pt_mb": 113.0, "overfit_risk": "Rất cao nếu <100 ảnh", "speed": "Chậm", "best_for": ["Cloud GPU", "Độ chính xác tối đa"]},
    # YOLO11
    {"id": "yolo11n.pt", "family": "yolo11", "size": "n", "label": "YOLO11n (Nano)", "params_m": 2.6, "pt_mb": 5.4, "overfit_risk": "Thấp (Tốt cho <100 ảnh)", "speed": "Siêu nhanh", "best_for": ["Android", "Windows CPU", "Raspberry Pi", "Data <100 ảnh"]},
    {"id": "yolo11s.pt", "family": "yolo11", "size": "s", "label": "YOLO11s (Small)", "params_m": 9.4, "pt_mb": 18.4, "overfit_risk": "Thấp–Vừa", "speed": "Rất nhanh", "best_for": ["Windows GPU", "Linux Jetson", "Android Flagship"]},
    {"id": "yolo11m.pt", "family": "yolo11", "size": "m", "label": "YOLO11m (Medium)", "params_m": 20.1, "pt_mb": 38.8, "overfit_risk": "Trung bình", "speed": "Nhanh (GPU)", "best_for": ["Windows RTX GPU", "Linux GPU"]},
    {"id": "yolo11l.pt", "family": "yolo11", "size": "l", "label": "YOLO11l (Large)", "params_m": 25.3, "pt_mb": 49.0, "overfit_risk": "Cao nếu <100 ảnh", "speed": "Trung bình", "best_for": ["Linux GPU Server"]},
    {"id": "yolo11x.pt", "family": "yolo11", "size": "x", "label": "YOLO11x (XLarge)", "params_m": 56.9, "pt_mb": 109.0, "overfit_risk": "Rất cao nếu <100 ảnh", "speed": "Chậm", "best_for": ["Cloud GPU"]},
    # YOLOv8
    {"id": "yolov8n.pt", "family": "yolov8", "size": "n", "label": "YOLOv8n (Nano)", "params_m": 3.2, "pt_mb": 6.2, "overfit_risk": "Thấp (Tốt cho <100 ảnh)", "speed": "Siêu nhanh", "best_for": ["Android SDK cũ/mới", "Windows CPU", "Data <100 ảnh"]},
    {"id": "yolov8s.pt", "family": "yolov8", "size": "s", "label": "YOLOv8s (Small)", "params_m": 11.2, "pt_mb": 21.5, "overfit_risk": "Thấp–Vừa", "speed": "Rất nhanh", "best_for": ["Windows Desktop", "Linux Jetson"]},
    {"id": "yolov8m.pt", "family": "yolov8", "size": "m", "label": "YOLOv8m (Medium)", "params_m": 25.9, "pt_mb": 49.7, "overfit_risk": "Trung bình", "speed": "Nhanh (GPU)", "best_for": ["Windows RTX GPU", "Linux GPU"]},
    {"id": "yolov8l.pt", "family": "yolov8", "size": "l", "label": "YOLOv8l (Large)", "params_m": 43.7, "pt_mb": 83.7, "overfit_risk": "Cao nếu <100 ảnh", "speed": "Trung bình", "best_for": ["Linux GPU Server"]},
    {"id": "yolov8x.pt", "family": "yolov8", "size": "x", "label": "YOLOv8x (XLarge)", "params_m": 68.2, "pt_mb": 130.5, "overfit_risk": "Rất cao nếu <100 ảnh", "speed": "Chậm", "best_for": ["Cloud GPU"]},
    # YOLO12
    {"id": "yolo12n.pt", "family": "yolo12", "size": "n", "label": "YOLO12n (Nano)", "params_m": 2.6, "pt_mb": 5.5, "overfit_risk": "Thấp", "speed": "Nhanh", "best_for": ["GPU Edge", "Cảnh nhiều nhiễu"]},
    {"id": "yolo12s.pt", "family": "yolo12", "size": "s", "label": "YOLO12s (Small)", "params_m": 9.3, "pt_mb": 18.6, "overfit_risk": "Trung bình", "speed": "Nhanh (GPU)", "best_for": ["Windows/Linux GPU"]},
]

PLATFORM_PROFILES: list[dict[str, Any]] = [
    {
        "id": "android_mobile",
        "os": "Android",
        "title": "Android (Điện thoại / Tablet phổ thông — ARM64)",
        "icon": "smartphone",
        "primary_model": "yolo26n.pt",
        "alternative_models": ["yolo11n.pt", "yolov8n.pt"],
        "recommended_imgsz": 416,
        "recommended_formats": ["ncnn", "litert", "onnx"],
        "recommended_precision": "fp16",
        "expected_fps": "35 – 75 FPS (ARM64 CPU / GPU Delegate)",
        "why": "YOLO26n loại bỏ DFL và chạy NMS-Free nên cực nhẹ (~5MB), tránh nghẽn cổ chai hậu xử lý trên chip Snapdragon/MediaTek. Nếu app Android đã tích hợp sẵn code YOLOv8 TFLite thì chọn YOLOv8n.",
        "format_guide": "Ưu tiên NCNN (nhanh nhất qua C++/JNI trên mọi chip ARM), hoặc LiteRT/TFLite (.tflite FP16/INT8) cho Kotlin/Java Android Studio, hoặc ONNX Runtime Mobile.",
        "tips": [
            "Dùng imgsz=320 hoặc 416 để đạt >45 FPS trên điện thoại tầm trung.",
            "Nếu nhận diện hình khối lớn gần camera, INT8 LiteRT giảm dung lượng còn ~3MB nhưng nhớ cung cấp data hiệu chuẩn (calibration).",
            "YOLO26n xuất không cần viết code NMS thủ công phức tạp trên Android.",
        ],
    },
    {
        "id": "android_flagship",
        "os": "Android",
        "title": "Android Flagship / Thiết bị AI công nghiệp (Snapdragon 8 / NPU)",
        "icon": "cpu",
        "primary_model": "yolo26s.pt",
        "alternative_models": ["yolo11s.pt", "yolov8s.pt"],
        "recommended_imgsz": 640,
        "recommended_formats": ["litert", "ncnn", "onnx"],
        "recommended_precision": "fp16",
        "expected_fps": "30 – 60 FPS (GPU/NNAPI Delegate)",
        "why": "Chip mạnh cho phép chạy bản Small (s) ở độ phân giải 640px để nhận diện góc cạnh hình khối ở khoảng cách xa hoặc ánh sáng yếu sắc nét hơn.",
        "format_guide": "Xuất LiteRT (.tflite FP16) bật GPU Delegate hoặc NCNN Vulkan.",
        "tips": [
            "Với tập <100 ảnh, vẫn nên giữ freeze=10 và weight_decay=0.001 khi dùng bản Small.",
        ],
    },
    {
        "id": "linux_edge_arm",
        "os": "Linux",
        "title": "Linux Nhúng ARM (Raspberry Pi 4/5, Orange Pi, RK3588)",
        "icon": "terminal",
        "primary_model": "yolo26n.pt",
        "alternative_models": ["yolo11n.pt", "yolov8n.pt"],
        "recommended_imgsz": 416,
        "recommended_formats": ["ncnn", "onnx"],
        "recommended_precision": "fp32",
        "expected_fps": "18 – 40 FPS trên RPi 5 CPU (NCNN)",
        "why": "YOLO26n được tối ưu đặc biệt cho CPU edge (nhanh hơn YOLO11n tới ~43% trên CPU). Rất hợp cho robot tự hành, băng chuyền phân loại hình khối chạy Raspberry Pi.",
        "format_guide": "NCNN cho tốc độ CPU ARM cao nhất; ONNX Runtime dễ cài đặt nhanh bằng Python (`pip install onnxruntime`).",
        "tips": [
            "Nếu dùng board Rockchip RK3588 (Orange Pi 5), có thể chuyển đổi từ ONNX sang RKNN để chạy trên NPU 6 TOPS.",
        ],
    },
    {
        "id": "linux_jetson",
        "os": "Linux",
        "title": "Linux NVIDIA Jetson (Orin Nano / Xavier / TX2)",
        "icon": "zap",
        "primary_model": "yolo26s.pt",
        "alternative_models": ["yolo11s.pt", "yolov8s.pt"],
        "recommended_imgsz": 640,
        "recommended_formats": ["onnx", "engine"],
        "recommended_precision": "fp16",
        "expected_fps": "60 – 150 FPS (TensorRT FP16)",
        "why": "Jetson có Tensor Cores hỗ trợ TensorRT tăng tốc 3–5 lần. Bản Small cho độ chính xác rất cao mà vẫn chạy siêu mượt realtime.",
        "format_guide": "Xuất .onnx từ máy train Windows rồi build .engine TensorRT trực tiếp trên Jetson (`trtexec` hoặc `yolo export format=engine`) vì .engine gắn chặt với kiến trúc GPU mục tiêu.",
        "tips": [
            "Lưu ý: File .engine build trên PC RTX 4060 không chạy trực tiếp trên Jetson — hãy xuất .onnx sang Jetson rồi convert sang .engine trên chính Jetson.",
        ],
    },
    {
        "id": "windows_gpu",
        "os": "Windows",
        "title": "Windows PC / Laptop có GPU NVIDIA (VD: RTX 3060 / 4060)",
        "icon": "monitor",
        "primary_model": "yolo26s.pt",
        "alternative_models": ["yolo26n.pt", "yolo11s.pt", "yolov8s.pt"],
        "recommended_imgsz": 640,
        "recommended_formats": ["pt", "tf32_bundle", "onnx", "engine"],
        "recommended_precision": "tf32",
        "expected_fps": "120 – 300+ FPS (TensorRT / PyTorch TF32)",
        "why": "GPU RTX dòng 30/40 (Ampere/Ada) hỗ trợ chuẩn TF32 Tensor Cores giúp tính toán gần chính xác tuyệt đối như FP32 nhưng nhanh gấp ~2–3x. Nếu dữ liệu <60 ảnh nên chọn bản Nano (`yolo26n.pt`) để tránh overfit.",
        "format_guide": "PyTorch (.pt) + TF32 cho ứng dụng Python; ONNX (.onnx) cho C# / C++ / DirectML; TensorRT (.engine) cho tốc độ tối đa trên cùng GPU NVIDIA.",
        "tips": [
            "Bật tùy chọn TF32 trong cấu hình train để tối ưu Tensor Cores trên RTX 4060.",
            "Nếu phân phối phần mềm Windows cho nhiều máy khác nhau (có máy NVIDIA, có máy AMD/Intel), hãy xuất ONNX chạy với DirectML.",
        ],
    },
    {
        "id": "windows_cpu",
        "os": "Windows",
        "title": "Windows PC / Laptop chỉ dùng CPU (Intel / AMD không card rời)",
        "icon": "laptop",
        "primary_model": "yolo26n.pt",
        "alternative_models": ["yolo11n.pt", "yolov8n.pt"],
        "recommended_imgsz": 416,
        "recommended_formats": ["openvino", "onnx"],
        "recommended_precision": "fp16",
        "expected_fps": "25 – 60 FPS (OpenVINO trên Intel Core i5/i7)",
        "why": "Máy văn phòng/máy công nghiệp Windows không có GPU rời cần model nhẹ nhất. YOLO26n kết hợp OpenVINO chạy mượt thời gian thực mà không cần cài CUDA.",
        "format_guide": "Máy chip Intel → OpenVINO (nhanh nhất, gấp 2–4x PyTorch CPU). Máy chip AMD hoặc cần chạy mọi loại máy → ONNX Runtime (DirectML / CPU).",
        "tips": [
            "Giảm imgsz từ 640 xuống 416 giúp tăng tốc gần gấp đôi trên CPU trong khi các vật thể lớn như khối lập phương/hình cầu hầu như không giảm độ chính xác.",
        ],
    },
    {
        "id": "linux_server",
        "os": "Linux",
        "title": "Linux Server / Cloud AI (x86_64 + GPU NVIDIA)",
        "icon": "server",
        "primary_model": "yolo26m.pt",
        "alternative_models": ["yolo11m.pt", "yolov8m.pt"],
        "recommended_imgsz": 640,
        "recommended_formats": ["onnx", "engine", "pt", "torchscript"],
        "recommended_precision": "tf32",
        "expected_fps": "150 – 400 FPS (Batch inference)",
        "why": "Server GPU mạnh phù hợp xử lý nhiều luồng camera đồng thời với độ chính xác cao.",
        "format_guide": "TensorRT (.engine FP16/TF32) hoặc ONNX + Triton Inference Server, TorchScript cho C++ backend.",
        "tips": [
            "Chỉ nâng lên bản Medium/Large khi tập ảnh đủ đa dạng (>150 ảnh); nếu <100 ảnh vẫn nên giữ Nano/Small để tránh học vẹt.",
        ],
    },
    {
        "id": "ios_macos",
        "os": "iOS / macOS",
        "title": "iPhone / iPad / Mac Apple Silicon (M1–M4)",
        "icon": "tablet",
        "primary_model": "yolo26n.pt",
        "alternative_models": ["yolo11n.pt", "yolo26s.pt"],
        "recommended_imgsz": 640,
        "recommended_formats": ["coreml", "onnx"],
        "recommended_precision": "fp16",
        "expected_fps": "50 – 120 FPS (Apple Neural Engine)",
        "why": "CoreML tận dụng trực tiếp Apple Neural Engine (ANE) cực tiết kiệm pin.",
        "format_guide": "CoreML (.mlpackage FP16) cho Swift/Xcode; lưu ý export CoreML ổn định nhất khi chạy trên macOS hoặc Python có gói coremltools.",
        "tips": [
            "YOLO26 bỏ DFL giúp đồ thị CoreML gọn hơn, ít lỗi tương thích lớp toán tử.",
        ],
    },
]

TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "small_shapes_default": {
        "name": "🎯 Tối ưu mặc định: Ít ảnh (<100 ảnh) — Hình khối / Vật thể đơn giản",
        "badge": "Mặc định khuyến nghị",
        "description": (
            "Thiết kế riêng cho tập dữ liệu nhỏ (15–100 ảnh) như khối lập phương, hình trụ tròn, hình cầu, linh kiện cơ khí. "
            "Kết hợp Transfer Learning + đóng băng Backbone (freeze=10) + AdamW lr thấp + Cosine LR + tăng cường hình học 3D "
            "(xoay, phối cảnh, lật, ánh sáng) và nhân bản dữ liệu tập train (không rò rỉ sang val) giúp chống học vẹt (overfitting) cực hiệu quả."
        ),
        "params": {
            "model": "yolo26n.pt",
            "epochs": 120,
            "patience": 35,
            "imgsz": 640,
            "batch": 8,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.001,
            "warmup_epochs": 5.0,
            "freeze": 10,
            "pretrained": True,
            "val_split": 0.2,
            "mosaic": 1.0,
            "close_mosaic": 20,
            "mixup": 0.15,
            "degrees": 15.0,
            "translate": 0.15,
            "scale": 0.5,
            "shear": 3.0,
            "perspective": 0.0005,
            "fliplr": 0.5,
            "flipud": 0.15,
            "hsv_h": 0.015,
            "hsv_s": 0.6,
            "hsv_v": 0.45,
            "erasing": 0.2,
            "cls_pw": 0.5,
            "offline_aug_multiplier": 3,
            "cache": "ram",
            "device": "auto",
            "amp": True,
            "allow_tf32": True,
            "workers": 0,
            "seed": 42,
        },
    },
    "ultra_few_shot": {
        "name": "🔬 Siêu ít ảnh (<30 ảnh) — Chống Overfit tối đa",
        "badge": "Few-shot",
        "description": (
            "Dành cho trường hợp chỉ có 10–30 ảnh gốc. Batch siêu nhỏ (4), đóng băng 10 tầng đầu, "
            "nhân bản thông minh 4x riêng tập train + xoay 25° + Regularization mạnh."
        ),
        "params": {
            "model": "yolo26n.pt",
            "epochs": 150,
            "patience": 45,
            "imgsz": 640,
            "batch": 4,
            "optimizer": "AdamW",
            "lr0": 0.0008,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.002,
            "warmup_epochs": 6.0,
            "freeze": 10,
            "pretrained": True,
            "val_split": 0.2,
            "mosaic": 1.0,
            "close_mosaic": 25,
            "mixup": 0.2,
            "degrees": 25.0,
            "translate": 0.18,
            "scale": 0.55,
            "shear": 4.0,
            "perspective": 0.0008,
            "fliplr": 0.5,
            "flipud": 0.25,
            "hsv_h": 0.02,
            "hsv_s": 0.65,
            "hsv_v": 0.5,
            "erasing": 0.25,
            "cls_pw": 0.7,
            "offline_aug_multiplier": 4,
            "cache": "ram",
            "device": "auto",
            "amp": True,
            "allow_tf32": True,
            "workers": 0,
            "seed": 42,
        },
    },
    "fast_cpu_test": {
        "name": "⚡ Huấn luyện nhanh (Chạy thử / Máy chỉ có CPU)",
        "badge": "Siêu nhanh",
        "description": (
            "Giảm imgsz xuống 416 và 45 epochs để kiểm tra nhanh pipeline hoặc chạy trên laptop CPU không có GPU rời."
        ),
        "params": {
            "model": "yolo26n.pt",
            "epochs": 45,
            "patience": 15,
            "imgsz": 416,
            "batch": 8,
            "optimizer": "AdamW",
            "lr0": 0.0012,
            "lrf": 0.05,
            "cos_lr": True,
            "weight_decay": 0.001,
            "warmup_epochs": 3.0,
            "freeze": 10,
            "pretrained": True,
            "val_split": 0.2,
            "mosaic": 0.8,
            "close_mosaic": 10,
            "mixup": 0.1,
            "degrees": 12.0,
            "translate": 0.12,
            "scale": 0.4,
            "shear": 2.0,
            "perspective": 0.0003,
            "fliplr": 0.5,
            "flipud": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.5,
            "hsv_v": 0.4,
            "erasing": 0.15,
            "cls_pw": 0.5,
            "offline_aug_multiplier": 2,
            "cache": "ram",
            "device": "auto",
            "amp": True,
            "allow_tf32": True,
            "workers": 0,
            "seed": 42,
        },
    },
    "high_accuracy_medium_data": {
        "name": "🏆 Độ chính xác cao (Dataset >150 ảnh / GPU mạnh)",
        "badge": "Production GPU",
        "description": (
            "Mở khóa toàn bộ mạng (freeze=0), dùng bản Small để đạt mAP tối đa khi đã có nhiều ảnh và góc chụp phong phú."
        ),
        "params": {
            "model": "yolo26s.pt",
            "epochs": 150,
            "patience": 40,
            "imgsz": 640,
            "batch": 16,
            "optimizer": "auto",
            "lr0": 0.002,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "freeze": 0,
            "pretrained": True,
            "val_split": 0.2,
            "mosaic": 1.0,
            "close_mosaic": 15,
            "mixup": 0.1,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 2.0,
            "perspective": 0.0003,
            "fliplr": 0.5,
            "flipud": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "erasing": 0.2,
            "cls_pw": 0.3,
            "offline_aug_multiplier": 1,
            "cache": "ram",
            "device": "auto",
            "amp": True,
            "allow_tf32": True,
            "workers": 0,
            "seed": 42,
        },
    },
}

HYPERPARAM_DOCS: list[dict[str, Any]] = [
    # --- BASIC ---
    {
        "key": "epochs",
        "group": "basic",
        "label": "Số vòng lặp huấn luyện (Epochs)",
        "type": "int",
        "min": 5,
        "max": 500,
        "step": 5,
        "default": 120,
        "short_note": "Số lần mô hình học qua toàn bộ tập ảnh train.",
        "impact_increase": "Tăng (↑): Mô hình có thêm thời gian hội tụ khi dữ liệu ít & tăng cường mạnh; nhưng quá cao (>250) dễ gây học vẹt (overfitting) và tốn thời gian.",
        "impact_decrease": "Giảm (↓): Huấn luyện nhanh hơn nhưng mô hình có thể chưa học hết đặc trưng góc cạnh/hình dạng (underfitting), mAP thấp.",
        "small_data_tip": "Với <100 ảnh kết hợp augmentation mạnh, nên để 100–150 epochs kèm Early Stopping (patience=35) để tự dừng tại điểm tốt nhất.",
    },
    {
        "key": "imgsz",
        "group": "basic",
        "label": "Kích thước ảnh đầu vào (Image Size)",
        "type": "select",
        "options": [320, 416, 512, 640, 800, 960, 1280],
        "default": 640,
        "short_note": "Độ phân giải (pixel) mà ảnh sẽ được resize về trước khi đưa vào mạng YOLO.",
        "impact_increase": "Tăng (↑ 800–1280): Nhận diện vật thể nhỏ/xa chính xác hơn, giữ rõ cạnh hình khối; nhưng tốn RAM/VRAM gấp 2–4 lần và giảm FPS trên Android/CPU.",
        "impact_decrease": "Giảm (↓ 320–416): Train và suy luận nhanh gấp 2–3 lần, rất hợp Android/Raspberry Pi; nhưng vật thể quá nhỏ dễ bị bỏ sót.",
        "small_data_tip": "640 là điểm vàng cho PC/GPU; chọn 416 nếu mục tiêu chạy realtime trên điện thoại Android tầm trung hoặc Raspberry Pi.",
    },
    {
        "key": "batch",
        "group": "basic",
        "label": "Kích thước lô (Batch Size)",
        "type": "select",
        "options": [2, 4, 8, 16, 32, -1],
        "option_labels": {"-1": "Tự động (AutoBatch)"},
        "default": 8,
        "short_note": "Số ảnh xử lý song song trong một bước cập nhật trọng số.",
        "impact_increase": "Tăng (↑ 16–32): Tận dụng GPU tốt hơn, gradient mượt hơn; nhưng với <100 ảnh chỉ có 2–4 bước cập nhật mỗi epoch khiến học kém tinh tế và dễ tràn VRAM.",
        "impact_decrease": "Giảm (↓ 4–8): Tạo nhiều bước cập nhật hơn mỗi epoch, nhiễu gradient nhỏ giúp chống overfitting rất tốt trên dataset ít ảnh; tốn ít VRAM/RAM.",
        "small_data_tip": "Rất quan trọng! Với <100 ảnh nên đặt batch = 8 (hoặc 4 nếu <35 ảnh) thay vì 16–32.",
    },
    {
        "key": "patience",
        "group": "basic",
        "label": "Dừng sớm tự động (Early Stopping Patience)",
        "type": "int",
        "min": 0,
        "max": 150,
        "step": 5,
        "default": 35,
        "short_note": "Tự động dừng train nếu mAP trên tập validation không cải thiện sau N epoch liên tiếp (0 = tắt).",
        "impact_increase": "Tăng (↑): Cho mô hình thêm cơ hội vượt qua vùng phẳng (plateau), đặc biệt khi bật Cosine LR hoặc sau khi tắt Mosaic.",
        "impact_decrease": "Giảm (↓ 10–15): Tiết kiệm thời gian nhưng dễ dừng quá sớm trước khi bộ trọng số kịp tinh chỉnh ở giai đoạn close_mosaic.",
        "small_data_tip": "Nên để 30–40 (lớn hơn close_mosaic=20) để chắc chắn mô hình trải qua giai đoạn tinh chỉnh cuối không có mosaic.",
    },
    {
        "key": "val_split",
        "group": "basic",
        "label": "Tỷ lệ tập kiểm thử (Validation Split)",
        "type": "float",
        "min": 0.1,
        "max": 0.4,
        "step": 0.05,
        "default": 0.2,
        "short_note": "Tỷ lệ ảnh tách riêng (có phân tầng theo từng class) để đo độ chính xác thực tế, KHÔNG đưa vào train.",
        "impact_increase": "Tăng (↑ 0.25–0.3): Đánh giá mAP ổn định, khách quan hơn; nhưng làm giảm số ảnh quý giá dùng để train khi data ít.",
        "impact_decrease": "Giảm (↓ 0.1–0.15): Giữ lại nhiều ảnh hơn cho train; nhưng nếu tập val chỉ có vài ảnh, chỉ số mAP có thể dao động mạnh.",
        "small_data_tip": "0.20 (80% Train / 20% Val) phân tầng đều mọi lớp là chuẩn an toàn nhất cho 30–100 ảnh.",
    },
    # --- TRANSFER LEARNING & OPTIMIZER ---
    {
        "key": "freeze",
        "group": "optimizer",
        "label": "Đóng băng tầng nền (Freeze Backbone Layers)",
        "type": "int",
        "min": 0,
        "max": 22,
        "step": 1,
        "default": 10,
        "short_note": "Khóa N tầng trích xuất đặc trưng đầu tiên (cạnh, góc, đường cong đã học từ COCO) khi huấn luyện.",
        "impact_increase": "Tăng (↑ 8–12): Chống overfitting cực mạnh khi chỉ có <100 ảnh, train nhanh hơn ~30% và giữ vững khả năng nhận diện biên dạng hình khối.",
        "impact_decrease": "Giảm (↓ 0): Cho phép toàn mạng thay đổi tự do — tốt khi có >200 ảnh hoặc ảnh chuyên ngành cực khác biệt (y tế, nhiệt), nhưng dễ học vẹt nếu <80 ảnh.",
        "small_data_tip": "Bí quyết số 1 cho <100 ảnh! Đặt freeze = 10 giữ nguyên bộ lọc cạnh/góc/bóng đổ từ COCO và chỉ huấn luyện phần Neck + Detection Head cho khối lập phương, hình trụ, hình cầu.",
    },
    {
        "key": "optimizer",
        "group": "optimizer",
        "label": "Thuật toán tối ưu (Optimizer)",
        "type": "select",
        "options": ["AdamW", "auto", "MuSGD", "SGD", "Adam"],
        "default": "AdamW",
        "short_note": "Thuật toán cập nhật trọng số mạng nơ-ron sau mỗi batch.",
        "impact_increase": "AdamW: Hội tụ rất êm, ít nhạy cảm với lr0, kết hợp Weight Decay tách biệt chống overfit tuyệt vời cho <100 ảnh.",
        "impact_decrease": "SGD / MuSGD: Cho độ tổng quát cao nhất trên tập lớn (>500 ảnh) nhưng dễ dao động mạnh nếu tập quá ít ảnh.",
        "small_data_tip": "Chọn AdamW cho dataset <100 ảnh; dùng 'auto' hoặc 'MuSGD' (YOLO26) khi dataset lớn.",
    },
    {
        "key": "lr0",
        "group": "optimizer",
        "label": "Tốc độ học ban đầu (Initial Learning Rate - lr0)",
        "type": "float",
        "min": 0.0001,
        "max": 0.05,
        "step": 0.0002,
        "default": 0.001,
        "short_note": "Độ lớn mỗi bước điều chỉnh trọng số lúc bắt đầu train.",
        "impact_increase": "Tăng (↑ >0.005 với AdamW): Học nhanh lúc đầu nhưng rất dễ phá vỡ trọng số pretrained, khiến loss nhảy loạn hoặc NaN.",
        "impact_decrease": "Giảm (↓ 0.0003–0.0008): Rất an toàn, giữ trọn tri thức pretrained; nhưng hội tụ chậm hơn (cần nhiều epochs hơn).",
        "small_data_tip": "AdamW + Pretrained nên dùng 0.0008 – 0.0015 (nếu chuyển sang SGD mới nâng lên 0.01).",
    },
    {
        "key": "lrf",
        "group": "optimizer",
        "label": "Tỷ lệ tốc độ học cuối (Final LR Factor - lrf)",
        "type": "float",
        "min": 0.001,
        "max": 0.5,
        "step": 0.005,
        "default": 0.01,
        "short_note": "Tốc độ học ở epoch cuối = lr0 × lrf.",
        "impact_increase": "Tăng (↑ 0.1–0.2): Tốc độ học cuối vẫn khá lớn, mô hình có thể rung nhẹ quanh điểm tối ưu.",
        "impact_decrease": "Giảm (↓ 0.01): Hạ dần bước nhảy rất mịn ở cuối quá trình, giúp khung bounding box ôm sát cạnh hình khối chính xác hơn.",
        "small_data_tip": "Giữ mặc định 0.01 cùng Cosine LR.",
    },
    {
        "key": "weight_decay",
        "group": "optimizer",
        "label": "Hệ số phạt trọng số (Weight Decay / L2)",
        "type": "float",
        "min": 0.0,
        "max": 0.02,
        "step": 0.0005,
        "default": 0.001,
        "short_note": "Phạt các trọng số quá lớn để ép mô hình học đặc trưng tổng quát thay vì thuộc lòng nền ảnh.",
        "impact_increase": "Tăng (↑ 0.001–0.003): Giảm overfitting rõ rệt khi ít ảnh; nhưng nếu quá cao (>0.01) mô hình bị 'cứng', khó học sát chi tiết.",
        "impact_decrease": "Giảm (↓ 0.0001): Mô hình bám sát dữ liệu train hơn, dễ đạt train loss rất thấp nhưng val mAP có thể tụt trên ảnh mới.",
        "small_data_tip": "0.001 (hoặc 0.002 nếu <30 ảnh) gấp đôi mặc định YOLO giúp chống học vẹt cực tốt.",
    },
    {
        "key": "cos_lr",
        "group": "optimizer",
        "label": "Lịch giảm LR hình Cosine (Cosine LR Scheduler)",
        "type": "bool",
        "default": True,
        "short_note": "Giảm tốc độ học theo đường cong Cosine mượt mà thay vì tuyến tính.",
        "impact_increase": "Bật (True): Giữ LR đủ cao ở giai đoạn giữa để học đặc trưng rồi hạ siêu mượt ở giai đoạn cuối → mAP50-95 thường tăng +1–3%.",
        "impact_decrease": "Tắt (False): Giảm tuyến tính đơn giản, đôi khi hạ quá sớm ở giữa quá trình.",
        "small_data_tip": "Luôn nên BẬT cho tập dữ liệu nhỏ.",
    },
    {
        "key": "cls_pw",
        "group": "optimizer",
        "label": "Cân bằng lớp tự động (Class Weight Power)",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "default": 0.5,
        "short_note": "Tự tăng trọng số loss cho lớp có ít ảnh hơn (0 = tắt, 1 = nghịch đảo hoàn toàn theo tần suất).",
        "impact_increase": "Tăng (↑ 0.5–1.0): Cứu các lớp bị thiếu mẫu (ví dụ 40 ảnh lập phương nhưng chỉ có 12 ảnh hình cầu) không bị mô hình bỏ qua.",
        "impact_decrease": "Giảm (↓ 0.0): Coi mọi lớp ngang nhau; nếu một lớp có quá ít ảnh, Recall lớp đó sẽ thấp.",
        "small_data_tip": "Đặt 0.5 giúp cân bằng mềm dẻo ngay cả khi số lượng ảnh mỗi hình khối không đều.",
    },
    # --- SMALL DATA & AUGMENTATION ---
    {
        "key": "offline_aug_multiplier",
        "group": "small_data",
        "label": "Nhân bản dữ liệu thông minh tập Train (Offline Expansion)",
        "type": "select",
        "options": [1, 2, 3, 4, 5],
        "option_labels": {"1": "1x (Không nhân bản)", "2": "2x (Gấp đôi tập train)", "3": "3x (Khuyên dùng <80 ảnh)", "4": "4x (Siêu ít ảnh <35 ảnh)", "5": "5x (Cực ít ảnh <20 ảnh)"},
        "default": 3,
        "short_note": "Tạo thêm biến thể ánh sáng, độ tương phản, nhiễu nhẹ, mờ chuyển động và xoay an toàn CHỈ trên tập Train sau khi chia Val (không bao giờ rò rỉ dữ liệu).",
        "impact_increase": "Tăng (↑ 3x–5x): Biến 30 ảnh thành 90–150 ảnh train đa dạng điều kiện ánh sáng/camera; thời gian mỗi epoch tăng tương ứng.",
        "impact_decrease": "Giảm (↓ 1x): Chỉ dùng tăng cường online mặc định của YOLO — phù hợp khi đã có sẵn >120 ảnh.",
        "small_data_tip": "Rất hữu ích cho <80 ảnh! Vì chỉ nhân bản trên tập train sau khi tách val nên điểm mAP đánh giá vẫn hoàn toàn trung thực.",
    },
    {
        "key": "mosaic",
        "group": "small_data",
        "label": "Ghép 4 ảnh Mosaic (Mosaic Augmentation)",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "default": 1.0,
        "short_note": "Ghép ngẫu nhiên 4 ảnh thành 1 khung hình để tăng mạnh sự đa dạng bố cục và tỷ lệ vật thể.",
        "impact_increase": "Tăng (↑ 0.8–1.0): Nhân 4 lần sự phong phú bối cảnh, giúp mô hình không bị phụ thuộc nền bàn/phông chụp khi ít ảnh.",
        "impact_decrease": "Giảm (↓ 0.0–0.3): Giữ nguyên khung hình gốc, hữu ích nếu vật thể luôn chiếm trọn khung hình và không nên bị cắt.",
        "small_data_tip": "Để 1.0 kết hợp close_mosaic=20 để vừa chống học vẹt nền vừa có 20 epoch cuối học lại ảnh nguyên vẹn.",
    },
    {
        "key": "close_mosaic",
        "group": "small_data",
        "label": "Tắt Mosaic ở N epoch cuối (Close Mosaic)",
        "type": "int",
        "min": 0,
        "max": 50,
        "step": 5,
        "default": 20,
        "short_note": "Tắt ghép Mosaic trong N epoch cuối để mô hình tinh chỉnh bounding box chuẩn trên ảnh thật.",
        "impact_increase": "Tăng (↑ 20–30): Giúp mô hình thích nghi kỹ lại với hình ảnh thực tế toàn khung hình, cải thiện mAP50-95.",
        "impact_decrease": "Giảm (↓ 0–5): Kết thúc nhanh nhưng box đôi khi hơi lệch do quen ảnh ghép cắt.",
        "small_data_tip": "20 epochs là tối ưu cho cấu hình 120 epochs.",
    },
    {
        "key": "mixup",
        "group": "small_data",
        "label": "Pha trộn 2 ảnh (MixUp Probability)",
        "type": "float",
        "min": 0.0,
        "max": 0.5,
        "step": 0.05,
        "default": 0.15,
        "short_note": "Chồng mờ hai ảnh lên nhau cùng nhãn của cả hai để làm mượt ranh giới quyết định.",
        "impact_increase": "Tăng (↑ 0.15–0.25): Tăng mạnh khả năng chống overfit và nhận diện khi vật thể bị che khuất/phản sáng; quá cao (>0.4) sẽ làm mờ biên dạng hình khối.",
        "impact_decrease": "Giảm (↓ 0.0): Ảnh sắc nét hơn nhưng ít tác dụng điều chuẩn (regularization) trên tập rất nhỏ.",
        "small_data_tip": "0.15 là mức lý tưởng cho các vật thể hình học cơ bản.",
    },
    {
        "key": "degrees",
        "group": "augmentation",
        "label": "Góc xoay ngẫu nhiên (Rotation ±Degrees)",
        "type": "float",
        "min": 0.0,
        "max": 90.0,
        "step": 5.0,
        "default": 15.0,
        "short_note": "Xoay ngẫu nhiên ảnh trong khoảng [-degrees, +degrees].",
        "impact_increase": "Tăng (↑ 15°–45°): Giúp nhận diện khối lập phương, hình trụ đặt nghiêng/ngang ở mọi góc camera; nhưng góc quá lớn với box ngang (HBB) sẽ làm box rộng thừa khoảng trống.",
        "impact_decrease": "Giảm (↓ 0°): Giữ box cực khít nhưng mô hình dễ trượt nếu thực tế camera cầm tay hơi nghiêng.",
        "small_data_tip": "15°–20° là mức an toàn cho bounding box chuẩn mà vẫn tăng mạnh đa dạng góc nhìn 3D.",
    },
    {
        "key": "scale",
        "group": "augmentation",
        "label": "Phóng to / Thu nhỏ (Scale Gain ±)",
        "type": "float",
        "min": 0.0,
        "max": 0.9,
        "step": 0.05,
        "default": 0.5,
        "short_note": "Co giãn kích thước ảnh ngẫu nhiên giúp nhận diện vật thể ở nhiều khoảng cách gần/xa.",
        "impact_increase": "Tăng (↑ 0.5–0.7): Nhận diện tốt cả khi đưa hình khối sát camera hay đặt xa vài mét.",
        "impact_decrease": "Giảm (↓ 0.1–0.2): Chỉ ổn định nếu khoảng cách camera thực tế luôn cố định (ví dụ camera gắn cố định trên băng chuyền).",
        "small_data_tip": "Giữ 0.5 cho camera cầm tay / điện thoại Android.",
    },
    {
        "key": "shear",
        "group": "augmentation",
        "label": "Biến dạng nghiêng góc nhìn (Shear ±Degrees)",
        "type": "float",
        "min": 0.0,
        "max": 15.0,
        "step": 1.0,
        "default": 3.0,
        "short_note": "Làm xiên nhẹ hình ảnh mô phỏng góc nhìn chéo vào vật thể 3D.",
        "impact_increase": "Tăng (↑ 3°–6°): Rất tốt cho khối lập phương/hình trụ để học biến dạng cạnh khi nhìn nghiêng; quá cao (>10°) có thể biến hình cầu thành elip dễ nhầm hình trụ.",
        "impact_decrease": "Giảm (↓ 0°): Bảo toàn tỷ lệ hình học tuyệt đối (quan trọng nếu cần phân biệt hình cầu với elipsoid).",
        "small_data_tip": "3.0° đủ mô phỏng góc nghiêng nhẹ mà không làm biến dạng hình cầu thành hình trụ.",
    },
    {
        "key": "perspective",
        "group": "augmentation",
        "label": "Biến dạng phối cảnh 3D (Perspective)",
        "type": "float",
        "min": 0.0,
        "max": 0.002,
        "step": 0.0001,
        "default": 0.0005,
        "short_note": "Mô phỏng thay đổi góc nghiêng 3D của ống kính camera.",
        "impact_increase": "Tăng (↑ 0.0005–0.001): Giúp mô hình bền vững hơn khi quay camera từ trên cao xuống hoặc từ bên hông.",
        "impact_decrease": "Giảm (↓ 0): Giữ nguyên hình học phẳng 2D.",
        "small_data_tip": "0.0005 là giá trị tinh tế, không làm méo nhãn.",
    },
    {
        "key": "fliplr",
        "group": "augmentation",
        "label": "Lật ngang (Horizontal Flip Probability)",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "default": 0.5,
        "short_note": "Xác suất lật gương trái-phải.",
        "impact_increase": "Tăng (0.5): Nhân đôi tức thì số góc nhìn hình khối mà không tốn công chụp thêm.",
        "impact_decrease": "Giảm (0.0): Chỉ tắt nếu nhận diện chữ viết, mũi tên chỉ hướng trái/phải.",
        "small_data_tip": "Luôn giữ 0.5 cho vật thể hình học.",
    },
    {
        "key": "flipud",
        "group": "augmentation",
        "label": "Lật dọc (Vertical Flip Probability)",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "default": 0.15,
        "short_note": "Xác suất lật ngược trên-dưới.",
        "impact_increase": "Tăng (↑ 0.15–0.5): Rất có lợi cho khối lập phương, hình cầu, hình trụ đối xứng hoặc camera nhìn từ trên xuống (top-down).",
        "impact_decrease": "Giảm (↓ 0.0): Tắt nếu vật thể có hướng trên-dưới cố định (như hình nón luôn dựng đứng, người, xe).",
        "small_data_tip": "0.15 tăng thêm góc chiếu bóng đổ đối xứng cho các khối cơ bản.",
    },
    {
        "key": "hsv_v",
        "group": "augmentation",
        "label": "Biến thiên độ sáng (HSV Brightness)",
        "type": "float",
        "min": 0.0,
        "max": 0.9,
        "step": 0.05,
        "default": 0.45,
        "short_note": "Thay đổi ngẫu nhiên độ sáng tối để mô hình chạy ổn cả ngoài trời lẫn trong phòng tối.",
        "impact_increase": "Tăng (↑ 0.45–0.65): Giúp nhận diện ổn định khi đổi nguồn sáng đèn LED/ánh nắng/bóng đổ.",
        "impact_decrease": "Giảm (↓ 0.1): Mô hình có thể giảm mạnh độ chính xác khi mang sang phòng khác sáng/tối hơn lúc chụp ảnh mẫu.",
        "small_data_tip": "0.45 giúp dataset ít ảnh không bị phụ thuộc ánh sáng phòng chụp.",
    },
    {
        "key": "hsv_s",
        "group": "augmentation",
        "label": "Biến thiên độ bão hòa màu (HSV Saturation)",
        "type": "float",
        "min": 0.0,
        "max": 0.9,
        "step": 0.05,
        "default": 0.6,
        "short_note": "Thay đổi độ đậm nhạt màu sắc vật thể.",
        "impact_increase": "Tăng (↑ 0.6–0.8): Ép mô hình tập trung vào HÌNH DẠNG thay vì chỉ nhớ màu sắc (ví dụ khối lập phương xanh vẫn nhận ra được dù đổi sang đỏ/xám).",
        "impact_decrease": "Giảm (↓ 0.1): Hữu ích nếu màu sắc chính là đặc trưng phân loại quan trọng.",
        "small_data_tip": "Để 0.6 (và tăng hsv_h nếu cùng một hình khối có nhiều màu khác nhau).",
    },
    {
        "key": "hsv_h",
        "group": "augmentation",
        "label": "Biến thiên tông màu (HSV Hue)",
        "type": "float",
        "min": 0.0,
        "max": 0.5,
        "step": 0.005,
        "default": 0.015,
        "short_note": "Dịch chuyển tông màu sắc tổng thể.",
        "impact_increase": "Tăng (↑ 0.1–0.3): Nếu bạn muốn train khối lập phương màu ĐỎ nhưng thực tế muốn nhận diện cả khối lập phương XANH/VÀNG, hãy tăng hsv_h lên 0.2–0.4!",
        "impact_decrease": "Giảm (↓ 0.015): Giữ màu tự nhiên, cần thiết khi phân biệt vật thể theo màu.",
        "small_data_tip": "Mẹo cực hay: Nếu ảnh mẫu của bạn chỉ có 1 màu cho mỗi hình khối nhưng thực tế nhiều màu, tăng hsv_h lên 0.25!",
    },
    {
        "key": "erasing",
        "group": "augmentation",
        "label": "Che khuất ngẫu nhiên (Random Erasing)",
        "type": "float",
        "min": 0.0,
        "max": 0.6,
        "step": 0.05,
        "default": 0.2,
        "short_note": "Xóa ngẫu nhiên một vùng nhỏ trên ảnh mô phỏng vật thể bị tay hoặc vật khác che một phần.",
        "impact_increase": "Tăng (↑ 0.2–0.4): Giúp nhận diện tốt cả khi tay đang cầm khối lập phương/hình cầu hoặc vật bị che góc.",
        "impact_decrease": "Giảm (↓ 0.0): Train dễ hội tụ hơn nếu vật thể luôn lộ hoàn toàn.",
        "small_data_tip": "0.2 giúp mô hình bền bỉ khi vật thể bị cầm nắm trên tay.",
    },
    # --- HARDWARE & PRECISION ---
    {
        "key": "device",
        "group": "hardware",
        "label": "Thiết bị tính toán (Device)",
        "type": "select",
        "options": ["auto", "0", "cpu"],
        "option_labels": {"auto": "Tự động (Ưu tiên GPU CUDA nếu có)", "0": "GPU CUDA 0", "cpu": "Chỉ dùng CPU"},
        "default": "auto",
        "short_note": "Chọn GPU NVIDIA CUDA hoặc CPU để thực hiện huấn luyện.",
        "impact_increase": "GPU CUDA: Tăng tốc huấn luyện gấp 10–30 lần so với CPU.",
        "impact_decrease": "CPU: Luôn chạy được trên mọi máy nhưng chậm hơn nhiều (nên dùng imgsz=416 và epochs=40–60).",
        "small_data_tip": "Giữ 'auto' để phần mềm tự chọn GPU nếu PyTorch CUDA sẵn sàng, ngược lại tự chuyển CPU an toàn.",
    },
    {
        "key": "amp",
        "group": "hardware",
        "label": "Độ chính xác hỗn hợp (AMP FP16)",
        "type": "bool",
        "default": True,
        "short_note": "Automatic Mixed Precision (FP16 + FP32) giảm VRAM và tăng tốc GPU.",
        "impact_increase": "Bật (True): Giảm ~40% VRAM và tăng tốc 1.5–2x trên GPU RTX; tự bỏ qua an toàn khi chạy CPU.",
        "impact_decrease": "Tắt (False): Chạy thuần FP32 (hoặc TF32 trên RTX) — dùng nếu gặp lỗi NaN loss trên GPU đời cũ (GTX 16xx).",
        "small_data_tip": "Bật True cho RTX 30/40 series; tắt nếu dùng GTX 1650/1660 bị NaN.",
    },
    {
        "key": "allow_tf32",
        "group": "hardware",
        "label": "Tăng tốc TensorFloat-32 (NVIDIA TF32)",
        "type": "bool",
        "default": True,
        "short_note": "Kích hoạt lõi Tensor Cores TF32 trên GPU NVIDIA Ampere/Ada (RTX 30xx/40xx) cho phép nhân ma trận & tích chập.",
        "impact_increase": "Bật (True): Tăng tốc tính toán FP32 lên tới 2–3x trên RTX 4060 mà giữ nguyên dải động (dynamic range) như FP32, không mất ổn định số học.",
        "impact_decrease": "Tắt (False): Dùng IEEE FP32 thuần túy chậm hơn, chỉ cần khi yêu cầu tái lập bit-exact tuyệt đối.",
        "small_data_tip": "Nên BẬT trên RTX 4060 để tận dụng tối đa phần cứng.",
    },
    {
        "key": "cache",
        "group": "hardware",
        "label": "Bộ nhớ đệm ảnh (Dataset Cache)",
        "type": "select",
        "options": ["ram", "disk", "false"],
        "option_labels": {"ram": "RAM (Nhanh nhất cho <500 ảnh)", "disk": "Ổ cứng (Disk)", "false": "Tắt"},
        "default": "ram",
        "short_note": "Nạp trước ảnh vào RAM để không phải đọc ổ cứng lặp lại mỗi epoch.",
        "impact_increase": "RAM: Với <100 ảnh chỉ tốn ~50–150MB RAM nhưng tăng tốc mỗi epoch rõ rệt.",
        "impact_decrease": "Tắt (False): Tiết kiệm RAM tối đa khi bộ dữ liệu có hàng chục nghìn ảnh.",
        "small_data_tip": "Dataset <100 ảnh luôn nên chọn 'ram'.",
    },
]

EXPORT_FORMATS: list[dict[str, Any]] = [
    {
        "id": "pt",
        "name": "PyTorch (.pt)",
        "ultralytics_fmt": "pt",
        "suffix": ".pt",
        "platforms": ["Windows GPU/CPU", "Linux GPU/CPU", "Python"],
        "precisions": ["fp32", "tf32"],
        "description": "Trọng số gốc chuẩn Ultralytics, giữ 100% độ chính xác và hỗ trợ suy luận TF32/FP16 trên PyTorch CUDA.",
        "requires": [],
    },
    {
        "id": "tf32_bundle",
        "name": "PyTorch TF32 Accelerated Package (.zip)",
        "ultralytics_fmt": "tf32_bundle",
        "suffix": "_tf32_bundle.zip",
        "platforms": ["Windows NVIDIA RTX", "Linux NVIDIA Ampere/Ada"],
        "precisions": ["tf32"],
        "description": "Gói triển khai tối ưu chuẩn TF32 (TensorFloat-32) gồm best.pt + script infer_tf32.py bật sẵn Tensor Cores TF32 + metadata lớp.",
        "requires": [],
    },
    {
        "id": "onnx",
        "name": "ONNX (.onnx)",
        "ultralytics_fmt": "onnx",
        "suffix": ".onnx",
        "platforms": ["Windows DirectML/CUDA", "Linux", "Android ORT", "C# / C++ / Web"],
        "precisions": ["fp32", "fp16"],
        "description": "Định dạng mở đa nền tảng phổ biến nhất. Chạy tốt trên Windows (DirectML/CUDA/CPU), Linux, C#, C++, Web và làm bước trung gian sang TensorRT/RKNN.",
        "requires": ["onnx"],
    },
    {
        "id": "litert",
        "name": "LiteRT / TFLite (.tflite)",
        "ultralytics_fmt": "litert",
        "suffix": ".tflite",
        "platforms": ["Android (NNAPI/GPU)", "iOS", "Edge TPU / Embedded"],
        "precisions": ["fp32", "fp16", "int8"],
        "description": "Chuẩn TensorFlow Lite / Google LiteRT cho ứng dụng di động Android & thiết bị nhúng. Hỗ trợ lượng tử hóa FP16/INT8 cực nhẹ.",
        "requires": ["tensorflow / ai-edge-litert"],
    },
    {
        "id": "ncnn",
        "name": "NCNN (_ncnn_model.zip)",
        "ultralytics_fmt": "ncnn",
        "suffix": "_ncnn_model",
        "platforms": ["Android ARM64", "Raspberry Pi / Linux ARM"],
        "precisions": ["fp32", "fp16"],
        "description": "Engine C++ tối ưu hàng đầu cho chip ARM (điện thoại Android & Raspberry Pi), không cần phụ thuộc nặng, tốc độ CPU/Vulkan cực cao.",
        "requires": ["ncnn", "pnnx"],
    },
    {
        "id": "engine",
        "name": "TensorRT (.engine — TF32 / FP16 / INT8)",
        "ultralytics_fmt": "engine",
        "suffix": ".engine",
        "platforms": ["Windows NVIDIA GPU", "Linux NVIDIA GPU", "Jetson Orin/Nano"],
        "precisions": ["tf32", "fp32", "fp16", "int8"],
        "description": "Engine tốc độ cao nhất cho GPU NVIDIA (tăng tốc 3–5x). Hỗ trợ TF32/FP32, FP16 và INT8 Tensor Cores (yêu cầu PyTorch CUDA + tensorrt trên máy chạy).",
        "requires": ["cuda", "tensorrt"],
    },
    {
        "id": "openvino",
        "name": "OpenVINO (_openvino_model.zip)",
        "ultralytics_fmt": "openvino",
        "suffix": "_openvino_model",
        "platforms": ["Windows Intel CPU/iGPU", "Linux Intel CPU/iGPU"],
        "precisions": ["fp32", "fp16", "int8"],
        "description": "Tối ưu chuyên sâu cho CPU và đồ họa tích hợp Intel trên Windows & Linux (nhanh hơn PyTorch CPU 2–4x).",
        "requires": ["openvino"],
    },
    {
        "id": "torchscript",
        "name": "TorchScript (.torchscript)",
        "ultralytics_fmt": "torchscript",
        "suffix": ".torchscript",
        "platforms": ["Linux C++ LibTorch", "Windows C++", "Mobile PyTorch"],
        "precisions": ["fp32", "fp16"],
        "description": "Đồ thị PyTorch biên dịch độc lập không cần Python runtime, tích hợp thẳng vào ứng dụng C++ LibTorch.",
        "requires": [],
    },
    {
        "id": "saved_model",
        "name": "TensorFlow SavedModel (FP32 _saved_model.zip)",
        "ultralytics_fmt": "saved_model",
        "suffix": "_saved_model",
        "platforms": ["TensorFlow Serving", "Linux / Windows Server"],
        "precisions": ["fp32", "fp16"],
        "description": "Định dạng TensorFlow chuẩn FP32 dùng cho hệ thống TensorFlow Serving / TFX doanh nghiệp.",
        "requires": ["tensorflow"],
    },
    {
        "id": "pb",
        "name": "TensorFlow GraphDef (.pb — FP32)",
        "ultralytics_fmt": "pb",
        "suffix": ".pb",
        "platforms": ["TensorFlow C++", "OpenCV DNN"],
        "precisions": ["fp32"],
        "description": "File đồ thị đóng băng TensorFlow FP32 đơn lẻ, dễ nạp qua OpenCV DNN hoặc TensorFlow C++.",
        "requires": ["tensorflow"],
    },
    {
        "id": "coreml",
        "name": "CoreML (.mlpackage.zip)",
        "ultralytics_fmt": "coreml",
        "suffix": ".mlpackage",
        "platforms": ["iOS (iPhone/iPad)", "macOS Apple Silicon"],
        "precisions": ["fp16", "fp32"],
        "description": "Tối ưu trực tiếp cho Apple Neural Engine (ANE) trên iPhone, iPad và Mac chip M-series.",
        "requires": ["coremltools"],
    },
    {
        "id": "mnn",
        "name": "MNN (.mnn)",
        "ultralytics_fmt": "mnn",
        "suffix": ".mnn",
        "platforms": ["Android ARM", "iOS", "Embedded Linux"],
        "precisions": ["fp32", "fp16"],
        "description": "Framework suy luận di động siêu nhẹ của Alibaba cho thiết bị biên tài nguyên thấp.",
        "requires": ["MNN"],
    },
]
