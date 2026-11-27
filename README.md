# DERICH YOLO Auto-Training Studio (WebSocket Web App)

Ứng dụng web kết nối **WebSocket thời gian thực** hỗ trợ tự động huấn luyện các thế hệ **YOLOv8 · YOLO11 · YOLO12 · YOLO26** (và trọng số `.pt` tùy chỉnh), tối ưu mặc định cho **dataset ít ảnh (<100 ảnh)** như nhận diện hình khối cơ bản (`cube`, `cylinder`, `sphere`), tích hợp nhiều nguồn đầu vào, tự động tiền xử lý, tự động đánh dấu (Auto-Label), tư vấn chọn model theo nền tảng (**Android / Linux / Windows / iOS**) và xuất đa định dạng (**ONNX, TF32, PT, LiteRT/TFLite, NCNN, TensorRT, OpenVINO, TorchScript...**).

## Yêu cầu môi trường chuẩn
- **Python**: `3.11.6` (`.python-version` / `requires-python = ">=3.11.6,<3.12"`)
- **CUDA Toolkit**: `11.8` (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`)
- **PyTorch**: `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, `torchaudio==2.7.1+cu118` *(phiên bản chính thức mới nhất của PyTorch hỗ trợ CUDA 11.8 trên Python 3.11; từ PyTorch 2.8 trở đi đã ngừng hỗ trợ CUDA 11.8)*

## Cách khởi chạy hợp nhất (Tự nhận diện Windows / Linux / macOS)

Chỉ cần **1 bộ mã nguồn duy nhất** cho cả **Windows và Linux** — trình khởi chạy `run.py` sẽ tự nhận diện hệ điều hành, tự tách/chọn virtualenv tương thích (`.venv` hoặc `.venv_linux`/`.venv_win` nếu dùng chung thư mục giữa 2 hệ điều hành), tự cấu hình `CUDA_PATH` / `LD_LIBRARY_PATH`, tự cài thư viện nếu còn thiếu và khởi động server:

### Cách 1: Chạy trực tiếp `run.py` (dùng chung cho cả Windows & Linux)
```bash
# Windows:
python run.py

# Linux / macOS:
python3 run.py
```
- Cài / cập nhật lại môi trường (tự nhận GPU NVIDIA → CUDA 11.8, hoặc thêm `--cpu` nếu chỉ dùng CPU):
```bash
python3 run.py --setup        # Linux / macOS
python run.py --setup         # Windows
python3 run.py --check        # Kiểm tra OS, Python, venv, GPU/CUDA
```

### Cách 2: Chạy nhanh bằng script theo hệ điều hành (đều gọi `run.py`)
- **Windows**: click đúp `start_app.bat` (hoặc `setup_env_cu118.bat` để chỉ cài môi trường).
- **Linux**:
```bash
chmod +x start_app.sh
./start_app.sh                # Chạy app (tự cài môi trường nếu chưa có)
./start_app.sh --setup        # Cài lại môi trường Python + PyTorch CUDA/CPU
./start_app.sh --host 0.0.0.0 # Cho phép truy cập qua mạng LAN
```

- Giao diện Web: `http://127.0.0.1:8000`
- Kênh WebSocket: `ws://127.0.0.1:8000/ws`
- Khi chạy trên Linux server không có màn hình (headless / SSH), `run.py` tự bật `--no-browser` và in địa chỉ truy cập.

### Cách 3: Dùng `uv` (tự động đọc `pyproject.toml` + index `cu118`)
```bash
uv sync --python 3.11.6
```

## Tối ưu mặc định cho tập dữ liệu ít ảnh (<100 ảnh)
Preset mặc định **Small-Data (`small_shapes_default`)** tự động áp dụng:
- **YOLO26n / YOLO11n** (ít tham số, chống học vẹt tốt).
- **Transfer Learning + `freeze=10`**: giữ nguyên bộ lọc cạnh/góc/bóng đổ đã học từ COCO, chỉ tinh chỉnh phần Neck & Head cho các hình khối của bạn.
- **AdamW (`lr0=0.001`) + Cosine LR + `weight_decay=0.001` + `batch=8`** + `patience=35`.
- **Tách Train/Val phân tầng** + **Nhân bản dữ liệu thông minh 3x riêng tập Train** (tuyệt đối không rò rỉ sang tập Val).
- **Tăng cường hình học 3D**: `mosaic=1.0`, `close_mosaic=20`, `mixup=0.15`, `degrees=15°`, `shear=3°`, `perspective=0.0005`, `fliplr=0.5`, `flipud=0.15`, `hsv_v=0.45`, `hsv_s=0.6`, `erasing=0.2`.

## Cài thêm thư viện cho các định dạng xuất tùy chọn
- **ONNX**: `pip install onnx onnxslim onnxruntime` (nếu cần GPU CUDA 11.8: `pip install onnxruntime-gpu==1.18.1`, hoặc `onnxruntime-directml` trên Windows)
- **LiteRT / TFLite (Android)**: `pip install ai-edge-litert tensorflow onnx2tf`
- **NCNN (Android ARM / Raspberry Pi)**: `pip install ncnn pnnx`
- **OpenVINO (Windows / Linux Intel CPU)**: `pip install openvino`
