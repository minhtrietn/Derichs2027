import cv2
import torch
from ultralytics import YOLO

def run_webcam():
    # 1. Dọn dẹp rác trên GPU trước khi khởi tạo mô hình
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Tải mô hình bản Nano (tự động tải file yolov8n.pt về nếu chưa có)
    print("Đang nạp mô hình vào GPU...")
    model = YOLO('yolov8n.pt')

    # 3. Thiết lập giới hạn luồng thu
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Không thể mở Webcam!")
        return

    print("Camera đang chạy. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Chạy suy luận tối ưu
        # - device='0': Bắt buộc chạy trên lõi GPU CUDA
        # - half=True: Dùng số thực 16-bit (FP16), giảm một nửa dung lượng VRAM tiêu thụ và tăng tốc
        # - verbose=False: Tắt in log ra terminal để tránh ngốn tài nguyên CPU
        results = model.predict(source=frame, device='0', half=True, verbose=False)

        # 5. Vẽ hộp bao chuẩn (Hình chữ nhật ngang) lên ảnh
        # Hàm plot() tự động đóng gói toàn bộ khâu vẽ bounding box, tên class và độ tin cậy
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Nano Detection - GPU", annotated_frame)

        # Giải phóng RAM lập tức cho vòng lặp tiếp theo
        del results

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()