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
    url = "http://192.168.0.104:8080/video"
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Không thể mở Webcam!")
        return

    print("Camera đang chạy. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất kết nối khung hình hoặc hết stream.")
            break

        # 1. Đưa khung hình vào mô hình YOLO để nhận diện (tắt verbose để đỡ rác terminal)
        results = model(frame, verbose=False)

        # 2. Trích xuất kết quả
        for result in results:
            # Lấy tọa độ [x_min, y_min, x_max, y_max] và chuyển thành list Python chuẩn
            boxes_list = result.boxes.xyxy.tolist()
            
            # Lấy thêm ID của vật thể (ví dụ: 0 là người, 2 là ô tô...)
            class_ids = result.boxes.cls.tolist()

            # Nếu nhận diện được ít nhất 1 vật thể thì in ra
            if len(boxes_list) > 0:
                print(f"Phát hiện {len(boxes_list)} vật thể:")
                print("Tọa độ box:", boxes_list)
                print("Class ID:", class_ids)
                print("-" * 30)

        # Ghi chú: Nếu bạn không muốn hiện cửa sổ video nữa, hãy thêm dấu # trước 2 dòng dưới đây
        # cv2.imshow('Camera', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
