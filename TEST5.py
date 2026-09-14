import cv2
import math
import numpy as np
import onnxruntime as ort


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)


def run_webcam(onnx_model_path, conf_thresh=0.4, iou_thresh=0.45):
    # 1. Tải mô hình ONNX một lần duy nhất trước khi mở camera
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # 2. Mở kết nối Webcam (số 0 là camera mặc định của máy)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    print("Đang chạy... Bấm phím 'q' trên bàn phím để thoát.")

    # 3. Vòng lặp đọc và xử lý liên tục từng khung hình
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tiền xử lý
        img_input, ratio, (dw, dh) = letterbox(frame, (640, 640))
        blob = img_input[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # Chạy suy luận
        outputs = session.run(None, {input_name: blob})[0]

        # Hậu xử lý
        predictions = np.squeeze(outputs).T
        num_classes = predictions.shape[1] - 5
        boxes, scores, class_ids = [], [], []

        for pred in predictions:
            class_scores = pred[4: 4 + num_classes]
            max_score = np.max(class_scores)
            if max_score >= conf_thresh:
                cx, cy, w, h = pred[0:4]
                angle_rad = pred[-1]

                cx, cy = (cx - dw) / ratio, (cy - dh) / ratio
                w, h = w / ratio, h / ratio
                angle_deg = math.degrees(angle_rad)

                boxes.append(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
                scores.append(float(max_score))
                class_ids.append(int(np.argmax(class_scores)))

        # Lọc các hộp trùng lặp
        indices = cv2.dnn.NMSBoxesRotated(boxes, scores, conf_thresh, iou_thresh)

        # Vẽ kết quả lên khung hình
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                points = np.int32(cv2.boxPoints(boxes[i]))
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"Class {class_ids[i]}: {scores[i]:.2f}",
                            (points[0][0], points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Hiển thị cửa sổ video
        cv2.imshow("YOLOv8 OBB - ONNX Webcam", frame)

        # Lắng nghe sự kiện bàn phím (thoát nếu bấm 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Đổi tên file này thành file .onnx thực tế của bạn
    run_webcam("runs/obb/yolov8_train/v1-3/weights/best.onnx")