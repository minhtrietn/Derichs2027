"""Multi-engine automatic image annotation: Smart CV Shape Boxer, Open-Vocabulary YOLOE/World, and Pretrained/Custom YOLO."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import VI_EN_LABEL_HINTS, configure_cuda_environment, ensure_directories
from app.dataset_manager import dataset_manager
from app.ws_manager import ws_manager


def _prepare_inference_device() -> tuple[Any, Any]:
    """Configure CUDA 11.8 DLL paths and return (torch_module, device) for YOLO inference."""
    configure_cuda_environment()
    import torch

    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            use_tf32 = major >= 8
        except Exception:
            use_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
        return torch, 0
    return torch, "cpu"


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _nms_rects(
    rects: list[tuple[float, float, float, float, float, int]],
    iou_thresh: float = 0.45,
) -> list[tuple[float, float, float, float, float, int]]:
    """Non-maximum suppression for (x1, y1, x2, y2, score, class_id)."""
    rects = sorted(rects, key=lambda r: r[4], reverse=True)
    kept: list[tuple[float, float, float, float, float, int]] = []
    for r in rects:
        if all(_iou_xyxy(r[:4], k[:4]) < iou_thresh for k in kept):
            kept.append(r)
    return kept


def _map_prompt_to_english(label: str) -> str:
    low = label.strip().lower()
    return VI_EN_LABEL_HINTS.get(low, label.strip())


def _find_shape_class_ids(classes: list[str]) -> dict[str, int]:
    """Locate cube / cylinder / sphere indices regardless of Vietnamese or English naming."""
    mapping: dict[str, int] = {}
    for idx, name in enumerate(classes):
        low = name.strip().lower()
        en = VI_EN_LABEL_HINTS.get(low, low)
        if any(k in low or k in en for k in ("cube", "lập phương", "lap phuong", "box", "vuông")):
            mapping.setdefault("cube", idx)
        elif any(k in low or k in en for k in ("cylinder", "trụ", "tru", "tube", "can")):
            mapping.setdefault("cylinder", idx)
        elif any(k in low or k in en for k in ("sphere", "cầu", "cau", "ball", "tròn", "circle")):
            mapping.setdefault("sphere", idx)
    return mapping


def _classify_contour_shape(contour: np.ndarray, classes: list[str], fallback_class_id: int) -> int:
    """Classify a foreground contour into cube, cylinder, or sphere using geometric descriptors."""
    shape_ids = _find_shape_class_ids(classes)
    if len(shape_ids) < 2:
        return fallback_class_id

    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if area <= 10 or peri <= 10:
        return fallback_class_id

    circularity = (4.0 * math.pi * area) / (peri * peri)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1.0, min(w, h))
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = area / hull_area
    extent = area / max(1.0, float(w * h))

    approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
    vertices = len(approx)

    # Strong circular symmetry -> sphere
    if circularity >= 0.855 and aspect <= 1.18 and "sphere" in shape_ids:
        return shape_ids["sphere"]

    # Elongated vertical/horizontal shape -> cylinder
    if aspect >= 1.34 and "cylinder" in shape_ids:
        return shape_ids["cylinder"]

    # Clean polygonal 4-7 corner silhouette (isometric cube has 6 outer vertices, frontal cube has 4)
    if vertices <= 7 and solidity >= 0.90 and extent >= 0.62 and "cube" in shape_ids:
        return shape_ids["cube"]

    # Distinguish remaining curved vs boxy silhouettes
    if circularity >= 0.82 and aspect <= 1.22 and "sphere" in shape_ids:
        return shape_ids["sphere"]
    if aspect >= 1.20 and "cylinder" in shape_ids:
        return shape_ids["cylinder"]
    if "cube" in shape_ids:
        return shape_ids["cube"]
    return fallback_class_id


def _detect_foreground_cv(
    img_bgr: np.ndarray,
    classes: list[str],
    class_mode: str = "auto_geometry",
    target_class_id: int = 0,
    min_area_ratio: float = 0.008,
    max_area_ratio: float = 0.90,
    sensitivity: float = 0.5,
) -> list[dict[str, Any]]:
    """Offline foreground object detector using border Lab color distance + adaptive edges."""
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample border pixels to model background distribution robustly
    border_px = np.concatenate(
        [
            lab[: max(4, h // 18), :, :].reshape(-1, 3),
            lab[-max(4, h // 18) :, :, :].reshape(-1, 3),
            lab[:, : max(4, w // 18), :].reshape(-1, 3),
            lab[:, -max(4, w // 18) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_median = np.median(border_px, axis=0)
    bg_mad = np.median(np.abs(border_px - bg_median), axis=0) + 4.0

    # Weighted perceptual color distance (emphasize chroma a/b slightly over shadow luminance L)
    diff = np.abs(lab - bg_median[None, None, :])
    dist = np.sqrt(
        (diff[:, :, 0] / (bg_mad[0] * 1.8 + 8.0)) ** 2
        + (diff[:, :, 1] / (bg_mad[1] * 1.1 + 5.0)) ** 2
        + (diff[:, :, 2] / (bg_mad[2] * 1.1 + 5.0)) ** 2
    )
    dist_u8 = np.clip(dist * 38.0, 0, 255).astype(np.uint8)

    # Combine Otsu color mask + adaptive threshold adjusted by sensitivity
    otsu_val, mask_otsu = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    manual_thr = max(18, int(otsu_val * (1.28 - 0.55 * sensitivity)))
    _, mask_color = cv2.threshold(dist_u8, manual_thr, 255, cv2.THRESH_BINARY)

    # Edge cue reinforcement
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray))
    lo = int(max(15, 0.55 * med))
    hi = int(min(220, 1.35 * med + 20))
    edges = cv2.Canny(gray, lo, hi)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=2)

    combined = cv2.bitwise_or(mask_color, cv2.bitwise_and(mask_otsu, edges_closed))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float, float, float, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        ratio = area / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Skip thin border artifacts
        if bw < 14 or bh < 14:
            continue
        if (bw / w > 0.96 and bh / h > 0.96):
            continue

        # Score by mean foreground distance inside bbox
        score = float(np.mean(dist[y : y + bh, x : x + bw]))
        if class_mode == "auto_geometry":
            cls_id = _classify_contour_shape(cnt, classes, target_class_id)
        else:
            cls_id = target_class_id
        candidates.append((float(x), float(y), float(x + bw), float(y + bh), score, cls_id))

    kept = _nms_rects(candidates, iou_thresh=0.4)
    # Limit to top 12 salient objects per frame
    kept = kept[:12]

    boxes: list[dict[str, Any]] = []
    for x1, y1, x2, y2, _score, cls_id in kept:
        boxes.append(
            {
                "class_id": cls_id,
                "cx": round(((x1 + x2) / 2.0) / w, 6),
                "cy": round(((y1 + y2) / 2.0) / h, 6),
                "w": round((x2 - x1) / w, 6),
                "h": round((y2 - y1) / h, 6),
            }
        )
    return boxes


class AutoLabeler:
    """Runs automated annotation over unlabeled (or all) images in a project with live WebSocket progress."""

    def run(
        self,
        project_id: str,
        engine: str = "cv_smart",
        overwrite_existing: bool = False,
        conf: float = 0.25,
        class_mode: str = "auto_geometry",
        target_class_id: int = 0,
        model_name: str = "yolo26n.pt",
        sensitivity: float = 0.5,
    ) -> dict[str, Any]:
        ensure_directories()
        meta = dataset_manager.get_project(project_id)
        classes: list[str] = meta.get("classes", ["object"])
        images = dataset_manager.list_images(project_id)

        targets = [im for im in images if overwrite_existing or not im["labeled"]]
        if not targets:
            return {
                "processed": 0,
                "labeled_images": 0,
                "total_boxes": 0,
                "engine": engine,
                "message": "Tất cả ảnh đều đã có nhãn (bật 'Ghi đè nhãn cũ' nếu muốn đánh dấu lại).",
                "project": meta,
            }

        ws_manager.log_sync(
            f"Bắt đầu Tự động đánh dấu ({engine}) cho {len(targets)} ảnh...",
            source="autolabel",
        )

        if engine == "cv_smart":
            return self._run_cv_smart(
                project_id=project_id,
                targets=targets,
                classes=classes,
                class_mode=class_mode,
                target_class_id=target_class_id,
                sensitivity=sensitivity,
            )
        if engine == "open_vocab":
            return self._run_open_vocab(
                project_id=project_id,
                targets=targets,
                classes=classes,
                conf=conf,
                model_name=model_name,
            )
        if engine == "yolo_model":
            return self._run_yolo_model(
                project_id=project_id,
                targets=targets,
                classes=classes,
                conf=conf,
                model_name=model_name,
                target_class_id=target_class_id,
            )
        raise ValueError(f"Engine tự động đánh dấu không hợp lệ: {engine}")

    def _run_cv_smart(
        self,
        project_id: str,
        targets: list[dict[str, Any]],
        classes: list[str],
        class_mode: str,
        target_class_id: int,
        sensitivity: float,
    ) -> dict[str, Any]:
        labeled_images = 0
        total_boxes = 0
        safe_cls = max(0, min(len(classes) - 1, int(target_class_id)))

        for idx, item in enumerate(targets):
            img_path = dataset_manager.get_image_path(project_id, item["filename"])
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            boxes = _detect_foreground_cv(
                img,
                classes=classes,
                class_mode=class_mode,
                target_class_id=safe_cls,
                sensitivity=sensitivity,
            )
            if boxes:
                dataset_manager.save_labels(project_id, item["filename"], boxes)
                labeled_images += 1
                total_boxes += len(boxes)

            ws_manager.broadcast_sync(
                "task_progress",
                {
                    "task": "autolabel",
                    "progress": int((idx + 1) / len(targets) * 100),
                    "message": f"CV Smart Auto-Label: {idx + 1}/{len(targets)} ảnh ({total_boxes} hộp)...",
                },
            )

        ws_manager.log_sync(
            f"CV Smart Auto-Label hoàn tất: gán nhãn {labeled_images}/{len(targets)} ảnh với {total_boxes} bounding boxes.",
            level="success",
            source="autolabel",
        )
        return {
            "processed": len(targets),
            "labeled_images": labeled_images,
            "total_boxes": total_boxes,
            "engine": "cv_smart",
            "message": f"Đã tự động đánh dấu {labeled_images}/{len(targets)} ảnh ({total_boxes} vật thể) bằng Smart CV Offline.",
            "project": dataset_manager.get_project(project_id),
        }

    def _run_open_vocab(
        self,
        project_id: str,
        targets: list[dict[str, Any]],
        classes: list[str],
        conf: float,
        model_name: str,
    ) -> dict[str, Any]:
        torch, device = _prepare_inference_device()
        from ultralytics import YOLO

        prompts = [_map_prompt_to_english(c) for c in classes]
        chosen = model_name if model_name and ("world" in model_name.lower() or "yoloe" in model_name.lower()) else "yolov8s-worldv2.pt"
        dev_label = f"CUDA {torch.version.cuda}" if device != "cpu" else "CPU"
        ws_manager.log_sync(
            f"Đang nạp mô hình Open-Vocabulary '{chosen}' ({dev_label}) với nhãn prompt: {prompts}...",
            source="autolabel",
        )
        try:
            model = YOLO(chosen)
            if hasattr(model, "set_classes"):
                model.set_classes(prompts)
        except Exception as exc:
            raise RuntimeError(
                f"Không thể nạp mô hình Open-Vocabulary '{chosen}' ({exc}). "
                "Kiểm tra kết nối mạng để tải trọng số lần đầu, hoặc dùng chế độ 'Smart CV Offline' không cần internet."
            ) from exc

        labeled_images = 0
        total_boxes = 0
        try:
            with torch.inference_mode():
                for idx, item in enumerate(targets):
                    img_path = dataset_manager.get_image_path(project_id, item["filename"])
                    results = model.predict(str(img_path), conf=conf, device=device, verbose=False)
                    boxes: list[dict[str, Any]] = []
                    if results and results[0].boxes is not None:
                        xywhn = results[0].boxes.xywhn.cpu().numpy()
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        for (cx, cy, bw, bh), cid in zip(xywhn, clss):
                            if 0 <= cid < len(classes):
                                boxes.append({"class_id": int(cid), "cx": float(cx), "cy": float(cy), "w": float(bw), "h": float(bh)})
                    if boxes:
                        dataset_manager.save_labels(project_id, item["filename"], boxes)
                        labeled_images += 1
                        total_boxes += len(boxes)
                    ws_manager.broadcast_sync(
                        "task_progress",
                        {
                            "task": "autolabel",
                            "progress": int((idx + 1) / len(targets) * 100),
                            "message": f"Open-Vocab AI: {idx + 1}/{len(targets)} ảnh ({total_boxes} hộp)...",
                        },
                    )
        finally:
            if device != "cpu":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        ws_manager.log_sync(
            f"Open-Vocab Auto-Label hoàn tất: {labeled_images}/{len(targets)} ảnh, {total_boxes} boxes.",
            level="success",
            source="autolabel",
        )
        return {
            "processed": len(targets),
            "labeled_images": labeled_images,
            "total_boxes": total_boxes,
            "engine": "open_vocab",
            "message": f"Đã tự động đánh dấu {labeled_images}/{len(targets)} ảnh ({total_boxes} boxes) bằng {chosen}.",
            "project": dataset_manager.get_project(project_id),
        }

    def _run_yolo_model(
        self,
        project_id: str,
        targets: list[dict[str, Any]],
        classes: list[str],
        conf: float,
        model_name: str,
        target_class_id: int,
    ) -> dict[str, Any]:
        torch, device = _prepare_inference_device()
        from ultralytics import YOLO

        dev_label = f"CUDA {torch.version.cuda}" if device != "cpu" else "CPU"
        ws_manager.log_sync(f"Đang nạp mô hình YOLO '{model_name}' ({dev_label}) để tự động gán nhãn (Active Learning)...", source="autolabel")
        model = YOLO(model_name)
        model_names = model.names or {}

        # Match model class names to project class names (case-insensitive / VI-EN aware)
        proj_lookup = {_map_prompt_to_english(c).lower(): i for i, c in enumerate(classes)}
        for i, c in enumerate(classes):
            proj_lookup[c.strip().lower()] = i

        labeled_images = 0
        total_boxes = 0
        safe_fallback = max(0, min(len(classes) - 1, int(target_class_id)))

        try:
            with torch.inference_mode():
                for idx, item in enumerate(targets):
                    img_path = dataset_manager.get_image_path(project_id, item["filename"])
                    results = model.predict(str(img_path), conf=conf, device=device, verbose=False)
                    boxes: list[dict[str, Any]] = []
                    if results and results[0].boxes is not None:
                        xywhn = results[0].boxes.xywhn.cpu().numpy()
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        for (cx, cy, bw, bh), mid in zip(xywhn, clss):
                            mname = str(model_names.get(int(mid), "")).strip().lower()
                            mapped_id = proj_lookup.get(mname, safe_fallback)
                            boxes.append({"class_id": mapped_id, "cx": float(cx), "cy": float(cy), "w": float(bw), "h": float(bh)})
                    if boxes:
                        dataset_manager.save_labels(project_id, item["filename"], boxes)
                        labeled_images += 1
                        total_boxes += len(boxes)

                    ws_manager.broadcast_sync(
                        "task_progress",
                        {
                            "task": "autolabel",
                            "progress": int((idx + 1) / len(targets) * 100),
                            "message": f"YOLO Model Auto-Label: {idx + 1}/{len(targets)} ảnh ({total_boxes} hộp)...",
                        },
                    )
        finally:
            if device != "cpu":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        ws_manager.log_sync(
            f"YOLO Model Auto-Label hoàn tất: {labeled_images}/{len(targets)} ảnh, {total_boxes} boxes.",
            level="success",
            source="autolabel",
        )
        return {
            "processed": len(targets),
            "labeled_images": labeled_images,
            "total_boxes": total_boxes,
            "engine": "yolo_model",
            "message": f"Đã tự động đánh dấu {labeled_images}/{len(targets)} ảnh ({total_boxes} boxes) bằng {model_name}.",
            "project": dataset_manager.get_project(project_id),
        }


auto_labeler = AutoLabeler()
