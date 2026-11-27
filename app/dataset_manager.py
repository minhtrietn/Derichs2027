"""Dataset project store, multi-source ingestion, auto-preprocessing, health analyzer, and leak-free small-data expansion."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from app.config import (
    PROJECTS_DIR,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_VIDEO_EXTS,
    ensure_directories,
)
from app.ws_manager import ws_manager


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_").lower()
    return slug[:40] or "project"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_dhash(img_bgr: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _blur_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class DatasetManager:
    """Manages dataset projects on disk with YOLO-standard labels and smart preprocessing."""

    def __init__(self) -> None:
        ensure_directories()

    def _project_dir(self, project_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "", project_id)
        if not safe:
            raise ValueError("Invalid project_id")
        return PROJECTS_DIR / safe

    def _meta_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _images_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "images"

    def _labels_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "labels"

    def _prepared_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "prepared"

    def list_projects(self) -> list[dict[str, Any]]:
        ensure_directories()
        projects: list[dict[str, Any]] = []
        for pdir in sorted(PROJECTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta_file = pdir / "project.json"
            if pdir.is_dir() and meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    stats = self._quick_stats(pdir.name)
                    meta.update(stats)
                    projects.append(meta)
                except Exception:
                    continue
        return projects

    def get_project(self, project_id: str) -> dict[str, Any]:
        meta_file = self._meta_path(project_id)
        if not meta_file.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta.update(self._quick_stats(project_id))
        return meta

    def create_project(
        self,
        name: str,
        classes: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        ensure_directories()
        pid = f"{_slugify(name)}_{uuid.uuid4().hex[:6]}"
        pdir = self._project_dir(pid)
        (pdir / "images").mkdir(parents=True, exist_ok=True)
        (pdir / "labels").mkdir(parents=True, exist_ok=True)

        clean_classes = [c.strip() for c in (classes or ["cube", "cylinder", "sphere"]) if c.strip()]
        if not clean_classes:
            clean_classes = ["cube", "cylinder", "sphere"]

        meta = {
            "id": pid,
            "name": name.strip() or "Dataset mới",
            "classes": clean_classes,
            "notes": notes,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._meta_path(pid).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.get_project(pid)

    def ensure_default_project(self) -> dict[str, Any]:
        projects = self.list_projects()
        if projects:
            return projects[0]
        return self.create_project(
            name="Geometric_Shapes_Demo",
            classes=["cube", "cylinder", "sphere"],
            notes="Dự án mặc định nhận diện hình khối cơ bản (khối lập phương, hình trụ tròn, hình cầu).",
        )

    def delete_project(self, project_id: str) -> None:
        pdir = self._project_dir(project_id)
        if pdir.exists():
            shutil.rmtree(pdir, ignore_errors=True)

    def update_classes(self, project_id: str, classes: list[str]) -> dict[str, Any]:
        meta = json.loads(self._meta_path(project_id).read_text(encoding="utf-8"))
        cleaned = []
        seen = set()
        for c in classes:
            name = c.strip()
            if name and name not in seen:
                cleaned.append(name)
                seen.add(name)
        if not cleaned:
            raise ValueError("Danh sách lớp (classes) không được để trống.")
        meta["classes"] = cleaned
        meta["updated_at"] = _now_iso()
        self._meta_path(project_id).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.get_project(project_id)

    def _quick_stats(self, project_id: str) -> dict[str, Any]:
        img_dir = self._images_dir(project_id)
        lbl_dir = self._labels_dir(project_id)
        images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS] if img_dir.exists() else []
        labeled = 0
        total_boxes = 0
        for img in images:
            boxes = self.read_labels(project_id, img.name)
            if boxes:
                labeled += 1
                total_boxes += len(boxes)
        return {
            "image_count": len(images),
            "labeled_count": labeled,
            "unlabeled_count": len(images) - labeled,
            "total_boxes": total_boxes,
        }

    def list_images(self, project_id: str) -> list[dict[str, Any]]:
        img_dir = self._images_dir(project_id)
        if not img_dir.exists():
            return []
        meta = json.loads(self._meta_path(project_id).read_text(encoding="utf-8"))
        classes = meta.get("classes", [])
        items: list[dict[str, Any]] = []
        for p in sorted(img_dir.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            boxes = self.read_labels(project_id, p.name)
            w, h = 0, 0
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                pass
            items.append(
                {
                    "filename": p.name,
                    "url": f"/api/projects/{project_id}/images/{p.name}",
                    "width": w,
                    "height": h,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "labeled": len(boxes) > 0,
                    "box_count": len(boxes),
                    "boxes": boxes,
                    "class_names": [
                        classes[b["class_id"]] if 0 <= b["class_id"] < len(classes) else str(b["class_id"])
                        for b in boxes
                    ],
                }
            )
        return items

    def get_image_path(self, project_id: str, filename: str) -> Path:
        safe = Path(str(filename).replace("\\", "/")).name
        path = self._images_dir(project_id) / safe
        if not path.exists():
            raise FileNotFoundError(f"Image '{safe}' not found")
        return path

    def delete_image(self, project_id: str, filename: str) -> bool:
        safe = Path(str(filename).replace("\\", "/")).name
        img = self._images_dir(project_id) / safe
        lbl = self._labels_dir(project_id) / f"{Path(safe).stem}.txt"
        existed = img.exists()
        img.unlink(missing_ok=True)
        lbl.unlink(missing_ok=True)
        return existed

    def delete_images(self, project_id: str, filenames: list[str]) -> int:
        removed = 0
        for fname in filenames:
            if self.delete_image(project_id, fname):
                removed += 1
        return removed

    def clear_images(self, project_id: str) -> None:
        for d in (self._images_dir(project_id), self._labels_dir(project_id)):
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink(missing_ok=True)

    def read_labels(self, project_id: str, filename: str) -> list[dict[str, Any]]:
        lbl_path = self._labels_dir(project_id) / f"{Path(filename).stem}.txt"
        if not lbl_path.exists():
            return []
        boxes: list[dict[str, Any]] = []
        for raw_line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.002, min(1.0, w))
                h = max(0.002, min(1.0, h))
                boxes.append({"class_id": cls_id, "cx": round(cx, 6), "cy": round(cy, 6), "w": round(w, 6), "h": round(h, 6)})
            except ValueError:
                continue
        return boxes

    def save_labels(self, project_id: str, filename: str, boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._labels_dir(project_id).mkdir(parents=True, exist_ok=True)
        lbl_path = self._labels_dir(project_id) / f"{Path(filename).stem}.txt"
        cleaned: list[dict[str, Any]] = []
        lines: list[str] = []
        for b in boxes:
            cls_id = int(b["class_id"])
            cx = max(0.001, min(0.999, float(b["cx"])))
            cy = max(0.001, min(0.999, float(b["cy"])))
            w = max(0.004, min(1.0, float(b["w"])))
            h = max(0.004, min(1.0, float(b["h"])))
            # Keep box inside [0, 1]
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(1.0, cx + w / 2.0)
            y2 = min(1.0, cy + h / 2.0)
            w = max(0.004, x2 - x1)
            h = max(0.004, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cleaned.append({"class_id": cls_id, "cx": round(cx, 6), "cy": round(cy, 6), "w": round(w, 6), "h": round(h, 6)})
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            lbl_path.unlink(missing_ok=True)
        return cleaned

    def _save_normalized_image_bytes(
        self,
        project_id: str,
        raw_bytes: bytes,
        original_name: str,
        max_side: int = 1600,
    ) -> str | None:
        """Decode image, apply EXIF auto-orient, convert RGB, optionally cap huge phone resolution."""
        try:
            with Image.open(io.BytesIO(raw_bytes)) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                if max_side > 0 and max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                stem = _slugify(Path(original_name).stem) or "img"
                digest = hashlib.md5(raw_bytes[:4096] + original_name.encode("utf-8", errors="ignore")).hexdigest()[:6]
                target_name = f"{stem}_{digest}.jpg"
                out_path = self._images_dir(project_id) / target_name
                im.save(out_path, format="JPEG", quality=93)
                return target_name
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Multi-source Data Ingestion
    # ------------------------------------------------------------------

    def ingest_image_files(self, project_id: str, file_pairs: list[tuple[str, bytes]]) -> dict[str, Any]:
        added = 0
        failed = 0
        saved_names: list[str] = []
        for fname, content in file_pairs:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTS:
                failed += 1
                continue
            saved = self._save_normalized_image_bytes(project_id, content, fname)
            if saved:
                added += 1
                saved_names.append(saved)
            else:
                failed += 1
        ws_manager.log_sync(f"Đã nhập {added} ảnh vào dự án {project_id} (bỏ qua lỗi: {failed}).", source="dataset")
        return {"added": added, "failed": failed, "files": saved_names, "project": self.get_project(project_id)}

    def ingest_webcam_frame(self, project_id: str, data_url: str, label_hint: str = "cam") -> dict[str, Any]:
        if "," in data_url:
            _, b64 = data_url.split(",", 1)
        else:
            b64 = data_url
        raw = base64.b64decode(b64)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        saved = self._save_normalized_image_bytes(project_id, raw, f"{label_hint}_{stamp}.jpg")
        if not saved:
            raise ValueError("Không thể giải mã khung hình webcam.")
        return {"filename": saved, "url": f"/api/projects/{project_id}/images/{saved}", "project": self.get_project(project_id)}

    def ingest_video(
        self,
        project_id: str,
        video_bytes: bytes,
        filename: str,
        fps_extract: float = 2.0,
        max_frames: int = 75,
        skip_blurry: bool = True,
        blur_threshold: float = 40.0,
        dedup: bool = True,
    ) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            raise ValueError(f"Định dạng video chưa hỗ trợ: {ext}")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        added = 0
        skipped_blur = 0
        skipped_dup = 0
        seen_hashes: list[int] = []
        stem = _slugify(Path(filename).stem)

        try:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                raise ValueError("Không thể mở file video.")
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(native_fps / max(0.2, fps_extract))))

            idx = 0
            while added < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    if skip_blurry and _blur_score(frame) < blur_threshold:
                        skipped_blur += 1
                    else:
                        dhash = _compute_dhash(frame)
                        if dedup and any(_hamming_distance(dhash, h) <= 5 for h in seen_hashes):
                            skipped_dup += 1
                        else:
                            seen_hashes.append(dhash)
                            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 93])
                            if ok_enc:
                                name = f"{stem}_f{idx:05d}.jpg"
                                saved = self._save_normalized_image_bytes(project_id, buf.tobytes(), name)
                                if saved:
                                    added += 1
                    if total_frames > 0 and idx % (step * 5) == 0:
                        ws_manager.broadcast_sync(
                            "task_progress",
                            {
                                "task": "video_extract",
                                "progress": min(99, int(idx / max(1, total_frames) * 100)),
                                "message": f"Đang trích xuất video: đã lấy {added}/{max_frames} khung hình rõ nét...",
                            },
                        )
                idx += 1
            cap.release()
        finally:
            tmp_path.unlink(missing_ok=True)

        ws_manager.broadcast_sync(
            "task_progress",
            {"task": "video_extract", "progress": 100, "message": f"Hoàn tất trích xuất {added} ảnh từ video."},
        )
        ws_manager.log_sync(
            f"Video '{filename}': trích xuất {added} ảnh (lọc mờ: {skipped_blur}, lọc trùng: {skipped_dup}).",
            source="dataset",
        )
        return {
            "added": added,
            "skipped_blur": skipped_blur,
            "skipped_dup": skipped_dup,
            "project": self.get_project(project_id),
        }

    def ingest_zip_archive(self, project_id: str, zip_bytes: bytes) -> dict[str, Any]:
        """Smart ZIP importer: supports YOLO, COCO JSON, Pascal VOC XML, and raw image folders."""
        with tempfile.TemporaryDirectory(prefix="derich_zip_") as tmpdir:
            tmp_root = Path(tmpdir)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    rel = Path(member.filename)
                    if rel.is_absolute() or ".." in rel.parts:
                        continue
                    dest = tmp_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            return self.ingest_local_folder(project_id, str(tmp_root))

    def ingest_local_folder(self, project_id: str, folder_path: str) -> dict[str, Any]:
        """Import images + optional YOLO .txt / VOC .xml / COCO .json labels from a local folder."""
        root = Path(folder_path).expanduser()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Thư mục không tồn tại: {folder_path}")

        meta = json.loads(self._meta_path(project_id).read_text(encoding="utf-8"))
        existing_classes: list[str] = list(meta.get("classes", []))

        # Detect external class definitions (data.yaml or classes.txt)
        imported_classes: list[str] | None = None
        for yml in root.rglob("*.yaml"):
            try:
                data = yaml.safe_load(yml.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict) and "names" in data:
                    names = data["names"]
                    if isinstance(names, dict):
                        imported_classes = [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
                    elif isinstance(names, list):
                        imported_classes = [str(x) for x in names]
                    break
            except Exception:
                continue
        if imported_classes is None:
            for cfile in root.rglob("classes.txt"):
                lines = [ln.strip() for ln in cfile.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                if lines:
                    imported_classes = lines
                    break

        # Detect COCO JSON files
        coco_boxes_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        coco_classes: list[str] = []
        for jfile in root.rglob("*.json"):
            try:
                obj = json.loads(jfile.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(obj, dict) and "images" in obj and "annotations" in obj and "categories" in obj:
                    cat_map: dict[int, int] = {}
                    for cat in obj["categories"]:
                        cname = str(cat["name"]).strip()
                        if cname not in coco_classes:
                            coco_classes.append(cname)
                        cat_map[int(cat["id"])] = coco_classes.index(cname)
                    img_map = {int(im["id"]): im for im in obj["images"]}
                    for ann in obj["annotations"]:
                        im_info = img_map.get(int(ann["image_id"]))
                        if not im_info or "bbox" not in ann:
                            continue
                        iw, ih = float(im_info["width"]), float(im_info["height"])
                        if iw <= 0 or ih <= 0:
                            continue
                        x, y, bw, bh = ann["bbox"]
                        cx = (x + bw / 2.0) / iw
                        cy = (y + bh / 2.0) / ih
                        cls_idx = cat_map.get(int(ann["category_id"]), 0)
                        coco_boxes_by_name[Path(str(im_info["file_name"]).replace("\\", "/")).name].append(
                            {"class_id": cls_idx, "cx": cx, "cy": cy, "w": bw / iw, "h": bh / ih}
                        )
            except Exception:
                continue

        if imported_classes:
            classes_final = imported_classes
        elif coco_classes:
            classes_final = coco_classes
        else:
            classes_final = existing_classes

        # Index label files (.txt and .xml) by stem
        txt_by_stem: dict[str, Path] = {}
        xml_by_stem: dict[str, Path] = {}
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf == ".txt" and p.name.lower() not in {"classes.txt", "readme.txt", "requirements.txt"}:
                txt_by_stem[p.stem] = p
            elif suf == ".xml":
                xml_by_stem[p.stem] = p

        added_images = 0
        added_labels = 0
        voc_new_classes: list[str] = list(classes_final)

        for img_path in sorted(root.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            raw = img_path.read_bytes()
            saved = self._save_normalized_image_bytes(project_id, raw, img_path.name)
            if not saved:
                continue
            added_images += 1

            boxes: list[dict[str, Any]] = []
            # Priority 1: YOLO .txt
            if img_path.stem in txt_by_stem:
                for line in txt_by_stem[img_path.stem].read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            boxes.append(
                                {
                                    "class_id": int(float(parts[0])),
                                    "cx": float(parts[1]),
                                    "cy": float(parts[2]),
                                    "w": float(parts[3]),
                                    "h": float(parts[4]),
                                }
                            )
                        except ValueError:
                            continue
            # Priority 2: COCO JSON
            elif img_path.name in coco_boxes_by_name:
                boxes = coco_boxes_by_name[img_path.name]
            # Priority 3: Pascal VOC XML
            elif img_path.stem in xml_by_stem:
                try:
                    tree = ET.parse(xml_by_stem[img_path.stem])
                    xroot = tree.getroot()
                    size_el = xroot.find("size")
                    iw = float(size_el.findtext("width", "0")) if size_el is not None else 0.0
                    ih = float(size_el.findtext("height", "0")) if size_el is not None else 0.0
                    if iw > 0 and ih > 0:
                        for obj_el in xroot.findall("object"):
                            cname = (obj_el.findtext("name") or "").strip()
                            if not cname:
                                continue
                            if cname not in voc_new_classes:
                                voc_new_classes.append(cname)
                            cls_id = voc_new_classes.index(cname)
                            bb = obj_el.find("bndbox")
                            if bb is None:
                                continue
                            xmin = float(bb.findtext("xmin", "0"))
                            ymin = float(bb.findtext("ymin", "0"))
                            xmax = float(bb.findtext("xmax", "0"))
                            ymax = float(bb.findtext("ymax", "0"))
                            boxes.append(
                                {
                                    "class_id": cls_id,
                                    "cx": ((xmin + xmax) / 2.0) / iw,
                                    "cy": ((ymin + ymax) / 2.0) / ih,
                                    "w": (xmax - xmin) / iw,
                                    "h": (ymax - ymin) / ih,
                                }
                            )
                except Exception:
                    pass

            if boxes:
                self.save_labels(project_id, saved, boxes)
                added_labels += 1

        if voc_new_classes:
            self.update_classes(project_id, voc_new_classes)

        ws_manager.log_sync(
            f"Đã nhập từ nguồn dữ liệu: {added_images} ảnh, {added_labels} file nhãn (hỗ trợ YOLO/COCO/VOC).",
            source="dataset",
        )
        return {
            "added_images": added_images,
            "added_labels": added_labels,
            "classes": voc_new_classes,
            "project": self.get_project(project_id),
        }

    # ------------------------------------------------------------------
    # Built-in 3D Shaded Geometric Shapes Demo Generator
    # ------------------------------------------------------------------

    def generate_demo_shapes(
        self,
        project_id: str,
        count: int = 45,
        include_labels: bool = True,
        img_size: int = 640,
    ) -> dict[str, Any]:
        """Generate realistic 3D-shaded cube, cylinder, and sphere images for immediate testing (<100 data)."""
        rng = random.Random(2026)
        np_rng = np.random.default_rng(2026)
        count = max(9, min(120, int(count)))

        self.update_classes(project_id, ["cube", "cylinder", "sphere"])

        palette = [
            (215, 78, 62),    # crimson red (BGR converted below)
            (58, 138, 224),   # cobalt blue
            (60, 178, 113),   # emerald green
            (234, 179, 44),   # amber yellow
            (156, 98, 212),   # violet
            (225, 128, 52),   # orange
            (72, 185, 198),   # teal
            (196, 196, 204),  # matte silver
        ]

        def make_bg(h: int, w: int) -> np.ndarray:
            c1 = np.array([rng.randint(185, 245), rng.randint(185, 245), rng.randint(185, 245)], dtype=np.float32)
            c2 = np.array([rng.randint(145, 225), rng.randint(145, 225), rng.randint(145, 225)], dtype=np.float32)
            if rng.random() < 0.3:
                # darker studio tabletop
                c1 *= 0.45
                c2 *= 0.35
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            t = (xx / w * rng.uniform(0.3, 0.8) + yy / h * rng.uniform(0.3, 0.8))
            t = np.clip(t, 0.0, 1.0)[..., None]
            bg = c1 * (1.0 - t) + c2 * t
            noise = np_rng.normal(0, rng.uniform(2.0, 6.0), (h, w, 3)).astype(np.float32)
            bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
            return bg

        def draw_shadow(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> None:
            overlay = canvas.copy()
            cv2.ellipse(overlay, (cx + int(rx * 0.14), cy + int(ry * 0.18)), (int(rx * 1.05), int(ry * 0.55)), 0, 0, 360, (22, 22, 26), -1)
            cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0, canvas)

        def draw_sphere(canvas: np.ndarray, cx: int, cy: int, r: int, rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
            draw_shadow(canvas, cx, cy + int(r * 0.78), int(r * 0.85), int(r * 0.38))
            y1, y2 = max(0, cy - r - 2), min(canvas.shape[0], cy + r + 2)
            x1, x2 = max(0, cx - r - 2), min(canvas.shape[1], cx + r + 2)
            yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
            dx = (xx - cx) / max(1, r)
            dy = (yy - cy) / max(1, r)
            r2 = dx * dx + dy * dy
            mask = r2 <= 1.0
            dz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
            lx, ly, lz = -0.45, -0.45, 0.77
            diff = np.clip(dx * lx + dy * ly + dz * lz, 0.0, 1.0)
            spec = np.power(np.clip(dx * (-0.3) + dy * (-0.3) + dz * 0.9, 0.0, 1.0), 18.0) * 0.48
            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
            shaded = np.clip((0.28 + 0.78 * diff[..., None]) * bgr + spec[..., None] * 255.0, 0, 255).astype(np.uint8)
            patch = canvas[y1:y2, x1:x2]
            patch[mask] = shaded[mask]
            return cx - r, cy - r, cx + r, cy + r

        def draw_cube(canvas: np.ndarray, cx: int, cy: int, size: int, rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
            half = size // 2
            dx = int(size * rng.uniform(0.24, 0.36))
            dy = int(size * rng.uniform(0.22, 0.34))
            sign = rng.choice([-1, 1])
            dx *= sign
            draw_shadow(canvas, cx, cy + half + 4, int(size * 0.65), int(size * 0.25))

            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
            front_col = tuple(int(v) for v in np.clip(bgr * 0.92, 0, 255))
            top_col = tuple(int(v) for v in np.clip(bgr * 1.15 + 18, 0, 255))
            side_col = tuple(int(v) for v in np.clip(bgr * 0.62, 0, 255))
            edge_col = tuple(int(v) for v in np.clip(bgr * 0.35, 0, 255))

            fx1, fy1 = cx - half, cy - half + dy // 2
            fx2, fy2 = cx + half, cy + half + dy // 2
            front = np.array([[fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]], dtype=np.int32)
            top = np.array([[fx1, fy1], [fx2, fy1], [fx2 + dx, fy1 - dy], [fx1 + dx, fy1 - dy]], dtype=np.int32)
            if dx > 0:
                side = np.array([[fx2, fy1], [fx2 + dx, fy1 - dy], [fx2 + dx, fy2 - dy], [fx2, fy2]], dtype=np.int32)
            else:
                side = np.array([[fx1, fy1], [fx1 + dx, fy1 - dy], [fx1 + dx, fy2 - dy], [fx1, fy2]], dtype=np.int32)

            cv2.fillConvexPoly(canvas, top, top_col, cv2.LINE_AA)
            cv2.fillConvexPoly(canvas, side, side_col, cv2.LINE_AA)
            cv2.fillConvexPoly(canvas, front, front_col, cv2.LINE_AA)
            for poly in (top, side, front):
                cv2.polylines(canvas, [poly], True, edge_col, 2, cv2.LINE_AA)

            all_pts = np.vstack([front, top, side])
            return int(all_pts[:, 0].min()), int(all_pts[:, 1].min()), int(all_pts[:, 0].max()), int(all_pts[:, 1].max())

        def draw_cylinder(canvas: np.ndarray, cx: int, cy: int, radius: int, height: int, rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
            rx = radius
            ry = max(8, int(radius * rng.uniform(0.30, 0.44)))
            top_y = cy - height // 2
            bot_y = cy + height // 2
            draw_shadow(canvas, cx, bot_y + ry // 2, int(rx * 1.15), int(ry * 0.9))

            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
            # Cylindrical horizontal shading gradient
            x1, x2 = max(0, cx - rx), min(canvas.shape[1], cx + rx)
            y1, y2 = max(0, top_y), min(canvas.shape[0], bot_y)
            if x2 > x1 and y2 > y1:
                xs = np.linspace(-1.0, 1.0, x2 - x1, dtype=np.float32)
                nx = xs
                nz = np.sqrt(np.clip(1.0 - nx * nx, 0.0, 1.0))
                diff = np.clip(nx * (-0.45) + nz * 0.88, 0.0, 1.0)
                spec = np.power(np.clip(nx * (-0.25) + nz * 0.96, 0.0, 1.0), 14.0) * 0.38
                strip = np.clip((0.28 + 0.76 * diff[:, None]) * bgr[None, :] + spec[:, None] * 255.0, 0, 255).astype(np.uint8)
                canvas[y1:y2, x1:x2] = strip[None, :, :]

            # Bottom rounded cap
            bot_col = tuple(int(v) for v in np.clip(bgr * 0.68, 0, 255))
            top_col = tuple(int(v) for v in np.clip(bgr * 1.14 + 14, 0, 255))
            edge_col = tuple(int(v) for v in np.clip(bgr * 0.36, 0, 255))
            cv2.ellipse(canvas, (cx, bot_y), (rx, ry), 0, 0, 180, bot_col, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, bot_y), (rx, ry), 0, 0, 180, edge_col, 2, cv2.LINE_AA)
            # Top ellipse cap
            cv2.ellipse(canvas, (cx, top_y), (rx, ry), 0, 0, 360, top_col, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, top_y), (rx, ry), 0, 0, 360, edge_col, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx - rx, top_y), (cx - rx, bot_y), edge_col, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx + rx, top_y), (cx + rx, bot_y), edge_col, 2, cv2.LINE_AA)
            return cx - rx, top_y - ry, cx + rx, bot_y + ry

        created = 0
        for i in range(count):
            img = make_bg(img_size, img_size)
            n_objs = 1 if i < count // 2 else rng.choice([1, 2, 2, 3])
            placed_rects: list[tuple[int, int, int, int]] = []
            boxes: list[dict[str, Any]] = []

            for k in range(n_objs):
                cls_id = (i + k) % 3  # balanced cube=0, cylinder=1, sphere=2
                color = rng.choice(palette)
                for _attempt in range(25):
                    if cls_id == 0:
                        sz = rng.randint(85, 165)
                        cx = rng.randint(115, img_size - 115)
                        cy = rng.randint(125, img_size - 115)
                        cand = (cx - sz, cy - sz, cx + sz, cy + sz)
                    elif cls_id == 1:
                        rad = rng.randint(36, 68)
                        ht = rng.randint(110, 200)
                        cx = rng.randint(95, img_size - 95)
                        cy = rng.randint(135, img_size - 135)
                        cand = (cx - rad - 15, cy - ht // 2 - 30, cx + rad + 15, cy + ht // 2 + 30)
                    else:
                        rad = rng.randint(48, 92)
                        cx = rng.randint(105, img_size - 105)
                        cy = rng.randint(105, img_size - 105)
                        cand = (cx - rad - 12, cy - rad - 12, cx + rad + 12, cy + rad + 12)

                    overlap = False
                    for r in placed_rects:
                        if not (cand[2] < r[0] or cand[0] > r[2] or cand[3] < r[1] or cand[1] > r[3]):
                            overlap = True
                            break
                    if not overlap:
                        if cls_id == 0:
                            x1, y1, x2, y2 = draw_cube(img, cx, cy, sz, color)
                        elif cls_id == 1:
                            x1, y1, x2, y2 = draw_cylinder(img, cx, cy, rad, ht, color)
                        else:
                            x1, y1, x2, y2 = draw_sphere(img, cx, cy, rad, color)
                        placed_rects.append((x1 - 12, y1 - 12, x2 + 12, y2 + 12))
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_size, x2), min(img_size, y2)
                        boxes.append(
                            {
                                "class_id": cls_id,
                                "cx": ((x1 + x2) / 2.0) / img_size,
                                "cy": ((y1 + y2) / 2.0) / img_size,
                                "w": (x2 - x1) / img_size,
                                "h": (y2 - y1) / img_size,
                            }
                        )
                        break

            # Slight realistic camera smoothing
            if rng.random() < 0.35:
                img = cv2.GaussianBlur(img, (3, 3), 0.6)

            fname = f"shape_demo_{i + 1:03d}.jpg"
            out_img = self._images_dir(project_id) / fname
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if include_labels:
                self.save_labels(project_id, fname, boxes)
            else:
                (self._labels_dir(project_id) / f"{Path(fname).stem}.txt").unlink(missing_ok=True)
            created += 1

        ws_manager.log_sync(
            f"Đã tạo bộ ảnh mẫu {created} hình khối 3D (cube, cylinder, sphere) — {'có sẵn nhãn' if include_labels else 'chưa gán nhãn để thử Auto-Label'}.",
            source="dataset",
        )
        return {"created": created, "include_labels": include_labels, "project": self.get_project(project_id)}

    # ------------------------------------------------------------------
    # Auto-Preprocessing & Health Diagnostics
    # ------------------------------------------------------------------

    def preprocess_project(
        self,
        project_id: str,
        remove_duplicates: bool = True,
        remove_blurry: bool = False,
        blur_threshold: float = 35.0,
        max_side: int = 1280,
        apply_clahe: bool = False,
        clean_labels: bool = True,
    ) -> dict[str, Any]:
        """Automatic dataset cleaning, deduplication, resolution normalization, contrast fix, and label sanitization."""
        img_dir = self._images_dir(project_id)
        meta = json.loads(self._meta_path(project_id).read_text(encoding="utf-8"))
        num_classes = len(meta.get("classes", []))

        files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS])
        report = {
            "total_scanned": len(files),
            "removed_corrupt": 0,
            "removed_duplicates": 0,
            "removed_blurry": 0,
            "resized_count": 0,
            "enhanced_count": 0,
            "labels_cleaned": 0,
        }
        seen_hashes: list[int] = []

        for idx, img_path in enumerate(files):
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                self.delete_image(project_id, img_path.name)
                report["removed_corrupt"] += 1
                continue

            if remove_duplicates:
                dh = _compute_dhash(img)
                if any(_hamming_distance(dh, h) <= 4 for h in seen_hashes):
                    self.delete_image(project_id, img_path.name)
                    report["removed_duplicates"] += 1
                    continue
                seen_hashes.append(dh)

            if remove_blurry and _blur_score(img) < blur_threshold:
                self.delete_image(project_id, img_path.name)
                report["removed_blurry"] += 1
                continue

            modified = False
            h, w = img.shape[:2]
            if max_side > 0 and max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                report["resized_count"] += 1
                modified = True

            if apply_clahe:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
                l_eq = clahe.apply(l_ch)
                img = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
                report["enhanced_count"] += 1
                modified = True

            if modified:
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 93])

            if clean_labels:
                boxes = self.read_labels(project_id, img_path.name)
                valid = [b for b in boxes if 0 <= b["class_id"] < max(1, num_classes) and b["w"] >= 0.005 and b["h"] >= 0.005]
                if len(valid) != len(boxes):
                    report["labels_cleaned"] += 1
                if valid or boxes:
                    self.save_labels(project_id, img_path.name, valid)

            if idx % 10 == 0 or idx == len(files) - 1:
                ws_manager.broadcast_sync(
                    "task_progress",
                    {
                        "task": "preprocess",
                        "progress": int((idx + 1) / max(1, len(files)) * 100),
                        "message": f"Đang tiền xử lý tự động {idx + 1}/{len(files)} ảnh...",
                    },
                )

        ws_manager.log_sync(
            f"Tiền xử lý hoàn tất: quét {report['total_scanned']} ảnh | xóa lỗi {report['removed_corrupt']} | "
            f"xóa trùng {report['removed_duplicates']} | xóa mờ {report['removed_blurry']} | resize {report['resized_count']}.",
            source="dataset",
        )
        report["project"] = self.get_project(project_id)
        return report

    def analyze_health(self, project_id: str) -> dict[str, Any]:
        """Inspect dataset readiness and produce concrete recommendations for <100-image training."""
        meta = self.get_project(project_id)
        classes: list[str] = meta.get("classes", [])
        images = self.list_images(project_id)

        class_counts: Counter[int] = Counter()
        blur_warnings = 0
        for im in images:
            for b in im["boxes"]:
                class_counts[b["class_id"]] += 1

        # Quick blur sample on up to 60 images
        img_dir = self._images_dir(project_id)
        for im in images[:60]:
            arr = cv2.imread(str(img_dir / im["filename"]))
            if arr is not None and _blur_score(arr) < 35.0:
                blur_warnings += 1

        per_class = [
            {"id": idx, "name": cname, "count": int(class_counts.get(idx, 0))}
            for idx, cname in enumerate(classes)
        ]
        total = len(images)
        labeled = sum(1 for im in images if im["labeled"])
        unlabeled = total - labeled

        warnings: list[str] = []
        recommendations: list[str] = []
        if total == 0:
            warnings.append("Chưa có ảnh nào. Hãy tải ảnh, trích xuất video, chụp webcam hoặc bấm 'Tạo bộ ảnh mẫu Hình khối 3D'.")
        elif labeled < 6:
            warnings.append(f"Mới có {labeled} ảnh đã gán nhãn (tối thiểu nên có ≥10 ảnh). Hãy chạy 'Tự động đánh dấu' trước khi train.")
        if unlabeled > 0:
            warnings.append(f"Còn {unlabeled} ảnh chưa được đánh dấu bounding box — bạn có thể bấm Tự động đánh dấu để gán nhãn hàng loạt.")
        if 0 < labeled < 100:
            recommendations.append(
                f"Tập dữ liệu nhỏ ({labeled} ảnh < 100 ảnh): Hệ thống đã tự động bật Preset Small-Data (YOLO26n + freeze=10 + AdamW lr=0.001 + nhân bản tập train 3x + xoay/phối cảnh 3D)."
            )
        if blur_warnings > 0:
            warnings.append(f"Phát hiện {blur_warnings} ảnh có độ sắc nét thấp (mờ) — có thể bật 'Tiền xử lý tự động' để lọc bớt.")

        nonzero = [c["count"] for c in per_class if c["count"] > 0]
        if nonzero and max(nonzero) / max(1, min(nonzero)) >= 2.5:
            warnings.append("Phân bố nhãn giữa các lớp bị lệch (>2.5x). Khuyến nghị giữ cls_pw=0.5 để cân bằng loss tự động.")
        empty_classes = [c["name"] for c in per_class if c["count"] == 0]
        if labeled > 0 and empty_classes:
            warnings.append(f"Các lớp chưa có mẫu nhãn nào: {', '.join(empty_classes)}.")

        if labeled < 30:
            regime = "ultra_small"
            regime_label = "Siêu ít ảnh (<30 ảnh) — Nên dùng Nano + freeze=10 + nhân bản 4x"
        elif labeled < 100:
            regime = "small"
            regime_label = "Ít ảnh (<100 ảnh) — Tối ưu mặc định Small-Data đã sẵn sàng"
        else:
            regime = "medium"
            regime_label = "Đủ dữ liệu (≥100 ảnh) — Có thể dùng Nano hoặc Small"

        return {
            "total_images": total,
            "labeled_images": labeled,
            "unlabeled_images": unlabeled,
            "total_boxes": sum(class_counts.values()),
            "per_class": per_class,
            "blur_warnings": blur_warnings,
            "regime": regime,
            "regime_label": regime_label,
            "warnings": warnings,
            "recommendations": recommendations,
            "ready_to_train": labeled >= 4,
        }

    # ------------------------------------------------------------------
    # Stratified Split & Leak-Free Offline Augmentation Expansion
    # ------------------------------------------------------------------

    def _augment_train_pair(
        self,
        img_bgr: np.ndarray,
        boxes: list[dict[str, Any]],
        rng: random.Random,
        variant_idx: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Generate a label-preserving photographic/geometric variant strictly for train split."""
        out = img_bgr.copy()
        new_boxes = [dict(b) for b in boxes]

        # Horizontal flip
        if variant_idx % 2 == 1 or rng.random() < 0.5:
            out = cv2.flip(out, 1)
            for b in new_boxes:
                b["cx"] = round(1.0 - b["cx"], 6)

        # Brightness & contrast jitter (simulates indoor/outdoor lighting shifts)
        alpha = rng.uniform(0.75, 1.28)
        beta = rng.uniform(-24, 24)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        # Mild HSV saturation / value shift
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-6, 6)) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.72, 1.30), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.80, 1.22), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Occasional mild blur or sharpen or camera noise
        roll = rng.random()
        if roll < 0.30:
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.4, 1.1))
        elif roll < 0.60:
            noise = np.random.default_rng(rng.randint(0, 999999)).normal(0, rng.uniform(3.0, 9.0), out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif roll < 0.80:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
            out = cv2.filter2D(out, -1, kernel)

        return out, new_boxes

    def prepare_yolo_dataset(
        self,
        project_id: str,
        val_split: float = 0.2,
        offline_aug_multiplier: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Create stratified train/val folders + data.yaml, expanding ONLY train split so val stays 100% leak-free."""
        meta = self.get_project(project_id)
        classes: list[str] = meta.get("classes", [])
        img_dir = self._images_dir(project_id)

        labeled_items: list[tuple[Path, list[dict[str, Any]]]] = []
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                boxes = self.read_labels(project_id, p.name)
                if boxes:
                    labeled_items.append((p, boxes))

        if len(labeled_items) < 4:
            raise ValueError(
                f"Cần tối thiểu 4 ảnh đã gắn nhãn để huấn luyện (hiện có {len(labeled_items)} ảnh có nhãn)."
            )

        rng = random.Random(seed)
        # Group by primary (rarest) class for stratified split
        by_class: dict[int, list[tuple[Path, list[dict[str, Any]]]]] = defaultdict(list)
        for item in labeled_items:
            primary_cls = item[1][0]["class_id"]
            by_class[primary_cls].append(item)

        train_orig: list[tuple[Path, list[dict[str, Any]]]] = []
        val_orig: list[tuple[Path, list[dict[str, Any]]]] = []

        for _cls_id, group in by_class.items():
            rng.shuffle(group)
            if len(group) == 1:
                train_orig.extend(group)
            else:
                n_val = max(1, int(round(len(group) * val_split)))
                n_val = min(n_val, len(group) - 1)
                val_orig.extend(group[:n_val])
                train_orig.extend(group[n_val:])

        # Safety guarantee: at least 1 val image
        if not val_orig and len(train_orig) >= 2:
            val_orig.append(train_orig.pop())

        prep = self._prepared_dir(project_id)
        if prep.exists():
            shutil.rmtree(prep, ignore_errors=True)
        for split in ("train", "val"):
            (prep / split / "images").mkdir(parents=True, exist_ok=True)
            (prep / split / "labels").mkdir(parents=True, exist_ok=True)

        def write_pair(split: str, stem: str, img_bgr: np.ndarray, boxes: list[dict[str, Any]]) -> None:
            cv2.imwrite(str(prep / split / "images" / f"{stem}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
            lines = [f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes]
            (prep / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Write untouched validation split
        for img_path, boxes in val_orig:
            arr = cv2.imread(str(img_path))
            if arr is not None:
                write_pair("val", img_path.stem, arr, boxes)

        # Write train split + leak-free offline expansion
        mult = max(1, min(6, int(offline_aug_multiplier)))
        # Auto-cap expansion if dataset is already large (>150 images)
        if len(train_orig) >= 150 and mult > 2:
            mult = 1

        train_total = 0
        for img_path, boxes in train_orig:
            arr = cv2.imread(str(img_path))
            if arr is None:
                continue
            write_pair("train", img_path.stem, arr, boxes)
            train_total += 1
            for v in range(1, mult):
                aug_img, aug_boxes = self._augment_train_pair(arr, boxes, rng, v)
                write_pair("train", f"{img_path.stem}_aug{v}", aug_img, aug_boxes)
                train_total += 1

        yaml_data = {
            "path": str(prep.resolve()),
            "train": "train/images",
            "val": "val/images",
            "names": {idx: name for idx, name in enumerate(classes)},
        }
        yaml_path = prep / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

        return {
            "yaml_path": str(yaml_path),
            "train_original": len(train_orig),
            "train_effective": train_total,
            "val_count": len(val_orig),
            "offline_multiplier": mult,
            "classes": classes,
        }


dataset_manager = DatasetManager()
