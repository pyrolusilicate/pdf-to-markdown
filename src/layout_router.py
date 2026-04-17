import argparse
import json
import os

import cv2
import fitz
import numpy as np
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image


class LayoutRouter:
    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        self.device = self._get_device()
        self.model = self._load_model()

        # Мусор, который мы игнорируем согласно заданию
        self.ignore_classes = {
            "page-header",
            "page-footer",
            "footnote",
            "watermark",
            "abandon",
        }

        # Кандидаты на VLM (фотографии, рисунки)
        self.vlm_candidates = {"picture", "figure", "image"}

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> YOLOv10:
        os.makedirs(self.weights_dir, exist_ok=True)
        weights_path = os.path.join(
            self.weights_dir, "doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        if not os.path.exists(weights_path):
            print("📥 Скачивание весов Layout...")
            hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                local_dir=self.weights_dir,
            )
        return YOLOv10(weights_path)

    def get_document_id(self, filename: str) -> str:
        base_name = os.path.basename(filename)
        try:
            return str(int(base_name.replace("document_", "").replace(".pdf", "")))
        except ValueError:
            return base_name.replace(".pdf", "")

    def _sort_reading_order(self, boxes: list, page_width: float) -> list:
        # Сортировка для многоколоночной верстки (arxiv_columns)
        parsed_boxes = []
        for box in boxes:
            coords = box.xyxy[0].tolist()
            parsed_boxes.append(
                {
                    "box": box,
                    "y1": coords[1],
                    "w": coords[2] - coords[0],
                    "center_x": (coords[0] + coords[2]) / 2,
                }
            )
        full_width, left_col, right_col = [], [], []
        for item in parsed_boxes:
            if item["w"] > page_width * 0.6:
                full_width.append(item)
            elif item["center_x"] < page_width / 2:
                left_col.append(item)
            else:
                right_col.append(item)
        full_width.sort(key=lambda x: x["y1"])
        left_col.sort(key=lambda x: x["y1"])
        right_col.sort(key=lambda x: x["y1"])
        return [item["box"] for item in (full_width + left_col + right_col)]

    def _crop_image(
        self, img: Image.Image, coords: list, padding: int = 0
    ) -> Image.Image:
        x1 = max(0, coords[0] - padding)
        y1 = max(0, coords[1] - padding)
        x2 = min(img.width, coords[2] + padding)
        y2 = min(img.height, coords[3] + padding)
        return img.crop((x1, y1, x2, y2))

    def build_routing_plan(
        self, pdf_path: str, output_dir: str = "data/output", visualize: bool = False
    ) -> dict:
        doc_id = self.get_document_id(pdf_path)
        temp_dir = os.path.join(output_dir, "temp", f"document_{doc_id}")
        os.makedirs(temp_dir, exist_ok=True)

        vis_dir = None
        if visualize:
            vis_dir = os.path.join("data", "visualization", f"document_{doc_id}")
            os.makedirs(vis_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        routing_plan = {"doc_id": doc_id, "pages": []}
        global_image_counter = 1
        scale_factor = 72 / 300  # Для конвертации 300 DPI в 72 DPI PDF

        print(f"\n🗺️ Анализ: {os.path.basename(pdf_path)} (ID: {doc_id})")

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            img_cv2 = cv2.cvtColor(
                img_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
            )
            img_rgb = (
                cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) if pix.n == 4 else img_array
            )
            pil_img = Image.fromarray(img_rgb)

            results = self.model.predict(
                img_cv2, imgsz=1024, conf=0.2, device=self.device, verbose=False
            )[0]

            if visualize and vis_dir:
                annotated_frame = results.plot(pil=True, line_width=2, font_size=12)
                cv2.imwrite(
                    os.path.join(vis_dir, f"page_{page_num + 1}.jpg"), annotated_frame
                )

            sorted_boxes = self._sort_reading_order(results.boxes, pil_img.width)
            page_plan = {"page_num": page_num + 1, "blocks": []}

            for box in sorted_boxes:
                coords = [int(c) for c in box.xyxy[0].tolist()]
                label = self.model.names[int(box.cls[0])].lower()

                if label in self.ignore_classes:
                    continue

                # --- ПРОВЕРКА РАСТРА ВНУТРИ РАМКИ ---
                clip_rect = fitz.Rect(
                    coords[0] * scale_factor,
                    coords[1] * scale_factor,
                    coords[2] * scale_factor,
                    coords[3] * scale_factor,
                )
                clip_dict = page.get_text("dict", clip=clip_rect)

                text_chars = 0
                image_area = 0
                box_area = clip_rect.get_area()

                for b in clip_dict.get("blocks", []):
                    if b["type"] == 0:  # Текстовый слой
                        for line in b.get("lines", []):
                            for span in line.get("spans", []):
                                text_chars += len(span.get("text", "").strip())
                    elif b["type"] == 1:  # Картинка в PDF
                        img_rect = fitz.Rect(b["bbox"])
                        image_area += clip_rect.intersect(img_rect).get_area()

                # Если текста < 5 символов или картинка занимает > 50% площади - это растр
                is_raster = (text_chars < 5) or (image_area > box_area * 0.5)

                block = {
                    "type": label,
                    "coords": coords,
                    "track": "PADDLE_OCR"
                    if (is_raster or label in self.vlm_candidates)
                    else "DOCLING_TEXT",
                    "content_path": None,
                    "md_image_name": None,
                }

                if label == "table" and not is_raster:
                    block["track"] = "DOCLING_TABLE"

                # Сохранение кропов для VLM или таблиц
                if block["track"] in ["PADDLE_OCR", "DOCLING_TABLE"]:
                    prefix = (
                        "table" if block["track"] == "DOCLING_TABLE" else "candidate"
                    )
                    fname = (
                        f"{prefix}_{global_image_counter}.png"
                        if prefix == "candidate"
                        else f"table_p{page_num + 1}_{coords[1]}.png"
                    )
                    temp_path = os.path.join(temp_dir, fname)
                    self._crop_image(
                        pil_img, coords, padding=10 if label == "table" else 0
                    ).save(temp_path)

                    block["content_path"] = temp_path
                    if block["track"] == "PADDLE_OCR":
                        block["md_image_name"] = (
                            f"doc_{doc_id}_image_{global_image_counter}.png"
                        )
                        global_image_counter += 1

                page_plan["blocks"].append(block)
            routing_plan["pages"].append(page_plan)

        doc.close()
        return routing_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layout Router CLI")
    parser.add_argument("--pdf", type=str, required=True, help="Путь к PDF")
    parser.add_argument(
        "--visualize", action="store_true", help="Включить визуализацию"
    )
    args = parser.parse_args()

    router = LayoutRouter()
    plan = router.build_routing_plan(args.pdf, visualize=args.visualize)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
