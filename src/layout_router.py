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

from config import LAYOUT_DPI, LAYOUT_TO_PDF, PDF_TO_LAYOUT
from device import get_torch_device, setup_environment

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(
    *args, **{**kwargs, "weights_only": False}
)

setup_environment()

weights_name_file = "doclayout_yolo_docstructbench_imgsz1280_2501.pt"


class LayoutRouter:
    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        self.device = get_torch_device()
        self.model = self._load_model()
        self.ignore_classes = {
            "page-header",
            "page-footer",
            "footnote",
            "watermark",
            "abandon",
        }
        self.vlm_candidates = {"picture", "figure", "image"}
        # Приоритет при NMS: при перекрытии оставляем класс с бо́льшим приоритетом
        self._class_priority = {
            "title": 10,
            "section-header": 9,
            "table": 8,
            "figure": 7,
            "picture": 7,
            "table_caption": 6,
            "figure_caption": 6,
            "caption": 6,
            "list-item": 5,
            "text": 4,
            "plain text": 4,
        }

    def _load_model(self) -> YOLOv10:
        os.makedirs(self.weights_dir, exist_ok=True)
        weights_path = os.path.join(self.weights_dir, weights_name_file)
        if not os.path.exists(weights_path):
            hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename=weights_name_file,
                local_dir=self.weights_dir,
            )
        return YOLOv10(weights_path)

    def get_document_id(self, filename: str) -> str:
        base_name = os.path.basename(filename)
        try:
            return str(int(base_name.replace("document_", "").replace(".pdf", "")))
        except ValueError:
            return base_name.replace(".pdf", "")

    def _sort_reading_order(
        self, boxes: list, page_width: float, page_height: float
    ) -> list:
        if not boxes:
            return []

        parsed_boxes = []
        for box in boxes:
            coords = box.xyxy[0].tolist()
            cls_name = self.model.names[int(box.cls[0])].lower()
            parsed_boxes.append(
                {
                    "box": box,
                    "x1": coords[0],
                    "y1": coords[1],
                    "x2": coords[2],
                    "y2": coords[3],
                    "w": coords[2] - coords[0],
                    "h": coords[3] - coords[1],
                    "cls": cls_name,
                }
            )

        # 1. Фильтрация колонтитулов
        margin_top = page_height * 0.05
        margin_bottom = page_height * 0.95
        filtered_boxes = []
        for b in parsed_boxes:
            if b["cls"] in ["page-header", "page-footer"]:
                continue
            if b["cls"] in ["text", "plain text", "title"]:
                if b["y2"] < margin_top or b["y1"] > margin_bottom:
                    continue
            filtered_boxes.append(b)

        # Первичная сортировка сверху вниз
        filtered_boxes.sort(key=lambda x: x["y1"])

        # 2. ЛОГИЧЕСКАЯ ПРЕ-СКЛЕЙКА (Figure/Table + Caption)
        # Связываем медиа-объекты с их подписями в единые неразрывные блоки
        logical_blocks = []
        used_indices = set()

        for i, b in enumerate(filtered_boxes):
            if i in used_indices:
                continue

            current_logical = [b]
            used_indices.add(i)

            base_box = b
            for j in range(i + 1, len(filtered_boxes)):
                if j in used_indices:
                    continue
                next_box = filtered_boxes[j]

                # Расстояние по вертикали
                y_gap = next_box["y1"] - base_box["y2"]

                # Доля пересечения по X
                overlap_x = max(
                    0,
                    min(base_box["x2"], next_box["x2"])
                    - max(base_box["x1"], next_box["x1"]),
                )
                min_w = min(base_box["w"], next_box["w"])
                x_ratio = overlap_x / min_w if min_w > 0 else 0

                is_parent_media = base_box["cls"] in [
                    "figure",
                    "picture",
                    "image",
                    "table",
                    "table_merged",
                    "table_borderless",
                ]
                # Расширяем: клеим и caption, и обычный текст, если он явно работает как подпись
                is_valid_child = next_box["cls"] in [
                    "figure_caption",
                    "table_caption",
                    "caption",
                    "text",
                    "plain text",
                ]

                # Если блок находится близко под картинкой/таблицей и сильно выровнен по ширине
                if y_gap < page_height * 0.08 and x_ratio > 0.5:
                    if is_parent_media and is_valid_child:
                        current_logical.append(next_box)
                        used_indices.add(j)
                        base_box = (
                            next_box  # Сдвигаем низ для возможной многострочной подписи
                        )
                    else:
                        break  # Обычные абзацы не склеиваем жестко
                elif y_gap >= page_height * 0.08:
                    pass  # Ушли слишком далеко вниз

            # Формируем габариты объединенного логического блока
            logical_blocks.append(
                {
                    "boxes": current_logical,
                    "x1": min(cb["x1"] for cb in current_logical),
                    "y1": min(cb["y1"] for cb in current_logical),
                    "x2": max(cb["x2"] for cb in current_logical),
                    "y2": max(cb["y2"] for cb in current_logical),
                    "w": max(cb["x2"] for cb in current_logical)
                    - min(cb["x1"] for cb in current_logical),
                    "h": max(cb["y2"] for cb in current_logical)
                    - min(cb["y1"] for cb in current_logical),
                }
            )

        # 3. Формирование горизонтальных полос (Bands) на основе Y-проекции
        bands = []
        current_band = []
        band_bottom = 0

        for lb in logical_blocks:
            lb_width = lb["w"]
            # Широкие блоки (> 55% страницы) или заголовки всегда разрывают полосу
            is_full_width = lb_width > page_width * 0.55
            is_title_class = any(
                b["cls"] in ["title", "section-header"] for b in lb["boxes"]
            )

            if is_full_width or is_title_class:
                # Если уже что-то копили - закрываем старую полосу
                if current_band:
                    bands.append(current_band)
                # Широкий элемент идет своей собственной полосой
                bands.append([lb])
                current_band = []
                band_bottom = 0
            else:
                if not current_band:
                    current_band.append(lb)
                    band_bottom = lb["y2"]
                else:
                    # Если верхняя граница блока лежит выше "дна" текущей полосы 
                    # (с допуском 3% высоты страницы на кривизну скана), он в той же полосе.
                    # Это позволяет собирать длинные и короткие колонки вместе!
                    if lb["y1"] <= band_bottom + (page_height * 0.03):
                        current_band.append(lb)
                        # Расширяем "дно" полосы
                        band_bottom = max(band_bottom, lb["y2"])
                    else:
                        # Ушли слишком далеко вниз - начинаем новую полосу
                        bands.append(current_band)
                        current_band = [lb]
                        band_bottom = lb["y2"]
        
        if current_band:
            bands.append(current_band)

        # 4. Формирование колонок внутри полос и итоговая распаковка
        final_sorted = []
        for band in bands:
            columns = []
            
            for lb in band:
                placed = False
                for col in columns:
                    col_x1 = min(cb["x1"] for cb in col)
                    col_x2 = max(cb["x2"] for cb in col)
                    col_w = col_x2 - col_x1
                    
                    overlap = max(0, min(lb["x2"], col_x2) - max(lb["x1"], col_x1))
                    min_w = min(lb["w"], col_w)
                    
                    # Если блок пересекается с существующей колонкой по оси X хотя бы на 30%
                    if min_w > 0 and (overlap / min_w) > 0.3:
                        col.append(lb)
                        placed = True
                        break
                
                # Если не пересекается ни с одной - это новая колонка
                if not placed:
                    columns.append([lb])
            
            # Сортируем колонки строго слева направо
            columns.sort(key=lambda col: min(cb["x1"] for cb in col))

            for col in columns:
                # Внутри каждой колонки сортируем блоки строго сверху вниз
                col.sort(key=lambda lb: lb["y1"])

                for lb in col:
                    # Распаковываем физические боксы (они уже отсортированы логикой на Шаге 2)
                    for phys_box in lb["boxes"]:
                        final_sorted.append(phys_box["box"])

        return final_sorted

    @staticmethod
    def _cluster_centers(centers: list[float], gap: float) -> list[dict]:
        """
        Жадная 1D-кластеризация: соседние точки, расстояние между которыми
        меньше `gap`, идут в один кластер. Возвращает [{center, count}, ...].
        """
        if not centers:
            return []
        clusters: list[list[float]] = [[centers[0]]]
        for c in centers[1:]:
            if c - clusters[-1][-1] <= gap:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [{"center": sum(cl) / len(cl), "count": len(cl)} for cl in clusters]

    def _apply_nms(self, boxes: list, iom_threshold: float = 0.7) -> list:
        """
        Intersection-over-Min-Area NMS: удаляет боксы, у которых
        >iom_threshold площади перекрывается с уже принятым боксом.
        Приоритет: класс-приоритет, затем confidence.
        """
        if not boxes:
            return boxes

        def priority(box):
            cls = self.model.names[int(box.cls[0])].lower()
            base_prio = self._class_priority.get(cls, 3)
            
            # --- УБИЙЦА ФЕЙКОВЫХ ФИГУР ---
            # Если YOLO придумал картинку/фигуру, но вес мусорный (< 0.4),
            # роняем ее приоритет до 2 (ниже, чем у plain text). 
            if cls in {"figure", "picture", "image"} and float(box.conf[0]) < 0.4:
                base_prio = 2
                
            return base_prio, float(box.conf[0])

        sorted_boxes = sorted(boxes, key=priority, reverse=True)
        kept: list = []

        for box in sorted_boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area == 0:
                continue

            dominated = False
            for kb in kept:
                kx1, ky1, kx2, ky2 = kb.xyxy[0].tolist()
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                min_area = min(area, k_area)
                if min_area > 0 and inter / min_area > iom_threshold:
                    dominated = True
                    break

            if not dominated:
                kept.append(box)

        return kept

    def _crop_image(
        self, img: Image.Image, coords: list, padding: int = 0
    ) -> Image.Image:
        x1 = max(0, coords[0] - padding)
        y1 = max(0, coords[1] - padding)
        x2 = min(img.width, coords[2] + padding)
        y2 = min(img.height, coords[3] + padding)
        return img.crop((x1, y1, x2, y2))
    
    @staticmethod
    def _is_image_noisy(img_gray: np.ndarray, noise_threshold: int = 3000) -> bool:
        """
        Быстрый детектор точечного шума на основе анализа связных компонент.
        Если на картинке больше `noise_threshold` микро-точек, она считается шумной.
        """
        # Быстрая бинаризация Оцу
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Ищем все связные компоненты
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        # Вытаскиваем столбец с площадями всех найденных объектов
        areas = stats[:, cv2.CC_STAT_AREA]
        
        # Считаем "пыль": объекты с площадью от 1 до 10 пикселей.
        # Индекс 0 пропускаем, так как это всегда фон.
        noise_dots_count = np.sum((areas[1:] > 0) & (areas[1:] <= 10))
        
        return noise_dots_count > noise_threshold

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
        routing_plan = {
            "doc_id": doc_id,
            "pdf_path": os.path.abspath(pdf_path),
            "pages": [],
        }
        global_image_counter = 1

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=LAYOUT_DPI)
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

            # --- ВНЕДРЕНИЕ ОЧИСТКИ ОТ ШУМА ---
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            
            is_noisy = self._is_image_noisy(gray, noise_threshold=7000)
            if is_noisy:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 2. Ищем все независимые пятна (компоненты связности)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                
                # 3. Достаем площади всех найденных пятен
                areas = stats[:, cv2.CC_STAT_AREA]
                
                # --- ДИНАМИЧЕСКИЙ РАСЧЕТ ПОРОГА (АДАПТИВНЫЙ ЛАСТИК) ---
                # Плотность шума прямо пропорциональна количеству мусорных компонент (num_labels).
                # Делим общее число объектов на эмпирический коэффициент (например, 800).
                # np.clip гарантирует, что мы не опустимся ниже 3 пикселей (на чистых листах)
                # и не поднимемся выше 18 пикселей (чтобы не удалить запятые и точки на грязных).
                
                dynamic_threshold = num_labels / 6000.0
                print(dynamic_threshold)
                aria = int(np.clip(dynamic_threshold, 3, 18))
                # --------------------------------------------------------
                
                # 4. Векторная магия Numpy
                valid_labels_mask = (areas > aria).astype(np.uint8) * 255
                valid_labels_mask[0] = 0  # Индекс 0 это фон, его тоже делаем черным в маске
                
                # Превращаем лейблы обратно в картинку-маску
                clean_mask = valid_labels_mask[labels]
                
                # 5. Накладываем идеальный белый фон поверх шума
                # Всё, что является полезным текстом, останется в ИДЕАЛЬНОМ оригинальном качестве
                img_cv2_ready = img_cv2.copy()
                img_cv2_ready[clean_mask == 0] = [255, 255, 255]
            else:
                # Страница чистая -> не трогаем пиксели, применяем только легкий блюр 
                # для сглаживания возможных артефактов PDF-рендеринга
                img_cv2_ready = cv2.medianBlur(img_cv2, 3)
            # =====================================================================
            # --- ВНЕДРЕНИЕ: MULTI-SCALE INFERENCE (ДВУХПРОХОДНЫЙ АНСАМБЛЬ) ---
            # =====================================================================

            def get_ioa(box_small, box_large):
                """Вычисляет, какой процент площади box_small находится внутри box_large"""
                x_left = max(box_small[0], box_large[0])
                y_top = max(box_small[1], box_large[1])
                x_right = min(box_small[2], box_large[2])
                y_bottom = min(box_small[3], box_large[3])

                if x_right < x_left or y_bottom < y_top:
                    return 0.0

                inter_area = (x_right - x_left) * (y_bottom - y_top)
                small_area = (box_small[2] - box_small[0]) * (
                    box_small[3] - box_small[1]
                )
                return inter_area / small_area if small_area > 0 else 0

            # 1. Глобальный проход (надежные границы, широкие таблицы не режутся)
            res_1280 = self.model.predict(
                img_cv2_ready,
                imgsz=1280,
                conf=0.165,
                device=self.device,
                verbose=False,
            )[0]

            # 2. Детальный проход (лупа для поиска разделений между таблицами)
            res_2400 = self.model.predict(
                img_cv2_ready,
                imgsz=2400,
                conf=0.165,
                device=self.device,
                verbose=False,
            )[0]

            final_boxes = []

            # Парсим боксы из 2400 для быстрого поиска
            boxes_2400_parsed = []
            for b in res_2400.boxes:
                boxes_2400_parsed.append(
                    {
                        "coords": b.xyxy[0].tolist(),
                        "cls": self.model.names[int(b.cls[0])].lower(),
                        "box_obj": b,
                    }
                )

            # Группа всех табличных классов
            table_classes = {"table", "table_merged", "table_borderless"}

            # Проходимся по глобальным якорям (1280)
            for b_1280 in res_1280.boxes:
                coords_1280 = b_1280.xyxy[0].tolist()
                cls_1280 = self.model.names[int(b_1280.cls[0])].lower()

                if cls_1280 in table_classes:
                    # Ищем все боксы из прохода 2400
                    internal_boxes = [
                        b2400
                        for b2400 in boxes_2400_parsed
                        if b2400["cls"] in table_classes
                        and get_ioa(b2400["coords"], coords_1280) > 0.8
                    ]

                    if len(internal_boxes) >= 2:
                        height_1280 = coords_1280[3] - coords_1280[1]
                        width_1280 = coords_1280[2] - coords_1280[0]
                        
                        covered_height = sum(b["coords"][3] - b["coords"][1] for b in internal_boxes)
                        min_width_2400 = min(b["coords"][2] - b["coords"][0] for b in internal_boxes)
                        
                        coverage_ratio = covered_height / height_1280 if height_1280 > 0 else 0
                        width_ratio = min_width_2400 / width_1280 if width_1280 > 0 else 0
                        
                        if coverage_ratio >= 0.95 and width_ratio > 0.945:
                            # Лупа отработала чисто, забираем разрезанные куски
                            for internal in internal_boxes:
                                final_boxes.append(internal["box_obj"])
                                if internal in boxes_2400_parsed:
                                    boxes_2400_parsed.remove(internal)
                        else:
                            # Лупа потеряла часть данных. Доверяем глобальному проходу 1280.
                            final_boxes.append(b_1280)
                            # КРИТИЧЕСКИЙ ФИКС: Обязательно вычищаем бракованные куски лупы, 
                            # чтобы они не убили правильный 1280 бокс внутри NMS!
                            for internal in internal_boxes:
                                if internal in boxes_2400_parsed:
                                    boxes_2400_parsed.remove(internal)
                    else:
                        # Нашли 1 или 0 внутренних кусков. Доверяем 1280.
                        final_boxes.append(b_1280)
                        # Тоже зачищаем от греха подальше
                        for internal in internal_boxes:
                            if internal in boxes_2400_parsed:
                                boxes_2400_parsed.remove(internal)
                else:
                    final_boxes.append(b_1280)

            # Спасаем только те боксы 2400, которые реально независимы от 1280
            for b2400 in boxes_2400_parsed:
                final_boxes.append(b2400["box_obj"])

            # =====================================================================

            if visualize and vis_dir:
                # Визуализируем сырой глобальный проход (для отладки)
                annotated_frame = res_1280.plot(pil=True, line_width=2, font_size=12)
                cv2.imwrite(
                    os.path.join(vis_dir, f"page_{page_num + 1}.jpg"), annotated_frame
                )

            # Передаем ансамбль боксов в NMS и далее в сортировщик!
            filtered_boxes = self._apply_nms(final_boxes)
            sorted_boxes = self._sort_reading_order(
                filtered_boxes, pil_img.width, pil_img.height
            )

            page_plan = {
                "page_num": page_num + 1,
                "width_px": pil_img.width,
                "height_px": pil_img.height,
                "width_pt": page.rect.width,
                "height_pt": page.rect.height,
                "blocks": [],
            }

            for box in sorted_boxes:
                coords = [int(c) for c in box.xyxy[0].tolist()]
                label = self.model.names[int(box.cls[0])].lower()

                if label in self.ignore_classes:
                    continue

                block = {
                    "type": label,
                    "coords": coords,
                    "track": "DOCLING_TEXT",  # Текст и заголовки
                    "content_path": None,
                    "md_image_name": None,
                }

                if label in self.vlm_candidates:
                    block["track"] = "PADDLE_OCR"  # VLM
                elif label == "table":
                    block["track"] = "DOCLING_TABLE"  # Таблицы

                # Кропы только для OCR и таблиц
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
                        pil_img, coords, padding=25 if label == "table" else 0
                    ).save(temp_path)

                    block["content_path"] = temp_path
                    if block["track"] == "PADDLE_OCR":
                        block["md_image_name"] = (
                            f"doc_{doc_id}_image_{global_image_counter}.png"
                        )
                        global_image_counter += 1

                page_plan["blocks"].append(block)

            # Ищем растровые объекты, которые пропустила YOLO
            physical_images = page.get_image_info(xrefs=True)
            for img in physical_images:
                img_rect = fitz.Rect(img["bbox"])

                if img_rect.width < 50 or img_rect.height < 50:
                    continue

                is_caught_by_yolo = False
                for box in sorted_boxes:
                    coords = [int(c) for c in box.xyxy[0].tolist()]
                    yolo_rect = fitz.Rect(
                        coords[0] * (PDF_TO_LAYOUT),
                        coords[1] * (PDF_TO_LAYOUT),
                        coords[2] * (PDF_TO_LAYOUT),
                        coords[3] * (PDF_TO_LAYOUT),
                    )
                    if (
                        yolo_rect.intersect(img_rect).get_area()
                        > img_rect.get_area() * 0.5
                    ):
                        is_caught_by_yolo = True
                        break

                if not is_caught_by_yolo:
                    missed_coords = [
                        int(img_rect.x0 * (LAYOUT_TO_PDF)),
                        int(img_rect.y0 * (LAYOUT_TO_PDF)),
                        int(img_rect.x1 * (LAYOUT_TO_PDF)),
                        int(img_rect.y1 * (LAYOUT_TO_PDF)),
                    ]

                    missed_coords = [
                        max(0, missed_coords[0]),
                        max(0, missed_coords[1]),
                        min(pil_img.width, missed_coords[2]),
                        min(pil_img.height, missed_coords[3]),
                    ]

                    if (
                        missed_coords[2] <= missed_coords[0]
                        or missed_coords[3] <= missed_coords[1]
                    ):
                        continue

                    block = {
                        "type": "missed_raster",
                        "coords": missed_coords,
                        "track": "PADDLE_OCR",
                        "content_path": None,
                        "md_image_name": None,
                    }

                    temp_path = os.path.join(
                        temp_dir, f"candidate_{global_image_counter}.png"
                    )
                    self._crop_image(pil_img, missed_coords, padding=5).save(temp_path)

                    block["content_path"] = temp_path
                    block["md_image_name"] = (
                        f"doc_{doc_id}_image_{global_image_counter}.png"
                    )
                    global_image_counter += 1

                    page_plan["blocks"].append(block)

            routing_plan["pages"].append(page_plan)

            if visualize and vis_dir:
                # Рисуем поверх оригинальной картинки
                img_draw = img_cv2.copy()

                # Отдельный счетчик для валидных (не игнорируемых) блоков
                valid_block_counter = 1

                for box in sorted_boxes:
                    label = self.model.names[int(box.cls[0])].lower()

                    # Пропускаем abandon и все остальные классы из ignore_classes
                    if label in self.ignore_classes:
                        continue

                    coords = [int(c) for c in box.xyxy[0].tolist()]

                    # Отрисовываем рамку (зеленая)
                    cv2.rectangle(
                        img_draw,
                        (coords[0], coords[1]),
                        (coords[2], coords[3]),
                        (0, 255, 0),
                        2,
                    )

                    # Рисуем порядковый номер и класс блока (например: "1 (text)")
                    order_text = f"{valid_block_counter} ({label})"
                    cv2.putText(
                        img_draw,
                        order_text,
                        (coords[0] + 5, coords[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # Немного уменьшили шрифт, чтобы влезло название класса
                        (0, 0, 255),  # Красный цвет (BGR)
                        2,  # Толщина линии
                    )

                    valid_block_counter += 1

                cv2.imwrite(
                    os.path.join(vis_dir, f"page_{page_num + 1}_order.jpg"), img_draw
                )

        doc.close()
        return routing_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutRouter")
    parser.add_argument("--pdf", type=str, required=True, help="Путь к PDF")
    parser.add_argument("--visualize", action="store_true", help="Визуализация")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[ERROR] Файл не найден по пути '{args.pdf}'")
        exit(1)

    router = LayoutRouter()
    plan = router.build_routing_plan(args.pdf, visualize=args.visualize)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
