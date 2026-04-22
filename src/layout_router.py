"""
Layout-роутер: DocLayout-YOLOv10 + детерминированный reading order.

Multi-scale inference (imgsz=1280 + imgsz=2400):
  * 1280 — надёжные глобальные якоря, широкие таблицы не режутся на части.
  * 2400 — «лупа»: разбивает сросшиеся соседние таблицы, если покрытие >=95%
    и ширина/высота субтаблиц >=94.5%.

NMS по IoM (intersection over min-area) с приоритетом класса:
``title > section-header > table > figure > text``. Бокс выбрасывается,
если >70% его площади перекрыто более приоритетным.

Reading order:
  1. Логические блоки: медиа (figure/table) + ближайшая подпись склеиваются.
  2. Полосы: блоки с пересечением по Y (допуск 3% высоты) объединяются.
  3. Колонки внутри полос (horizontal overlap > 30%), слева-направо.
  4. Внутри колонки — сверху-вниз.

Также добавляем растровые объекты PDF, которые YOLO пропустил (по
``page.get_image_info``) — в итоговый план как type=``picture``.

Возвращаемая структура:
    {
      "doc_id": "51",
      "pdf_path": "...",
      "pages": [
        {"page_num": 1, "width": 2480, "height": 3508, "blocks": [
            {"type": "title", "coords": [x1,y1,x2,y2], "conf": 0.93,
             "md_image_name": null},
            ...
        ]},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np
import torch
from PIL import Image

from config import LAYOUT_DPI, LAYOUT_TO_PDF, PDF_TO_LAYOUT, WEIGHTS_DIR
from device import get_torch_device

# PyTorch >=2.6 делает weights_only=True по умолчанию и ломает загрузку YOLO.
# Мононакладка: форсим weights_only=False на уровне torch.load.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

# Короткие псевдонимы для проекции координат: «точки <-> пиксели»
# (имена `_PX_TO_PT` / `_PT_TO_PX` более читаемы в вызовах, чем сырые коэффициенты).
_PX_TO_PT = PDF_TO_LAYOUT
_PT_TO_PX = LAYOUT_TO_PDF

_WEIGHTS_FILE = "doclayout_yolo_docstructbench_imgsz1280_2501.pt"

_FIGURE_LABELS = {"picture", "figure", "image"}
_TEXT_LABELS = {
    "text", "plain text", "plain-text", "title", "section-header", "list-item",
}
_TABLE_LABELS = {"table", "table_merged", "table_borderless"}

# Что игнорируем на уровне класса — полностью, без попыток обработать.
_IGNORE_CLASSES = {"page-header", "page-footer", "footnote", "watermark", "abandon"}

# Приоритет классов для NMS: чем выше число, тем больше шансов выжить при коллизии.
_CLASS_PRIORITY = {
    "title": 10,
    "section-header": 9,
    "table": 8,
    "table_merged": 8,
    "table_borderless": 8,
    "figure": 7,
    "picture": 7,
    "image": 7,
    "table_caption": 6,
    "figure_caption": 6,
    "caption": 6,
    "list-item": 5,
    "text": 4,
    "plain text": 4,
    "plain-text": 4,
}


# ---------------------------------------------------------------------------


class LayoutRouter:
    """
    DocLayout-YOLOv10 + reading-order постобработка.

    Одна публичная точка входа — ``build_routing_plan``. Модель подгружается
    лениво при первом вызове (~400MB весов) и кэшируется в инстансе.
    """

    def __init__(self, weights_dir: str = WEIGHTS_DIR):
        """
        Args:
            weights_dir: куда скачивать веса YOLO, если их нет локально.
        """
        self.weights_dir = weights_dir
        self.device = get_torch_device()
        self._model = None

    def _load(self) -> None:
        """Лениво грузит DocLayout-YOLOv10 (скачивает из HF при отсутствии)."""
        if self._model is not None:
            return
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        os.makedirs(self.weights_dir, exist_ok=True)
        weights_path = os.path.join(self.weights_dir, _WEIGHTS_FILE)
        if not os.path.exists(weights_path):
            hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename=_WEIGHTS_FILE,
                local_dir=self.weights_dir,
            )
        self._model = YOLOv10(weights_path)
        print(f"  [YOLO] DocLayout ready ({self.device})")

    # ---- публичный API ---------------------------------------------------

    def build_routing_plan(
        self, pdf_path: str, output_dir: str, visualize: bool = False
    ) -> dict:
        """
        Строит план обработки документа: per-page список блоков с reading order.

        Args:
            pdf_path: путь к PDF.
            output_dir: база для вспомогательных файлов (визуализации и пр.).
            visualize: если True, пишет в ``data/visualization/document_<id>/``
                аннотированные PNG-страницы с индексами блоков.

        Returns:
            dict со структурой, описанной в module-docstring.
        """
        self._load()
        doc_name = Path(pdf_path).stem
        doc_id = _doc_id_from_name(doc_name)

        vis_dir: Optional[str] = None
        if visualize:
            vis_dir = os.path.join("data", "visualization", f"document_{doc_id}")
            os.makedirs(vis_dir, exist_ok=True)

        pdf_doc = fitz.open(pdf_path)
        pages_out: list[dict] = []
        try:
            for page_idx, page in enumerate(pdf_doc):
                page_num_1 = page_idx + 1

                scale = LAYOUT_DPI / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img_rgb = img_rgb[:, :, :3]
                img_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                pil_img = Image.fromarray(img_rgb)
                page_w, page_h = pil_img.size

                gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

                # Шумоподавление только если страница реально «пыльная»:
                # median blur на чистой странице смазывает мелкий текст.
                img_ready = (
                    cv2.medianBlur(img_cv2, 3) if _is_image_noisy(gray) else img_cv2
                )

                raw_boxes = self._multi_scale_predict(img_ready)
                nms_boxes = self._apply_nms(raw_boxes)
                sorted_boxes = self._sort_reading_order(nms_boxes, page_w, page_h)

                if visualize and vis_dir:
                    _visualize(
                        img_cv2, sorted_boxes, self._model.names, vis_dir, page_num_1
                    )

                blocks: list[dict] = []
                saved_coords: list[list[int]] = []

                for box in sorted_boxes:
                    coords = [int(c) for c in box.xyxy[0].tolist()]
                    label = self._model.names[int(box.cls[0])].lower()

                    if label in _IGNORE_CLASSES:
                        continue
                    if _is_box_empty(coords, gray):
                        continue
                    # Цельностраничные боксы — шум (фон / граница).
                    box_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                    if box_area / (page_w * page_h) > 0.75:
                        continue

                    md_image_name: Optional[str] = None
                    if label in _FIGURE_LABELS:
                        # Реальное имя присваивается в pipeline после проверки,
                        # что картинка не шумная и стоит её сохранять.
                        md_image_name = "__figure__"
                    # saved_coords нужен, чтобы _find_missed_rasters не добавил
                    # растр поверх уже покрытой таблицей/текстом области.
                    saved_coords.append(coords)

                    blocks.append({
                        "type": label,
                        "coords": coords,
                        "conf": float(box.conf[0]),
                        "md_image_name": md_image_name,
                    })

                missed = _find_missed_rasters(
                    page, pil_img, sorted_boxes, saved_coords
                )
                for blk in missed:
                    blocks.append(blk)

                pages_out.append({
                    "page_num": page_num_1,
                    "width": page_w,
                    "height": page_h,
                    "blocks": blocks,
                })
        finally:
            pdf_doc.close()

        return {"doc_id": doc_id, "pdf_path": pdf_path, "pages": pages_out}

    # ---- inference -------------------------------------------------------

    def _multi_scale_predict(self, img: np.ndarray) -> list:
        """
        Двухпроходный YOLO-ансамбль (imgsz=1280 + imgsz=2400).

        Если 2400-лупа нашла >=2 субтаблиц внутри 1280-таблицы и суммарно они
        покрывают >=95% по одной оси с >=94.5% по другой — это сросшиеся соседние
        таблицы, берём субтаблицы. Иначе доверяем 1280-якорю.
        """
        res_1280 = self._model.predict(
            img, imgsz=1280, conf=0.165, device=self.device, verbose=False
        )[0]
        res_2400 = self._model.predict(
            img, imgsz=2400, conf=0.165, device=self.device, verbose=False
        )[0]

        table_cls = _TABLE_LABELS
        parsed_2400 = [
            {
                "coords": b.xyxy[0].tolist(),
                "cls": self._model.names[int(b.cls[0])].lower(),
                "box": b,
            }
            for b in (res_2400.boxes or [])
        ]
        used_2400: set[int] = set()
        final: list = []

        for b in (res_1280.boxes or []):
            coords = b.xyxy[0].tolist()
            label = self._model.names[int(b.cls[0])].lower()

            if label not in table_cls:
                final.append(b)
                continue

            internal = [
                i for i, p in enumerate(parsed_2400)
                if i not in used_2400
                and p["cls"] in table_cls
                and _ioa(p["coords"], coords) > 0.8
            ]

            if len(internal) >= 2:
                h = coords[3] - coords[1]
                w = coords[2] - coords[0]
                sub_h = [
                    parsed_2400[i]["coords"][3] - parsed_2400[i]["coords"][1]
                    for i in internal
                ]
                sub_w = [
                    parsed_2400[i]["coords"][2] - parsed_2400[i]["coords"][0]
                    for i in internal
                ]
                # Вертикальный сплит (таблицы стопкой): сумма высот >=95%, каждая >=94.5% ширины.
                vert_ok = (
                    h > 0
                    and sum(sub_h) / h >= 0.95
                    and (w == 0 or min(sub_w) / w > 0.945)
                )
                # Горизонтальный сплит (таблицы рядом): симметрично по осям.
                horiz_ok = (
                    w > 0
                    and sum(sub_w) / w >= 0.95
                    and (h == 0 or min(sub_h) / h > 0.945)
                )
                if vert_ok or horiz_ok:
                    for i in internal:
                        final.append(parsed_2400[i]["box"])
                        used_2400.add(i)
                    continue
                # Лупа дала неполное покрытие -> доверяем якорю 1280.
                for i in internal:
                    used_2400.add(i)
            else:
                for i in internal:
                    used_2400.add(i)

            final.append(b)

        # Независимые 2400-боксы (не поглощённые никаким 1280-якорем).
        for i, p in enumerate(parsed_2400):
            if i not in used_2400:
                final.append(p["box"])

        return final

    # ---- NMS -------------------------------------------------------------

    def _apply_nms(self, boxes: list, iom_threshold: float = 0.7) -> list:
        """
        NMS по IoM с приоритетом класса.

        Бокс удаляется, если >``iom_threshold`` его площади перекрыто боксом
        более высокого приоритета. Приоритет figure с низкой уверенностью
        принудительно занижаем, чтобы он не «съедал» текстовый блок поверх.
        """
        if not boxes:
            return []

        def _priority(b) -> tuple[int, float]:
            cls = self._model.names[int(b.cls[0])].lower()
            p = _CLASS_PRIORITY.get(cls, 3)
            if cls in _FIGURE_LABELS and float(b.conf[0]) < 0.4:
                p = 2
            return p, float(b.conf[0])

        kept: list = []
        for box in sorted(boxes, key=_priority, reverse=True):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area == 0:
                continue
            dominated = False
            for kb in kept:
                kx1, ky1, kx2, ky2 = kb.xyxy[0].tolist()
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
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

    # ---- reading order ---------------------------------------------------

    def _sort_reading_order(self, boxes: list, page_w: int, page_h: int) -> list:
        """
        Band-based reading order для мульти-колоночных макетов.

        Пайплайн сортировки:
          1. Логические блоки: медиа + ближайшая подпись склеиваются в один.
          2. Полосы (bands): блоки с пересечением по Y (допуск 3% высоты
             страницы) объединяются в горизонтальные группы.
          3. Колонки внутри полос (overlap > 30% по X), слева-направо.
          4. Внутри колонки — сверху-вниз.
        """
        if not boxes:
            return []

        parsed = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            parsed.append({
                "box": box,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": x2 - x1, "h": y2 - y1,
                "cls": self._model.names[int(box.cls[0])].lower(),
            })

        # Убираем колонтитулы по позиции (дополнительный фильтр к ignore_classes)
        m_top = page_h * 0.05
        m_bot = page_h * 0.95
        filtered = []
        for b in parsed:
            if b["cls"] in {"page-header", "page-footer"}:
                continue
            if b["cls"] in {"text", "plain text", "plain-text", "title"}:
                if b["y2"] < m_top or b["y1"] > m_bot:
                    continue
            filtered.append(b)
        filtered.sort(key=lambda x: x["y1"])

        # Логические блоки: медиа + ближайшая подпись
        logical_blocks: list[dict] = []
        used: set[int] = set()
        media_cls = _FIGURE_LABELS | _TABLE_LABELS
        caption_cls = {"figure_caption", "table_caption", "caption", "text", "plain text"}
        for i, b in enumerate(filtered):
            if i in used:
                continue
            group = [b]
            used.add(i)
            base = b
            for j in range(i + 1, len(filtered)):
                if j in used:
                    continue
                nxt = filtered[j]
                y_gap = nxt["y1"] - base["y2"]
                if y_gap >= page_h * 0.08:
                    break
                ov_x = max(0.0, min(base["x2"], nxt["x2"]) - max(base["x1"], nxt["x1"]))
                min_w = min(base["w"], nxt["w"])
                x_ratio = ov_x / min_w if min_w > 0 else 0.0
                if base["cls"] in media_cls and nxt["cls"] in caption_cls and x_ratio > 0.5:
                    group.append(nxt)
                    used.add(j)
                    base = nxt
                else:
                    break
            logical_blocks.append({
                "boxes": group,
                "x1": min(g["x1"] for g in group),
                "y1": min(g["y1"] for g in group),
                "x2": max(g["x2"] for g in group),
                "y2": max(g["y2"] for g in group),
                "w": max(g["x2"] for g in group) - min(g["x1"] for g in group),
                "h": max(g["y2"] for g in group) - min(g["y1"] for g in group),
            })

        # Полосы
        bands: list[list[dict]] = []
        current_band: list[dict] = []
        band_bot = 0.0
        for lb in logical_blocks:
            is_full = lb["w"] > page_w * 0.55
            is_title = any(b["cls"] in {"title", "section-header"} for b in lb["boxes"])
            if is_full or is_title:
                if current_band:
                    bands.append(current_band)
                bands.append([lb])
                current_band = []
                band_bot = 0.0
            else:
                if not current_band:
                    current_band.append(lb)
                    band_bot = lb["y2"]
                elif lb["y1"] <= band_bot + page_h * 0.03:
                    current_band.append(lb)
                    band_bot = max(band_bot, lb["y2"])
                else:
                    bands.append(current_band)
                    current_band = [lb]
                    band_bot = lb["y2"]
        if current_band:
            bands.append(current_band)

        # Колонки и финальная распаковка
        final: list = []
        for band in bands:
            columns: list[list[dict]] = []
            for lb in band:
                placed = False
                for col in columns:
                    col_x1 = min(c["x1"] for c in col)
                    col_x2 = max(c["x2"] for c in col)
                    col_w = col_x2 - col_x1
                    ov = max(0.0, min(lb["x2"], col_x2) - max(lb["x1"], col_x1))
                    min_w = min(lb["w"], col_w)
                    if min_w > 0 and ov / min_w > 0.3:
                        col.append(lb)
                        placed = True
                        break
                if not placed:
                    columns.append([lb])
            columns.sort(key=lambda col: min(c["x1"] for c in col))
            for col in columns:
                col.sort(key=lambda lb: lb["y1"])
                for lb in col:
                    for phys in lb["boxes"]:
                        final.append(phys["box"])
        return final


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _ioa(box_small: list, box_large: list) -> float:
    """Intersection over Area of box_small."""
    x1 = max(box_small[0], box_large[0])
    y1 = max(box_small[1], box_large[1])
    x2 = min(box_small[2], box_large[2])
    y2 = min(box_small[3], box_large[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    small_area = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
    return inter / small_area if small_area > 0 else 0.0


def _is_image_noisy(gray: np.ndarray, threshold: int = 5000) -> bool:
    """True если страница содержит >threshold точечных артефактов (пыль сканера)."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    return int(np.sum((areas[1:] > 0) & (areas[1:] <= 10))) > threshold


def _is_box_empty(coords: list[int], gray: np.ndarray) -> bool:
    """True если внутри бокса нет реального контента (только фон или пыль)."""
    x1, y1, x2, y2 = coords
    m = 5
    x1c = min(x1 + m, x2); y1c = min(y1 + m, y2)
    x2c = max(x1, x2 - m); y2c = max(y1, y2 - m)
    crop = gray[y1c:y2c, x1c:x2c]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return True
    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if n <= 1:
        return True
    return int(np.max(stats[1:, cv2.CC_STAT_AREA])) < 35


def _find_missed_rasters(
    page: fitz.Page,
    pil_img: Image.Image,
    sorted_boxes: list,
    saved_coords: list[list[int]],
) -> list[dict]:
    """
    Находит растровые объекты PDF, которые YOLO пропустил.

    YOLO иногда не детектит мелкие/низкоконтрастные картинки в цифровом PDF;
    PyMuPDF даёт их напрямую через ``page.get_image_info``. Мы добавляем
    только те, что НЕ поймал YOLO (IoA с существующим YOLO-боксом <=50%)
    и которые не дублируют уже сохранённые координаты.
    """
    results: list[dict] = []
    page_area_pt = page.rect.width * page.rect.height

    for img_info in page.get_image_info(xrefs=True):
        img_rect = fitz.Rect(img_info["bbox"])
        if img_rect.width < 50 or img_rect.height < 50:
            continue
        if page_area_pt > 0 and img_rect.get_area() / page_area_pt > 0.75:
            continue

        caught = False
        for box in sorted_boxes:
            px = [int(c) for c in box.xyxy[0].tolist()]
            yolo_rect = fitz.Rect(
                px[0] * _PX_TO_PT, px[1] * _PX_TO_PT,
                px[2] * _PX_TO_PT, px[3] * _PX_TO_PT,
            )
            if yolo_rect.intersect(img_rect).get_area() > img_rect.get_area() * 0.5:
                caught = True
                break
        if caught:
            continue

        mc = [
            max(0, int(img_rect.x0 * _PT_TO_PX)),
            max(0, int(img_rect.y0 * _PT_TO_PX)),
            min(pil_img.width, int(img_rect.x1 * _PT_TO_PX)),
            min(pil_img.height, int(img_rect.y1 * _PT_TO_PX)),
        ]
        if mc[2] <= mc[0] or mc[3] <= mc[1]:
            continue
        if _is_duplicate(mc, saved_coords):
            continue

        saved_coords.append(mc)
        results.append({
            "type": "picture",
            "coords": mc,
            "conf": 1.0,
            "md_image_name": "__figure__",
        })

    return results


def _is_duplicate(
    coords: list[int], saved: list[list[int]], threshold: float = 0.85
) -> bool:
    """True, если ``coords`` сильно пересекается (IoM > threshold) с уже сохранённым."""
    for s in saved:
        x1 = max(coords[0], s[0]); y1 = max(coords[1], s[1])
        x2 = min(coords[2], s[2]); y2 = min(coords[3], s[3])
        if x2 > x1 and y2 > y1:
            inter = (x2 - x1) * (y2 - y1)
            a1 = (coords[2] - coords[0]) * (coords[3] - coords[1])
            a2 = (s[2] - s[0]) * (s[3] - s[1])
            if min(a1, a2) > 0 and inter / min(a1, a2) > threshold:
                return True
    return False


def _visualize(
    img_cv2: np.ndarray, boxes: list, names: dict, vis_dir: str, page_num: int
) -> None:
    """Сохраняет аннотированную страницу (bbox'ы + индекс + класс) для дебага."""
    draw = img_cv2.copy()
    for idx, box in enumerate(boxes, 1):
        label = names[int(box.cls[0])].lower()
        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            draw, f"{idx} ({label})", (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
    cv2.imwrite(os.path.join(vis_dir, f"page_{page_num}_order.jpg"), draw)


def _doc_id_from_name(stem: str) -> str:
    """document_051 -> '51' (без ведущих нулей)."""
    m = re.search(r"(\d+)", stem)
    return str(int(m.group(1))) if m else stem


# Экспорт для pipeline.py
FIGURE_LABELS = _FIGURE_LABELS
TEXT_LABELS = _TEXT_LABELS
TABLE_LABELS = _TABLE_LABELS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutRouter — DocLayout-YOLOv10")
    parser.add_argument("--pdf", required=True, help="Путь к PDF")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[ERROR] Файл не найден: {args.pdf}")
        raise SystemExit(1)

    router = LayoutRouter()
    plan = router.build_routing_plan(args.pdf, args.output_dir, visualize=args.visualize)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
