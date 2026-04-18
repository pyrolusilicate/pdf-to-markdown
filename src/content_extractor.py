"""
Извлечение контента из PDF.

Треки:
- VECTOR_TEXT  — PyMuPDF text dict по bbox
- VECTOR_TABLE — Docling конвертит весь PDF один раз, таблицы сопоставляются
                 с YOLO-блоками по IoU bbox
- RASTER_*     — VLM (DeepSeek-OCR-2), вызывается из pipeline.py
- SMART_FIGURE — VLM-классификация + маршрутизация
"""

from __future__ import annotations

import re
from typing import Optional

import cv2
import fitz
import numpy as np
from PIL import Image

# Слой YOLO работает при 400 DPI — все coords в плане именно в этих пикселях.
RENDER_DPI = 400
PDF_DPI = 72.0
_PX_TO_PT = PDF_DPI / RENDER_DPI


def _to_pdf_rect(coords_px: list) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = coords_px
    return (x1 * _PX_TO_PT, y1 * _PX_TO_PT, x2 * _PX_TO_PT, y2 * _PX_TO_PT)


# ---------------------------------------------------------------------------
# Текст (PyMuPDF)
# ---------------------------------------------------------------------------


def extract_text_block(
    pdf_doc: fitz.Document, page_num: int, coords: list
) -> tuple[str, float]:
    """Возвращает (text, avg_font_size) для bbox страницы (coords в пикселях 400 DPI)."""
    page = pdf_doc[page_num]
    rect = fitz.Rect(*_to_pdf_rect(coords))
    text_dict = page.get_text("dict", clip=rect)

    lines, sizes = [], []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    parts.append(t)
                    sizes.append(span.get("size", 12.0))
            if parts:
                lines.append(" ".join(parts))

    avg = sum(sizes) / len(sizes) if sizes else 12.0
    return "\n".join(lines), avg


def has_vector_text(
    pdf_doc: fitz.Document, page_num: int, coords: list, *, min_chars: int = 4
) -> bool:
    """True, если в bbox страницы есть текстовый слой PDF."""
    page = pdf_doc[page_num]
    rect = fitz.Rect(*_to_pdf_rect(coords))
    text = page.get_text("text", clip=rect) or ""
    return len(text.strip()) >= min_chars


def render_block_image(
    pdf_doc: fitz.Document, page_num: int, coords: list, dpi: int = 300
) -> Image.Image:
    """Рендерит bbox страницы в PIL Image с заданным DPI (для VLM-инференса)."""
    page = pdf_doc[page_num]
    scale = dpi / PDF_DPI
    rect = fitz.Rect(*_to_pdf_rect(coords))
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------------------------------------------------------------------
# Заголовки
# ---------------------------------------------------------------------------


def collect_heading_sizes(pdf_doc: fitz.Document, routing_plan: dict) -> list[float]:
    """Уникальные размеры шрифтов во всех блоках title/section-header, по убыванию."""
    sizes: set[float] = set()
    for page_data in routing_plan.get("pages", []):
        pnum = page_data["page_num"] - 1
        page = pdf_doc[pnum]
        for block in page_data.get("blocks", []):
            if block["type"] not in ("title", "section-header"):
                continue
            rect = fitz.Rect(*_to_pdf_rect(block["coords"]))
            td = page.get_text("dict", clip=rect)
            for b in td.get("blocks", []):
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.add(round(span.get("size", 12.0), 1))
    return sorted(sizes, reverse=True)


def detect_heading_level(font_size: float, heading_sizes: list[float]) -> int:
    """1–4 по размеру шрифта относительно всех заголовков документа."""
    for i, s in enumerate(heading_sizes[:4]):
        if font_size >= s * 0.88:
            return i + 1
    return 4


# ---------------------------------------------------------------------------
# Таблицы: Docling кеш + сопоставление по IoU
# ---------------------------------------------------------------------------


class DoclingTableStore:
    """
    Однократная конвертация PDF через Docling. Таблицы индексируются по странице,
    каждая ищется по IoU bbox.

    Docling bbox хранятся в PDF points. У нас coords YOLO-блока в пикселях 400 DPI —
    конвертим в PDF points (×72/400) и считаем IoU.
    """

    def __init__(self, pdf_path: str):
        self._tables_by_page: dict[int, list[tuple[tuple, str]]] = {}
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Docling не установлен: `pip install docling docling-core`."
            ) from exc

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        # Высоты страниц нужны, чтобы инвертировать BOTTOMLEFT → TOPLEFT.
        page_heights = (
            {
                p.page_no: float(p.size.height)
                for p in getattr(doc, "pages", {}).values()
            }
            if hasattr(doc, "pages")
            else {}
        )

        for table in getattr(doc, "tables", []):
            md = _finalize_table_markdown(table)
            if not md:
                continue
            for prov in getattr(table, "prov", []) or []:
                page_no = int(getattr(prov, "page_no", 0))
                bbox = getattr(prov, "bbox", None)
                if bbox is None or page_no <= 0:
                    continue
                tl = _bbox_to_topleft(bbox, page_heights.get(page_no))
                if tl is None:
                    continue
                self._tables_by_page.setdefault(page_no, []).append((tl, md))

    def find(
        self, page_num_1based: int, coords_px: list, *, min_iou: float = 0.25
    ) -> Optional[str]:
        """Возвращает markdown таблицы с максимальным IoU с данным bbox, или None."""
        candidates = self._tables_by_page.get(page_num_1based, [])
        if not candidates:
            return None

        q = _to_pdf_rect(coords_px)
        best_md, best_iou = None, 0.0
        for bbox, md in candidates:
            iou = _iou(bbox, q)
            if iou > best_iou:
                best_iou = iou
                best_md = md
        return best_md if best_iou >= min_iou else None


def _bbox_to_topleft(bbox, page_height_pt: Optional[float]) -> Optional[tuple]:
    """Приводит Docling bbox к (l, t, r, b) в TOPLEFT и в PDF points."""
    try:
        l = float(bbox.l)
        r = float(bbox.r)
        t = float(bbox.t)
        b = float(bbox.b)
    except AttributeError:
        return None

    origin = getattr(bbox, "coord_origin", None)
    origin_name = getattr(origin, "name", str(origin) if origin else "")

    if "BOTTOM" in origin_name.upper() and page_height_pt:
        # y снизу вверх → инвертируем
        new_t = page_height_pt - max(t, b)
        new_b = page_height_pt - min(t, b)
        t, b = new_t, new_b
    else:
        if t > b:
            t, b = b, t

    if l > r:
        l, r = r, l
    return (l, t, r, b)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _finalize_table_markdown(table_item) -> str:
    """
    Docling умеет сам экспортировать таблицу в markdown. Поверх этого
    добиваем требования хакатона:
      1. Многоуровневые заголовки объединяются через `_`.
      2. Объединённые ячейки дублируются (forward-fill).
    """
    # Путь 1: есть структурированный grid — обрабатываем вручную.
    grid = _table_to_grid(table_item)
    if grid:
        return format_table_markdown(grid)

    # Путь 2: фолбэк — доклинговский экспорт (уже markdown).
    try:
        md = table_item.export_to_markdown()
    except Exception:
        md = ""
    return str(md).strip()


def _table_to_grid(table_item) -> Optional[list[list[str]]]:
    """Извлекает 2D-массив ячеек с учётом rowspan/colspan."""
    data = getattr(table_item, "data", None)
    if data is None:
        return None
    nrows = int(getattr(data, "num_rows", 0) or 0)
    ncols = int(getattr(data, "num_cols", 0) or 0)
    if nrows == 0 or ncols == 0:
        return None

    grid: list[list[str]] = [["" for _ in range(ncols)] for _ in range(nrows)]
    cells = getattr(data, "table_cells", None) or getattr(data, "cells", None) or []

    for cell in cells:
        r0 = int(getattr(cell, "start_row_offset_idx", getattr(cell, "row_id", 0)) or 0)
        c0 = int(getattr(cell, "start_col_offset_idx", getattr(cell, "col_id", 0)) or 0)
        r1 = int(getattr(cell, "end_row_offset_idx", r0 + 1) or r0 + 1)
        c1 = int(getattr(cell, "end_col_offset_idx", c0 + 1) or c0 + 1)
        raw = getattr(cell, "text", "") or ""
        text = str(raw).strip().replace("\n", " ")

        # Дублируем содержимое объединённых ячеек (требование ТЗ)
        for rr in range(r0, min(r1, nrows)):
            for cc in range(c0, min(c1, ncols)):
                if not grid[rr][cc]:
                    grid[rr][cc] = text

    return grid


# ---------------------------------------------------------------------------
# Markdown таблицы (объединённые шапки + forward-fill)
# ---------------------------------------------------------------------------


def _forward_fill(table: list[list]) -> list[list[str]]:
    """Дублирует значения объединённых ячеек (None/пусто → последнее значение в строке)."""
    result = []
    for row in table:
        new_row, last = [], ""
        for cell in row:
            if cell is None or (isinstance(cell, str) and not cell.strip()):
                new_row.append(last)
            else:
                val = str(cell).strip().replace("\n", " ")
                last = val
                new_row.append(val)
        result.append(new_row)
    return result


def format_table_markdown(table: list[list]) -> str:
    """Markdown-таблица с многоуровневыми заголовками (header1_header2)."""
    if not table:
        return ""

    filled = _forward_fill(table)
    if not filled or not filled[0]:
        return ""

    header_rows = 1
    if len(filled) >= 3:
        r0, r1 = filled[0], filled[1]
        non_numeric = all(not re.fullmatch(r"[\d\s.,+\-/%]+", c) or c == "" for c in r1)
        repeats = sum(1 for c in r1 if c in r0 and c != "")
        if non_numeric and repeats > max(1, len(r1) * 0.25):
            header_rows = 2

    if header_rows == 2:
        headers = [
            f"{h1}_{h2}" if (h1 and h2 and h1 != h2) else (h1 or h2)
            for h1, h2 in zip(filled[0], filled[1])
        ]
        data_rows = filled[2:]
    else:
        headers = filled[0]
        data_rows = filled[1:]

    n = len(headers)
    if n == 0:
        return ""

    def pad(row: list, width: int) -> list[str]:
        row = list(row) + [""] * width
        return [str(c) for c in row[:width]]

    lines = [
        "| " + " | ".join(pad(headers, n)) + " |",
        "| " + " | ".join(["---"] * n) + " |",
    ]
    for row in data_rows:
        lines.append("| " + " | ".join(pad(row, n)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Форматирование текстового блока
# ---------------------------------------------------------------------------


def format_text_markdown(text: str, block_type: str, heading_level: int = 0) -> str:
    """Применяет Markdown-разметку к извлечённому тексту."""
    text = text.strip()
    if not text:
        return ""

    if block_type in ("title", "section-header") and heading_level > 0:
        single_line = " ".join(l.strip() for l in text.splitlines() if l.strip())
        return "#" * heading_level + " " + single_line

    if block_type == "list-item":
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^[•▪▸\-\*]\s", line) or re.match(r"^\d+[.)]\s", line):
                out.append(line)
            else:
                out.append(f"- {line}")
        return "\n".join(out)

    return text


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def estimate_text_density(img: Image.Image) -> float:
    """Доля 'чернильных' пикселей — для эвристической классификации."""
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return float(binary.sum()) / (255.0 * binary.size)


def filter_noise_lines(text: str, min_chars: int = 3) -> str:
    """Убирает коротко-мусорные строки (остатки буквиц/рамок)."""
    if not text:
        return ""
    lines = [
        l for l in text.splitlines() if len(l.strip().replace(" ", "")) >= min_chars
    ]
    return "\n".join(lines).strip()
