"""
Главный пайплайн: PDF → Markdown + images/ → submission.zip

Архитектура (гибрид):
    Router    : DocLayout-YOLOv10 (layout_router.py) — всегда.
                Определяет bbox блоков, фильтрует вотермарки/колонтитулы/blank,
                задаёт reading order (топо-сорт по колонкам).
    Fast Track: Docling DocumentConverter — один convert(pdf) на документ.
                Текст и таблицы, сопоставляются с YOLO-блоками по IoM > 0.6.
    Fallback  : olmOCR-2-7B-1025 BF16 — crop YOLO-блока → markdown,
                только если Docling вернул пусто/битое.

Треки YOLO → обработка:
    title/section-header/list-item/text  → Docling text items (IoM-матч)
    table                                → Docling TableItem + format_table_markdown
    picture/figure/image                 → PNG + olmOCR OCR-текст

Использование:
    python src/pipeline.py --all
    python src/pipeline.py --pdf data/raw/document_001.pdf
    python src/pipeline.py --all --no-vlm         # только YOLO + Docling
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from content_extractor import (
    crop_pdf_bbox,
    cyrillic_ratio,
    filter_noise_lines,
    format_table_markdown,
    format_text_markdown,
    repetition_ratio,
    table_stats,
)
from coord_projection import iom, points_to_pixels
from device import setup_environment
from layout_router import FIGURE_LABELS, TABLE_LABELS, LayoutRouter

setup_environment()

OUTPUT_DIR = "data/output"
IMAGE_MAX_SIDE = 150     # финальное сжатие сохранённых картинок
OLM_RENDER_SIDE = 1288   # olmOCR требует длинную сторону == 1288

IOM_MATCH_THRESHOLD = 0.6   # порог IoM для матчинга Docling-item ↔ YOLO-block

# Водяные знаки которые могут просочиться через YOLO-фильтр
_WM_ONLY_RE = re.compile(
    r"^(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)\s*$",
    re.I | re.UNICODE,
)
_WM_PREFIX_RE = re.compile(
    r"^(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)\s+",
    re.I | re.UNICODE,
)

# Docling внутренние ссылки на изображения (page_0_0_1280_960.png) — не наш формат
_DOCLING_IMG_REF_RE = re.compile(r"!\[[^\]]*\]\(page_\d+[^\)]*\.(?:png|jpg|jpeg)\)", re.I)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(self, output_dir: str = OUTPUT_DIR, use_vlm: bool = True):
        self.output_dir = output_dir
        self.use_vlm = use_vlm

        self.router = LayoutRouter()

        from docling_engine import DoclingEngine
        self.docling = DoclingEngine.get()

        self.olm = None
        if use_vlm:
            try:
                from olm_engine import OLMEngine
                self.olm = OLMEngine.get()
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] olmOCR недоступен: {exc}")
                self.olm = None

        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str) -> str:
        doc_name = Path(pdf_path).stem
        doc_id = _doc_id_from_name(doc_name)  # без ведущих нулей

        # 1. YOLO routing
        plan = self.router.build_routing_plan(pdf_path, self.output_dir)

        # 2. Docling один проход (Fast Track)
        docling_doc = self.docling.convert(pdf_path)

        # 3. Индекс Docling-items в пиксельных координатах YOLO-растра
        docling_idx = _build_docling_index(docling_doc)

        # 4. Per-block: IoM-матч с Docling, fallback на olmOCR
        pdf_doc = fitz.open(pdf_path)
        self._doc_img_counter = 0   # счётчик реально сохранённых PNG для этого документа
        flat: list[tuple[str, str, int]] = []  # (yolo_type, md, page_num)

        try:
            for page_data in plan["pages"]:
                page_num_1 = page_data["page_num"]
                page_num_0 = page_num_1 - 1

                page_items = docling_idx.get(page_num_1, [])

                for block in page_data["blocks"]:
                    md = self._process_block(
                        block, page_num_0, pdf_doc, page_items, doc_id
                    )
                    if md and md.strip():
                        flat.append((block["type"], md.strip(), page_num_1))

                # Освобождаем VRAM между страницами
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            pdf_doc.close()

        # 5. Склейка таблиц через разрыв страницы
        parts = _merge_cross_page_tables(flat)

        # 6. Сборка и пост-обработка
        markdown = _postprocess_document("\n\n".join(parts))
        md_path = os.path.join(self.output_dir, f"{doc_name}.md")
        Path(md_path).write_text(markdown, encoding="utf-8")
        print(f"  ✓ {doc_name}.md")
        return md_path

    def process_all(self, raw_dir: str = "data/raw") -> str:
        pdfs = sorted(Path(raw_dir).glob("document_*.pdf"))
        print(f"Найдено {len(pdfs)} PDF файлов")
        for i, pdf_path in enumerate(pdfs, 1):
            print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
            try:
                self.process_pdf(str(pdf_path))
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ ОШИБКА: {exc}")
        zip_path = self._create_zip()
        print(f"\nГотово: {zip_path}")
        return zip_path

    # ------------------------------------------------------------------
    # Обработка блока
    # ------------------------------------------------------------------

    def _process_block(
        self,
        block: dict,
        page_num_0: int,
        pdf_doc: fitz.Document,
        page_items: list,
        doc_id: str,
    ) -> str:
        btype = block["type"]

        if btype in FIGURE_LABELS:
            return self._process_figure(block, page_num_0, pdf_doc, doc_id)
        if btype in TABLE_LABELS:
            return self._process_table(block, page_num_0, pdf_doc, page_items)
        # текстовый (title/section-header/list-item/text/plain-text)
        return self._process_text(block, page_num_0, pdf_doc, page_items)

    # ---- текстовые блоки ----------------------------------------------

    def _process_text(
        self,
        block: dict,
        page_num_0: int,
        pdf_doc: fitz.Document,
        page_items: list,
    ) -> str:
        coords = block["coords"]
        btype = block["type"]

        # 1. Docling Fast Track: собираем тексты по IoM-матчу
        matched = _match_items_by_iom(coords, page_items, kind_filter=("text",))
        text_parts = []
        for _, _, item in matched:
            t = getattr(item, "text", "") or ""
            t = _DOCLING_IMG_REF_RE.sub("", t).strip()  # убираем внутренние ссылки Docling
            if t.strip():
                text_parts.append(t.strip())
        text = "\n".join(text_parts)
        text = filter_noise_lines(text, min_chars=3)

        # 2. Docling пусто → olmOCR crop
        if not text and self.olm is not None:
            img = crop_pdf_bbox(pdf_doc, page_num_0, coords, max_side=OLM_RENDER_SIDE)
            if img is not None:
                try:
                    raw = self.olm.page_to_markdown(img).strip()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [olmOCR] text crop failed: {exc}")
                    raw = ""
                if raw and _validate_text(raw, ""):
                    text = filter_noise_lines(raw, min_chars=3)

        if not text:
            return ""

        # 3. Markdown-форматирование
        heading_level = 0
        if btype == "title":
            heading_level = 1
        elif btype == "section-header":
            # Берём уровень из Docling если есть (до 4 уровней по заданию)
            docling_level = 0
            for _, _, item in matched:
                lvl = getattr(item, "level", None)
                if lvl and isinstance(lvl, int) and 1 <= lvl <= 4:
                    docling_level = lvl
                    break
            heading_level = docling_level if docling_level else 2

        if heading_level > 0:
            # Если bbox захватил несколько Docling-элементов (заголовок + тело),
            # отделяем первую строку как заголовок, остальные — как обычный текст
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if len(lines) > 1:
                heading_md = format_text_markdown(lines[0], btype, heading_level)
                body_text = "\n".join(lines[1:])
                return f"{heading_md}\n\n{body_text}"

        return format_text_markdown(text, btype, heading_level)

    # ---- таблицы ------------------------------------------------------

    def _process_table(
        self,
        block: dict,
        page_num_0: int,
        pdf_doc: fitz.Document,
        page_items: list,
    ) -> str:
        coords = block["coords"]

        # 1. Docling TableFormer: матчим по IoM
        matched = _match_items_by_iom(coords, page_items, kind_filter=("table",))
        if matched:
            # Берём таблицу с лучшим IoM (матч упорядочен по убыванию)
            _, _, tbl_item = matched[0]
            grid = _docling_table_to_grid(tbl_item)
            if grid:
                md = format_table_markdown(grid)
                if md and _validate_table(md, ""):
                    return md

        # 2. Fallback: olmOCR crop
        if self.olm is not None:
            img = crop_pdf_bbox(pdf_doc, page_num_0, coords, max_side=OLM_RENDER_SIDE)
            if img is not None:
                try:
                    raw = self.olm.page_to_markdown(img).strip()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [olmOCR] table crop failed: {exc}")
                    raw = ""
                md = _postprocess_vlm_table(raw)
                if md and _validate_table(md, ""):
                    return md
                # olmOCR вернул текст без табличной структуры — только если нет HTML
                if raw and _validate_text(raw, "") and "<" not in raw:
                    return filter_noise_lines(raw, min_chars=3)

        return ""

    # ---- картинки -----------------------------------------------------

    def _process_figure(
        self,
        block: dict,
        page_num_0: int,
        pdf_doc: fitz.Document,
        doc_id: str,
    ) -> str:
        coords = block["coords"]
        # md_image_name от роутера — всегда "__figure__" (имя назначается ниже при сохранении)
        if not block.get("md_image_name"):
            return ""

        # Рендерим bbox в PIL для OCR и (возможно) сохранения
        pil = crop_pdf_bbox(pdf_doc, page_num_0, coords, max_side=OLM_RENDER_SIDE, pad_pts=0.0)
        if pil is None:
            return ""

        # Шумовое изображение (чёрные точки / пыль): почти полностью белое → пропускаем
        try:
            import numpy as np
            arr = np.array(pil.convert("L"))
            if float(np.mean(arr > 230)) > 0.985:
                return ""
        except Exception:
            pass

        # Сначала olmOCR — определяем тип контента
        if self.olm is not None:
            try:
                raw = self.olm.page_to_markdown(pil).strip()
            except Exception as exc:  # noqa: BLE001
                print(f"  [olmOCR] figure OCR failed: {exc}")
                raw = ""
            if raw and _validate_text(raw, ""):
                is_table = "<table" in raw.lower() or raw.lstrip().startswith("|")
                if is_table:
                    # image_with_table: только текст, без PNG и без image ref
                    tbl = _postprocess_vlm_table(raw)
                    result = tbl if tbl else re.sub(r"<[^>]+>", " ", raw).strip()
                    return filter_noise_lines(result, min_chars=3)
                # Много слов → скан / рукопись / rasterized_pdf: только текст, без PNG
                cleaned = filter_noise_lines(raw[:800], min_chars=3)
                if len(cleaned.split()) > 20:
                    return cleaned
                # Мало слов → подпись на реальном рисунке
                ocr_text = cleaned
            else:
                ocr_text = ""
        else:
            ocr_text = ""

        # Реальный рисунок: назначаем имя только сейчас (счётчик по реально сохранённым PNG)
        self._doc_img_counter += 1
        md_image_name = f"doc_{doc_id}_image_{self._doc_img_counter}.png"
        dest = os.path.join(self.images_dir, md_image_name)
        try:
            thumb = pil.convert("RGB")
            thumb.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.NEAREST)
            thumb.save(dest, "PNG", optimize=True, compress_level=9)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] save figure: {exc}")
            self._doc_img_counter -= 1  # откатываем счётчик — PNG не сохранился
            return ""

        alt = _first_clean_line(ocr_text) or "image"
        for ch in "![]()":
            alt = alt.replace(ch, "")
        alt = alt.strip()[:120] or "image"

        image_md = f"![{alt}](images/{md_image_name})"
        if ocr_text:
            return f"{image_md}\n{ocr_text}"
        return image_md

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def _create_zip(self) -> str:
        zip_path = os.path.join(self.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for md_file in sorted(Path(self.output_dir).glob("document_*.md")):
                zf.write(md_file, md_file.name)
            img_dir = Path(self.images_dir)
            if img_dir.exists():
                for img_file in sorted(img_dir.glob("*.png")):
                    zf.write(img_file, f"images/{img_file.name}")
        return zip_path


# ---------------------------------------------------------------------------
# Docling → пиксельный индекс
# ---------------------------------------------------------------------------


def _build_docling_index(doc) -> dict[int, list[tuple]]:
    """
    Строит {page_num_1based: [(bbox_px, kind, item), ...]}.
    bbox_px — в пикселях YOLO-растра (LAYOUT_DPI), top-left origin.
    kind ∈ {'text', 'table', 'picture'}.
    """
    index: dict[int, list[tuple]] = {}

    pages = getattr(doc, "pages", {}) or {}

    def _page_height_pts(page_num_1: int) -> float:
        try:
            p = pages.get(page_num_1) if isinstance(pages, dict) else pages[page_num_1 - 1]
            size = getattr(p, "size", None)
            return float(getattr(size, "height", 0.0)) if size else 0.0
        except Exception:
            return 0.0

    def _add(bbox_pts, origin: str, page_num_1: int, kind: str, item):
        if not bbox_pts:
            return
        h = _page_height_pts(page_num_1) if origin == "bottom" else None
        try:
            bbox_px = points_to_pixels(bbox_pts, h, origin=origin)
        except Exception:
            return
        index.setdefault(page_num_1, []).append((bbox_px, kind, item))

    def _bbox_from_prov(prov):
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None, None, None
        page_no = getattr(prov, "page_no", None)
        l = float(getattr(bbox, "l", 0.0))
        t = float(getattr(bbox, "t", 0.0))
        r = float(getattr(bbox, "r", 0.0))
        b = float(getattr(bbox, "b", 0.0))
        origin_attr = getattr(bbox, "coord_origin", None)
        origin_val = str(origin_attr).upper() if origin_attr is not None else "TOPLEFT"
        origin = "bottom" if "BOTTOM" in origin_val else "top"
        # В BOTTOMLEFT b > t; в TOPLEFT t < b. Нормализуем порядок:
        y_low, y_high = (min(t, b), max(t, b))
        return page_no, (l, y_low, r, y_high), origin

    # Тексты
    for item in getattr(doc, "texts", None) or []:
        for prov in (getattr(item, "prov", None) or []):
            page_no, bbox, origin = _bbox_from_prov(prov)
            if page_no is not None and bbox is not None:
                _add(bbox, origin, page_no, "text", item)

    # Таблицы
    for item in getattr(doc, "tables", None) or []:
        for prov in (getattr(item, "prov", None) or []):
            page_no, bbox, origin = _bbox_from_prov(prov)
            if page_no is not None and bbox is not None:
                _add(bbox, origin, page_no, "table", item)

    # Картинки (редко нужны — Docling picture ≠ YOLO figure, но полезно для IoM-проверки)
    for item in getattr(doc, "pictures", None) or []:
        for prov in (getattr(item, "prov", None) or []):
            page_no, bbox, origin = _bbox_from_prov(prov)
            if page_no is not None and bbox is not None:
                _add(bbox, origin, page_no, "picture", item)

    return index


def _match_items_by_iom(
    block_bbox_px: list,
    page_items: list,
    *,
    kind_filter: tuple = ("text", "table", "picture"),
    threshold: float = IOM_MATCH_THRESHOLD,
) -> list[tuple[float, str, object]]:
    """
    Возвращает [(iom_score, kind, item), ...] отсортированный по убыванию IoM,
    только элементы с IoM ≥ threshold и нужным kind.
    """
    bbox_a = tuple(block_bbox_px)  # type: ignore[arg-type]
    matches: list[tuple[float, str, object]] = []
    for bbox_b, kind, item in page_items:
        if kind not in kind_filter:
            continue
        score = iom(bbox_a, bbox_b)
        if score >= threshold:
            matches.append((score, kind, item))
    matches.sort(key=lambda x: -x[0])
    return matches


# ---------------------------------------------------------------------------
# Docling TableItem → grid
# ---------------------------------------------------------------------------


def _docling_table_to_grid(tbl) -> Optional[list[list[str]]]:
    """
    Конвертирует Docling TableItem в list[list[str]] для format_table_markdown.
    Объединённые ячейки раскрываются: текст дублируется во все охватываемые позиции.
    """
    try:
        data = getattr(tbl, "data", None)
        if data is None:
            return None
        num_rows = getattr(data, "num_rows", 0)
        num_cols = getattr(data, "num_cols", 0)
        if num_rows == 0 or num_cols == 0:
            return None
        grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
        for cell in (getattr(data, "table_cells", None) or []):
            r0 = getattr(cell, "start_row_offset_idx", 0)
            c0 = getattr(cell, "start_col_offset_idx", 0)
            r1 = getattr(cell, "end_row_offset_idx", r0 + 1)
            c1 = getattr(cell, "end_col_offset_idx", c0 + 1)
            text = (getattr(cell, "text", "") or "").strip().replace("\n", " ")
            for r in range(r0, min(r1, num_rows)):
                for c in range(c0, min(c1, num_cols)):
                    grid[r][c] = text
        return grid if any(any(c for c in row) for row in grid) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# olmOCR таблицы: HTML/pipe → pipe-markdown через format_table_markdown
# ---------------------------------------------------------------------------


_TD_RE = re.compile(r"<(td|th)([^>]*)>(.*?)</(td|th)>", re.S | re.I)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_HTML_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S | re.I)


def _parse_html_table(html: str) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    n_header_rows = 0
    for tr in _TR_RE.finditer(html):
        cells = []
        all_th = True
        for td in _TD_RE.finditer(tr.group(1)):
            tag = td.group(1).lower()
            text = _clean_cell(_TAG_STRIP_RE.sub("", td.group(3)).strip())
            cells.append(text)
            if tag != "th":
                all_th = False
        if cells:
            rows.append(cells)
            if all_th and len(rows) == n_header_rows + 1:
                n_header_rows += 1
    return rows, n_header_rows


_CELL_LEAD_JUNK_RE = re.compile(r"""^['"'\[\]]+""")
_CELL_TRAIL_JUNK_RE = re.compile(r"""['"'\[\];]+$""")
_NUM_SPACE_COMMA_RE = re.compile(r"(\d)\s+([,.])\s*(\d)")
_PCT_DIGIT_RE = re.compile(r"(%|руб\.?)(\d+)")
_DIGIT_COLON_RE = re.compile(r"(\d):$")


def _clean_cell(cell: str) -> str:
    """Убирает OCR-артефакты olmOCR из одной ячейки таблицы."""
    cell = _CELL_LEAD_JUNK_RE.sub("", cell).strip()
    cell = _CELL_TRAIL_JUNK_RE.sub("", cell).strip()
    # 1896 ,15 → 1896,15  /  1896 .15 → 1896.15
    cell = _NUM_SPACE_COMMA_RE.sub(r"\1\2\3", cell)
    # 57.75%6 → 57.75%  (цифра приклеилась после % или руб)
    cell = _PCT_DIGIT_RE.sub(r"\1", cell)
    # 4207: → 4207  (хвостовое двоеточие после числа)
    cell = _DIGIT_COLON_RE.sub(r"\1", cell)
    return cell


def _pipe_rows_to_md(lines: list[str]) -> str:
    """Конвертирует список pipe-строк в pipe-markdown таблицу."""
    rows = []
    for ln in lines:
        if re.match(r"^\|\s*:?-{2,}", ln.strip()):
            continue
        cells = [_clean_cell(c.strip()) for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2:
        return format_table_markdown(rows) or ""
    return ""


def _postprocess_vlm_table(md: str) -> str:
    """
    Нормализует markdown-таблицы из olmOCR в pipe-markdown.
    Если olmOCR вернул несколько таблиц подряд — разделяет их через пустую строку.
    """
    if not md:
        return ""

    parts: list[str] = []

    if "<table" in md.lower():
        # Обрабатываем каждый <table>...</table> блок отдельно
        for block in _HTML_TABLE_RE.findall(md):
            rows, n_hdr = _parse_html_table(block)
            if len(rows) >= 2:
                converted = format_table_markdown(rows, n_header_rows=n_hdr)
                if converted:
                    parts.append(converted)
        return "\n\n".join(parts)

    # pipe-markdown: группируем непрерывные |...| блоки, разбитые пустыми/нетабличными строками
    current: list[str] = []
    for ln in md.splitlines():
        if ln.strip().startswith("|"):
            current.append(ln)
        else:
            if current:
                converted = _pipe_rows_to_md(current)
                if converted:
                    parts.append(converted)
                current = []
    if current:
        converted = _pipe_rows_to_md(current)
        if converted:
            parts.append(converted)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Склейка cross-page таблиц
# ---------------------------------------------------------------------------


def _merge_two_tables(t1: str, t2: str) -> str:
    lines1 = [l for l in t1.splitlines() if l.strip()]
    lines2 = [l for l in t2.splitlines() if l.strip()]
    if not lines1 or not lines2:
        return (t1 + "\n\n" + t2).strip()

    header1 = lines1[0].strip()
    data2: list[str] = []
    for i, ln in enumerate(lines2):
        if i == 0 and ln.strip() == header1:
            continue
        if re.match(r"^\|\s*[-:]+", ln.strip()):
            continue
        data2.append(ln)
    return "\n".join(lines1 + data2)


def _merge_cross_page_tables(flat: list[tuple[str, str, int]]) -> list[str]:
    """Два подряд *_TABLE НА РАЗНЫХ (соседних) СТРАНИЦАХ → склеиваем (разрыв страницы)."""
    result: list[str] = []
    i = 0
    while i < len(flat):
        track, md, page = flat[i]
        while (
            track in TABLE_LABELS
            and i + 1 < len(flat)
            and flat[i + 1][0] in TABLE_LABELS
            and flat[i + 1][2] != page  # только cross-page, не same-page!
        ):
            i += 1
            md = _merge_two_tables(md, flat[i][1])
            page = flat[i][2]
        result.append(md)
        i += 1
    return result


# ---------------------------------------------------------------------------
# Валидаторы (для olmOCR-output)
# ---------------------------------------------------------------------------


def _validate_text(text: str, ref_text: str) -> bool:
    if not text:
        return False
    if repetition_ratio(text) > 0.5:
        return False
    if ref_text and cyrillic_ratio(ref_text) > 0.3 and cyrillic_ratio(text) < 0.1:
        return False
    return True


def _validate_table(md: str, ref_text: str) -> bool:
    if not md:
        return False
    stats = table_stats(md)
    if stats["n_cols"] < 2 or stats["n_rows"] < 1:
        return False
    if stats["n_cols"] > 15 or stats["n_rows"] > 150:
        return False
    if stats["max_cell"] > 300:
        return False
    if stats["row_repeat_ratio"] > 0.4 and stats["n_rows"] > 3:
        return False
    if ref_text and cyrillic_ratio(ref_text) > 0.3 and cyrillic_ratio(md) < 0.1:
        return False
    return True


# ---------------------------------------------------------------------------
# Пост-обработчик документа
# ---------------------------------------------------------------------------


_MD_IMAGE_LINE_RE = re.compile(r"^!?\[.*?\]\(.*?\)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _first_clean_line(text: str) -> str:
    """Первая чистая строка (для alt)."""
    for line in text.splitlines():
        line = line.strip()
        if not line or _MD_IMAGE_LINE_RE.match(line):
            continue
        if line.startswith(("#", "|", "!", "[")):
            continue
        return line[:80]
    return ""


def _doc_id_from_name(stem: str) -> str:
    """document_051 → '51' (без ведущих нулей)."""
    m = re.search(r"(\d+)", stem)
    if m:
        return str(int(m.group(1)))
    return stem


def _postprocess_document(text: str) -> str:
    """
    Rule-based чистка:
      0. Вотермарки (ЧЕРНОВИК/DRAFT) в нетабличных строках.
      1. Trailing whitespace.
      2. 3+ пустых строки → 2.
      3. Дубли соседних блоков.
      4. Дубли соседних заголовков.
      5. Нормализация уровней заголовков (min → #).
    """
    if not text:
        return text

    # -1. Убираем Docling-внутренние ссылки на изображения (page_N_N_N_N.png)
    text = _DOCLING_IMG_REF_RE.sub("", text)

    # 0. Вотермарки (нетабличные строки и внутри ячеек таблиц)
    lines_wm = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|"):
            # Чистим ЧЕРНОВИК/DRAFT из ячеек: | ЧЕРНОВИК | → |  |
            ln = re.sub(
                r"(?<=\|)\s*(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)\s*(?=\|)",
                "  ",
                ln,
                flags=re.I | re.UNICODE,
            )
            lines_wm.append(ln)
        elif _WM_ONLY_RE.match(s):
            pass
        else:
            ln = _WM_PREFIX_RE.sub("", ln)
            lines_wm.append(ln)
    text = "\n".join(lines_wm)

    # 1. Trailing whitespace
    lines = [ln.rstrip() for ln in text.splitlines()]

    # 2. Blank runs
    cleaned: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(ln)
        else:
            blank_run = 0
            cleaned.append(ln)
    text = "\n".join(cleaned)

    # 3. Дубли соседних блоков
    blocks = text.split("\n\n")
    deduped: list[str] = []
    prev_block = None
    for blk in blocks:
        norm = blk.strip()
        if norm and norm == prev_block:
            continue
        deduped.append(blk)
        if norm:
            prev_block = norm
    text = "\n\n".join(deduped)

    # 4. Дубли соседних заголовков
    lines = text.splitlines()
    result: list[str] = []
    prev_heading: Optional[str] = None
    for ln in lines:
        m = _HEADING_RE.match(ln)
        if m:
            heading_text = m.group(2).strip().lower()
            if heading_text == prev_heading:
                continue
            prev_heading = heading_text
        else:
            if ln.strip():
                prev_heading = None
        result.append(ln)
    text = "\n".join(result)

    # 5. Нормализация уровней
    lines = text.splitlines()
    min_level = 6
    for ln in lines:
        m = _HEADING_RE.match(ln)
        if m:
            min_level = min(min_level, len(m.group(1)))
    if min_level > 1:
        shift = min_level - 1
        normalized: list[str] = []
        for ln in lines:
            m = _HEADING_RE.match(ln)
            if m:
                new_level = max(1, len(m.group(1)) - shift)
                normalized.append("#" * new_level + " " + m.group(2))
            else:
                normalized.append(ln)
        text = "\n".join(normalized)

    return text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF → Markdown (DocLayout-YOLO + Docling + olmOCR)"
    )
    parser.add_argument("--pdf", type=str, help="Путь к одному PDF")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-vlm", action="store_true", help="Отключить olmOCR fallback")
    args = parser.parse_args()

    pipeline = Pipeline(output_dir=args.output_dir, use_vlm=not args.no_vlm)

    if args.all:
        pipeline.process_all(args.raw_dir)
    elif args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Файл не найден: {args.pdf}")
            sys.exit(1)
        pipeline.process_pdf(args.pdf)
        pipeline._create_zip()
    else:
        print("Используйте --pdf <path> или --all")
        parser.print_help()
