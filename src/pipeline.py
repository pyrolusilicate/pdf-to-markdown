"""
Главный пайплайн: PDF -> Markdown + images/ -> submission.zip.

Гибридная архитектура из трёх движков:

  Router     : DocLayout-YOLOv10 (layout_router.py).
               Определяет bbox блоков, отбрасывает вотермарки/колонтитулы/пустые
               места, задаёт reading order (band -> columns -> top-down).

  Fast Track : Docling DocumentConverter - один convert(pdf) на документ.
               Отдаёт текст и таблицы; сопоставляем с YOLO-блоками по IoM,
               форматирование берём из реальных типов Docling-элементов
               (SectionHeaderItem, ListItem, TableItem и т.д.).

  Fallback   : olmOCR-2-7B-1025 BF16 (Qwen2.5-VL) - crop YOLO-блока -> markdown.
               Срабатывает, если Docling не выдал контент (пустой OCR на скане,
               битый текстовый слой, figure с встроенным текстом).

Маппинг YOLO-класса -> обработчик:
  title/section-header/list-item/text  -> _process_text
  table                                -> _process_table
  picture/figure/image                 -> _process_figure (OCR + PNG)

Использование (CLI):
  python src/pipeline.py --all
  python src/pipeline.py --pdf data/raw/document_001.pdf
  python src/pipeline.py --all --no-vlm        # Docling-only, без olmOCR
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

from config import (
    IMAGE_MAX_SIDE,
    IOM_MATCH_THRESHOLD,
    MAX_AREA_RATIO,
    OLM_RENDER_SIDE,
    OUTPUT_DIR,
)
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


# Водяные знаки, которые YOLO-классификатор иногда не отбрасывает как abandon.
_WM_ONLY_RE = re.compile(
    r"^(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)\s*$",
    re.I | re.UNICODE,
)
_WM_PREFIX_RE = re.compile(
    r"^(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)\s+",
    re.I | re.UNICODE,
)

# Внутренние Docling-ссылки на изображения (page_0_0_1280_960.png) нужно убирать,
# так как наш экспорт использует собственные имена doc_<id>_image_<N>.png.
_DOCLING_IMG_REF_RE = re.compile(
    r"!\[[^\]]*\]\(page_\d+[^\)]*\.(?:png|jpg|jpeg)\)", re.I
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Оркестратор PDF -> Markdown.

    Запускает YOLO-роутер, вызывает Docling и (опционально) olmOCR, собирает
    markdown и сохраняет картинки. Один экземпляр может обработать много
    документов подряд - движки закэшированы как singletons и не перезагружаются.
    """

    def __init__(self, output_dir: str = OUTPUT_DIR, use_vlm: bool = True):
        """
        Args:
            output_dir: директория для .md и images/, создаётся при отсутствии.
            use_vlm: инициализировать olmOCR fallback. На Mac/CPU ставь False.
        """
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
        """
        Обрабатывает один PDF и пишет ``<stem>.md`` в ``self.output_dir``.

        Пайплайн: YOLO routing -> Docling convert -> per-block match/fallback ->
        склейка cross-page таблиц -> rule-based post-processing.

        Returns:
            Абсолютный путь к созданному .md файлу.
        """
        doc_name = Path(pdf_path).stem
        doc_id = _doc_id_from_name(doc_name)

        plan = self.router.build_routing_plan(pdf_path, self.output_dir)
        docling_doc = self.docling.convert(pdf_path)
        docling_idx = _build_docling_index(docling_doc)

        pdf_doc = fitz.open(pdf_path)
        # Счётчик реально сохранённых PNG для doc_<id>_image_<N>.png.
        self._doc_img_counter = 0
        # Docling-элементы, уже использованные в этом документе: защита от
        # повторного попадания одного абзаца в два YOLO-блока.
        self._used_item_ids: set[int] = set()
        # (yolo_type, markdown, page_num) - плоский поток перед склейкой.
        flat: list[tuple[str, str, int]] = []

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
                        flat.append((block["type"], md.rstrip(), page_num_1))

                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            pdf_doc.close()

        flat = _drop_figures_without_caption(flat, self.images_dir)
        parts = _merge_cross_page_tables(flat)

        markdown = _postprocess_document("\n\n".join(parts))
        md_path = os.path.join(self.output_dir, f"{doc_name}.md")
        Path(md_path).write_text(markdown, encoding="utf-8")
        print(f"  ✓ {doc_name}.md")
        return md_path

    def process_all(self, raw_dir: str = "data/raw") -> str:
        """
        Обрабатывает все "document_*.pdf" из "raw_dir" и пакует результат.

        Ошибки одного документа не прерывают работу - логируются и пропускаются,
        чтобы на 100 PDF один битый файл не ломал весь прогон.

        Returns:
            Путь к "submission.zip".
        """
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
        """Маршрутизация YOLO-блока в соответствующий handler по его типу."""
        btype = block["type"]

        if btype in FIGURE_LABELS:
            return self._process_figure(block, page_num_0, pdf_doc, doc_id)
        if btype in TABLE_LABELS:
            return self._process_table(block, page_num_0, pdf_doc, page_items)
        return self._process_text(block, page_num_0, pdf_doc, page_items)

    # ---- текстовые блоки ----------------------------------------------

    def _process_text(
        self,
        block: dict,
        page_num_0: int,
        pdf_doc: fitz.Document,
        page_items: list,
    ) -> str:
        """
        Обработка текстового YOLO-блока (title / section-header / list-item / text).

        1. Fast Track: матчинг с Docling-text-items по IoM; форматирование
           берётся из реального Docling-типа (SectionHeaderItem.level, ListItem).
        2. Fallback: если Docling-матчей нет, рендерим crop и гоним olmOCR.
        3. Markdown: приоритет форматирования - Docling-тип -> YOLO-btype ->
           эвристика по первой строке.
        """
        coords = block["coords"]
        btype = block["type"]

        matched = _match_items_by_iom(coords, page_items, kind_filter=("text",))
        # Восстанавливаем порядок чтения (сортируем по Docling, а не по площади)
        docling_order = {id(itm): idx for idx, (_, _, itm) in enumerate(page_items)}
        matched.sort(key=lambda x: docling_order.get(id(x[2]), 0))

        text_parts = []
        docling_heading_level: int = 0
        docling_is_list: bool = False

        for _, _, item in matched:
            item_id = id(item)
            if item_id in self._used_item_ids:
                continue
            t = getattr(item, "text", "") or ""
            t = _DOCLING_IMG_REF_RE.sub("", t).rstrip()
            if not t:
                continue
            self._used_item_ids.add(item_id)
            text_parts.append(t)

            # Форматирование определяет ПЕРВЫЙ непустой матч - он же и самый
            # крупный по IoM (matches отсортированы по score).
            if docling_heading_level == 0 and not docling_is_list:
                itype = type(item).__name__
                if itype == "SectionHeaderItem":
                    lvl = getattr(item, "level", None)
                    docling_heading_level = (
                        max(3, int(lvl)) if isinstance(lvl, int) and 1 <= lvl <= 6 else 2
                    )
                elif itype == "ListItem":
                    docling_is_list = True

        text = "\n".join(text_parts)
        text = filter_noise_lines(text, min_chars=3)

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

        heading_level = 0

        if btype == "title":
            first_line = text.splitlines()[0].strip() if text.strip() else ""
            # Пронумерованный пункт («1. Введение») - подраздел ####, а не ###.
            heading_level = 4 if re.match(r"^\d+[\d\.]*\s", first_line) else 3
        elif docling_heading_level:
            heading_level = docling_heading_level
        elif btype == "section-header":
            first_line = text.splitlines()[0].strip() if text.strip() else ""
            heading_level = 4 if re.match(r"^\d+[\d\.]*\s", first_line) else 3

        if heading_level > 0:
            # Иногда YOLO-bbox захватывает и заголовок, и первый абзац тела -
            # разделяем их на два markdown-блока.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) > 1:
                heading_md = format_text_markdown(
                    lines[0], "section-header", heading_level
                )
                body_text = _as_list_if_needed(lines[1:]) or "\n".join(lines[1:])
                return f"{heading_md}\n\n{body_text}"
            return format_text_markdown(text, "section-header", heading_level)

        if docling_is_list or btype == "list-item":
            return format_text_markdown(text, "list-item", 0)

        return _as_list_if_needed(text.splitlines()) or format_text_markdown(
            text, btype, 0
        )

    # ---- таблицы ------------------------------------------------------

    def _process_table(
        self,
        block: dict,
        page_num_0: int,
        pdf_doc: fitz.Document,
        page_items: list,
    ) -> str:
        """
        Обработка YOLO-table: TableFormer (Docling) -> pipe-markdown; иначе olmOCR.

        Используется только лучший Docling-матч по IoM, остальные не пробуются -
        в 99% случаев top-match и есть таблица, а попытки fallback-матчей дают
        дубли / артефакты.
        """
        coords = block["coords"]

        matched = _match_items_by_iom(coords, page_items, kind_filter=("table",))
        if matched:
            for _, _, tbl_item in matched:
                tbl_id = id(tbl_item)
                if tbl_id in self._used_item_ids:
                    continue
                self._used_item_ids.add(tbl_id)
                grid = _docling_table_to_grid(tbl_item)
                if grid:
                    md = format_table_markdown(grid)
                    if md and _validate_table(md, ""):
                        return md
                break

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
                # VLM вернул текст без табличной структуры - принимаем как
                # текстовый абзац, но только если нет обрезанного HTML.
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
        """
        Обработка figure/picture/image-блока.

        Логика ветвится по содержимому:
          * почти-белый crop (>98%) -> шум сканера, пропуск;
          * olmOCR распознал таблицу -> только text/markdown, PNG не сохраняем;
          * olmOCR распознал много текста (>20 слов) -> это скан страницы,
            сохраняем только текст;
          * иначе -> сохраняем PNG (с опциональным OCR-подписью внизу).
        """
        coords = block["coords"]
        if not block.get("md_image_name"):
            return ""

        pil = crop_pdf_bbox(
            pdf_doc, page_num_0, coords, max_side=OLM_RENDER_SIDE, pad_pts=0.0
        )
        if pil is None:
            return ""

        _white_ratio = 0.0
        try:
            import numpy as np
            arr = np.array(pil.convert("L"))
            _white_ratio = float(np.mean(arr > 230))
            if _white_ratio > 0.98:
                return ""
        except Exception:
            pass

        if self.olm is not None:
            try:
                raw = self.olm.page_to_markdown(pil).strip()
            except Exception as exc:  # noqa: BLE001
                print(f"  [olmOCR] figure OCR failed: {exc}")
                raw = ""
            if raw and _validate_text(raw, ""):
                is_table = "<table" in raw.lower() or raw.lstrip().startswith("|")
                if is_table:
                    tbl = _postprocess_vlm_table(raw)
                    result = tbl if tbl else re.sub(r"<[^>]+>", " ", raw).strip()
                    return filter_noise_lines(result, min_chars=3)
                cleaned = filter_noise_lines(raw[:800], min_chars=3)
                if len(cleaned.split()) > 20:
                    return cleaned
                ocr_text = cleaned
            else:
                ocr_text = ""
        else:
            ocr_text = ""

        if not ocr_text and _white_ratio > 0.95:
            return ""

        # Имя присваиваем только сейчас: счётчик считает РЕАЛЬНО сохранённые PNG,
        # а не все YOLO-figure блоки (многие из них окажутся шумом / скан-текстом).
        self._doc_img_counter += 1
        md_image_name = f"doc_{doc_id}_image_{self._doc_img_counter}.png"
        dest = os.path.join(self.images_dir, md_image_name)
        try:
            thumb = pil.convert("RGB")
            thumb.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.NEAREST)
            thumb.save(dest, "PNG", optimize=True, compress_level=9)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] save figure: {exc}")
            self._doc_img_counter -= 1
            return ""

        alt = "text" if ocr_text else "image"
        image_md = f"![{alt}](images/{md_image_name})"
        if ocr_text:
            return f"{image_md}\n\n{ocr_text}"
        return image_md

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def _create_zip(self) -> str:
        """Пакует все document_*.md + images/*.png в submission.zip."""
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
# Docling -> пиксельный индекс
# ---------------------------------------------------------------------------


def _build_docling_index(doc) -> dict[int, list[tuple]]:
    """
    Строит индекс Docling-элементов по страницам в координатах YOLO-растра.

    Returns:
        {page_num_1based: [(bbox_px, kind, item), ...]}, где bbox_px
        - кортеж (x1, y1, x2, y2) в пикселях LAYOUT_DPI с top-left origin,
        а kind in {'text', 'table', 'picture'}.

    Docling отдаёт bbox в PDF points с разным origin (TOPLEFT / BOTTOMLEFT);
    нормализуем к единой системе через points_to_pixels.
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

    # Картинки (редко нужны - Docling picture != YOLO figure, но полезно для IoM-проверки)
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
    Сопоставляет YOLO-блок с Docling-items одной страницы по IoM.

    Возвращает список (iom_score, kind, item), отсортированный по убыванию
    IoM; только элементы с IoM >= threshold и нужным kind.

    Отсекаем Docling-items, у которых bbox > MAX_AREA_RATIO * YOLO-bbox:
    это «whole-section» элементы (страница целиком, раздел), и без фильтра они
    склеивают весь текст в первый попавшийся YOLO-блок, блокируя остальные
    через _used_item_ids.
    """
    x1a, y1a, x2a, y2a = block_bbox_px[:4]
    yolo_area = max(1.0, (x2a - x1a) * (y2a - y1a))

    matches: list[tuple[float, str, object]] = []
    for bbox_b, kind, item in page_items:
        if kind not in kind_filter:
            continue
        x1b, y1b, x2b, y2b = bbox_b[:4]
        docling_area = max(1.0, (x2b - x1b) * (y2b - y1b))
        if docling_area / yolo_area > MAX_AREA_RATIO:
            continue
        score = iom(tuple(block_bbox_px), bbox_b)
        if score >= threshold:
            matches.append((score, kind, item))
    matches.sort(key=lambda x: -x[0])
    return matches


# ---------------------------------------------------------------------------
# Docling TableItem -> grid
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
# olmOCR таблицы: HTML/pipe -> pipe-markdown через format_table_markdown
# ---------------------------------------------------------------------------


_TD_RE = re.compile(r"<(td|th)([^>]*)>(.*?)</(td|th)>", re.S | re.I)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_HTML_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S | re.I)


def _parse_html_table(html: str) -> tuple[list[list[str]], int]:
    """Парсит <table> -> (rows, n_header_rows); хедером считается префикс <th>-строк."""
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
    # 1896 ,15 -> 1896,15  /  1896 .15 -> 1896.15
    cell = _NUM_SPACE_COMMA_RE.sub(r"\1\2\3", cell)
    # 57.75%6 -> 57.75%  (цифра приклеилась после % или руб)
    cell = _PCT_DIGIT_RE.sub(r"\1", cell)
    # 4207: -> 4207  (хвостовое двоеточие после числа)
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
    Если olmOCR вернул несколько таблиц подряд - разделяет их через пустую строку.
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
    """
    Склеивает две pipe-markdown таблицы (продолжение через разрыв страницы).

    Из второй таблицы удаляются: повтор хедера (если совпадает) и разделительная
    строка "| --- |" - чтобы получить один непрерывный markdown.
    """
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


_RIS_CAPTION_RE = re.compile(r"^Рис\s*\.\s*\d+", re.I | re.UNICODE)
_IMG_REF_LINE_RE = re.compile(r"^!\[(?:text|image)\]\(images/(doc_\d+_image_\d+\.png)\)", re.I)
_IMG_ANY_REF_RE = re.compile(r"(images/)(doc_\d+_image_)(\d+)(\.png)", re.I)


def _drop_figures_without_caption(
    flat: list[tuple[str, str, int]],
    images_dir: str,
) -> list[tuple[str, str, int]]:
    """
    Удаляет PNG и ссылки для figure-блоков без подписи «Рис. N.» в следующих 3 блоках.
    После удаления переименовывает оставшиеся PNG чтобы нумерация шла без пробелов (1,2,3...).
    """
    result = list(flat)

    # Шаг 1: определяем какие PNG-ссылки удалять, а какие оставить
    to_drop: set[str] = set()   # имена PNG которые надо удалить
    to_keep: list[str] = []     # имена PNG в порядке появления (только выжившие)

    for idx in range(len(result)):
        ftype, fmd, _ = result[idx]
        if ftype not in FIGURE_LABELS:
            continue
        first_line = fmd.strip().splitlines()[0].strip() if fmd.strip() else ""
        m = _IMG_REF_LINE_RE.match(first_line)
        if not m:
            continue
        png_name = m.group(1)

        has_ris = any(
            _RIS_CAPTION_RE.match(result[j][1].strip())
            for j in range(idx + 1, min(idx + 6, len(result)))
        )
        if has_ris:
            to_keep.append(png_name)
        else:
            to_drop.add(png_name)

    # Шаг 2: строим карту переименования для выживших PNG (убираем пробелы в нумерации)
    rename_map: dict[str, str] = {}
    for new_idx, old_name in enumerate(to_keep, start=1):
        # old_name: doc_5_image_3.png -> new_name: doc_5_image_1.png (после удалений)
        new_name = _IMG_ANY_REF_RE.sub(
            lambda mo, n=new_idx: f"{mo.group(2)}{n}{mo.group(4)}",
            f"images/{old_name}",
        ).removeprefix("images/")
        rename_map[old_name] = new_name

    # Шаг 3: удаляем шумовые PNG
    for png_name in to_drop:
        try:
            os.remove(os.path.join(images_dir, png_name))
        except OSError:
            pass

    # Шаг 4: переименовываем выжившие PNG (в два прохода чтобы избежать коллизий)
    tmp_names: dict[str, str] = {}
    for old_name, new_name in rename_map.items():
        if old_name == new_name:
            continue
        old_path = os.path.join(images_dir, old_name)
        tmp_path = old_path + ".tmp_rename"
        try:
            os.rename(old_path, tmp_path)
            tmp_names[tmp_path] = os.path.join(images_dir, new_name)
        except OSError:
            pass
    for tmp_path, new_path in tmp_names.items():
        try:
            os.rename(tmp_path, new_path)
        except OSError:
            pass

    # Шаг 5: обновляем ссылки в flat
    def _remap_md(md: str, drop: set[str], rmap: dict[str, str]) -> str:
        for old_name in drop:
            # Убираем строку с удалённой ссылкой
            md = "\n".join(
                ln for ln in md.splitlines()
                if old_name not in ln
            )
        for old_name, new_name in rmap.items():
            md = md.replace(f"images/{old_name}", f"images/{new_name}")
        return md.strip()

    result = [
        (ftype, _remap_md(fmd, to_drop, rename_map), fpage)
        for ftype, fmd, fpage in result
    ]
    return result


def _merge_cross_page_tables(flat: list[tuple[str, str, int]]) -> list[str]:
    """Два подряд *_TABLE НА РАЗНЫХ (соседних) СТРАНИЦАХ -> склеиваем (разрыв страницы)."""
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
    """
    Анти-галлюцинационный фильтр для olmOCR-текста.

    Отбрасываем, если:
      * повтор одной строки > 50% (циклическая генерация VLM);
      * исходный текст русский, а модель вывела почти чистый английский.
    """
    if not text:
        return False
    if repetition_ratio(text) > 0.5:
        return False
    if ref_text and cyrillic_ratio(ref_text) > 0.3 and cyrillic_ratio(text) < 0.1:
        return False
    return True


def _validate_table(md: str, ref_text: str) -> bool:
    """
    Структурная валидация markdown-таблицы.

    Ориентиры: минимум 2 колонки, разумные размеры, нет дублирующихся строк
    (признак VLM-галлюцинации), нет перевода RU->EN.
    """
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
# Форматирование списков
# ---------------------------------------------------------------------------

# Признаки пункта списка: маркер (•▪◦▸-*)
_LIST_BULLET_RE = re.compile(r"^\s*[•▪◦▸\-\*]\s")

def _as_list_if_needed(lines: list[str] | str) -> str:
    """
    Форматирует только маркированные списки. Нумерованные игнорируются.
    Сохраняет вложенность.
    """
    if isinstance(lines, str):
        lines = lines.splitlines()
        
    # Важно: фильтруем пустые строки, но не делаем strip() для элементов
    valid_lines = [line for line in lines if line.strip()]
    
    # МИКРОФИКС: Разрешаем списки из 1 пункта (на случай, если YOLO отрезал его в отдельный блок).
    if not valid_lines:
        return ""
    
    # Проверяем только на наличие маркера
    is_item = [bool(_LIST_BULLET_RE.match(line)) for line in valid_lines]
    if sum(is_item) < len(valid_lines):
        return ""
        
    result = []
    for line in valid_lines:
        # Сохраняем оригинальный отступ
        indent_match = re.match(r"^(\s*)", line)
        indent = indent_match.group(1) if indent_match else ""
        clean_line = line.lstrip()

        # Умная генерация отступа, если движок съел пробелы, но оставил спецсимволы
        if not indent:
            if clean_line.startswith('◦'): indent = "  "
            elif clean_line.startswith('▪') or clean_line.startswith('▸'): indent = "    "

        # Нормализуем маркер в "- " с сохранением отступа
        result.append(f"{indent}- {clean_line[2:]}")
        
    return "\n".join(result)

# ---------------------------------------------------------------------------
# Пост-обработчик документа
# ---------------------------------------------------------------------------


_MD_IMAGE_LINE_RE = re.compile(r"^!?\[.*?\]\(.*?\)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _first_clean_line(text: str) -> str:
    """Первая содержательная строка текста (не markdown-разметка) - до 80 символов."""
    for line in text.splitlines():
        line = line.strip()
        if not line or _MD_IMAGE_LINE_RE.match(line):
            continue
        if line.startswith(("#", "|", "!", "[")):
            continue
        return line[:80]
    return ""


def _doc_id_from_name(stem: str) -> str:
    """document_051 -> 51 (без ведущих нулей)."""
    m = re.search(r"(\d+)", stem)
    if m:
        return str(int(m.group(1)))
    return stem


def _postprocess_document(text: str) -> str:
    """
    Rule-based чистка финального markdown.

    Шаги:
      -1. Убрать внутренние Docling-ссылки на картинки (page_*.png).
       0. Водяные знаки (ЧЕРНОВИК/DRAFT/…) - в строках и внутри ячеек таблиц.
       1. Trailing whitespace.
       2. Свернуть прогоны пустых строк до максимум двух.
       3. Дедупликация соседних блоков.
      3.5. Переупорядочить «![...]» -> «Рис. N.» -> OCR-текст (если порядок
           после OCR оказался перевёрнут).
      3.6. Удалить ссылки на картинки без подписи «Рис. N.» - шум.
       4. Дедупликация соседних заголовков.
    """
    if not text:
        return text

    text = _DOCLING_IMG_REF_RE.sub("", text)

    # Водяные знаки. Внутри таблиц заменяем в ячейках, чтобы не ломать структуру;
    # обычные строки-вотермарки выкидываем целиком.
    lines_wm = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|"):
            ln = re.sub(
                r"(?<=\|)\s*(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|ОБРАЗЕЦ|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)\s*(?=\|)",
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

    lines = [ln.rstrip() for ln in text.splitlines()]

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
    blocks = deduped

    # Склеиваем разорванные куски списков (если YOLO разбил список на два блока)
    list_merged = []
    _ANY_LIST_RE = re.compile(r"^\s*[-*]\s")
    for blk in blocks:
        if not list_merged:
            list_merged.append(blk)
            continue
        
        prev = list_merged[-1]
        prev_last_line = prev.splitlines()[-1] if prev.strip() else ""
        curr_first_line = blk.splitlines()[0] if blk.strip() else ""
        
        # Если конец прошлого блока и начало текущего - это пункты списка, склеиваем в один список!
        if _ANY_LIST_RE.match(prev_last_line) and _ANY_LIST_RE.match(curr_first_line):
            list_merged[-1] = prev + "\n" + blk
        else:
            list_merged.append(blk)
    blocks = list_merged

    # OCR иногда подсовывает описание ДО подписи «Рис. N.» - исправляем порядок.
    _RIS_RE = re.compile(r"^Рис\s*\.\s*\d+", re.I | re.UNICODE)
    _IMG_BLOCK_RE = re.compile(r"^!\[(?:text|image)\]\(images/", re.I)
    reordered: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i].strip()
        if (
            i + 2 < len(blocks)
            and _IMG_BLOCK_RE.match(b)
            and not _RIS_RE.match(blocks[i + 1].strip())
            and _RIS_RE.match(blocks[i + 2].strip())
        ):
            reordered.extend([blocks[i], blocks[i + 2], blocks[i + 1]])
            i += 3
        else:
            reordered.append(blocks[i])
            i += 1

    # В этом датасете у значимых картинок ВСЕГДА есть подпись «Рис. N.»;
    # если её нет - это или шум, или поймана зря (логотип, декоративный элемент).
    filtered: list[str] = []
    i = 0
    while i < len(reordered):
        b = reordered[i].strip()
        if _IMG_BLOCK_RE.match(b):
            next_b = reordered[i + 1].strip() if i + 1 < len(reordered) else ""
            if _RIS_RE.match(next_b):
                filtered.append(reordered[i])
        else:
            filtered.append(reordered[i])
        i += 1

    text = "\n\n".join(filtered)

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

    return text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF -> Markdown (DocLayout-YOLO + Docling + olmOCR)"
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
