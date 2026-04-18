"""
Главный пайплайн: PDF → Markdown + images/ → submission.zip

Архитектура:
    Layout Engine  : DocLayout-YOLO (src/layout_router.py)
    Vector Engine  : PyMuPDF (текст)
    Raster Engine  : DeepSeek-OCR-2 (src/vlm_engine.py) — все таблицы через VLM

Треки, в которые перераспределяются блоки YOLO после is_vector-анализа:
    VECTOR_TEXT  — текст/заголовок/список, поверх которого есть текстовый слой PDF
    RASTER_TEXT  — текст/заголовок/список без текстового слоя (скан)
    RASTER_TABLE — любая таблица (DeepSeek → markdown)
    SMART_FIGURE — figure/picture/image: VLM-классификация → markdown/ocr/PNG

Использование:
    python src/pipeline.py --all
    python src/pipeline.py --pdf data/raw/document_001.pdf
    python src/pipeline.py --all --no-vlm         # только векторные треки
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from content_extractor import (
    collect_heading_sizes,
    detect_heading_level,
    extract_text_block,
    filter_noise_lines,
    format_table_markdown,
    format_text_markdown,
    has_vector_text,
    render_block_image,
)
from device import setup_environment
from layout_router import LayoutRouter

setup_environment()

OUTPUT_DIR = "data/output"
IMAGE_MAX_SIDE = 2400
VLM_RENDER_DPI = 300

FIGURE_LABELS = {"picture", "figure", "image", "missed_raster"}
TEXT_LABELS = {"text", "plain text", "title", "section-header", "list-item"}


# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        use_vlm: bool = True,
    ):
        self.output_dir = output_dir
        self.use_vlm = use_vlm

        self.router = LayoutRouter()

        self.vlm = None
        if use_vlm:
            try:
                from vlm_engine import VLMEngine

                self.vlm = VLMEngine.get()
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] VLM недоступен: {exc}")
                self.vlm = None

        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str) -> str:
        plan = self.router.build_routing_plan(pdf_path, self.output_dir)
        doc_name = Path(pdf_path).stem
        pdf_doc = fitz.open(pdf_path)

        self._enrich_plan_with_tracks(plan, pdf_doc)

        heading_sizes = collect_heading_sizes(pdf_doc, plan)

        chunks: list[str] = []
        for page_data in plan["pages"]:
            chunks.append(
                self._process_page(page_data, pdf_doc, heading_sizes)
            )
            # Очищаем VRAM между страницами (особенно важно на Colab/40GB).
            if self.vlm is not None:
                self.vlm.release_page_cache()
            else:
                gc.collect()

        pdf_doc.close()

        markdown = "\n\n".join(c for c in chunks if c.strip())
        md_path = os.path.join(self.output_dir, f"{doc_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        self._cleanup_temp(plan)
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
    # Роутинг: добавление is_vector и правильных треков
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_plan_with_tracks(plan: dict, pdf_doc: fitz.Document) -> None:
        """
        Layout-роутер не знает про векторный слой PDF. Проходим по плану и:
          - помечаем каждый блок `is_vector`
          - перевешиваем `track` согласно ТЗ (5 треков)
        """
        for page_data in plan["pages"]:
            page_num = page_data["page_num"] - 1
            for block in page_data["blocks"]:
                btype = block["type"]
                coords = block["coords"]

                if btype in FIGURE_LABELS:
                    block["track"] = "SMART_FIGURE"
                    block["is_vector"] = False
                    continue

                is_vec = has_vector_text(pdf_doc, page_num, coords, min_chars=4)
                block["is_vector"] = is_vec

                if btype == "table":
                    block["track"] = "RASTER_TABLE"
                elif btype in TEXT_LABELS:
                    block["track"] = "VECTOR_TEXT" if is_vec else "RASTER_TEXT"
                else:
                    # Незнакомый label — обрабатываем как текст.
                    block["track"] = "VECTOR_TEXT" if is_vec else "RASTER_TEXT"

    # ------------------------------------------------------------------
    # Страница / блок
    # ------------------------------------------------------------------

    def _process_page(
        self,
        page_data: dict,
        pdf_doc: fitz.Document,
        heading_sizes: list[float],
    ) -> str:
        page_num = page_data["page_num"] - 1
        parts: list[str] = []
        for block in page_data["blocks"]:
            md = self._process_block(block, page_num, pdf_doc, heading_sizes)
            if md and md.strip():
                parts.append(md.strip())
        return "\n\n".join(parts)

    def _process_block(
        self,
        block: dict,
        page_num: int,
        pdf_doc: fitz.Document,
        heading_sizes: list[float],
    ) -> str:
        track = block["track"]
        btype = block["type"]
        coords = block["coords"]

        if track == "VECTOR_TEXT":
            return self._vector_text(btype, coords, page_num, pdf_doc, heading_sizes)

        if track == "RASTER_TABLE":
            return self._raster_table(page_num, coords, pdf_doc)

        if track == "RASTER_TEXT":
            return self._raster_text(btype, coords, page_num, pdf_doc)

        if track == "SMART_FIGURE":
            return self._smart_figure(block, page_num, pdf_doc)

        return ""

    # ------------------------------------------------------------------
    # VECTOR_TEXT
    # ------------------------------------------------------------------

    def _vector_text(
        self,
        btype: str,
        coords: list,
        page_num: int,
        pdf_doc: fitz.Document,
        heading_sizes: list[float],
    ) -> str:
        text, font_size = extract_text_block(pdf_doc, page_num, coords)
        text = filter_noise_lines(text, min_chars=3)
        if not text:
            return ""

        heading_level = 0
        if btype in ("title", "section-header"):
            heading_level = detect_heading_level(font_size, heading_sizes)
        return format_text_markdown(text, btype, heading_level)

    # ------------------------------------------------------------------
    # RASTER_TEXT
    # ------------------------------------------------------------------

    def _raster_text(
        self,
        btype: str,
        coords: list,
        page_num: int,
        pdf_doc: fitz.Document,
    ) -> str:
        if self.vlm is None:
            return ""
        img = render_block_image(pdf_doc, page_num, coords, dpi=VLM_RENDER_DPI)
        # extract_markdown лучше справляется с русским и смешанными блоками,
        # чем free_ocr (который оптимизирован под короткие блоки на латинице).
        text = self.vlm.extract_markdown(img).strip()
        text = filter_noise_lines(text, min_chars=3)
        if not text:
            return ""

        heading_level = 0
        if btype == "title":
            heading_level = 1
        elif btype == "section-header":
            heading_level = 2
        return format_text_markdown(text, btype, heading_level)

    # ------------------------------------------------------------------
    # RASTER_TABLE
    # ------------------------------------------------------------------

    def _raster_table(self, page_num: int, coords: list, pdf_doc: fitz.Document) -> str:
        if self.vlm is None:
            return ""
        img = render_block_image(pdf_doc, page_num, coords, dpi=VLM_RENDER_DPI)
        md = self.vlm.extract_markdown(img).strip()
        return _postprocess_table_markdown(md)

    # ------------------------------------------------------------------
    # SMART_FIGURE
    # ------------------------------------------------------------------

    def _smart_figure(self, block: dict, page_num: int, pdf_doc: fitz.Document) -> str:
        coords = block["coords"]
        md_image_name = block.get("md_image_name")
        content_path = block.get("content_path")

        pil_img: Optional[Image.Image] = None
        if content_path and os.path.exists(content_path):
            try:
                pil_img = Image.open(content_path).convert("RGB")
            except Exception:
                pil_img = None
        if pil_img is None:
            pil_img = render_block_image(pdf_doc, page_num, coords, dpi=VLM_RENDER_DPI)

        # Всегда сохраняем изображение — метрика проверяет наличие файла.
        alt = "image"
        if self.vlm is not None:
            try:
                alt = self.vlm.short_caption(pil_img) or "image"
            except Exception:
                pass
        image_md = self._save_figure(pil_img, md_image_name, alt=alt)

        if self.vlm is None:
            return image_md

        kind = "picture"
        try:
            kind = self.vlm.classify(pil_img)
        except Exception:
            pass

        if kind == "table":
            md = self.vlm.extract_markdown(pil_img).strip()
            md = _postprocess_table_markdown(md)
            if md:
                return md  # таблица — только markdown, без image-ссылки

        # Для текста и рукописного — OCR через extract_markdown (лучше для русского).
        if kind in ("text", "handwritten"):
            text = ""
            try:
                text = self.vlm.extract_markdown(pil_img).strip()
                text = filter_noise_lines(text, min_chars=3)
            except Exception:
                pass
            if text:
                return f"{image_md}\n\n{text}"

        return image_md

    def _save_figure(
        self,
        pil_img: Image.Image,
        md_image_name: Optional[str],
        alt: str,
    ) -> str:
        if not md_image_name:
            return ""
        dest = os.path.join(self.images_dir, md_image_name)
        try:
            img = pil_img
            if max(img.width, img.height) > IMAGE_MAX_SIDE:
                img = img.copy()
                img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
            img.save(dest, "PNG", optimize=True)
        except Exception:
            pass

        alt = (alt or "image").strip() or "image"
        alt = alt.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        return f"![{alt}](images/{md_image_name})"

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def _cleanup_temp(self, plan: dict) -> None:
        temp_doc_dir = os.path.join(
            self.output_dir, "temp", f"document_{plan['doc_id']}"
        )
        shutil.rmtree(temp_doc_dir, ignore_errors=True)

    def _create_zip(self) -> str:
        zip_path = os.path.join(self.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for md_file in sorted(Path(self.output_dir).glob("document_*.md")):
                zf.write(md_file, md_file.name)
            img_dir = Path(self.images_dir)
            if img_dir.exists():
                for img_file in sorted(img_dir.glob("*.png")):
                    zf.write(img_file, f"images/{img_file.name}")
        print(f"ZIP: {zip_path} ({Path(zip_path).stat().st_size // 1024} KB)")
        return zip_path


# ---------------------------------------------------------------------------
# VLM-таблицы: пост-обработка (единый формат с векторными)
# ---------------------------------------------------------------------------


_TD_RE = re.compile(r"<(td|th)([^>]*)>(.*?)</(td|th)>", re.S | re.I)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _parse_html_table(html: str) -> tuple[list[list[str]], int]:
    """
    Парсит HTML <table> в (rows, n_header_rows).
    n_header_rows — количество строк, где все ячейки были <th>.
    """
    rows: list[list[str]] = []
    n_header_rows = 0
    for tr in _TR_RE.finditer(html):
        cells = []
        all_th = True
        for td in _TD_RE.finditer(tr.group(1)):
            tag = td.group(1).lower()
            text = _TAG_STRIP_RE.sub("", td.group(3)).strip()
            cells.append(text)
            if tag != "th":
                all_th = False
        if cells:
            rows.append(cells)
            if all_th and len(rows) == n_header_rows + 1:
                n_header_rows += 1
    return rows, n_header_rows


def _postprocess_table_markdown(md: str) -> str:
    """
    Нормализует вывод DeepSeek-OCR в pipe-markdown.
    DeepSeek может вернуть HTML или pipe-markdown.
    Прогоняем через format_table_markdown (forward_fill + multilevel headers).
    """
    if not md:
        return ""

    # Путь 1: HTML-таблица
    if "<table" in md.lower():
        rows, n_hdr = _parse_html_table(md)
        if len(rows) >= 2:
            return format_table_markdown(rows, n_header_rows=n_hdr) or md.strip()

    # Путь 2: pipe-markdown
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) >= 2:
        rows = []
        for ln in lines:
            if re.match(r"^\|\s*:?-{2,}", ln):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
        if len(rows) >= 2:
            return format_table_markdown(rows) or md.strip()

    return md.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF → Markdown (YOLO + PyMuPDF + DeepSeek-OCR)"
    )
    parser.add_argument("--pdf", type=str, help="Путь к одному PDF")
    parser.add_argument(
        "--all", action="store_true", help="Обработать все PDF из raw-dir"
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-vlm", action="store_true", help="Отключить DeepSeek-OCR")
    args = parser.parse_args()

    pipeline = Pipeline(
        output_dir=args.output_dir,
        use_vlm=not args.no_vlm,
    )

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
