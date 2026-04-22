"""
Docling DocumentConverter wrapper.

Docling (IBM) — основной движок Fast Track. Делает всё сразу:
  - layout-анализ (heron-101)
  - извлечение текста с сохранением позиций (prov/bbox)
  - TableFormer для структуры таблиц (state-of-art TEDS)
  - EasyOCR (ru+en) для сканов
  - рендер picture-объектов
"""

from __future__ import annotations

import gc
from typing import Optional

from device import is_cuda_available


class DoclingEngine:
    """Singleton-обёртка над ``docling.DocumentConverter``."""

    _instance: Optional["DoclingEngine"] = None

    @classmethod
    def get(cls) -> "DoclingEngine":
        """Возвращает процесс-глобальный экземпляр, создавая при первом вызове."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Конфигурирует DocumentConverter.

        Настройки подобраны под задачу: RU+EN OCR, TableFormer с cell-matching,
        сохранение картинок как PIL (generate_picture_images), масштаб в 2 раза
        больше для читаемых превью.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            EasyOcrOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        device = AcceleratorDevice.CUDA if is_cuda_available() else AcceleratorDevice.CPU

        opts = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=EasyOcrOptions(
                lang=["ru", "en"],
                use_gpu=is_cuda_available(),
            ),
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=4,
            ),
            generate_picture_images=True,
            images_scale=2.0,
        )
        opts.table_structure_options.do_cell_matching = True

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        device_name = device.value if hasattr(device, "value") else device
        print(f"  [Docling] ready ({device_name})")

    def convert(self, pdf_path: str):
        """Конвертирует PDF и возвращает ``DoclingDocument`` (со всеми items)."""
        result = self._converter.convert(pdf_path)
        return result.document

    @staticmethod
    def page_is_sparse(doc, page_num: int, *, min_chars: int = 30) -> bool:
        """
        Признак «пустой» страницы — кандидата на olmOCR-fallback.

        Суммируем длину text-items на странице; если на ней есть табличные или
        picture-элементы, она автоматически не sparse (TableFormer / layout
        уже извлекли структуру — fallback не нужен).
        """
        try:
            texts = []
            has_structure = False
            for item, _ in doc.iterate_items():
                prov = getattr(item, "prov", None)
                if not prov:
                    continue
                if not any(p.page_no == page_num for p in prov):
                    continue
                label = getattr(item, "label", "")
                label_str = str(label).lower()
                if "table" in label_str or "picture" in label_str:
                    has_structure = True
                    break
                text = getattr(item, "text", "") or ""
                if text.strip():
                    texts.append(text)
            if has_structure:
                return False
            total = sum(len(t) for t in texts)
            return total < min_chars
        except Exception:
            return False

    def release(self) -> None:
        """Освобождает VRAM между документами (до загрузки olmOCR на ту же GPU)."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
