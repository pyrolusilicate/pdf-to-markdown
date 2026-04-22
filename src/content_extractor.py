"""
Вспомогательные валидаторы, форматтеры и crop-операции для пайплайна.

Используется в pipeline.py:
  - cyrillic_ratio, repetition_ratio, table_stats — анти-галлюцинационные метрики 
  - format_table_markdown — unified pipe-markdown (forward-fill + multi-level)
  - format_text_markdown — применяет markdown к извлечённому тексту
  - filter_noise_lines — чистка штампов/коротких фрагментов
  - crop_pdf_bbox — рендер bbox страницы в PIL.Image для olmOCR
"""

from __future__ import annotations

import re
from typing import Optional

import fitz
from PIL import Image

from config import (
    CROP_PAD_PTS,
    PDF_TO_LAYOUT,
    MIN_LINES_REP_CHECK,
    MIN_VALID_CHARS,
    OLM_RENDER_SIDE,
)

# ---------------------------------------------------------------------------
# Регулярные выражения (скомпилированы на уровне модуля для скорости)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[\d\s.,+\-/%№()]+$")
_STAMP_FRAG_RE = re.compile(r"^[А-ЯЁA-Z]{2,6}$")
_WATERMARK_RE = re.compile(
    r"^(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|ОБРАЗЕЦ|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)[\s\d\W]*$",
    re.I | re.UNICODE,
)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


# ---------------------------------------------------------------------------
# Crop bbox из PDF (для olmOCR fallback)
# ---------------------------------------------------------------------------


def crop_pdf_bbox(
    pdf_doc: fitz.Document,
    page_num: int,
    coords_px: list,
    *,
    max_side: int = OLM_RENDER_SIDE,
    pad_pts: float = CROP_PAD_PTS,
) -> Optional[Image.Image]:
    """
    Рендерит bbox страницы PDF в PIL.Image.

    Результат масштабируется так, чтобы длинная сторона соответствовала max_side
    (оптимизация под требования olmOCR).

    Args:
        pdf_doc (fitz.Document): Открытый документ PyMuPDF.
        page_num (int): Номер страницы (0-indexed).
        coords_px (list | tuple): Координаты (x1, y1, x2, y2) в пикселях YOLO-растра.
        max_side (int): Максимальный размер длинной стороны итогового изображения.
        pad_pts (float): Padding в PDF points.

    Returns:
        Optional[Image.Image]: Объект PIL.Image или None в случае ошибки рендера.
    """
    try:
        page = pdf_doc[page_num]
    except Exception:
        return None

    # Конвертация пикселей в PDF points
    x1, y1, x2, y2 = coords_px
    rect = fitz.Rect(
        x1 * PDF_TO_LAYOUT - pad_pts, y1 * PDF_TO_LAYOUT - pad_pts,
        x2 * PDF_TO_LAYOUT + pad_pts, y2 * PDF_TO_LAYOUT + pad_pts
    )

    # Вычисление масштаба рендера
    bbox_w_pts = max(1.0, rect.width)
    bbox_h_pts = max(1.0, rect.height)
    longest_pts = max(bbox_w_pts, bbox_h_pts)
    render_scale = max_side / longest_pts

    mat = fitz.Matrix(render_scale, render_scale)
    try:
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    except Exception:
        return None
        
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------------------------------------------------------------------
# Markdown-таблицы: forward-fill + multi-level header
# ---------------------------------------------------------------------------


def _forward_fill(table: list[list]) -> list[list[str]]:
    """
    Дублирует значения объединённых ячеек слева направо.

    Args:
        table (list[list[Any]]): Исходная таблица с возможными None значениями.

    Returns:
        list[list[str]]: Таблица, где пустые ячейки заполнены значениями слева.
    """
    result: list[list[str]] = []
    for row in table:
        new_row: list[str] = []
        last_val = ""
        for cell in row:
            if cell is None or (isinstance(cell, str) and not cell.strip()):
                new_row.append(last_val)
            else:
                val = str(cell).strip().replace("\n", " ")
                last_val = val
                new_row.append(val)
        result.append(new_row)
    return result


def _is_text_row(row: list[str]) -> bool:
    """
    Проверяет, состоит ли строка преимущественно из текстовых (не числовых) данных.
    
    Args:
        row (list[str]): 

    Returns:
        bool: True, если состоит из текстовых, иначе False.
    """
    non_empty = [c for c in row if c.strip()]
    if not non_empty:
        return False
    return all(not _NUMERIC_RE.fullmatch(c) for c in non_empty)


def format_table_markdown(table: list[list], n_header_rows: int = 0) -> str:
    """
    Генерирует Markdown-таблицу с поддержкой многоуровневых заголовков.

    Args:
        table (list[list[Any]]): Двумерный массив данных таблицы.
        n_header_rows (int): Количество строк заголовка (0, 1 или 2). Если 0,
            функция попытается автоматически определить двухуровневый заголовок.

    Returns:
        str: Отформатированная таблица в формате pipe-markdown.
    """
    if not table:
        return ""

    filled = _forward_fill(table)
    if not filled or not filled[0]:
        return ""

    # Определение количества строк заголовка
    if n_header_rows >= 2:
        header_rows = 2
    elif n_header_rows == 1:
        header_rows = 1
    else:
        header_rows = 1
        if len(filled) >= 3 and _is_text_row(filled[0]) and _is_text_row(filled[1]):
            header_rows = 2

    # Формирование заголовков и данных
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
        """Дополняет строку пустыми ячейками до нужной ширины."""
        row = list(row) + [""] * width
        return [str(c) for c in row[:width]]

    # Сборка строк Markdown
    lines = [
        "| " + " | ".join(pad(headers, n)) + " |",
        "| " + " | ".join(["---"] * n) + " |",
    ]
    for row in data_rows:
        lines.append("| " + " | ".join(pad(row, n)) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Форматирование текстовых блоков
# ---------------------------------------------------------------------------


def format_text_markdown(text: str, block_type: str, heading_level: int = 0) -> str:
    """
    Применяет Markdown-разметку к извлечённому тексту на основе типа YOLO-блока.

    Args:
        text (str): Исходный сырой текст.
        block_type (str): Тип блока ('title', 'section-header', 'list-item' и т.д.).
        heading_level (int): Уровень заголовка (1-6). Используется, если блок - заголовок.

    Returns:
        str: Текст, обернутый в соответствующую Markdown-разметку.
    """
    text = text.strip()
    if not text:
        return ""

    if block_type in ("title", "section-header") and heading_level > 0:
        single_line = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return "#" * heading_level + " " + single_line

    if block_type == "list-item":
        out = []
        for line in text.splitlines():
            # Если строка пустая, пропускаем, но не используем strip() на самой line,
            # чтобы не потерять отступы
            if not line.strip():
                continue
            
            # Сохраняем оригинальный отступ
            indent_match = re.match(r"^(\s*)", line)
            indent = indent_match.group(1) if indent_match else ""
            clean_line = line.lstrip() # Строка без отступов для проверок
            
            # Если это реальный маркер списка — нормализуем
            if re.match(r"^[•▪◦▸\-\*]\s", clean_line):
                # Страховка: если пробелов нет, но маркер вложенный, делаем отступ сами
                if not indent:
                    if clean_line.startswith('◦'):
                        indent = "  "
                    elif clean_line.startswith('▪') or clean_line.startswith('▸'):
                        indent = "    "
                out.append(f"{indent}- {clean_line[2:]}")
                
            # Если начинается с цифры или буквы со скобкой то без тире
            elif re.match(r"^\d", clean_line) or re.match(r"^[A-Za-zА-Яа-яЁё]\)", clean_line):
                out.append(f"{indent}{clean_line}")
            else:
                out.append(f"{indent}- {clean_line}")
        return "\n".join(out)

    return text


# ---------------------------------------------------------------------------
# Фильтрация мусорных строк (штампы/водяные знаки)
# ---------------------------------------------------------------------------


def filter_noise_lines(text: str, min_chars: int = MIN_VALID_CHARS) -> str:
    """
    Удаляет слишком короткие строки, водяные знаки и штампы.

    Args:
        text (str): Исходный многострочный текст.
        min_chars (int): Минимально допустимое количество символов без пробелов.

    Returns:
        str: Очищенный текст.
    """
    if not text:
        return ""
        
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped.replace(" ", "")) < min_chars:
            continue
        if _STAMP_FRAG_RE.match(stripped):
            continue
        if _WATERMARK_RE.match(stripped):
            continue
        lines.append(line)
        
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Анти-галлюцинационные метрики
# ---------------------------------------------------------------------------


def cyrillic_ratio(text: str) -> float:
    """
    Вычисляет долю кириллических символов среди всех букв.

    Args:
        text (str): Текст для анализа.

    Returns:
        float: Значение от 0.0 до 1.0. Вернет 0.0, если букв нет вообще.
    """
    if not text:
        return 0.0
        
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
        
    cyr = _CYRILLIC_RE.findall(text)
    return len(cyr) / len(letters)


def repetition_ratio(text: str) -> float:
    """
    Вычисляет частоту повторения самой частой непустой строки (признак цикла VLM).

    Args:
        text (str): Текст, сгенерированный моделью.

    Returns:
        float: Значение от 0.0 до 1.0. Возвращает 0.0, если строк меньше 4.
    """
    if not text:
        return 0.0
        
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) < MIN_LINES_REP_CHECK:
        return 0.0
        
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
        
    return max(counts.values()) / len(lines)


def table_stats(md: str) -> dict:
    """
    Собирает структурную статистику по pipe-markdown таблице.
    Полезно для выявления сломанных или галлюцинированных таблиц.

    Args:
        md (str): Текст таблицы в формате Markdown.

    Returns:
        dict: Словарь с метриками (n_cols, n_rows, empty_ratio, max_cell, 
              row_repeat_ratio).
    """
    stats = {
        "n_cols": 0,
        "n_rows": 0,
        "empty_ratio": 0.0,
        "max_cell": 0,
        "row_repeat_ratio": 0.0,
    }
    if not md:
        return stats

    rows: list[list[str]] = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s.startswith("|"):
            continue
        if re.match(r"^\|\s*:?-{2,}", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return stats

    n_cols = max(len(r) for r in rows)
    n_rows = len(rows)
    total = sum(len(r) for r in rows) or 1
    empty = sum(1 for r in rows for c in r if not c)
    max_cell = max((len(c) for r in rows for c in r), default=0)

    row_keys = ["|".join(r) for r in rows]
    row_counts: dict[str, int] = {}
    for k in row_keys:
        row_counts[k] = row_counts.get(k, 0) + 1

    row_repeat = (max(row_counts.values()) / len(row_keys)) if row_keys else 0.0

    stats.update(
        n_cols=n_cols,
        n_rows=n_rows,
        empty_ratio=empty / total,
        max_cell=max_cell,
        row_repeat_ratio=row_repeat,
    )
    return stats
