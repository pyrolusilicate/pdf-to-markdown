"""
Проекция координат между системами YOLO и Docling. Обеспечивает
бесшовную трансляцию bboxes между пиксельной системой координат
растрированных страниц (YOLO) и векторной системой PDF points
(Docling и PyMuPDF).

Содержит функции для расчета метрик Intersection over Union
(IoU) и Intersection over Minimum (IoM), критичных для матчинга
элементов макета.
"""

from __future__ import annotations

from typing import Optional

from config import LAYOUT_DPI, PDF_DPI


Bbox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def points_to_pixels(
    bbox_pts: Bbox,
    page_height_pts: Optional[float] = None,
    *,
    origin: str = "top",
) -> Bbox:
    """
    Проецирует координаты из PDF points в пиксели растра.
    Учитывает разницу в системах координат Docling и
    стандартной оси PyMuPDF.

    Args:
        bbox_pts (Bbox): Координаты (x1, y1, x2, y2) в PDF points.
        page_height_pts (Optional[float]): Высота страницы в points. Обязательна,
            если origin установлен в "bottom".
        origin (str): Точка отсчета оси Y. "top" или "bottom". По умолчанию "top".

    Returns:
        Bbox: Масштабированные координаты (x1, y1, x2, y2) в пикселях.

    Raises:
        ValueError: Если origin='bottom', но page_height_pts не передан.
    """
    scale = LAYOUT_DPI / PDF_DPI
    x1, y1, x2, y2 = bbox_pts
    if origin == "bottom":
        if page_height_pts is None:
            raise ValueError("origin='bottom' требует page_height_pts")
        # Инверсия Y и нормализация порядка (y1 < y2 в целевой системе)
        y1, y2 = page_height_pts - y2, page_height_pts - y1
    return (x1 * scale, y1 * scale, x2 * scale, y2 * scale)


def pixels_to_points(bbox_px: Bbox) -> Bbox:
    """
    Осуществляет обратную проекцию координат из пикселей в PDF points.
    Предполагается система координат top-down.

    Args:
        bbox_px (Bbox): Координаты (x1, y1, x2, y2) в пикселях.

    Returns:
        Bbox: Масштабированные координаты в PDF points.
    """
    scale = PDF_DPI / LAYOUT_DPI
    x1, y1, x2, y2 = bbox_px
    return (x1 * scale, y1 * scale, x2 * scale, y2 * scale)


# ---------------------------------------------------------------------------
# Метрики пересечения
# ---------------------------------------------------------------------------


def _area(bbox: Bbox) -> float:
    """
    Вычисляет площадь bounding box.
    
    Args:
        bbox (Bbox): Координаты (x1, y1, x2, y2) в пикселях.

    Returns:
        float: Площадь в пикселях.
    """
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height


def _intersection(bbox_a: Bbox, bbox_b: Bbox) -> float:
    """
    Вычисляет площадь прямоугольника пересечения двух bounding boxes.
    
    Args:
        bbox_a (Bbox): Первый bounding box.
        bbox_b (Bbox): Второй bounding box.

    Returns:
        float: Площадь прямоугольника пересечения в пикселях.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
        
    return (ix2 - ix1) * (iy2 - iy1)


def iou(bbox_a: Bbox, bbox_b: Bbox) -> float:
    """
    Вычисляет метрику Intersection over Union.

    Args:
        bbox_a (Bbox): Первый bounding box.
        bbox_b (Bbox): Второй bounding box.

    Returns:
        float: Значение IoU от 0.0 до 1.0.
    """
    inter = _intersection(bbox_a, bbox_b)
    if inter <= 0.0:
        return 0.0
        
    union = _area(bbox_a) + _area(bbox_b) - inter
    return inter / union if union > 0.0 else 0.0


def iom(bbox_a: Bbox, bbox_b: Bbox) -> float:
    """
    Вычисляет метрику Intersection over Minimum Area (IoM).

    Args:
        bbox_a (Bbox): Первый bounding box.
        bbox_b (Bbox): Второй bounding box.

    Returns:
        float: Значение IoM от 0.0 до 1.0.
    """
    inter = _intersection(bbox_a, bbox_b)
    if inter <= 0.0:
        return 0.0
        
    min_area = min(_area(bbox_a), _area(bbox_b))
    return inter / min_area if min_area > 0.0 else 0.0


def horizontal_overlap(bbox_a: Bbox, bbox_b: Bbox) -> float:
    """
    Вычисляет долю горизонтального пересечения блоков.

    Используется для определения того, находятся ли элементы в одной колонке.
    Отношение считается от ширины более узкого блока.

    Args:
        bbox_a (Bbox): Первый bounding box.
        bbox_b (Bbox): Второй bounding box.

    Returns:
        float: Доля пересечения от 0.0 до 1.0.
    """
    ax1, _, ax2, _ = bbox_a
    bx1, _, bx2, _ = bbox_b
    
    ix1, ix2 = max(ax1, bx1), min(ax2, bx2)
    if ix2 <= ix1:
        return 0.0
        
    min_width = min(ax2 - ax1, bx2 - bx1)
    return (ix2 - ix1) / min_width if min_width > 0.0 else 0.0