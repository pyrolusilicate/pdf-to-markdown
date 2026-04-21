"""
Выбор устройства и базовые настройки окружения.

Обеспечивает детерминированный выбор доступного акселератора и соответствующих
типов данных для корректной работы моделей.
"""

from __future__ import annotations

import os
import warnings

import torch


def get_torch_device() -> str:
    """
    Определяет наиболее производительное доступное устройство для PyTorch.

    Приоритет выбора:
    1. CUDA (cuda:0) - основная конфигурация, необходима для работы VLM.
    2. MPS (Apple Silicon) - fallback-вариант для легковесных задач (YOLO).
    3. CPU - базовый fallback при отсутствии графических ускорителей.

    Returns:
        str: Строковый идентификатор устройства.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Определяет оптимальный тип данных тензоров PyTorch в зависимости от устройства.

    Обеспечивает баланс между потреблением VRAM и точностью вычислений.

    Args:
        device (str): Идентификатор целевого устройства.

    Returns:
        torch.dtype: torch.bfloat16 для CUDA, torch.float16 для MPS, 
                     torch.float32 для CPU.
    """
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def is_cuda_available() -> bool:
    """
    Проверяет физическую доступность CUDA NVIDIA.

    Returns:
        bool: True, если доступен хотя бы один GPU с CUDA, иначе False.
    """
    return torch.cuda.is_available()


def setup_environment() -> None:
    """
    Инициализирует глобальные переменные окружения и подавляет системный шум.

    Настраивает аллокатор памяти PyTorch для снижения
    фрагментации VRAM. Отключает некритичные предупреждения от transformers,
    tokenizers и PyMuPDF для чистоты пайплайна.
    """
    # Настройки библиотек и аллокатора памяти
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYMUPDF_MESSAGE", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Фильтрация нерелевантных предупреждений
    warnings.filterwarnings("ignore", message=r".*pin_memory.*MPS.*")
    warnings.filterwarnings("ignore", message=r".*`do_sample` is set to `False`.*temperature.*")
    warnings.filterwarnings("ignore", message=r".*temperature.*do_sample.*")
    warnings.filterwarnings("ignore", message=r".*pymupdf_layout.*")
