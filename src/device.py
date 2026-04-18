"""
Выбор устройства и базовые настройки окружения.

DeepSeek-OCR-2 требует CUDA + bf16, поэтому основная конфигурация — CUDA.
Для нетребовательных веток (YOLO layout на MPS/CPU) оставлен фолбэк.
"""

from __future__ import annotations

import os
import warnings

import torch


def get_torch_device() -> str:
    """cuda:0 → mps → cpu. Для VLM подходит только cuda."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str):
    """bf16 на CUDA, fp16 на MPS, fp32 на CPU."""
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def setup_environment() -> None:
    """Снижает шум от transformers/tokenizers и убирает устаревшие warnings."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    warnings.filterwarnings("ignore", message=r".*pin_memory.*MPS.*")
