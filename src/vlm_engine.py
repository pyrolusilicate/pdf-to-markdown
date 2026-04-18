"""
DeepSeek-OCR-2 wrapper.

Используется для всего, что не извлекается из векторного слоя:
растровые таблицы (RASTER_TABLE), сканы/rasterized страницы (RASTER_TEXT),
рукописный текст и классификация фигур (SMART_FIGURE).

Требования (жёсткие, задаются самой моделью через trust_remote_code):
    torch==2.6.0, transformers==4.46.3, tokenizers==0.20.3,
    einops, addict, easydict, flash-attn==2.7.3  (CUDA only)

Модель грузится лениво — первый вызов любой публичной функции инициализирует
её, последующие вызовы переиспользуют singleton.
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import tempfile
import uuid
from typing import Optional, Union

from PIL import Image

from device import get_torch_device, is_cuda_available, setup_environment

setup_environment()

_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"

# Размеры входа. По рекомендации README:
#   "Base"     — base_size=1024, image_size=1024, crop_mode=False  (один блок)
#   "Gundam"   — base_size=1024, image_size=640,  crop_mode=True   (страница)
#   "Small"    — base_size=640,  image_size=640,  crop_mode=False  (короткий OCR)
_SIZES_PAGE = {"base_size": 1024, "image_size": 640, "crop_mode": True}
_SIZES_BLOCK = {"base_size": 1024, "image_size": 1024, "crop_mode": False}
_SIZES_SMALL = {"base_size": 640, "image_size": 640, "crop_mode": False}

# Промпты. Все начинаются с `<image>\n` — этого требует чат-шаблон DeepSeek-OCR.
PROMPT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown. "
PROMPT_FREE_OCR = "<image>\nFree OCR. "
PROMPT_PARSE_FIGURE = "<image>\nParse the figure. "
PROMPT_DESCRIBE_RU = "<image>\nОпиши изображение одной фразой на русском (до 10 слов). "
PROMPT_CLASSIFY = (
    "<image>\nWhat is this image? Answer with a single word: "
    "'text', 'table', 'handwritten', or 'picture'."
)

# ---------------------------------------------------------------------------
# post-processing

_GROUNDING_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
_REF_RE = re.compile(r"<\|/?ref\|>")
_TAG_RE = re.compile(r"<\|/?grounding\|>")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.M)


def _strip_special_tokens(text: str) -> str:
    """Убирает `<|grounding|>`, `<|ref|>...<|/ref|>`, `<|det|>...<|/det|>`."""
    text = _GROUNDING_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------


class VLMEngine:
    """Singleton-обёртка над DeepSeek-OCR-2."""

    _instance: Optional["VLMEngine"] = None

    def __init__(self, model_id: str = _MODEL_ID):
        self.model_id = model_id
        self.device = get_torch_device()
        self._model = None
        self._tokenizer = None
        self._tmp_root = tempfile.mkdtemp(prefix="dsocr_")

    # ---- lifecycle ----------------------------------------------------

    @classmethod
    def get(cls, model_id: str = _MODEL_ID) -> "VLMEngine":
        if cls._instance is None:
            cls._instance = cls(model_id)
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        if not is_cuda_available():
            raise RuntimeError(
                "DeepSeek-OCR-2 требует CUDA + flash-attn. "
                "Запусти на Colab/сервере с GPU."
            )

        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"  [VLM] Загрузка {self.model_id} на {self.device} (bfloat16)...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_id,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        self._model = model.eval().cuda().to(torch.bfloat16)
        print("  [VLM] Готово.")

    def release_page_cache(self) -> None:
        """Вызывать между страницами: освобождает VRAM без выгрузки весов."""
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- low-level inference -----------------------------------------

    def _infer(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        *,
        base_size: int,
        image_size: int,
        crop_mode: bool,
    ) -> str:
        """
        Вызывает `model.infer(...)` и возвращает очищенный markdown.

        DeepSeek-OCR-2 принимает только путь к файлу, поэтому PIL-картинки
        сохраняем во временный PNG.
        """
        self._load()

        tmp_img_path: Optional[str] = None
        if isinstance(image, Image.Image):
            tmp_img_path = os.path.join(self._tmp_root, f"{uuid.uuid4().hex}.png")
            image.save(tmp_img_path, "PNG")
            image_file = tmp_img_path
        elif isinstance(image, str):
            if not os.path.exists(image):
                return ""
            image_file = image
        else:
            return ""

        out_dir = os.path.join(self._tmp_root, uuid.uuid4().hex)
        os.makedirs(out_dir, exist_ok=True)

        try:
            self._model.infer(
                self._tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=out_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
            )
            # Первичный источник — результат, записанный в файл (без служебных токенов).
            mmd_path = os.path.join(out_dir, "result.mmd")
            if os.path.exists(mmd_path):
                with open(mmd_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # На некоторых промптах файл не создаётся — как запасной вариант
                # вызываем без save_results и читаем возвращаемое значение.
                text = (
                    self._model.infer(
                        self._tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path=out_dir,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=False,
                    )
                    or ""
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [VLM] infer failed: {exc}")
            text = ""
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            if tmp_img_path:
                try:
                    os.remove(tmp_img_path)
                except OSError:
                    pass

        return _strip_special_tokens(str(text))

    # ---- высокоуровневое API -----------------------------------------

    def extract_markdown(self, image: Union[str, Image.Image]) -> str:
        """Полный разбор блока в markdown (таблицы, смешанный контент)."""
        return self._infer(image, PROMPT_MARKDOWN, **_SIZES_BLOCK)

    def extract_page_markdown(self, image: Union[str, Image.Image]) -> str:
        """Разбор целой страницы с многоколоночной разметкой ("Gundam")."""
        return self._infer(image, PROMPT_MARKDOWN, **_SIZES_PAGE)

    def free_ocr(self, image: Union[str, Image.Image]) -> str:
        """Быстрый plain-text OCR для коротких блоков."""
        return self._infer(image, PROMPT_FREE_OCR, **_SIZES_SMALL)

    def parse_figure(self, image: Union[str, Image.Image]) -> str:
        """Разбор сложной фигуры/диаграммы в структурированный текст."""
        return self._infer(image, PROMPT_PARSE_FIGURE, **_SIZES_BLOCK)

    def short_caption(self, image: Union[str, Image.Image]) -> str:
        """Короткий русский alt-текст (≤10 слов)."""
        raw = self._infer(image, PROMPT_DESCRIBE_RU, **_SIZES_SMALL)
        raw = raw.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        first = raw.splitlines()[0] if raw else ""
        return first[:120].strip() or "image"

    def classify(self, image: Union[str, Image.Image]) -> str:
        """Возвращает 'text' | 'table' | 'handwritten' | 'picture'."""
        raw = self._infer(image, PROMPT_CLASSIFY, **_SIZES_SMALL).lower()
        for key in ("handwritten", "table", "text", "picture"):
            if key in raw:
                return key
        return "picture"
