"""
olmOCR-2-7B-1025 wrapper (AllenAI, на базе Qwen2.5-VL-7B).

Используется как fallback для двух сценариев:
  - Docling вернул пустой/битый результат на блок (скан без OCR-слоя,
    сильно повреждённая страница);
  - определить содержимое figure-блока (таблица, текст, картинка).

Требования:
  - torch>=2.5.1, transformers>=4.49, olmocr>=0.4 (для промпта);
  - GPU с >=14GB VRAM для BF16;
  - flash-attn — опционально, ~2 ускорение decode.

Для детерминизма использованы ``temperature=0.1`` + ``do_sample=False``.
"""

from __future__ import annotations

import base64
import gc
import io
import re
from typing import Optional

from PIL import Image

from config import OLM_RENDER_SIDE
from device import is_cuda_available

_MODEL_ID = "allenai/olmOCR-2-7B-1025"
_PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
_MAX_NEW_TOKENS = 4096


def _resize_longest(img: Image.Image, max_side: int) -> Image.Image:
    """Масштабирует изображение так, чтобы длинная сторона == ``max_side``."""
    w, h = img.size
    longest = max(w, h)
    if longest == max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _pil_to_b64(img: Image.Image) -> str:
    """PIL.Image -> base64-PNG ASCII-строка для chat-template image_url."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)


def _strip_yaml_frontmatter(text: str) -> str:
    """Убирает YAML-метаданные из ответа olmOCR, оставляя только markdown."""
    m = _FRONTMATTER_RE.match(text.strip())
    if m:
        return m.group(2).strip()
    return text.strip()


class OLMEngine:
    """Singleton-обёртка над olmOCR-2-7B через HF transformers."""

    _instance: Optional["OLMEngine"] = None

    @classmethod
    def get(cls) -> "OLMEngine":
        """Процесс-глобальный экземпляр; модель грузится лениво при первом вызове."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Конструктор только регистрирует слоты; реальная загрузка — в ``_load``."""
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Ленивая загрузка модели и процессора. Идемпотентна."""
        if self._model is not None:
            return
        if not is_cuda_available():
            raise RuntimeError(
                "olmOCR-2-7B требует CUDA. Запусти на сервере с GPU."
            )
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(_PROCESSOR_ID)
        print(f"  [olmOCR] {_MODEL_ID} ready ({attn_impl})")

    def page_to_markdown(self, pil_image: Image.Image) -> str:
        """
        Распознаёт изображение -> markdown (без YAML-frontmatter).

        Изображение принудительно ресайзится к ``OLM_RENDER_SIDE`` (требование
        модели). Параметры генерации детерминированы, чтобы одинаковый вход
        давал одинаковый выход.
        """
        self._load()
        import torch

        img = _resize_longest(pil_image, OLM_RENDER_SIDE)
        b64 = _pil_to_b64(img)

        try:
            from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
            prompt_text = build_no_anchoring_v4_yaml_prompt()
        except Exception:
            # Fallback-промпт на случай отсутствия пакета olmocr на сервере.
            prompt_text = (
                "Attached is one page of a document. Return the full text and "
                "tables as clean markdown. Preserve the original language. "
                "Start the response with a YAML frontmatter block "
                "(primary_language, is_rotation_valid, rotation_correction, "
                "is_table, is_diagram) followed by --- and then the markdown."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        raw = self._processor.decode(generated, skip_special_tokens=True)
        return _strip_yaml_frontmatter(raw)

    def release(self) -> None:
        """Освобождает VRAM; полезно если далее идут не-GPU операции."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
