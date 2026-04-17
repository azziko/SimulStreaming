#!/usr/bin/env python3

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.preload_lingua import NEMO_TO_LINGUA, LINGUA_TO_NEMO, build_detector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_REGISTRY: dict[str, str] = {
    "gemma-2b":  "google/gemma-3-2b-it",
    "qwen-0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2-1.5B-Instruct",
}

LANG_NAMES: dict[str, str] = {
    "bg": "Bulgarian", "hr": "Croatian",  "cs": "Czech",     "da": "Danish",
    "nl": "Dutch",     "en": "English",   "et": "Estonian",  "fi": "Finnish",
    "fr": "French",    "de": "German",    "el": "Greek",     "hu": "Hungarian",
    "it": "Italian",   "lv": "Latvian",   "lt": "Lithuanian","mt": "Maltese",
    "pl": "Polish",    "pt": "Portuguese","ro": "Romanian",  "sk": "Slovak",
    "sl": "Slovenian", "es": "Spanish",   "sv": "Swedish",   "ru": "Russian",
    "uk": "Ukrainian",
}


class LLMCascadeProcessor:

    def __init__(
        self,
        model_key: str,
        target_lang: str,
        lingua_lang_codes: list[str],
        device: str = "cuda",
        min_chars: int = 8,
        max_new_tokens: int = 128,
    ) -> None:

        self.model_key = model_key
        self.model_id = MODEL_REGISTRY[model_key]
        self.target_lang = target_lang
        self.lingua_lang_codes = lingua_lang_codes
        self.device = device
        self.min_chars = min_chars
        self.max_new_tokens = max_new_tokens
        self._detector = None
        self._tokenizer = None
        self._llm = None

    def preload(self) -> None:
        self._preload_lingua()
        self._preload_llm()
        logger.info(
            "LLMCascadeProcessor ready — model=%s, target=%s",
            self.model_id, self.target_lang,
        )

    def _preload_lingua(self) -> None:
        logger.info("Building Lingua detector for: %s", self.lingua_lang_codes)
        t0 = time.perf_counter()
        self._detector = build_detector(self.lingua_lang_codes)
        logger.info("Lingua detector ready in %.3f s.", time.perf_counter() - t0)

    def _preload_llm(self) -> None:
        logger.info("Loading LLM: %s …", self.model_id)
        t0 = time.perf_counter()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._llm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        ).to(self.device)
        self._llm.eval()

        # Warm-up forward pass
        self._run_llm("Hello.", "en")
        logger.info("LLM ready in %.3f s.", time.perf_counter() - t0)

    def _detect_lang(self, text: str) -> Optional[str]:
        result = self._detector.detect_language_of(text)

        return LINGUA_TO_NEMO.get(result)

    def _run_llm(self, text: str, src_lang: str) -> str:
        src_name = LANG_NAMES.get(src_lang, src_lang.upper())
        tgt_name = LANG_NAMES.get(self.target_lang, self.target_lang.upper())
        prompt = (
            f"Translate the following {src_name} text to {tgt_name}. "
            "Output only the translation, no explanations or extra text.\n\n"
            f"{src_name} text: {text}\n"
            f"{tgt_name} translation:"
        )

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                input_text = self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a professional translator."},
                        {"role": "user", "content": prompt}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                input_text = prompt
        else:
            input_text = prompt

        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._llm.device)

        with torch.inference_mode():
            output_ids = self._llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def process(self, text: str) -> str:
        """
        Check the language of *text* and translate if it doesn't match
        *target_lang*. Returns the (possibly translated) text.

        Segments shorter than *min_chars* are passed through unchanged.
        """
        text_stripped = text.strip()
        if len(text_stripped) < self.min_chars:
            return text

        detected = self._detect_lang(text_stripped)
        if detected is None or detected == self.target_lang:
            return text

        logger.debug("[cascade] detected=%s → translating: %r", detected, text_stripped)
        translated = self._run_llm(text_stripped, detected)

        # Preserve leading whitespace from _modify_emit_text
        leading = " " if text.startswith(" ") else ""
        return leading + translated