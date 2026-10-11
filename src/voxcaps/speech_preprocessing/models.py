"""Actual model-family adapters used by the paper's speech preprocessing."""

from __future__ import annotations

import re
from pathlib import Path


def strip_sentence_punctuation(text: str) -> str:
    return re.sub(r"(?<!\d)[.,!?;:，。！？；：、…]|[.,!?;:，。！？；：、…](?!\d)", "", text)


class SpeechModels:
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cpu")
        from funasr import AutoModel

        self.primary = AutoModel(
            model=config["fun_asr_nano_model"],
            device=self.device,
            hub=config.get("hub", "hf"),
            trust_remote_code=True,
            disable_update=True,
            disable_pbar=True,
            **config.get("fun_asr_kwargs", {}),
        )
        self.secondary = AutoModel(
            model=config["sensevoice_model"],
            device=self.device,
            hub=config.get("hub", "hf"),
            disable_update=True,
            disable_pbar=True,
        )
        import torch
        from qwen_asr import Qwen3ForcedAligner

        self.aligner = Qwen3ForcedAligner.from_pretrained(
            config["forced_aligner_model"],
            device_map=self.device,
            dtype=getattr(torch, config.get("aligner_dtype", "bfloat16")),
            local_files_only=True,
            attn_implementation=config.get("aligner_attention", "sdpa"),
        )
        from .diarization import Diarization3Dspeaker

        self.diarizer = Diarization3Dspeaker(
            device=self.device,
            model_cache_dir=config.get("speaker_model_cache"),
            include_overlap=False,
        )
        from .uvr.api import build_uvr_model

        self.separator = build_uvr_model(
            device=self.device,
            model_name=config["uvr_model_name"],
            model_root=config["uvr_model_root"],
            allow_legacy_fallback=False,
            precision=config.get("uvr_precision", "fp32"),
        )

    def separate(self, mixture: Path, destination: Path) -> Path:
        import soundfile as sf

        from .uvr.api import run_uvr_model

        result = run_uvr_model(str(mixture), self.separator)
        vocals = result["outputs"]["vocals"].detach().cpu().numpy()
        sf.write(str(destination), vocals.T, result["sr"], subtype="PCM_24")
        return destination

    def diarize(self, vocal: Path) -> list[list]:
        return self.diarizer(str(vocal))

    def transcribe_and_align(
        self, vocal: Path, language: str, trusted_text: str | None = None
    ) -> dict:
        language_name = {"zh": "中文", "en": "英文"}[language]
        primary = self.primary.generate(
            input=[str(vocal)],
            cache={},
            batch_size=1,
            language=language_name,
            **self.config.get("asr_generate_kwargs", {}),
        )[0]["text"]
        secondary = self.secondary.generate(
            input=[str(vocal)], cache={}, language="auto", use_itn=False
        )[0]["text"]
        secondary = re.sub(r"<\|[^>]+\|>", "", secondary)
        clean = strip_sentence_punctuation(primary if trusted_text is None else trusted_text)
        result = self.aligner.align(
            audio=str(vocal), text=clean, language={"zh": "Chinese", "en": "English"}[language]
        )[0]
        words = [
            {"text": item.text, "start_time": item.start_time, "end_time": item.end_time}
            for item in result
        ]
        return {
            "text": clean,
            "primary_asr": primary,
            "secondary_asr": secondary,
            "language": language,
            "words": words,
            "asr_disagreement": lexical(primary) != lexical(secondary),
        }


def lexical(text: str) -> str:
    return "".join(char.casefold() for char in text if char.isalnum())
