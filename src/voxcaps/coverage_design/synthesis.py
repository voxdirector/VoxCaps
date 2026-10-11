"""CosyVoice 3 targeted synthesis with teacher-only pronunciation inpainting."""

from __future__ import annotations

import re
from pathlib import Path

from .manifest import SYNTHETIC_SUBSETS

INITIALS = re.compile(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])")


def teacher_text(
    student_text: str, indices: list[int], readings: dict[int, str] | None = None
) -> str:
    """Replace selected Han characters by CosyVoice 3 initial/final tokens.

    `readings` uses tone-mark syllables for explicit lexical disambiguation.
    The original student text is never modified in the manifest.
    """
    if not isinstance(student_text, str) or not student_text.strip():
        raise ValueError("student_text_is_required")
    if len(set(indices)) != len(indices) or any(
        not isinstance(i, int) or not 0 <= i < len(student_text) for i in indices
    ):
        raise ValueError("invalid_pronunciation_indices")
    readings = readings or {}
    output = list(student_text)
    if indices:
        from pypinyin import Style, pinyin

        contextual = pinyin(
            student_text, style=Style.TONE, heteronym=False, errors=lambda chars: list(chars)
        )
        # One item per original character is required so that replacements keep
        # their lexical positions. Unsupported segmentation is rejected.
        if len(contextual) != len(student_text):
            raise ValueError("pinyin_character_alignment_mismatch")
        for index in indices:
            if not "\u3400" <= student_text[index] <= "\u9fff":
                raise ValueError("pinyin_replacement_requires_han_character")
            syllable = readings.get(index, contextual[index][0])
            if not isinstance(syllable, str) or not re.fullmatch(
                r"[a-züāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+", syllable
            ):
                raise ValueError("pronunciation_override_requires_tone_mark_pinyin")
            match = INITIALS.match(syllable)
            initial = "" if match is None else match[0]
            final = syllable[len(initial) :]
            if not final:
                raise ValueError("empty_pinyin_final")
            output[index] = (f"[{initial}]" if initial else "") + f"[{final}]"
    return "".join(output)


def validate_job(job: dict) -> None:
    if job.get("synthetic_subset") not in SYNTHETIC_SUBSETS:
        raise ValueError("synthesis_requires_one_of_three_paper_subsets")
    if not isinstance(job.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]):
        raise ValueError("synthesis_id_must_be_safe_filename")
    for name in ("student_text", "reference_audio", "reference_transcript"):
        if not isinstance(job.get(name), str) or not job[name].strip():
            raise ValueError(f"synthesis_missing:{name}")
    if job["synthetic_subset"] == "elderly_timbres" and job.get("reference_age_group") != "elderly":
        raise ValueError("elderly_subset_requires_audited_elderly_reference")


def synthesize_job(job: dict, model, base: Path, output: Path) -> dict:
    validate_job(job)
    reference = base / job["reference_audio"]
    if not reference.is_file():
        raise ValueError("synthesis_reference_audio_missing")
    indices = job.get("pronunciation_indices", [])
    if job["synthetic_subset"] != "difficult_pronunciation" and indices:
        raise ValueError("pronunciation_hints_belong_to_difficult_subset")
    overrides = {int(key): value for key, value in job.get("reading_overrides", {}).items()}
    rendered = teacher_text(job["student_text"], indices, overrides)
    prompt = "You are a helpful assistant.<|endofprompt|>" + job["reference_transcript"]
    destination = output / (job["id"] + ".wav")
    if destination.exists():
        raise ValueError("synthesis_output_already_exists")
    import torch
    import torchaudio

    chunks = [
        part["tts_speech"]
        for part in model.inference_zero_shot(
            rendered, prompt, str(reference), stream=False, text_frontend=False
        )
    ]
    if not chunks:
        raise ValueError("teacher_returned_no_audio")
    output.mkdir(parents=True, exist_ok=True)
    waveform = torch.cat(chunks, dim=-1)
    torchaudio.save(
        str(destination), waveform.cpu(), model.sample_rate, encoding="PCM_S", bits_per_sample=24
    )
    return {
        "id": job["id"],
        "source_type": "targeted_synthesis",
        "synthetic_subset": job["synthetic_subset"],
        "targeted_short": job["synthetic_subset"] == "short_utterances",
        "audio": destination.name,
        "transcript": job["student_text"],
        "student_text": job["student_text"],
        "transcript_source": "trusted",
        "languages": job.get("languages", []),
        "duration": waveform.shape[-1] / model.sample_rate,
        "teacher": {"model_family": "CosyVoice3", "pronunciation_inpainting": bool(indices)},
    }


def load_teacher(model_dir: Path):
    if not (model_dir / "cosyvoice3.yaml").is_file():
        raise ValueError("cosyvoice3_model_directory_required")
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel(model_dir=str(model_dir), load_trt=False, load_vllm=False, fp16=False)
