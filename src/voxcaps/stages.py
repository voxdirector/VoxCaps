"""Four explicit streaming stage runners with opt-in model execution."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .pipeline import prepare_manifest, read_jsonl

TRANSPORT_KEYS = {
    "endpoint",
    "model",
    "key_env",
    "auth_scheme",
    "timeout",
    "max_tokens",
    "stream",
    "modalities",
}


def transport_config(config: dict, *, audio: bool = False) -> dict:
    if config.get("model_family") != "Qwen3.5-Omni":
        raise ValueError("paper_annotation_requires_qwen35_omni_model_family")
    return {
        key: value
        for key, value in config.items()
        if key in TRANSPORT_KEYS or audio and key == "media_format"
    }


def write_manifest(path: Path, records) -> int:
    if path.exists():
        raise ValueError("stage_manifest_already_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + "-", suffix=".tmp")
    temporary = Path(name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if path.exists():
            raise ValueError("stage_manifest_already_exists")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def stage1(manifest: Path, output: Path, *, teacher_dir: Path | None = None):
    from .coverage_design.manifest import PAPER_SUBSET_COUNTS, coverage_metadata
    from .coverage_design.synthesis import load_teacher, synthesize_job

    media, languages, subsets = Counter(), Counter(), Counter()

    def records():
        teacher = None
        for record in read_jsonl(manifest):
            if record.get("source_type") == "targeted_synthesis" and "audio" not in record:
                if teacher_dir is None:
                    raise ValueError("cosyvoice3_teacher_directory_required_for_synthesis")
                if teacher is None:
                    teacher = load_teacher(teacher_dir)
                record = synthesize_job(record, teacher, manifest.parent, output / "synthetic")
                record["audio"] = str((output / "synthetic" / record["audio"]).resolve())
            else:
                record["audio"] = str((manifest.parent / record["audio"]).resolve())
                if not Path(record["audio"]).is_file():
                    raise ValueError("coverage_source_audio_missing")
            meta = coverage_metadata(record)
            record.update(meta)
            media[meta["media_type"] or "unspecified"] += 1
            languages.update(meta["languages"])
            if meta["synthetic_subset"]:
                subsets[meta["synthetic_subset"]] += 1
            yield record

    count = write_manifest(output / "manifest.jsonl", records())
    report = {
        "examples": count,
        "media": dict(media),
        "languages": dict(languages),
        "synthetic_subsets": dict(subsets),
        "paper_subset_counts": PAPER_SUBSET_COUNTS,
        "paper_count_differences": {
            key: subsets[key] - value for key, value in PAPER_SUBSET_COUNTS.items()
        },
    }
    (output / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def stage2(manifest: Path, output: Path, config: dict, *, models=None):
    from .speech_preprocessing.models import SpeechModels
    from .speech_preprocessing.segmentation import preprocess_source

    if models is None:
        models = SpeechModels(config)

    def records():
        for index, record in enumerate(read_jsonl(manifest)):
            yield from preprocess_source(
                record, manifest.parent, output / f"source_{index:06d}", models
            )

    return {"clips": write_manifest(output / "manifest.jsonl", records())}


def stage3(manifest: Path, output: Path, config: dict, *, backend=None):
    from .scope_annotation.annotator import annotate_record
    from .scope_annotation.provider import request_annotation

    if backend is None:
        transport = transport_config(config, audio=True)

        def backend(audio, system, user):
            return request_annotation(audio, system, user, **transport)

    records = (annotate_record(record, manifest.parent, backend) for record in read_jsonl(manifest))
    return {"annotated": write_manifest(output / "manifest.jsonl", records)}


def stage4(manifest: Path, output: Path, config: dict, *, requester=None):
    from .data_refinement.quality import FilterConfig

    if requester is None and config.get("description_model"):
        from .scope_annotation.provider import request_text_json

        transport = transport_config(config["description_model"])

        def requester(system, text):
            return request_text_json(system, text, **transport)

    quality = config.get("quality_models")
    dnsmos = None
    squim = None

    def records():
        nonlocal dnsmos, squim
        for record in read_jsonl(manifest):
            if "quality_score" not in record:
                if not quality:
                    raise ValueError("cached_quality_scores_or_quality_models_required")
                if squim is None:
                    from .data_refinement.squim import SQUIMScorer

                    squim = SQUIMScorer(
                        Path(quality["squim_reference_audio"]), quality.get("device", "cpu")
                    )
                record["quality_score"] = squim(manifest.parent / record["audio"])["quality_score"]
            if "dnsmos" not in record and quality:
                if dnsmos is None:
                    from .data_refinement.dnsmos import ComputeScore

                    dnsmos = ComputeScore(
                        quality["dnsmos_primary_model"], quality["dnsmos_p808_model"]
                    )
                record["dnsmos"] = dnsmos(str(manifest.parent / record["audio"]), 16000, False)
            for field in ("audio", "analysis_audio"):
                if record.get(field):
                    record[field] = str((manifest.parent / record[field]).resolve())
            yield record

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix="refinement-input-") as scratch:
        temporary = Path(scratch) / "manifest.jsonl"
        write_manifest(temporary, records())
        report = prepare_manifest(
            temporary, output, FilterConfig(**config.get("filters", {})), requester=requester
        )
        report["data_root"] = os.path.relpath(manifest.parent.resolve(), output.resolve())
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
