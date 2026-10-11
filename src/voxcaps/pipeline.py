"""Offline refinement from local audio and cached preprocessing/annotation."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from .coverage_design.manifest import coverage_metadata
from .data_refinement.attributes import extract_attributes
from .data_refinement.description import refine_turn_punctuation
from .data_refinement.quality import FilterConfig, filter_reasons
from .data_refinement.scene import (
    TURN_RE,
    AnnotationError,
    lexical_text,
    normalize_annotation,
)
from .data_refinement.speaker_tags import canonicalize_speaker_tag_tokens
from .speech_preprocessing.audio import wav_duration


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid_json_at_line:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"record_must_be_object_at_line:{line_number}")
                yield record


def prepare_record(
    record: dict, base: Path, config: FilterConfig | None = None, *, requester=None
) -> dict:
    identifier = record.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise AnnotationError("missing_record_id")
    audio = record.get("audio")
    if not isinstance(audio, str) or not audio:
        raise AnnotationError("missing_original_mixture_audio")
    audio_path = base / audio
    if not audio_path.is_file():
        raise AnnotationError("original_mixture_audio_not_found")
    duration = wav_duration(audio_path)
    if "duration" in record and (
        isinstance(record["duration"], bool)
        or not isinstance(record["duration"], (int, float))
        or not math.isfinite(record["duration"])
        or abs(record["duration"] - duration) > 0.02
    ):
        raise AnnotationError("duration_does_not_match_waveform")
    metadata = coverage_metadata(record)
    try:
        scene, audit = normalize_annotation(record.get("annotation"))
    except AnnotationError:
        if requester is None:
            raise
        from .data_refinement.tag_repair import repair_annotation

        repaired = repair_annotation(record["annotation"], record, requester)
        scene, audit = normalize_annotation(repaired)
        audit.insert(0, "evidence_constrained_tag_repair")
    reasons = filter_reasons(
        duration,
        record.get("quality_score"),
        len(scene.speakers),
        targeted_short=metadata["targeted_short"],
        config=config,
    )
    if reasons:
        raise AnnotationError(";".join(reasons))
    if record.get("transcript") is not None:
        transcript = record["transcript"]
        spoken = "".join(match[2] for match in TURN_RE.finditer(scene.content))
        if not isinstance(transcript, str) or lexical_text(transcript) != lexical_text(spoken):
            raise AnnotationError("transcript_does_not_match_scene")
    if record.get("student_text") is not None:
        spoken = "".join(match[2] for match in TURN_RE.finditer(scene.content))
        if lexical_text(record["student_text"]) != lexical_text(spoken):
            raise AnnotationError("synthetic_student_orthography_changed")
    if record.get("analysis_audio"):
        vocal_path = base / record["analysis_audio"]
        if not vocal_path.is_file() or abs(wav_duration(vocal_path) - duration) > 0.02:
            raise AnnotationError("analysis_audio_timeline_mismatch")
        if vocal_path.resolve() == audio_path.resolve():
            raise AnnotationError("analysis_audio_must_be_distinct_from_original_mixture")
    if "turns" in record:
        turns = copy.deepcopy(record["turns"])
        if not isinstance(turns, list):
            raise AnnotationError("turns_must_be_list")
        if "renumbered_by_first_appearance" in audit:
            raw_content = record["annotation"].get("local", record["annotation"].get("content"))
            canonical = canonicalize_speaker_tag_tokens(
                raw_content.replace("&lt;", "<").replace("&gt;", ">")
            )
            # Open spans suffice, including a terminal span whose close was repaired.
            old_ids = list(
                dict.fromkeys("S" + m[1] for m in re.finditer(r"<S([1-9][0-9]*)>", canonical))
            )
            mapping = {old: "S" + str(index + 1) for index, old in enumerate(old_ids)}
            for turn in turns:
                turn["speaker_id"] = mapping.get(turn.get("speaker_id"), turn.get("speaker_id"))
        audit.extend(refine_turn_punctuation(scene, turns, duration))
    else:
        audit.append("punctuation_not_refined:no_alignment")
    return {
        "schema_version": 1,
        "id": identifier,
        "waveform_target": audio,
        "annotation_audio": audio,
        "analysis_audio": record.get("analysis_audio"),
        "duration": duration,
        "scene": scene.to_dict(),
        "description": scene.render(),
        "attributes": extract_attributes(
            scene, cached=record.get("attribute_annotations"), requester=requester
        ),
        "quality_score": record["quality_score"],
        "dnsmos": record.get("dnsmos"),
        "raw_annotation": record["annotation"],
        "fixture_only": record.get("fixture_only") is True,
        "coverage": metadata,
        "student_text": record.get("student_text"),
        "audit": audit,
    }


def prepare_manifest(
    manifest: Path, output: Path, config: FilterConfig | None = None, *, requester=None
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    accepted_path, rejected_path, report_path = [
        output / name for name in ("accepted.jsonl", "rejected.jsonl", "report.json")
    ]
    if any(path.exists() for path in (accepted_path, rejected_path, report_path)):
        raise ValueError("output_already_exists:choose_a_new_directory")
    counts = Counter()
    seen = set()
    temporary = [
        output / (path.name + ".tmp") for path in (accepted_path, rejected_path, report_path)
    ]
    try:
        with (
            temporary[0].open("w", encoding="utf-8") as accepted,
            temporary[1].open("w", encoding="utf-8") as rejected,
        ):
            for record in read_jsonl(manifest):
                identifier = record.get("id")
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise ValueError("missing_or_duplicate_manifest_id")
                seen.add(identifier)
                counts["total"] += 1
                try:
                    result = prepare_record(record, manifest.parent, config, requester=requester)
                except (AnnotationError, OSError, ValueError) as error:
                    reasons = str(error).split(";")
                    rejected.write(
                        json.dumps({"id": identifier, "reasons": reasons}, ensure_ascii=False)
                        + "\n"
                    )
                    counts["rejected"] += 1
                    counts.update("reason:" + reason for reason in reasons)
                else:
                    accepted.write(json.dumps(result, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
        report = {
            "total": counts["total"],
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "data_root": os.path.relpath(manifest.parent.resolve(), output.resolve()),
            "rejection_reasons": {
                k[7:]: v for k, v in sorted(counts.items()) if k.startswith("reason:")
            },
        }
        temporary[2].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for source, destination in zip(temporary, (accepted_path, rejected_path, report_path)):
            source.replace(destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return report
