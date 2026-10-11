"""Command line entry points for local preparation and optional inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data_refinement.quality import FilterConfig
from .data_refinement.ranking import rank_preferences, rank_scores
from .pipeline import prepare_manifest, read_jsonl
from .scope_annotation.annotator import annotation_prompts
from .speech_preprocessing.audio import segment_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoxCaps local annotation and refinement")
    commands = parser.add_subparsers(dest="command", required=True)
    synthesize = commands.add_parser("synthesize", help="Generate CosyVoice 3 speech from JSONL")
    synthesize.add_argument("manifest", type=Path)
    synthesize.add_argument("--output", type=Path, required=True)
    synthesize.add_argument("--teacher-dir", type=Path, required=True)
    setup = commands.add_parser(
        "check-models", help="Check model paths and dependencies without loading weights"
    )
    setup.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    setup.add_argument("--config", type=Path)
    setup.add_argument("--teacher-dir", type=Path)
    for number, help_text in (
        (1, "Coverage design: ingest media and targeted CosyVoice 3 synthesis"),
        (2, "Speech preprocessing: UVR, diarization, dual ASR and alignment"),
        (3, "Scope-structured annotation: mixture, transcript and persona matrix"),
        (4, "Data refinement: automatic quality, normalization and attributes"),
    ):
        stage = commands.add_parser(f"stage{number}", help=help_text)
        stage.add_argument("manifest", type=Path)
        stage.add_argument("--output", type=Path, required=True)
        if number == 1:
            stage.add_argument("--teacher-dir", type=Path)
        else:
            stage.add_argument("--config", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Refine cached annotations and word alignments")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path)
    prompt = commands.add_parser(
        "prompt", help="Inspect actual matrix-conditioned annotation prompts"
    )
    prompt.add_argument(
        "record", type=Path, help="One JSON record with transcript/media_type/languages"
    )
    rank = commands.add_parser("rank", help="Optionally order existing scores or preference pairs")
    rank.add_argument("input", type=Path)
    rank.add_argument("--mode", choices=("scores", "preferences"), default="scores")
    rank.add_argument("--score-field", default="score")
    rank.add_argument("--ascending", action="store_true")
    segment = commands.add_parser(
        "segment", help="Cut original and analysis PCM-WAV using the same time interval"
    )
    segment.add_argument("--mixture", type=Path, required=True)
    segment.add_argument("--vocal", type=Path, required=True)
    segment.add_argument("--start", type=float, required=True)
    segment.add_argument("--end", type=float, required=True)
    segment.add_argument("--output", type=Path, required=True)
    annotate = commands.add_parser(
        "annotate", help="One explicit request to a compatible audio annotation model"
    )
    annotate.add_argument("record", type=Path)
    annotate.add_argument("--endpoint", required=True, help="Complete chat/completions URL")
    annotate.add_argument("--model", required=True)
    annotate.add_argument("--key-env", default="VOXCAPS_API_KEY")
    score = commands.add_parser(
        "score", help="Optional torchaudio SQUIM scoring; may download model weights"
    )
    score.add_argument("audio", type=Path)
    score.add_argument("--reference-audio", type=Path, required=True)
    score.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthesize":
            from .coverage_design.synthesis import synthesize_manifest

            result = synthesize_manifest(args.manifest, args.output, args.teacher_dir)
        elif args.command == "check-models":
            from .model_setup import check_models

            config = json.loads(args.config.read_text()) if args.config else {}
            if args.stage != 1 and args.config is None:
                raise ValueError("model_config_is_required")
            result = check_models(args.stage, config, teacher_dir=args.teacher_dir)
        elif args.command.startswith("stage"):
            from . import stages

            runner = getattr(stages, args.command)
            if args.command == "stage1":
                result = runner(args.manifest, args.output, teacher_dir=args.teacher_dir)
            else:
                result = runner(args.manifest, args.output, json.loads(args.config.read_text()))
        elif args.command == "prepare":
            config = FilterConfig(**json.loads(args.config.read_text())) if args.config else None
            result = prepare_manifest(args.manifest, args.output, config)
        elif args.command == "prompt":
            system, user = annotation_prompts(json.loads(args.record.read_text()))
            result = {"system_prompt": system, "user_prompt": user}
        elif args.command == "rank":
            records = read_jsonl(args.input)
            result = (
                rank_scores(records, field=args.score_field, descending=not args.ascending)
                if args.mode == "scores"
                else rank_preferences(records)
            )
        elif args.command == "segment":
            result = segment_pair(args.mixture, args.vocal, args.start, args.end, args.output)
        elif args.command == "annotate":
            from .scope_annotation.provider import request_annotation

            record = json.loads(args.record.read_text())
            system, user = annotation_prompts(record)
            result = request_annotation(
                args.record.parent / record["audio"],
                system,
                user,
                endpoint=args.endpoint,
                model=args.model,
                key_env=args.key_env,
            )
        else:
            from .data_refinement.squim import score_audio

            result = score_audio(args.audio, args.reference_audio, args.device)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as error:
        print(f"voxcaps: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "check-models":
        return 0 if result["ready"] else 2
    return 2 if args.command in {"prepare", "stage4"} and result["accepted"] == 0 else 0
