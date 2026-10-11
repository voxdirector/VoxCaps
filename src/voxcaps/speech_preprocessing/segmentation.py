"""Turn-boundary clip assembly and dual-ASR/forced-alignment orchestration."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

from .audio import segment_pair, wav_duration


def assemble_clips(
    turns: list[list], *, min_duration: float = 1, max_single: float = 60, max_multi: float = 120
) -> list[dict]:
    """Pack chronological whole turns without cutting at arbitrary times.

    A turn longer than the applicable maximum is rejected for resegmentation.
    The paper does not prescribe an optimization objective for packing turns;
    this implementation deterministically fills each clip up to its limit.
    """
    if not 0 < min_duration <= max_single <= max_multi:
        raise ValueError("invalid_segment_limits")
    clips = []
    current = []

    def flush():
        if current and current[-1][1] - current[0][0] >= min_duration:
            clips.append({"start": current[0][0], "end": current[-1][1], "turns": list(current)})

    for turn in turns:
        if (
            len(turn) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in turn[:2])
            or not 0 <= turn[0] < turn[1]
        ):
            raise ValueError("invalid_diarization_turn")
        if current and turn[0] < current[-1][1]:
            raise ValueError("overlapping_or_unordered_turns_require_review")
        if turn[1] - turn[0] > max_single:
            raise ValueError("long_single_turn_requires_vad_resegmentation")
        candidate = current + [turn]
        maximum = max_multi if len({t[2] for t in candidate}) > 1 else max_single
        if current and turn[1] - current[0][0] > maximum:
            flush()
            current = [turn]
        else:
            current.append(turn)
    flush()
    return clips


def preprocess_source(record: dict, base: Path, output: Path, models) -> list[dict]:
    mixture = base / record["audio"]
    if not mixture.is_file():
        raise ValueError("source_audio_not_found")
    if output.exists():
        raise ValueError("preprocessing_output_already_exists")
    output.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(dir=output, prefix="analysis-") as scratch:
            scratch = Path(scratch)
            vocal = models.separate(mixture, scratch / "vocal.wav")
            synthetic = record.get("source_type") == "targeted_synthesis"
            if synthetic:
                if wav_duration(mixture) > 60:
                    raise ValueError("synthetic_single_speaker_clip_above_maximum")
                turns = [[0, wav_duration(mixture), 0]]
                clips = [{"start": 0, "end": wav_duration(mixture), "turns": turns}]
            else:
                turns = models.diarize(vocal)
                clips = assemble_clips(turns)
            results = []
            for index, clip in enumerate(clips):
                folder = output / f"clip_{index:06d}"
                segment_pair(mixture, vocal, clip["start"], clip["end"], folder)
                mapping = {}
                aligned = []
                for turn_index, (start, end, speaker) in enumerate(clip["turns"]):
                    mapping.setdefault(speaker, "S" + str(len(mapping) + 1))
                    tmp = scratch / f"turn_{index}_{turn_index}"
                    segment_pair(mixture, vocal, start, end, tmp)
                    languages = record.get("languages") or ["zh"]
                    if isinstance(languages, str):
                        languages = [languages]
                    language = languages[0]
                    if synthetic:
                        result = models.transcribe_and_align(
                            tmp / "vocal.wav", language, trusted_text=record["student_text"]
                        )
                    else:
                        result = models.transcribe_and_align(tmp / "vocal.wav", language)
                    if not result["text"].strip():
                        raise ValueError("diarized_turn_has_empty_transcript")
                    offset = start - clip["start"]
                    for word in result["words"]:
                        word["start_time"] += offset
                        word["end_time"] += offset
                    result.update(speaker_id=mapping[speaker], end_time=end - clip["start"])
                    aligned.append(result)
                retained = {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "quality_score",
                        "dnsmos",
                        "annotation",
                        "attribute_annotations",
                        "human_review",
                        "human_valid",
                    }
                }
                results.append(
                    {
                        **retained,
                        "id": record["id"] + f"_{index:06d}",
                        "audio": str((folder / "mixture.wav").resolve()),
                        "analysis_audio": str((folder / "vocal.wav").resolve()),
                        "duration": clip["end"] - clip["start"],
                        "turns": aligned,
                        "transcript": record["student_text"]
                        if synthetic
                        else "".join(t["text"] for t in aligned),
                        "transcript_source": "trusted" if synthetic else "asr",
                        "source_start": clip["start"],
                        "source_end": clip["end"],
                        "asr_disagreement": any(t["asr_disagreement"] for t in aligned),
                    }
                )
            return results
    except Exception:
        shutil.rmtree(output)
        raise
