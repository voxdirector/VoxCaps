"""Apply alignment evidence to speech spans without editing event narration."""

from __future__ import annotations

import math
from types import SimpleNamespace

from .punctuation import punctuate_by_alignment_pause
from .scene import TURN_RE, AnnotationError, Scene, lexical_text


def refine_turn_punctuation(scene: Scene, turns: list[dict], duration: float) -> list[str]:
    spans = list(TURN_RE.finditer(scene.content))
    if len(turns) != len(spans):
        raise AnnotationError("alignment_turn_count_mismatch")
    replacements = []
    for index, (span, turn) in enumerate(zip(spans, turns)):
        if turn.get("speaker_id") != "S" + span[1]:
            raise AnnotationError("alignment_speaker_mismatch")
        text = turn.get("text")
        if not isinstance(text, str) or lexical_text(text) != lexical_text(span[2]):
            raise AnnotationError("transcript_does_not_match_speech_span")
        lang = turn.get("language")
        if lang not in {"zh", "en"}:
            raise AnnotationError("pause_refinement_supports_zh_en_only")
        words = turn.get("words")
        if not isinstance(words, list) or not words:
            raise AnnotationError("missing_word_alignment")
        items = []
        last_start = -1
        for word in words:
            start, end = word.get("start_time"), word.get("end_time")
            if any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                for v in (start, end)
            ):
                raise AnnotationError("invalid_alignment_time")
            if not 0 <= start < end <= duration + 1e-6 or start < last_start:
                raise AnnotationError("invalid_alignment_interval")
            if not isinstance(word.get("text"), str) or not word["text"].strip():
                raise AnnotationError("empty_alignment_word")
            items.append(SimpleNamespace(**word))
            last_start = start
        if lexical_text("".join(w.text for w in items)) != lexical_text(text):
            raise AnnotationError("alignment_lexical_content_mismatch")
        turn_end = turn.get("end_time", items[-1].end_time)
        if (
            not isinstance(turn_end, (float, int))
            or not math.isfinite(turn_end)
            or not items[-1].end_time <= turn_end <= duration + 1e-6
        ):
            raise AnnotationError("invalid_turn_end_time")
        if index + 1 < len(turns):
            following = turns[index + 1].get("words", [])
            if following and turn_end > following[0].get("start_time", duration) + 1e-6:
                raise AnnotationError("overlapping_turn_alignment")
        output = punctuate_by_alignment_pause(
            asr_text=span[2].replace("<|sp|>", ""),
            align_items=items,
            lang=lang,
            require_positive_alignment_duration_for_pause=True,
            audio_duration_s=turn_end,
        )
        if lexical_text(output) != lexical_text(span[2]):
            raise AnnotationError("punctuation_changed_lexical_content")
        replacements.append((span.start(2), span.end(2), output))
    for start, end, output in reversed(replacements):
        scene.content = scene.content[:start] + output + scene.content[end:]
    return ["alignment_guided_punctuation"]
