"""Mixture-plus-transcript annotation with the paper's three output scopes."""

from __future__ import annotations

from ..speech_preprocessing.models import strip_sentence_punctuation
from .prompts import (
    SCENE_PROMPT_RULES,
    build_language_prompt_hint,
    build_scene_prompt_hint,
    build_voxcaps_audio_prompts,
    parse_lang_values,
)


def annotation_prompts(record: dict) -> tuple[str, str]:
    scene = record.get("media_type")
    if scene is not None and not isinstance(scene, str):
        raise ValueError("media_type_must_be_string")
    scene = next(
        (rule["display_name"] for rule in SCENE_PROMPT_RULES if rule["name"] == scene), scene
    )
    if scene == "animation":
        scene = "动漫/动画"
    languages = record.get("languages") or []
    if isinstance(languages, str):
        languages = [languages]
    if not isinstance(languages, list) or not all(isinstance(value, str) for value in languages):
        raise ValueError("annotation_languages_must_be_strings")
    scenarios = record.get("advertising_scenarios") or []
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    if not isinstance(scenarios, list) or not all(isinstance(value, str) for value in scenarios):
        raise ValueError("advertising_scenarios_must_be_strings")
    transcript = record.get("transcript")
    source = record.get("transcript_source", "asr")
    if source not in {"asr", "trusted"}:
        raise ValueError("transcript_source_must_be_asr_or_trusted")
    if transcript is not None and not isinstance(transcript, str):
        raise ValueError("transcript_must_be_string")
    if transcript and source == "asr":
        transcript = strip_sentence_punctuation(transcript)
    return build_voxcaps_audio_prompts(
        known_text=transcript,
        known_text_source=source,
        scene_prompt_hint=build_scene_prompt_hint(
            [scene] if scene else [], ad_scenario_values=scenarios
        ),
        language_prompt_hint=build_language_prompt_hint(parse_lang_values(",".join(languages))),
    )


def annotate_record(record: dict, base, backend) -> dict:
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        raise ValueError("annotation_record_id_is_required")
    if not isinstance(record.get("audio"), str) or not record["audio"]:
        raise ValueError("annotation_audio_path_is_required")
    audio = (base / record["audio"]).resolve()
    if not audio.is_file():
        raise ValueError("annotation_audio_not_found")
    retained = {**record, "audio": str(audio)}
    if record.get("analysis_audio"):
        if not isinstance(record["analysis_audio"], str):
            raise ValueError("analysis_audio_path_must_be_string")
        analysis = (base / record["analysis_audio"]).resolve()
        if not analysis.is_file():
            raise ValueError("analysis_audio_not_found")
        retained["analysis_audio"] = str(analysis)
    system, user = annotation_prompts(record)
    annotation = backend(audio, system, user)
    if set(annotation) != {"environment", "speakers", "content"}:
        raise ValueError("annotator_must_return_exactly_three_scopes")
    return {**retained, "annotation": annotation}
