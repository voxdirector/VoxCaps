"""Coverage metadata and corpus accounting for the paper's two source routes."""

from __future__ import annotations

from collections import Counter

MEDIA_FAMILIES = ("audiobook", "advertising", "animation", "short_drama", "podcast", "film")
SYNTHETIC_SUBSETS = ("elderly_timbres", "short_utterances", "difficult_pronunciation")
PAPER_SUBSET_COUNTS = dict.fromkeys(SYNTHETIC_SUBSETS, 100_000)
CONTEXT_AXES = (
    "speaker_density",
    "recording_condition",
    "character_type",
    "location",
    "sound_field",
    "persistent_ambience",
    "delivery_pattern",
    "local_acoustic_events",
)


def coverage_metadata(record: dict) -> dict:
    source = record.get("source_type", "collected")
    if source not in {"collected", "targeted_synthesis"}:
        raise ValueError("unsupported_source_route")
    subset = record.get("synthetic_subset")
    if source == "targeted_synthesis" and subset is not None and subset not in SYNTHETIC_SUBSETS:
        raise ValueError("unknown_paper_subset")
    short = record.get("targeted_short", subset == "short_utterances")
    if (
        not isinstance(short, bool)
        or short
        and (source != "targeted_synthesis" or subset != "short_utterances")
    ):
        raise ValueError("invalid_short_utterance_exemption")
    languages = record.get("languages") or []
    if isinstance(languages, str):
        languages = [languages]
    if not isinstance(languages, list) or not all(isinstance(x, str) for x in languages):
        raise ValueError("languages_must_be_strings")
    axes = record.get("context_axes") or {}
    if not isinstance(axes, dict):
        raise ValueError("context_axes_must_be_an_object")
    return {
        "source_type": source,
        "synthetic_subset": subset,
        "targeted_short": short,
        "media_type": record.get("media_type"),
        "languages": languages,
        "context_axes": {
            **{key: value for key, value in axes.items() if key in CONTEXT_AXES},
            **{key: record[key] for key in CONTEXT_AXES if key in record},
        },
    }


def coverage_report(records: list[dict]) -> dict:
    media, languages, subsets = Counter(), Counter(), Counter()
    for record in records:
        meta = coverage_metadata(record)
        media[meta["media_type"] or "unspecified"] += 1
        languages.update(meta["languages"])
        if meta["synthetic_subset"]:
            subsets[meta["synthetic_subset"]] += 1
    return {
        "examples": len(records),
        "media": dict(media),
        "languages": dict(languages),
        "synthetic_subsets": dict(subsets),
        "paper_subset_counts": PAPER_SUBSET_COUNTS,
        "paper_count_differences": {
            key: subsets[key] - count for key, count in PAPER_SUBSET_COUNTS.items()
        },
    }
