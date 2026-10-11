"""Recovered lightweight speaker markup validation."""

from __future__ import annotations

import re

_TAG_CANDIDATE_RE = re.compile(
    r"<[^<>]*?/?\s*S\s*\d[^<>]*(?:>|$)",
    flags=re.IGNORECASE,
)

_RELAXED_TAG_RE = re.compile(
    r"<\s*(/?)\s*S\s*(\d+)\s*>",
    flags=re.IGNORECASE,
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_PAUSE_TOKEN_RE = re.compile(r"<\|sp\|>", flags=re.IGNORECASE)


def _speaker_tag_validation_view(value: str) -> str:
    """Hide pause tokens only while validating surrounding tag grammar."""
    return _PAUSE_TOKEN_RE.sub("", str(value or ""))


def _pause_token_splits_angle_token(value: str) -> bool:
    """Return whether a pause token begins inside an unclosed angle token."""
    text = str(value or "")
    for match in _PAUSE_TOKEN_RE.finditer(text):
        if text.rfind("<", 0, match.start()) > text.rfind(">", 0, match.start()):
            return True
    return False


def speaker_items_from_field(value) -> list[str]:
    """Parse the speaker field exactly as caption construction does."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        if ";" in value:
            return [item.strip() for item in value.split(";") if item.strip()]
        return [value.strip()] if value.strip() else []
    if value is None or not str(value).strip():
        return []
    return [str(value).strip()]


def count_speakers_field(value) -> int:
    """Count speakers using the same representation accepted by the builder."""
    return len(speaker_items_from_field(value))


def _speaker_tag_candidates(value: str) -> list[str]:
    """Return fragments that look like numbered speaker tags."""
    return [match.group(0) for match in _TAG_CANDIDATE_RE.finditer(str(value or ""))]


def _parse_speaker_tag_token(token: str) -> tuple[bool, int] | None:
    """Parse a speaker tag as (is_closing, speaker_id)."""
    parsed = _RELAXED_TAG_RE.fullmatch(str(token))
    if parsed is None:
        return None
    return bool(parsed.group(1)), int(parsed.group(2))


def canonicalize_speaker_tag_tokens(value: str) -> str:
    """Normalize unambiguous positive speaker-ID tags while preserving other text."""
    if not isinstance(value, str) or not value:
        return value

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        parsed = _parse_speaker_tag_token(token)
        if parsed is None:
            return token
        is_closing, speaker_id = parsed
        if speaker_id < 1:
            return token
        slash = "/" if is_closing else ""
        return f"<{slash}S{speaker_id}>"

    return _TAG_CANDIDATE_RE.sub(replace, value)


def speaker_tag_structure_error(value: str) -> str:
    """Return the first malformed, unsupported, or misnested tag error."""
    raw_text = str(value or "")
    if not raw_text:
        return ""
    if _CONTROL_RE.search(raw_text):
        return "forbidden_control_character"
    if _pause_token_splits_angle_token(raw_text):
        return "pause_token_inside_tag"
    text = _speaker_tag_validation_view(raw_text)

    stack: list[int] = []
    for token in _speaker_tag_candidates(text):
        parsed = _parse_speaker_tag_token(token)
        if parsed is None:
            return f"malformed_token={token[:80]!r}"

        is_closing, speaker_id = parsed
        if speaker_id < 1:
            return f"unsupported_speaker_id=S{speaker_id}"

        if is_closing:
            if not stack:
                return f"unexpected_close=S{speaker_id}"
            if stack[-1] != speaker_id:
                return f"mismatched_close=open_S{stack[-1]} close_S{speaker_id}"
            stack.pop()
            continue

        if stack:
            return f"nested_open=open_S{stack[-1]} nested_S{speaker_id}"
        stack.append(speaker_id)

    return f"unclosed_open=S{stack[-1]}" if stack else ""
