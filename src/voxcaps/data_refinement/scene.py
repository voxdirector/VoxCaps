"""Legacy annotation adaptation and conservative, audited markup repair."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .speaker_tags import canonicalize_speaker_tag_tokens, speaker_tag_structure_error

TURN_RE = re.compile(r"<S([1-9][0-9]*)>(.*?)</S\1>", re.DOTALL)
TAG_RE = re.compile(r"<(/?)(S[1-9][0-9]*|Audio)>")
PREFIX_RE = re.compile(r"^(?:Subject|Speaker|S)\s*([1-9][0-9]*)\s*[:：]\s*(.+)$", re.DOTALL | re.I)


class AnnotationError(ValueError):
    pass


@dataclass
class Scene:
    environment: str
    speakers: list[dict[str, str]]
    content: str

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        inventory = "\n".join(f"{s['id']}: {s['description']}" for s in self.speakers)
        return (
            f"<Environment>{self.environment}</Environment>\n"
            f"<Speakers>{inventory}</Speakers>\n"
            f"<Content>{self.content}</Content>"
        )


def lexical_text(text: str) -> str:
    """Compare spoken lexical content without punctuation or explicit pauses."""
    text = text.replace("<|sp|>", "")
    return "".join(c.casefold() for c in text if c.isalnum())


def repair_markup(text: str) -> tuple[str, list[str]]:
    repairs = []
    if re.search(r"&(?:lt|gt);", text):
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        repairs.append("decoded_angle_entities")
    normalized = canonicalize_speaker_tag_tokens(text)
    normalized = re.sub(r"<\s*(/?)\s*audio\s*>", r"<\1Audio>", normalized, flags=re.I)
    if normalized != text:
        repairs.append("canonicalized_tag_tokens")
    # Only close a single terminal open span. Mismatches, nesting and ambiguous
    # interior boundaries are rejected rather than assigning new speech/events.
    stack = []
    for match in TAG_RE.finditer(normalized):
        closing, name = match.groups()
        if closing:
            if not stack or stack[-1] != name:
                raise AnnotationError(f"mismatched_tag:{name}")
            stack.pop()
        else:
            if stack:
                raise AnnotationError("nested_event_or_speech_tag")
            stack.append(name)
    if len(stack) == 1:
        normalized += f"</{stack[0]}>"
        repairs.append(f"closed_terminal_tag:{stack[0]}")
    return normalized, repairs


def validate_markup(text: str, *, allow_speech: bool = True) -> None:
    error = speaker_tag_structure_error(text)
    if error:
        raise AnnotationError(error)
    stack = []
    for token in re.finditer(r"<[^<>]*>|<[^<>]*$", text):
        if token.group() == "<|sp|>":
            continue
        match = TAG_RE.fullmatch(token.group())
        if not match:
            raise AnnotationError(f"unsupported_tag:{token.group()[:40]}")
        closing, name = match.groups()
        if not allow_speech and name.startswith("S"):
            raise AnnotationError("speech_in_environment")
        if closing:
            if not stack or stack[-1][0] != name:
                raise AnnotationError(f"mismatched_tag:{name}")
            _, start = stack.pop()
            if not text[start : token.start()].strip():
                raise AnnotationError(f"empty_tag:{name}")
        else:
            if stack:
                raise AnnotationError("nested_event_or_speech_tag")
            stack.append((name, token.end()))
    if stack:
        raise AnnotationError(f"unclosed_tag:{stack[-1][0]}")


def normalize_annotation(raw: dict) -> tuple[Scene, list[str]]:
    if not isinstance(raw, dict):
        raise AnnotationError("annotation_must_be_object")
    legacy = "local" in raw
    required = (
        {"env", "bgm", "speakers", "local"} if legacy else {"environment", "speakers", "content"}
    )
    if not required.issubset(raw):
        raise AnnotationError("missing_annotation_fields")
    environment = raw["env"] if legacy else raw["environment"]
    content = raw["local"] if legacy else raw["content"]
    if not isinstance(environment, str) or not isinstance(content, str) or not content.strip():
        raise AnnotationError("invalid_environment_or_content")
    audit = []
    if legacy:
        bgm = raw["bgm"]
        if not isinstance(bgm, str):
            raise AnnotationError("bgm_must_be_string")
        if bgm.strip().lower() not in {"", "无", "none", "no music"}:
            environment = environment.rstrip() + "\n" + bgm.strip()
        audit.append("merged_legacy_env_bgm")
    environment, fixes = repair_markup(environment)
    audit.extend("environment:" + f for f in fixes)
    content, fixes = repair_markup(content)
    audit.extend("content:" + f for f in fixes)
    validate_markup(environment, allow_speech=False)
    validate_markup(content)

    if not isinstance(raw["speakers"], list):
        raise AnnotationError("speakers_must_be_list")
    speakers = {}
    for item in raw["speakers"]:
        if isinstance(item, str):
            match = PREFIX_RE.fullmatch(item.strip())
            if not match:
                raise AnnotationError("speaker_description_requires_explicit_id")
            key, description = "S" + match[1], match[2]
        elif isinstance(item, dict):
            key, description = item.get("id"), item.get("description")
        else:
            raise AnnotationError("invalid_speaker_entry")
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"S[1-9][0-9]*", key)
            or not isinstance(description, str)
            or not description.strip()
        ):
            raise AnnotationError("invalid_speaker_id_or_description")
        if key in speakers:
            raise AnnotationError("duplicate_speaker_id")
        if re.search(r"[<>]", description):
            raise AnnotationError("speaker_profile_must_be_plain_description")
        speakers[key] = description.strip()
    first = list(dict.fromkeys("S" + m[1] for m in TURN_RE.finditer(content)))
    if set(first) != set(speakers):
        raise AnnotationError("speaker_inventory_does_not_match_spoken_turns")
    mapping = {old: "S" + str(i + 1) for i, old in enumerate(first)}
    if any(old != new for old, new in mapping.items()):

        def replace_id(match):
            old = "S" + match[2]
            return match[1] + mapping[old][1:]

        pattern = re.compile(r"(Subject|Speaker|(?<![A-Za-z])S)([1-9][0-9]*)(?![0-9A-Za-z])", re.I)
        content = pattern.sub(replace_id, content)
        speakers = {mapping[old]: pattern.sub(replace_id, speakers[old]) for old in first}
        audit.append("renumbered_by_first_appearance")
    else:
        speakers = {key: speakers[key] for key in first}
    scene = Scene(
        environment.strip(),
        [{"id": k, "description": v} for k, v in speakers.items()],
        content.strip(),
    )
    return scene, audit
