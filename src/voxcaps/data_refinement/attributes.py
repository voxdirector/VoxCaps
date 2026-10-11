"""Description-to-attribute extraction with evidence validation and no rule fallback."""

from __future__ import annotations

import json
import re

from .scene import TURN_RE, AnnotationError, Scene

ATTRIBUTE_PROMPT = """从规范化的声音描述中提取性别、年龄、情绪和人设标签。
只使用描述明确支持的信息，不根据名字、被提及的人物或说话内容猜测。
性别、年龄和人设属于说话人级；情绪属于当前话轮级。
每个非 null 标签必须含 value 和 evidence，evidence 必须逐字来自给定说话人描述或该话轮的相邻表达描述。
不确定或没有证据的标签填 null。允许任意有证据的人设，不限定词表。
返回 JSON：{"speakers":[{"id":"S1","gender":null,"age":null,"persona":null}],
"turns":[{"turn_index":0,"speaker_id":"S1","emotion":null}]}。
每个说话人和话轮各出现一次，顺序与输入一致；不要输出其他条目。"""


def attribute_context(scene: Scene) -> dict:
    turns = []
    previous_end = 0
    for index, match in enumerate(TURN_RE.finditer(scene.content)):
        context = scene.content[previous_end : match.start()]
        context = re.sub(r"<Audio>.*?</Audio>", "", context, flags=re.S)
        turns.append(
            {"turn_index": index, "speaker_id": "S" + match[1], "delivery_description": context}
        )
        previous_end = match.end()
    return {"speakers": scene.speakers, "turns": turns}


def validate_label(label, context: str):
    if label is None:
        return None
    if not isinstance(label, dict) or set(label) != {"value", "evidence"}:
        raise AnnotationError("attribute_requires_value_and_evidence")
    if not all(isinstance(label[k], str) and label[k].strip() for k in label):
        raise AnnotationError("empty_attribute_value_or_evidence")
    if label["evidence"] not in context:
        raise AnnotationError("attribute_evidence_not_in_its_scope")
    return label


def extract_attributes(scene: Scene, *, cached: dict | None = None, requester=None) -> dict:
    context = attribute_context(scene)
    if cached is None:
        if requester is None:
            raise AnnotationError("attribute_annotations_or_model_requester_required")
        cached = requester(ATTRIBUTE_PROMPT, json.dumps(context, ensure_ascii=False))
    if (
        not isinstance(cached, dict)
        or not isinstance(cached.get("speakers"), list)
        or not isinstance(cached.get("turns"), list)
    ):
        raise AnnotationError("invalid_attribute_response")
    if len(cached["speakers"]) != len(scene.speakers) or len(cached["turns"]) != len(
        context["turns"]
    ):
        raise AnnotationError("attribute_inventory_mismatch")
    speakers, turns = [], []
    for speaker, result in zip(scene.speakers, cached["speakers"]):
        if not isinstance(result, dict) or set(result) != {"id", "gender", "age", "persona"}:
            raise AnnotationError("attribute_speaker_schema_mismatch")
        if result.get("id") != speaker["id"]:
            raise AnnotationError("attribute_speaker_mismatch")
        speakers.append(
            {
                "id": speaker["id"],
                **{
                    key: validate_label(result.get(key), speaker["description"])
                    for key in ("gender", "age", "persona")
                },
            }
        )
    for turn, result in zip(context["turns"], cached["turns"]):
        if not isinstance(result, dict) or set(result) != {"turn_index", "speaker_id", "emotion"}:
            raise AnnotationError("attribute_turn_schema_mismatch")
        if (
            result.get("turn_index") != turn["turn_index"]
            or result.get("speaker_id") != turn["speaker_id"]
        ):
            raise AnnotationError("attribute_turn_mismatch")
        turns.append(
            {
                "turn_index": turn["turn_index"],
                "speaker_id": turn["speaker_id"],
                "emotion": validate_label(result.get("emotion"), turn["delivery_description"]),
            }
        )
    return {"method": "description_attribute_extraction", "speakers": speakers, "turns": turns}
