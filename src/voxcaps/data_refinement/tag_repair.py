"""Evidence-constrained repair of tag pairing and placement."""

from __future__ import annotations

import copy
import json
import re

from .scene import AnnotationError, normalize_annotation

REPAIR_PROMPT = """修正声音描述中 <S{id}> 和 <Audio> 标签的配对与位置。
使用给定的转写、说话人轮次和词级对齐约束，保持全部叙述文字和台词不变，只移动、补全、删除或规范化标签。
不得改写或新增人设、台词、音效、音乐、事件。不要把表达描述放进语音标签。
返回原有三字段 JSON：environment、speakers、content。说话人清单保持不变。
语音标签内的词面内容必须与转写一致，按首次发声连续编号。"""


def plain_text(value: str) -> str:
    value = value.replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"<[^<>]*>|<[^<>]*$", "", value)


def repair_annotation(raw: dict, record: dict, requester) -> dict:
    # Compatibility conversion changes field layout, not evidence text.
    converted = copy.deepcopy(raw)
    if "local" in converted:
        env = converted["env"]
        if converted.get("bgm", "").strip().lower() not in {"", "无", "none", "no music"}:
            env += "\n" + converted["bgm"].strip()
        converted = {
            "environment": env,
            "speakers": converted["speakers"],
            "content": converted["local"],
        }
    repaired = requester(
        REPAIR_PROMPT,
        json.dumps(
            {
                "annotation": converted,
                "transcript": record.get("transcript"),
                "turns": record.get("turns"),
            },
            ensure_ascii=False,
        ),
    )
    if repaired.get("speakers") != converted["speakers"]:
        raise AnnotationError("tag_repair_changed_speaker_inventory")
    for name in ("environment", "content"):
        if not isinstance(repaired.get(name), str) or plain_text(repaired[name]) != plain_text(
            converted[name]
        ):
            raise AnnotationError("tag_repair_changed_description_text")
    normalize_annotation(repaired)
    return repaired
