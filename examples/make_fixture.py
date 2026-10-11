"""Create a technical smoke-test fixture, not real speech or evaluated scores."""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


def create(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name, frequency in (("mixture.wav", 440), ("vocal.wav", 220)):
        path = destination / name
        if path.exists():
            raise ValueError("fixture_destination_already_populated")
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 32000, "NONE", "not compressed"))
            audio.writeframes(
                b"".join(
                    struct.pack("<h", int(1000 * math.sin(2 * math.pi * frequency * i / 16000)))
                    for i in range(32000)
                )
            )
    record = {
        "id": "technical-fixture",
        "fixture_only": True,
        "audio": "mixture.wav",
        "analysis_audio": "vocal.wav",
        "duration": 2,
        "media_type": "short_drama",
        "languages": ["zh"],
        "transcript": "你好世界",
        "transcript_source": "trusted",
        "quality_score": {"pesq": 2.0, "stoi": 0.85, "si_sdr": 0.0, "mos": 2.5},
        "annotation": {
            "env": "室内，背景有<Audio>微弱的雨声</Audio>。",
            "bgm": "轻柔的钢琴音乐。",
            "speakers": ["Subject1: 青年女性，新闻播报员，音色明亮，吐字清晰。"],
            "local": "Subject1 温柔地说：< s 1 >你好世界</ s 1 >接着是<audio>短促的敲门声</audio>。",
        },
        "turns": [
            {
                "speaker_id": "S1",
                "text": "你好世界",
                "language": "zh",
                "end_time": 1.9,
                "words": [
                    {"text": "你好", "start_time": 0.1, "end_time": 0.6},
                    {"text": "世界", "start_time": 0.9, "end_time": 1.4},
                ],
            }
        ],
        "attribute_annotations": {
            "speakers": [
                {
                    "id": "S1",
                    "gender": {"value": "女性", "evidence": "女性"},
                    "age": {"value": "青年", "evidence": "青年"},
                    "persona": {"value": "新闻播报员", "evidence": "新闻播报员"},
                }
            ],
            "turns": [
                {
                    "turn_index": 0,
                    "speaker_id": "S1",
                    "emotion": {"value": "温柔", "evidence": "温柔"},
                }
            ],
        },
    }
    manifest = destination / "manifest.jsonl"
    manifest.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    (destination / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(create(parser.parse_args().output))
