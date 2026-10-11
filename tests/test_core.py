from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from voxcaps.cli import annotation_prompts
from voxcaps.data_refinement.attributes import extract_attributes
from voxcaps.data_refinement.description import refine_turn_punctuation
from voxcaps.data_refinement.punctuation import punctuate_by_alignment_pause
from voxcaps.data_refinement.quality import filter_reasons
from voxcaps.data_refinement.scene import AnnotationError, lexical_text, normalize_annotation
from voxcaps.pipeline import prepare_manifest, prepare_record
from voxcaps.scope_annotation.prompts import (
    AD_SCENARIO_COMBINATION_HINTS,
    AD_SCENARIO_DOMAIN_HINTS,
    AD_SCENARIO_PERFORMANCE_HINTS,
    SCENE_PROMPT_RULES,
)
from voxcaps.speech_preprocessing.audio import segment_pair


def raw(content="Subject1说：<S1>你好</S1>", speakers=None):
    return {
        "env": "<Audio>雨声</Audio>",
        "bgm": "钢琴音乐",
        "speakers": speakers if speakers is not None else ["Subject1: 青年女性，新闻播报员。"],
        "local": content,
    }


def write_wav(path: Path, frames=32000):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, frames, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * frames)


def unknown_attributes(ids=("S1",), turn_ids=("S1",)):
    return {
        "speakers": [{"id": key, "gender": None, "age": None, "persona": None} for key in ids],
        "turns": [
            {"turn_index": i, "speaker_id": key, "emotion": None} for i, key in enumerate(turn_ids)
        ],
    }


class SceneTests(unittest.TestCase):
    def test_legacy_merge_retains_music_and_raw_speech(self):
        scene, audit = normalize_annotation(raw())
        self.assertIn("钢琴音乐", scene.environment)
        self.assertEqual(scene.speakers[0]["id"], "S1")
        self.assertEqual(scene.content, "Subject1说：<S1>你好</S1>")
        self.assertIn("merged_legacy_env_bgm", audit)

    def test_unambiguous_repair_and_audit(self):
        scene, audit = normalize_annotation(raw("&lt; s 1 &gt;你好</ s 1 ><audio>敲门声"))
        self.assertEqual(scene.content, "<S1>你好</S1><Audio>敲门声</Audio>")
        self.assertIn("content:closed_terminal_tag:Audio", audit)

    def test_mismatched_and_nested_tags_rejected(self):
        for text in ("<S1>你好</S2>", "<S1>你<Audio>雨声</Audio>好</S1>", "<S1>你好<S2>您好</S2>"):
            with self.subTest(text=text), self.assertRaises(AnnotationError):
                normalize_annotation(raw(text))

    def test_inventory_must_match_actual_speech(self):
        with self.assertRaises(AnnotationError):
            normalize_annotation(raw(speakers=["Subject1: 女性。", "Subject2: 男性。"]))
        with self.assertRaises(AnnotationError):
            normalize_annotation(raw("<S2>您好</S2>"))

    def test_numbering_follows_first_appearance(self):
        scene, audit = normalize_annotation(
            raw(
                "Subject2:<S2>早</S2>Subject1:<S1>好</S1>",
                ["Subject1: 老年男性。", "Subject2: 青年女性。与Subject1不同。"],
            )
        )
        self.assertTrue(scene.content.startswith("Subject1:<S1>早</S1>"))
        self.assertIn("与Subject2不同", scene.speakers[0]["description"])
        self.assertIn("renumbered_by_first_appearance", audit)

    def test_pause_inside_tag_and_unsupported_id_rejected(self):
        for text in ("<S<|sp|>1>你好</S1>", "<S5>你好</S5>"):
            with self.subTest(text=text), self.assertRaises(AnnotationError):
                normalize_annotation(raw(text))

    def test_empty_speech_tag_rejected(self):
        with self.assertRaises(AnnotationError):
            normalize_annotation(raw("<S1></S1>"))

    def test_audio_only_scene(self):
        scene, _ = normalize_annotation(raw("<Audio>响亮的钟声</Audio>", []))
        self.assertEqual(scene.speakers, [])


class PunctuationTests(unittest.TestCase):
    def punctuate(self, text, words, lang="zh", **kwargs):
        return punctuate_by_alignment_pause(
            asr_text=text, align_items=[SimpleNamespace(**w) for w in words], lang=lang, **kwargs
        )

    def test_pause_bands_and_tail(self):
        words = [
            {"text": c, "start_time": s, "end_time": e}
            for c, s, e in (("甲", 0, 0.1), ("乙", 0.2, 0.3), ("丙", 0.6, 0.7), ("丁", 1.3, 1.4))
        ]
        self.assertEqual(
            self.punctuate("甲乙丙丁", words, audio_duration_s=2), "甲<|sp|>乙，丙。丁。"
        )

    def test_english_contraction_preserved(self):
        words = [
            {"text": "I", "start_time": 0, "end_time": 0.2},
            {"text": "can't", "start_time": 0.5, "end_time": 0.7},
            {"text": "go", "start_time": 0.75, "end_time": 1},
        ]
        result = self.punctuate("I can't go!", words, "en", audio_duration_s=1.7)
        self.assertIn("can't", result)
        self.assertEqual(lexical_text(result), lexical_text("I can't go!"))
        self.assertTrue(result.endswith("!"))

    def test_imperfect_alignment_does_not_drop_words(self):
        result = self.punctuate("你好世界", [{"text": "你好", "start_time": 0, "end_time": 0.2}])
        self.assertEqual(lexical_text(result), "你好世界")

    def test_only_speech_span_is_refined(self):
        scene, _ = normalize_annotation(raw("温柔地说：<S1>你好</S1><Audio>敲门声</Audio>"))
        refine_turn_punctuation(
            scene,
            [
                {
                    "speaker_id": "S1",
                    "text": "你好",
                    "language": "zh",
                    "end_time": 1.5,
                    "words": [{"text": "你好", "start_time": 0.1, "end_time": 0.7}],
                }
            ],
            2,
        )
        self.assertEqual(scene.content, "温柔地说：<S1>你好。</S1><Audio>敲门声</Audio>")

    def test_mismatched_transcript_rejected(self):
        scene, _ = normalize_annotation(raw())
        with self.assertRaises(AnnotationError):
            refine_turn_punctuation(
                scene, [{"speaker_id": "S1", "text": "再见", "language": "zh", "words": []}], 2
            )

    def test_invalid_alignment_times_rejected(self):
        for start, end in ((-1, 0.7), (0.7, 0.2), (0, float("nan")), (0, 3)):
            scene, _ = normalize_annotation(raw())
            with self.subTest(times=(start, end)), self.assertRaises(AnnotationError):
                refine_turn_punctuation(
                    scene,
                    [
                        {
                            "speaker_id": "S1",
                            "text": "你好",
                            "language": "zh",
                            "words": [{"text": "你好", "start_time": start, "end_time": end}],
                        }
                    ],
                    2,
                )


class QualityTests(unittest.TestCase):
    boundary = {"pesq": 2.0, "stoi": 0.85, "si_sdr": 0.0, "mos": 2.5}

    def test_inclusive_thresholds(self):
        self.assertEqual(filter_reasons(1, self.boundary, 1), [])
        self.assertEqual(filter_reasons(120, self.boundary, 2), [])

    def test_single_speaker_and_short_exemption(self):
        self.assertIn("duration_above_maximum", filter_reasons(61, self.boundary, 1))
        self.assertIn("duration_below_minimum", filter_reasons(0.2, self.boundary, 1))
        self.assertEqual(filter_reasons(0.2, self.boundary, 1, targeted_short=True), [])

    def test_missing_nonfinite_low_scores_rejected(self):
        self.assertIn(
            "missing_or_invalid_quality:mos", filter_reasons(2, {**self.boundary, "mos": None}, 1)
        )
        self.assertIn(
            "missing_or_invalid_quality:stoi",
            filter_reasons(2, {**self.boundary, "stoi": float("nan")}, 1),
        )
        self.assertIn(
            "quality_below_minimum:pesq", filter_reasons(2, {**self.boundary, "pesq": 1.99}, 1)
        )


class AttributeTests(unittest.TestCase):
    def test_evidence_from_description_not_spoken_dialogue(self):
        scene, _ = normalize_annotation(raw("S1生气地说：<S1>你是一位温柔的公主</S1>"))
        cached = {
            "speakers": [
                {
                    "id": "S1",
                    "gender": {"value": "女性", "evidence": "女性"},
                    "age": None,
                    "persona": {"value": "新闻播报员", "evidence": "新闻播报员"},
                }
            ],
            "turns": [
                {
                    "turn_index": 0,
                    "speaker_id": "S1",
                    "emotion": {"value": "生气", "evidence": "生气"},
                }
            ],
        }
        labels = extract_attributes(scene, cached=cached)
        self.assertEqual(labels["speakers"][0]["gender"]["evidence"], "女性")
        self.assertEqual(labels["speakers"][0]["persona"]["value"], "新闻播报员")
        self.assertEqual(labels["turns"][0]["emotion"]["value"], "生气")

    def test_unknown_and_negated_persona_stay_unknown(self):
        scene, _ = normalize_annotation(raw(speakers=["Subject1: 声音清晰，并非霸道总裁。"]))
        labels = extract_attributes(scene, cached=unknown_attributes())["speakers"][0]
        self.assertIsNone(labels["gender"])
        self.assertIsNone(labels["age"])
        self.assertIsNone(labels["persona"])

    def test_demographics_ignore_comparison_to_other_voice(self):
        scene, _ = normalize_annotation(raw(speakers=["Subject1: 青年男性，音色像女性和老年人。"]))
        cached = unknown_attributes()
        cached["speakers"][0].update(
            gender={"value": "男性", "evidence": "青年男性"},
            age={"value": "青年", "evidence": "青年男性"},
        )
        labels = extract_attributes(scene, cached=cached)["speakers"][0]
        self.assertEqual(labels["gender"]["value"], "男性")
        self.assertEqual(labels["age"]["value"], "青年")


class PromptTests(unittest.TestCase):
    def test_asr_sentence_punctuation_removed_and_media_routed(self):
        _, prompt = annotation_prompts(
            {
                "media_type": "short_drama",
                "languages": ["zh"],
                "transcript": "你好，世界！",
                "transcript_source": "asr",
            }
        )
        self.assertIn("你好世界", prompt)
        self.assertNotIn("你好，世界！", prompt)
        self.assertIn("当前分类标签：短剧/长剧/影视剧", prompt)
        self.assertIn("当前任务的语音语种固定为：中文", prompt)

    def test_lexical_apostrophe_decimal_and_trusted_punctuation_preserved(self):
        _, prompt = annotation_prompts(
            {"transcript": "I can't pay 3.14!", "transcript_source": "asr"}
        )
        self.assertIn("I can't pay 3.14", prompt)
        _, trusted = annotation_prompts({"transcript": "你好！", "transcript_source": "trusted"})
        self.assertIn("你好！", trusted)


class PipelineTests(unittest.TestCase):
    def test_alignment_ids_renumbered_with_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_wav(base / "mix.wav")
            record = {
                "id": "r",
                "audio": "mix.wav",
                "annotation": raw("Subject2:<S2>你好</S2>", ["Subject2: 青年女性。"]),
                "quality_score": QualityTests.boundary,
                "turns": [
                    {
                        "speaker_id": "S2",
                        "text": "你好",
                        "language": "zh",
                        "words": [{"text": "你好", "start_time": 0.1, "end_time": 0.7}],
                    }
                ],
            }
            result = prepare_record({**record, "attribute_annotations": unknown_attributes()}, base)
            self.assertIn("<S1>你好</S1>", result["scene"]["content"])
            self.assertEqual(record["turns"][0]["speaker_id"], "S2")

    def test_short_exemption_requires_synthetic_source_and_fixture_flag_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_wav(base / "mix.wav", 3200)
            record = {
                "id": "r",
                "audio": "mix.wav",
                "annotation": raw(),
                "quality_score": QualityTests.boundary,
                "targeted_short": True,
                "fixture_only": True,
            }
            with self.assertRaises(ValueError):
                prepare_record(record, base)
            result = prepare_record(
                {
                    **record,
                    "source_type": "targeted_synthesis",
                    "synthetic_subset": "short_utterances",
                    "attribute_annotations": unknown_attributes(),
                },
                base,
            )
            self.assertTrue(result["fixture_only"])

    def test_original_mixture_retained_and_rejection_audited(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_wav(base / "mix.wav")
            write_wav(base / "vocal.wav")
            record = {
                "id": "good",
                "audio": "mix.wav",
                "analysis_audio": "vocal.wav",
                "annotation": raw(),
                "transcript": "你好",
                "quality_score": QualityTests.boundary,
                "attribute_annotations": unknown_attributes(),
            }
            bad = {**record, "id": "bad", "quality_score": {}}
            manifest = base / "input.jsonl"
            manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (record, bad)))
            report = prepare_manifest(manifest, base / "output")
            self.assertEqual((report["accepted"], report["rejected"]), (1, 1))
            output = json.loads((base / "output/accepted.jsonl").read_text())
            self.assertEqual(output["waveform_target"], "mix.wav")
            self.assertTrue(
                (base / "output" / report["data_root"] / output["waveform_target"]).is_file()
            )
            self.assertEqual(output["annotation_audio"], "mix.wav")
            self.assertEqual(output["analysis_audio"], "vocal.wav")
            self.assertNotIn("human_valid", output)
            self.assertEqual(output["raw_annotation"], record["annotation"])
            with self.assertRaises(ValueError):
                prepare_manifest(manifest, base / "output")

    def test_duration_and_transcript_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_wav(base / "mix.wav")
            record = {
                "id": "r",
                "audio": "mix.wav",
                "annotation": raw(),
                "quality_score": QualityTests.boundary,
            }
            for addition in ({"duration": 5}, {"transcript": "再见"}):
                with self.subTest(addition=addition), self.assertRaises(AnnotationError):
                    prepare_record({**record, **addition}, base)

    def test_pcm_cut_preserves_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_wav(base / "mix.wav")
            write_wav(base / "vocal.wav")
            segment_pair(base / "mix.wav", base / "vocal.wav", 0.5, 1.5, base / "cut")
            with wave.open(str(base / "cut/mixture.wav"), "rb") as audio:
                self.assertEqual(audio.getnframes(), 16000)
                self.assertEqual(audio.readframes(16000), b"\x01\x00" * 16000)

    def test_exported_matrix_matches_prompt_source(self):
        path = Path(__file__).resolve().parents[1] / "configs/03_scope_annotation/style_matrix.json"
        matrix = json.loads(path.read_text())
        expected = {
            "schema_version": 1,
            "scenes": SCENE_PROMPT_RULES,
            "advertising_performance": AD_SCENARIO_PERFORMANCE_HINTS,
            "advertising_combinations": AD_SCENARIO_COMBINATION_HINTS,
            "advertising_domains": AD_SCENARIO_DOMAIN_HINTS,
        }
        self.assertEqual(matrix, json.loads(json.dumps(expected)))


if __name__ == "__main__":
    unittest.main()
