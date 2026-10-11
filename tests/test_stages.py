"""Data-flow tests use explicit mock models, not real model-quality evidence."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import types
from unittest.mock import patch
from pathlib import Path

from voxcaps.coverage_design.manifest import PAPER_SUBSET_COUNTS, coverage_metadata
from voxcaps.coverage_design.synthesis import teacher_text, validate_job, load_teacher
from voxcaps.data_refinement.attributes import attribute_context, extract_attributes
from voxcaps.data_refinement.scene import AnnotationError, normalize_annotation
from voxcaps.scope_annotation.provider import audio_content
from voxcaps.speech_preprocessing.segmentation import assemble_clips
from voxcaps.stages import stage1, stage2, stage3, stage4
import test_core
from test_core import raw, write_wav, unknown_attributes


class CoverageTests(unittest.TestCase):
    def test_three_equal_synthetic_subsets(self):
        self.assertEqual(len(PAPER_SUBSET_COUNTS), 3)
        self.assertEqual(set(PAPER_SUBSET_COUNTS.values()), {100000})
        self.assertTrue(coverage_metadata({"source_type": "targeted_synthesis", "synthetic_subset": "short_utterances"})["targeted_short"])
        with self.assertRaises(ValueError):
            coverage_metadata({"source_type": "collected", "targeted_short": True})

    def test_elderly_job_requires_audited_reference(self):
        job = {"id": "test", "student_text": "你好", "reference_audio": "ref.wav", "reference_transcript": "您好",
               "synthetic_subset": "elderly_timbres"}
        with self.assertRaises(ValueError): validate_job(job)
        validate_job({**job, "reference_age_group": "elderly"})

    def test_plain_student_text_preserved_without_hints(self):
        original = "欢迎 Brand 来到这里！"
        self.assertEqual(teacher_text(original, []), original)

    def test_pinyin_inpainting_uses_cosyvoice_tokens_and_leaves_student_text(self):
        try: import pypinyin
        except ImportError: self.skipTest("optional pypinyin package unavailable")
        original = "Brand给予好评。"
        result = teacher_text(original, [5], {5: "jǐ"})
        self.assertEqual(result, "Brand[j][ǐ]予好评。")
        self.assertEqual(original, "Brand给予好评。")

    def test_cosyvoice3_constructor_matches_upstream_signature(self):
        module = types.ModuleType("cosyvoice.cli.cosyvoice")
        def factory(model_dir, load_trt=False, load_vllm=False, fp16=False):
            return (model_dir, load_trt, load_vllm, fp16)
        module.AutoModel = factory
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"cosyvoice.cli.cosyvoice": module}):
            path = Path(folder)
            (path / "cosyvoice3.yaml").write_text("test: fixture\n")
            self.assertEqual(load_teacher(path), (str(path), False, False, False))


class PreprocessingTests(unittest.TestCase):
    def test_single_and_multi_duration_limits_at_turn_boundaries(self):
        single = assemble_clips([[0, 30, 0], [30, 59, 0], [59, 90, 0]])
        self.assertEqual([(c["start"], c["end"]) for c in single], [(0, 59), (59, 90)])
        multi = assemble_clips([[0, 50, 0], [50, 100, 1], [100, 119, 0]])
        self.assertEqual(len(multi), 1)
        self.assertEqual(multi[0]["end"], 119)

    def test_overlapping_and_long_turns_require_review(self):
        for turns in ([[0, 4, 0], [3, 5, 1]], [[0, 61, 0]]):
            with self.assertRaises(ValueError): assemble_clips(turns)


class AnnotationTests(unittest.TestCase):
    def test_gateway_uses_recovered_audio_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "mix.wav"
            write_wav(audio)
            content = audio_content(audio, media_format="video_url")
            self.assertEqual(content["type"], "video_url")
            self.assertTrue(content["video_url"]["url"].startswith("data:audio/wav;base64,"))

    def test_paper_has_no_four_speaker_limit(self):
        scene, _ = normalize_annotation({"environment": "安静。", "speakers": [
            {"id": "S5", "description": "青年女性。"}], "content": "S5说：<S5>你好</S5>"})
        self.assertEqual(scene.speakers[0]["id"], "S1")
        self.assertIn("<S1>你好</S1>", scene.content)


class RefinementTests(unittest.TestCase):
    def test_no_attribute_rule_fallback(self):
        scene, _ = normalize_annotation(raw())
        with self.assertRaises(AnnotationError): extract_attributes(scene)

    def test_persona_not_limited_to_old_keyword_list(self):
        scene, _ = normalize_annotation(raw(speakers=["Subject1: 青年女性，温和的湖族长老。"] ))
        labels = unknown_attributes()
        labels["speakers"][0]["persona"] = {"value": "湖族长老", "evidence": "温和的湖族长老"}
        self.assertEqual(extract_attributes(scene, cached=labels)["speakers"][0]["persona"]["value"], "湖族长老")

    def test_attribute_evidence_cannot_come_from_spoken_dialogue(self):
        scene, _ = normalize_annotation(raw("平静地说：<S1>你很生气</S1>"))
        labels = unknown_attributes()
        labels["turns"][0]["emotion"] = {"value": "生气", "evidence": "生气"}
        with self.assertRaises(AnnotationError): extract_attributes(scene, cached=labels)




class FourStageTests(unittest.TestCase):
    def test_all_four_stages_keep_mixture_and_run_refinement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root / "source.wav")
            manifest = root / "sources.jsonl"
            manifest.write_text(json.dumps({"id": "fixture", "audio": "source.wav", "fixture_only": True,
                                           "languages": ["zh"], "media_type": "short_drama"}) + "\n")
            stage1(manifest, root / "s1")
            calls = []

            class Models:
                def separate(self, mixture, destination):
                    calls.append(("separate", mixture.name))
                    shutil.copyfile(mixture, destination)
                    payload = destination.read_bytes()
                    destination.write_bytes(payload[:44] + b"\x02\x00" * ((len(payload) - 44) // 2))
                    return destination
                def diarize(self, vocal):
                    calls.append(("diarize", vocal.name))
                    return [[0, 2, 0]]
                def transcribe_and_align(self, vocal, language):
                    calls.append(("asr_align", vocal.name))
                    return {"text": "你好", "primary_asr": "你好", "secondary_asr": "你好", "language": language,
                            "asr_disagreement": False, "words": [{"text": "你好", "start_time": .1, "end_time": 1.2}]}
            stage2(root / "s1/manifest.jsonl", root / "s2", {}, models=Models())
            self.assertEqual(calls, [("separate", "source.wav"), ("diarize", "vocal.wav"), ("asr_align", "vocal.wav")])

            def annotate(audio, system, user):
                self.assertEqual(audio.name, "mixture.wav")
                self.assertIn("environment、speakers、content", system)
                return {"environment": "<Audio>雨声</Audio>", "speakers": ["Subject1: 青年女性。"],
                        "content": "温柔地说：<S1>你好</S1>"}
            stage3(root / "s2/manifest.jsonl", root / "s3", {}, backend=annotate)
            record = json.loads((root / "s3/manifest.jsonl").read_text())
            record.update(quality_score=test_core.QualityTests.boundary, attribute_annotations=unknown_attributes())
            (root / "s3/manifest.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
            result = stage4(root / "s3/manifest.jsonl", root / "s4", {})
            self.assertEqual(result["accepted"], 1)
            prepared = json.loads((root / "s4/accepted.jsonl").read_text())
            self.assertEqual(prepared["waveform_target"], prepared["annotation_audio"])
            self.assertNotEqual(prepared["waveform_target"], prepared["analysis_audio"])
            self.assertEqual(Path(prepared["waveform_target"]).read_bytes(), (root / "source.wav").read_bytes())
            self.assertNotEqual(Path(prepared["waveform_target"]).read_bytes(), Path(prepared["analysis_audio"]).read_bytes())
            self.assertTrue(prepared["fixture_only"])
            self.assertFalse(list((root / "s4").glob("refinement-input-*")))


if __name__ == "__main__": unittest.main()
