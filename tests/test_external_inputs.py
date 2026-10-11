from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_core import write_wav

from voxcaps.coverage_design.synthesis import synthesize_manifest
from voxcaps.pipeline import read_jsonl
from voxcaps.scope_annotation.annotator import annotation_prompts
from voxcaps.scope_annotation.provider import request_annotation
from voxcaps.stages import stage3


class ExternalInputTests(unittest.TestCase):
    def synthesis_run(self, root: Path, *, duplicate: bool = False):
        reference = root / "reference.wav"
        write_wav(reference)
        jobs = [
            {
                "id": identifier,
                "student_text": text,
                "reference_audio": reference.name,
                "reference_transcript": "external reference transcript",
            }
            for identifier, text in (("first", "caller text A"), ("second", "caller text B"))
        ]
        if duplicate:
            jobs[1]["id"] = jobs[0]["id"]
        manifest = root / "jobs.jsonl"
        manifest.write_text("".join(json.dumps(job) + "\n" for job in jobs))
        waveform = SimpleNamespace(shape=(1, 32000), cpu=lambda: waveform)
        calls = []

        def infer(text, prompt, audio, **kwargs):
            calls.append((text, prompt, audio))
            return [{"tts_speech": waveform}]

        teacher = SimpleNamespace(sample_rate=16000, inference_zero_shot=infer)
        torch = SimpleNamespace(cat=lambda chunks, dim: waveform)
        torchaudio = SimpleNamespace(save=lambda path, *args, **kwargs: write_wav(Path(path)))
        with (
            patch.dict(sys.modules, {"torch": torch, "torchaudio": torchaudio}),
            patch("voxcaps.coverage_design.synthesis.load_teacher", return_value=teacher) as loader,
        ):
            if duplicate:
                with self.assertRaisesRegex(ValueError, "duplicate_synthesis_id"):
                    synthesize_manifest(manifest, root / "output", root / "teacher")
                self.assertFalse((root / "output/manifest.jsonl").exists())
                self.assertEqual(len(calls), 1)
            else:
                result = synthesize_manifest(manifest, root / "output", root / "teacher")
                rows = list(read_jsonl(Path(result["manifest"])))
                self.assertEqual(result["generated"], 2)
                self.assertEqual(
                    [row["student_text"] for row in rows], [job["student_text"] for job in jobs]
                )
                self.assertEqual([call[0] for call in calls], [job["student_text"] for job in jobs])
                self.assertTrue(all(call[2] == str(reference) for call in calls))
                self.assertTrue(all((root / "output" / row["audio"]).is_file() for row in rows))
                self.assertTrue(all(row["synthetic_subset"] is None for row in rows))
            loader.assert_called_once()

    def test_synthesis_uses_external_text_and_reference_without_category(self):
        with tempfile.TemporaryDirectory() as folder:
            self.synthesis_run(Path(folder))

    def test_duplicate_input_ids_do_not_overwrite_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            self.synthesis_run(Path(folder), duplicate=True)

    def test_missing_reference_is_rejected_before_loading_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "jobs.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "missing",
                        "student_text": "caller text",
                        "reference_audio": "missing.wav",
                        "reference_transcript": "caller reference",
                    }
                )
                + "\n"
            )
            with patch("voxcaps.coverage_design.synthesis.load_teacher") as loader:
                with self.assertRaisesRegex(ValueError, "reference_audio_missing"):
                    synthesize_manifest(manifest, root / "output", root / "teacher")
                loader.assert_not_called()

    def test_qwen_request_contains_input_audio_and_our_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "input.wav"
            write_wav(audio)
            system, user = annotation_prompts({"media_type": "animation", "languages": ["zh"]})
            captured = {}
            expected = {"environment": "", "speakers": [], "content": ""}

            class Response(io.BytesIO):
                headers = {"Content-Type": "application/json"}

            def fake_open(request, timeout):
                captured.update(json.loads(request.data))
                data = {"choices": [{"message": {"content": json.dumps(expected)}}]}
                return Response(json.dumps(data).encode())

            with (
                patch.dict(os.environ, {"TEST_CREDENTIAL": "fixture-value"}),
                patch("voxcaps.scope_annotation.provider.urlopen", fake_open),
            ):
                result = request_annotation(
                    audio,
                    system,
                    user,
                    endpoint="https://example.org/chat/completions",
                    model="qwen3.5-omni-plus",
                    key_env="TEST_CREDENTIAL",
                )
            self.assertEqual(result, expected)
            self.assertEqual(captured["messages"][0]["content"], system)
            parts = captured["messages"][1]["content"]
            self.assertEqual([part["type"] for part in parts], ["text", "input_audio"])
            self.assertEqual(parts[0]["text"], user)
            self.assertEqual(
                base64.b64decode(parts[1]["input_audio"]["data"].split(",", 1)[1]),
                audio.read_bytes(),
            )

    def test_audio_jsonl_can_be_annotated_without_preprocessing_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "input.wav"
            write_wav(audio)
            manifest = root / "audio.jsonl"
            manifest.write_text(json.dumps({"id": "external", "audio": audio.name}) + "\n")
            calls = []

            def backend(path, system, user):
                calls.append(path)
                return {"environment": "", "speakers": [], "content": ""}

            result = stage3(manifest, root / "output", {}, backend=backend)
            self.assertEqual(result["annotated"], 1)
            self.assertEqual(calls, [audio.resolve()])
            annotated = list(read_jsonl(root / "output/manifest.jsonl"))[0]
            self.assertEqual(Path(annotated["audio"]), audio.resolve())

    def test_duplicate_audio_ids_stop_before_second_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root / "input.wav")
            row = json.dumps({"id": "external", "audio": "input.wav"}) + "\n"
            manifest = root / "audio.jsonl"
            manifest.write_text(row * 2)
            backend = unittest.mock.Mock(
                return_value={"environment": "", "speakers": [], "content": ""}
            )
            with self.assertRaisesRegex(ValueError, "duplicate_annotation_id"):
                stage3(manifest, root / "output", {}, backend=backend)
            backend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
