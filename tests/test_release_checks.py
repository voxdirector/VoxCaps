from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from test_core import write_wav

from voxcaps.coverage_design.manifest import coverage_metadata
from voxcaps.coverage_design.synthesis import teacher_text
from voxcaps.model_setup import check_models
from voxcaps.scope_annotation import prompts
from voxcaps.scope_annotation.annotator import annotate_record, annotation_prompts
from voxcaps.scope_annotation.provider import request_json
from voxcaps.stages import stage1, stage3, stage4

ROOT = Path(__file__).resolve().parents[1]


class ReleaseChecks(unittest.TestCase):
    def test_saved_matrix_matches_runtime_prompt_content(self):
        matrix = json.loads((ROOT / "configs/03_scope_annotation/style_matrix.json").read_text())
        for key, constant in (
            ("scenes", "SCENE_PROMPT_RULES"),
            ("advertising_performance", "AD_SCENARIO_PERFORMANCE_HINTS"),
            ("advertising_combinations", "AD_SCENARIO_COMBINATION_HINTS"),
            ("advertising_domains", "AD_SCENARIO_DOMAIN_HINTS"),
        ):
            self.assertEqual(matrix[key], json.loads(json.dumps(getattr(prompts, constant))))

    def test_nullable_optional_annotation_fields_use_audio_prompt(self):
        self.assertEqual(
            annotation_prompts({"languages": None, "advertising_scenarios": None}),
            annotation_prompts({}),
        )
        with self.assertRaisesRegex(ValueError, "media_type"):
            annotation_prompts({"media_type": 3})

    def test_original_and_analysis_paths_remain_usable_after_annotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("original.wav", "analysis.wav"):
                write_wav(root / name)
            row = {"id": "input", "audio": "original.wav", "analysis_audio": "analysis.wav"}
            backend = Mock(return_value={"environment": "", "speakers": [], "content": ""})
            result = annotate_record(row, root, backend)
            self.assertEqual(Path(result["audio"]), (root / "original.wav").resolve())
            self.assertEqual(Path(result["analysis_audio"]), (root / "analysis.wav").resolve())
            self.assertEqual(row["audio"], "original.wav")

    def test_coverage_rejects_duplicate_ids_before_downstream_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root / "input.wav")
            manifest = root / "input.jsonl"
            row = json.dumps({"id": "same", "audio": "input.wav"}) + "\n"
            manifest.write_text(row * 2)
            with self.assertRaisesRegex(ValueError, "duplicate_coverage_id"):
                stage1(manifest, root / "out")
            self.assertFalse((root / "out/manifest.jsonl").exists())

    def test_context_axes_survive_saved_manifest_metadata(self):
        first = coverage_metadata({"location": "indoors", "languages": None})
        second = coverage_metadata(first)
        self.assertEqual(second["context_axes"], {"location": "indoors"})

    def test_existing_refinement_output_is_checked_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out"
            output.mkdir()
            (output / "accepted.jsonl").write_text("existing\n")
            with patch("voxcaps.data_refinement.squim.SQUIMScorer") as scorer:
                with self.assertRaisesRegex(ValueError, "output_already_exists"):
                    stage4(Path(folder) / "missing.jsonl", output, {"quality_models": {}})
                scorer.assert_not_called()
            self.assertEqual((output / "accepted.jsonl").read_text(), "existing\n")

    def test_unsupported_and_mismatched_audio_runtimes_fail_preflight(self):
        for versions in (("2.9.0", "2.9.0"), ("2.8.0", "2.7.1")):
            with patch("voxcaps.model_setup.version", side_effect=versions):
                result = check_models(1, {})
            compatibility = next(
                item for item in result["checks"] if item["kind"] == "compatibility"
            )
            self.assertFalse(compatibility["ok"])

    def test_supported_runtime_accepts_local_build_suffixes(self):
        with patch("voxcaps.model_setup.version", side_effect=("2.8.0+cpu", "2.8.0+cpu")):
            result = check_models(1, {})
        compatibility = next(item for item in result["checks"] if item["kind"] == "compatibility")
        self.assertTrue(compatibility["ok"])

    def test_unused_and_boolean_pronunciation_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not_selected"):
            teacher_text("给予", [0], {1: "yǔ"})
        with self.assertRaisesRegex(ValueError, "invalid_pronunciation_indices"):
            teacher_text("给予", [True])

    def test_model_requests_are_paced_and_not_retried(self):
        class Response(io.BytesIO):
            headers = {"Content-Type": "application/json"}

        payload = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()
        with (
            patch.dict(os.environ, {"TEST_CREDENTIAL": "fixture-value"}),
            patch("voxcaps.scope_annotation.provider._NEXT_REQUEST_AT", 0.0),
            patch(
                "voxcaps.scope_annotation.provider.time.monotonic",
                side_effect=(10.0, 10.0, 10.1, 11.0),
            ),
            patch("voxcaps.scope_annotation.provider.time.sleep") as sleep,
            patch(
                "voxcaps.scope_annotation.provider.urlopen",
                side_effect=lambda *args, **kwargs: Response(payload),
            ) as opener,
        ):
            for _ in range(2):
                request_json(
                    [],
                    endpoint="https://example.org/chat/completions",
                    model="qwen3.5-omni-plus",
                    key_env="TEST_CREDENTIAL",
                )
            self.assertEqual(opener.call_count, 2)
            self.assertAlmostEqual(sleep.call_args.args[0], 0.9)

    def test_failed_request_stops_batch_without_retrying(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root / "input.wav")
            manifest = root / "input.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps({"id": name, "audio": "input.wav"}) + "\n"
                    for name in ("first", "second")
                )
            )
            config = {
                "model_family": "Qwen3.5-Omni",
                "model": "qwen3.5-omni-plus",
                "endpoint": "https://example.org/chat/completions",
                "key_env": "TEST_CREDENTIAL",
            }
            error = HTTPError(config["endpoint"], 429, "fixture", {}, None)
            with (
                patch.dict(os.environ, {"TEST_CREDENTIAL": "fixture-value"}),
                patch("voxcaps.scope_annotation.provider._NEXT_REQUEST_AT", 0.0),
                patch("voxcaps.scope_annotation.provider.urlopen", side_effect=error) as opener,
            ):
                with self.assertRaisesRegex(RuntimeError, "http_error:429"):
                    stage3(manifest, root / "output", config)
                opener.assert_called_once()
            self.assertFalse((root / "output/manifest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
