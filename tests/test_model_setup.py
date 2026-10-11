from __future__ import annotations

import ast
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from voxcaps.model_setup import check_models
from voxcaps.scope_annotation.provider import audio_content, request_json, response_text

ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def __init__(self, data: bytes, content_type: str):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


def load_function(path: Path, name: str, globals_: dict):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
    exec(code, globals_)
    return globals_[name]


class ProviderTests(unittest.TestCase):
    def test_official_audio_input_contains_data_url(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.wav"
            path.write_bytes(b"RIFF-test-fixture")
            part = audio_content(path)
            self.assertEqual(part["type"], "input_audio")
            self.assertEqual(part["input_audio"]["format"], "wav")
            self.assertTrue(part["input_audio"]["data"].startswith("data:audio/wav;base64,"))

    def test_stream_collects_content_and_ignores_reasoning_usage(self):
        events = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "not output"}}]},
            {"choices": [{"index": 0, "delta": {"content": '{"content":'}}]},
            {"choices": [{"index": 0, "delta": {"content": '"你好"}'}}]},
            {"choices": [], "usage": {"total_tokens": 10}},
        ]
        data = b"".join(
            ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode() for event in events
        )
        data += b"data: [DONE]\n\n"
        self.assertEqual(
            json.loads(response_text(Response(data, "text/event-stream"))), {"content": "你好"}
        )

    def test_stream_rejects_truncated_output(self):
        data = b'data: {"choices":[{"delta":{"content":"{"},"finish_reason":"length"}]}\n\n'
        with self.assertRaisesRegex(ValueError, "truncated"):
            response_text(Response(data, "text/event-stream"))

    def test_request_uses_streaming_text_only_payload(self):
        captured = {}

        def fake_open(request, timeout):
            captured.update(json.loads(request.data))
            return Response(
                b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\ndata: [DONE]\n\n',
                "text/event-stream",
            )

        with (
            patch.dict(os.environ, {"TEST_CREDENTIAL": "fixture-value"}),
            patch("voxcaps.scope_annotation.provider.urlopen", fake_open),
        ):
            result = request_json(
                [{"role": "user", "content": "test"}],
                endpoint="https://example.org/chat/completions",
                model="qwen3.5-omni-plus",
                key_env="TEST_CREDENTIAL",
            )
        self.assertEqual(result, {})
        self.assertTrue(captured["stream"])
        self.assertEqual(captured["modalities"], ["text"])
        self.assertTrue(captured["stream_options"]["include_usage"])

    def test_nonstream_gateway_response_remains_supported(self):
        data = json.dumps(
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        ).encode()
        self.assertEqual(response_text(Response(data, "application/json")), "{}")


class DependencyTests(unittest.TestCase):
    def dns_method(self, waveform, source_rate=16000):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy package unavailable")
        source = ROOT / "src/voxcaps/data_refinement/dnsmos.py"
        cls = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "ComputeScore"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
        )

        class AudioReader:
            @staticmethod
            def read(path):
                return waveform, source_rate

        class Resampler:
            @staticmethod
            def resample(audio, *, orig_sr, target_sr):
                return audio

        namespace = {"np": np, "sf": AudioReader, "librosa": Resampler, "INPUT_LENGTH": 9.01}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["__call__"], np

    def test_dnsmos_empty_audio_does_not_enter_repeat_loop(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy package unavailable")
        method, _ = self.dns_method(np.zeros(0))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            method(None, "fixture.wav", 16000, False)

    def test_dnsmos_stereo_is_downmixed_before_inference(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy package unavailable")
        method, np = self.dns_method(np.ones((32000, 2)))

        class Objective:
            def run(self, names, inputs):
                assert inputs["input_1"].ndim == 2
                return [np.array([[3.0, 3.0, 3.0]])]

        class Subjective:
            def run(self, names, inputs):
                return [np.array([[3.0]])]

        class Scorer:
            onnx_sess = Objective()
            p808_onnx_sess = Subjective()

            @staticmethod
            def audio_melspec(audio):
                return np.zeros((900, 120))

            @staticmethod
            def get_polyfit_val(sig, bak, ovr, personal):
                return sig, bak, ovr

        result = method(Scorer(), "fixture.wav", 16000, False)
        self.assertEqual(result["len_in_sec"], 2)
        self.assertEqual(result["OVRL"], 3)

    def test_vad_silent_context_restores_stdout_after_exception(self):
        source = ROOT / "src/voxcaps/speech_preprocessing/speakerlab/utils/utils.py"
        silent_print = load_function(
            source, "silent_print", {"contextmanager": contextmanager, "os": os, "sys": sys}
        )
        previous = sys.stdout
        with self.assertRaises(RuntimeError):
            with silent_print():
                raise RuntimeError("fixture")
        self.assertIs(sys.stdout, previous)

    def test_uvr_loader_selects_checkpoint_instead_of_licence_file(self):
        source = ROOT / "src/voxcaps/speech_preprocessing/uvr/src/mdxc.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MDXC")

        class Base:
            def __init__(self, *args, device=None, **kwargs):
                self.device = device

            def update_metadata(self, value):
                self.other_metadata.update(value)

        class API:
            def load_mdxc_models_data(self, *args, **kwargs):
                return {}

            def get_model_hash_from_path(self, *args, **kwargs):
                return "fixture-hash"

            def load_mdxc_model_data(self, *args, **kwargs):
                return {}

            def load_modle(self, *args, **kwargs):
                return object()

        with tempfile.TemporaryDirectory() as folder:
            weight_dir = Path(folder)
            (weight_dir / "LICENSE").write_text("fixture")
            checkpoint = weight_dir / "MDX23C-8KFFT-InstVoc_HQ.ckpt"
            checkpoint.write_bytes(b"fixture")
            namespace = {
                "BaseModel": Base,
                "mdxc_api": API(),
                "MODELPARAMS_ROOT": weight_dir,
                "resolve_uvr_weight_dir": lambda *args, **kwargs: weight_dir,
            }
            module = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__", names=[ast.alias(name="annotations")], level=0
                    ),
                    cls,
                ],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
            model = namespace["MDXC"]("MDX23C-8KFFT-InstVoc_HQ", {}, device="cpu")
            self.assertEqual(model.model_path, checkpoint)

    def test_official_config_no_longer_points_to_missing_remote_model_py(self):
        config = json.loads(
            (ROOT / "configs/02_speech_preprocessing/config.example.json").read_text()
        )
        self.assertNotIn("remote_code", config.get("fun_asr_kwargs", {}))

    def test_preflight_reports_missing_models_without_importing_them(self):
        config = json.loads(
            (ROOT / "configs/02_speech_preprocessing/config.example.json").read_text()
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("voxcaps.model_setup.importlib.util.find_spec", return_value=None),
        ):
            config = {
                **config,
                "uvr_model_root": folder,
                "fun_asr_nano_model": folder,
                "sensevoice_model": folder,
                "forced_aligner_model": folder,
                "speaker_model_cache": folder,
            }
            result = check_models(2, config)
        self.assertFalse(result["ready"])
        self.assertTrue(
            any(item["item"] == "torch" and not item["ok"] for item in result["checks"])
        )

    def test_vad_configuration_without_weights_is_not_ready(self):
        config = json.loads(
            (ROOT / "configs/02_speech_preprocessing/config.example.json").read_text()
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("voxcaps.model_setup.importlib.util.find_spec", return_value=object()),
        ):
            for key in (
                "uvr_model_root",
                "fun_asr_nano_model",
                "sensevoice_model",
                "forced_aligner_model",
                "speaker_model_cache",
            ):
                config[key] = str(Path(folder) / key)
            for item in check_models(2, config)["checks"]:
                if item["kind"] == "model_file":
                    path = Path(item["path"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
            self.assertTrue(check_models(2, config)["ready"])
            vad_dir = (
                Path(config["speaker_model_cache"]) / "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            )
            (vad_dir / "model.pt").unlink()
            result = check_models(2, config)
        self.assertFalse(result["ready"])
        failed = [item["item"] for item in result["checks"] if not item["ok"]]
        self.assertEqual(failed, ["FSMN VAD: model.pt"])


if __name__ == "__main__":
    unittest.main()
