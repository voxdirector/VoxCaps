"""Check configured model files and dependencies without loading model weights."""

from __future__ import annotations

import importlib.util
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

STAGE2_IMPORTS = (
    "torch",
    "torchaudio",
    "numpy",
    "librosa",
    "soundfile",
    "scipy",
    "sklearn",
    "funasr",
    "qwen_asr",
    "modelscope",
    "ml_collections",
    "yaml",
    "audiofile",
    "audioread",
    "six",
)


def check_models(stage: int, config: dict, *, teacher_dir: Path | None = None) -> dict:
    checks = []

    def audio_runtime():
        versions = {}
        parsed = {}
        for name in ("torch", "torchaudio"):
            try:
                value = version(name)
            except PackageNotFoundError:
                value = None
            versions[name] = value
            match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value or "")
            parsed[name] = tuple(map(int, match.groups())) if match else None
        minimum = (2, 6, 0) if stage == 2 else (2, 1, 0)
        compatible = (
            parsed["torch"] is not None
            and parsed["torch"] == parsed["torchaudio"]
            and minimum <= parsed["torch"] < (2, 9, 0)
        )
        checks.append(
            {
                "item": "matching supported torch/torchaudio versions",
                "kind": "compatibility",
                "versions": versions,
                "ok": compatible,
            }
        )

    def dependency(name):
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        checks.append({"item": name, "kind": "dependency", "ok": found})

    def file(name, path):
        path = Path(path)
        checks.append({"item": name, "kind": "model_file", "path": str(path), "ok": path.is_file()})

    def endpoint(settings):
        value = settings.get("endpoint", "")
        url = urlsplit(value)
        configured = (
            url.scheme in {"http", "https"}
            and bool(url.netloc)
            and "WORKSPACE_ID" not in value
            and not url.username
            and not url.password
            and not url.fragment
        )
        checks.append({"item": "annotation endpoint", "kind": "api", "ok": configured})
        checks.append(
            {
                "item": "Qwen3.5-Omni model configuration",
                "kind": "api",
                "ok": settings.get("model_family") == "Qwen3.5-Omni"
                and isinstance(settings.get("model"), str)
                and bool(settings["model"].strip())
                and settings.get("auth_scheme", "bearer") in {"bearer", "azure"},
            }
        )
        checks.append(
            {
                "item": "streaming text output",
                "kind": "api",
                "ok": settings.get("stream", True) is True
                and settings.get("modalities", ["text"]) == ["text"],
            }
        )
        key_env = settings.get("key_env", "DASHSCOPE_API_KEY")
        checks.append(
            {"item": key_env, "kind": "credential_environment", "ok": bool(os.environ.get(key_env))}
        )

    if stage == 1:
        for name in ("torch", "torchaudio", "pypinyin", "cosyvoice", "matcha"):
            dependency(name)
        audio_runtime()
        root = teacher_dir or Path("models/Fun-CosyVoice3-0.5B")
        for name in (
            "cosyvoice3.yaml",
            "llm.pt",
            "flow.pt",
            "hift.pt",
            "campplus.onnx",
            "speech_tokenizer_v3.onnx",
            "CosyVoice-BlankEN/config.json",
            "CosyVoice-BlankEN/model.safetensors",
            "CosyVoice-BlankEN/tokenizer_config.json",
            "CosyVoice-BlankEN/merges.txt",
            "CosyVoice-BlankEN/vocab.json",
        ):
            file("CosyVoice 3: " + name, root / name)
    elif stage == 2:
        for name in STAGE2_IMPORTS:
            dependency(name)
        audio_runtime()
        model_name = config["uvr_model_name"]
        file(
            "UVR MDXC",
            Path(config["uvr_model_root"]) / "mdxc/weights" / model_name / (model_name + ".ckpt"),
        )
        for key in ("fun_asr_nano_model", "sensevoice_model"):
            for name in ("config.yaml", "configuration.json", "model.pt"):
                file(key + ": " + name, Path(config[key]) / name)
        for name in ("config.json", "tokenizer.json"):
            file("Fun-ASR-Nano: " + name, Path(config["fun_asr_nano_model"]) / "Qwen3-0.6B" / name)
        for name in ("am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model"):
            file("SenseVoice: " + name, Path(config["sensevoice_model"]) / name)
        for name in (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ):
            file("Qwen3-ForcedAligner: " + name, Path(config["forced_aligner_model"]) / name)
        cache = Path(config.get("speaker_model_cache", "models/speaker_models"))
        file(
            "CAM++",
            cache / "iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt",
        )
        for name in ("configuration.json", "config.yaml", "model.pt", "am.mvn"):
            file("FSMN VAD: " + name, cache / "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" / name)
        remote_code = config.get("fun_asr_kwargs", {}).get("remote_code")
        if remote_code:
            file("custom Fun-ASR code", remote_code)
    elif stage == 3:
        endpoint(config)
    elif stage == 4:
        for name in ("torch", "torchaudio", "numpy", "librosa", "soundfile", "onnxruntime"):
            dependency(name)
        audio_runtime()
        quality = config.get("quality_models", {})
        for key in ("squim_reference_audio", "dnsmos_primary_model", "dnsmos_p808_model"):
            if quality.get(key):
                file(key, quality[key])
            else:
                checks.append({"item": key, "kind": "model_file", "ok": False})
        if config.get("description_model"):
            endpoint(config["description_model"])
    else:
        raise ValueError("stage_must_be_1_to_4")
    return {"stage": stage, "ready": all(item["ok"] for item in checks), "checks": checks}
