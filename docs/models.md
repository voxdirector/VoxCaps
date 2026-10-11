# Models and setup

Run these commands from the package directory. Download complete model folders to the paths used in `configs/`.

| Component | Model or service | Source | Used by |
| --- | --- | --- | --- |
| Teacher synthesis | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | [Weights](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512), [runtime installation](https://github.com/QwenAudio/CosyVoice#install) | [Synthesis](../src/voxcaps/coverage_design/synthesis.py) |
| Vocal separation | `MDX23C-8KFFT-InstVoc_HQ.ckpt` | [UVR model release](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) | [UVR MDXC](../src/voxcaps/speech_preprocessing/uvr/src/mdxc.py) |
| Speaker embeddings | `iic/speech_campplus_sv_zh_en_16k-common_advanced`, revision `v1.0.0` | [ModelScope](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced) | [Diarization](../src/voxcaps/speech_preprocessing/diarization.py) |
| Voice activity detection | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`, revision `v2.0.4` | [ModelScope](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) | [Diarization](../src/voxcaps/speech_preprocessing/diarization.py) |
| Primary ASR | `FunAudioLLM/Fun-ASR-Nano-2512` | [Weights](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512), [FunASR runtime](https://github.com/modelscope/FunASR) | [Speech models](../src/voxcaps/speech_preprocessing/models.py) |
| Complementary ASR | `FunAudioLLM/SenseVoiceSmall` | [Weights and model terms](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | [Speech models](../src/voxcaps/speech_preprocessing/models.py) |
| Word alignment | `Qwen/Qwen3-ForcedAligner-0.6B` | [Weights](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), [runtime](https://github.com/QwenLM/Qwen3-ASR) | [Speech models](../src/voxcaps/speech_preprocessing/models.py) |
| Description annotation | `qwen3.5-omni-plus` through Model Studio | [Official API instructions](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) | [Provider](../src/voxcaps/scope_annotation/provider.py) |
| Quality estimates | TorchAudio `SQUIM_OBJECTIVE`, `SQUIM_SUBJECTIVE` | [Models and usage](https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html) | [SQUIM scorer](../src/voxcaps/data_refinement/squim.py) |
| Complementary quality | DNSMOS `sig_bak_ovr.onnx`, `model_v8.onnx` | [Microsoft DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | [DNSMOS scorer](../src/voxcaps/data_refinement/dnsmos.py) |

## Runtime dependencies

```bash
python -m pip install -e '.[models]'
```

Use matching PyTorch and TorchAudio 2.6 to 2.8 builds for preprocessing. TorchAudio is capped below 2.9 to retain [SQUIM](https://github.com/pytorch/audio/blob/v2.8.0/src/torchaudio/pipelines/__init__.py) and the [PCM audio IO interface](https://docs.pytorch.org/audio/2.8/generated/torchaudio.save.html). Fun-ASR-Nano needs FunASR 1.4 or later.

For stage 1, follow the [CosyVoice 3 runtime installation](https://github.com/QwenAudio/CosyVoice#install) in a separate environment, then install VoxCaps with its synthesis dependencies:

```bash
python -m pip install -e '.[synthesis]'
export PYTHONPATH="/path/to/CosyVoice:/path/to/CosyVoice/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
```

Replace `/path/to/CosyVoice` with your runtime checkout. Stages exchange saved manifests across environments.

## Hugging Face weights

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local-dir models/Fun-CosyVoice3-0.5B --max-workers 1
hf download FunAudioLLM/Fun-ASR-Nano-2512 \
  --local-dir models/Fun-ASR-Nano-2512 --max-workers 1
hf download FunAudioLLM/SenseVoiceSmall \
  --local-dir models/SenseVoiceSmall --max-workers 1
hf download Qwen/Qwen3-ForcedAligner-0.6B \
  --local-dir models/Qwen3-ForcedAligner-0.6B --max-workers 1
```

Keep the tokenizer and configuration subfolders. Record the revision for reproducible runs; `hf download` accepts `--revision`.

## UVR checkpoint

```bash
mkdir -p models/uvr/mdxc/weights/MDX23C-8KFFT-InstVoc_HQ
curl --fail --location \
  https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt \
  --output models/uvr/mdxc/weights/MDX23C-8KFFT-InstVoc_HQ/MDX23C-8KFFT-InstVoc_HQ.ckpt
```

Set `uvr_model_root` to `models/uvr`. The package includes the matching YAML and checkpoint-hash registry.

## CAM++ and VAD

```bash
modelscope download iic/speech_campplus_sv_zh_en_16k-common_advanced \
  --revision v1.0.0 --max-workers 1 \
  --local-dir models/speaker_models/iic/speech_campplus_sv_zh_en_16k-common_advanced
modelscope download iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --revision v2.0.4 --max-workers 1 \
  --local-dir models/speaker_models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
```

Set `speaker_model_cache` to `models/speaker_models`. CAM++ uses `campplus_cn_en_common.pt`; VAD uses `model.pt`, `am.mvn` and its configuration files. Both specified revisions contain these files. Empty caches are downloaded automatically.

## DNSMOS and SQUIM

```bash
mkdir -p models/DNSMOS
curl --fail --location \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx \
  --output models/DNSMOS/sig_bak_ovr.onnx
curl --fail --location \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx \
  --output models/DNSMOS/model_v8.onnx
```

SQUIM downloads its [objective weights](https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth) and [subjective weights](https://download.pytorch.org/torchaudio/models/squim_subjective_bvcc_daps.pth) on first use. Set `squim_reference_audio` to an unrelated clean-speech WAV, such as `references/nonmatching_clean_speech.wav`, which you supply separately.

## Qwen3.5-Omni API

Qwen3.5-Omni runs through the hosted API. Set `DASHSCOPE_API_KEY` using a Model Studio key for the endpoint's region.

In stage-3 and stage-4 configurations, replace the sample endpoint with your workspace's endpoint. The example uses Singapore. Keep `model: qwen3.5-omni-plus`, `stream: true` and `modalities: ["text"]`. Stage 3 uses `input_audio` WAV data; stage 4 sends text for refinement.

Requests run serially, with at least one second between starts. A failed request stops the batch.

## Check before a run

```bash
voxcaps check-models --stage 1 --teacher-dir models/Fun-CosyVoice3-0.5B
voxcaps check-models --stage 2 --config configs/02_speech_preprocessing/config.example.json
voxcaps check-models --stage 3 --config configs/03_scope_annotation/config.example.json
voxcaps check-models --stage 4 --config configs/04_data_refinement/config.example.json
```

The check reports missing dependencies, model files, references and API settings without inference or API requests. It never prints credentials. See [upstream terms and attribution](../THIRD_PARTY_NOTICES.md).
