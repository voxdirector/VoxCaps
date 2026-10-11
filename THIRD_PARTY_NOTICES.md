# Third-party code and models

This package includes the code listed below, with upstream copyright notices retained. Import paths and input/output adapters were changed for the four-stage pipeline. Downloaded model weights retain their upstream terms and are not included here.

| Component | Included code | Upstream and terms |
| --- | --- | --- |
| 3D-Speaker and CAM++ | `speech_preprocessing/diarization.py`, `speech_preprocessing/speakerlab/` | [3D-Speaker](https://github.com/modelscope/3D-Speaker), [Apache 2.0 licence](https://github.com/modelscope/3D-Speaker/blob/main/LICENSE) |
| UVR MDXC | `speech_preprocessing/uvr/` | [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), MIT as stated in its [README](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/README.md). UVR credits ZFTurbo for the MDX23C weights and its core developers Anjok07 and aufr33. |
| DNSMOS | `data_refinement/dnsmos.py` | [Microsoft DNS-Challenge scorer](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py), [upstream licence](https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE) |

The following runtimes/services are called by adapters; their implementation code and weights are supplied separately:

| Model/runtime | Upstream |
| --- | --- |
| CosyVoice 3 | [Runtime](https://github.com/QwenAudio/CosyVoice), [weights](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| Fun-ASR-Nano | [FunASR](https://github.com/modelscope/FunASR), [weights](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| SenseVoice | [Runtime](https://github.com/QwenAudio/SenseVoice), [model terms](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| Qwen3-ForcedAligner | [Runtime](https://github.com/QwenLM/Qwen3-ASR), [weights](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) |
| Qwen3.5-Omni | [Model Studio API](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) |
| SQUIM | [TorchAudio models and usage](https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html) |

See [Models and setup](docs/models.md) for download commands, exact model IDs and configuration paths. Code licences and model-weight terms may differ; SenseVoice has a separate model licence.
