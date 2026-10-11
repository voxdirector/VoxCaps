from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import librosa
import numpy as np
import soundfile as sf
import torch


def normalize_device(device="cpu"):
    requested = torch.device(device or "cpu")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested_cuda_device_unavailable")
    return requested


def resolve_uvr_weight_dir(arch, model_name, model_root=None, *, allow_legacy_fallback=False):
    if model_root is None:
        raise ValueError("explicit_uvr_model_root_is_required")
    path = Path(model_root) / arch / "weights" / model_name
    if not path.is_dir():
        raise FileNotFoundError("uvr_weight_directory_missing")
    return path


AudioInput = Union[str, Path, np.ndarray, torch.Tensor]


def load_audio_input(
    audio: AudioInput, sample_rate: Optional[int] = None
) -> Tuple[np.ndarray, int, Optional[Path]]:
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio input not found: {audio_path}")
        wav, sr = librosa.load(str(audio_path), sr=None, mono=False)
        if wav.ndim == 1:
            wav = wav[None, :]
        return wav.astype(np.float32), int(sr), audio_path

    if sample_rate is None:
        raise ValueError("内存音频输入必须显式提供 sample_rate。")

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().cpu().float().numpy()
    elif isinstance(audio, np.ndarray):
        wav = audio.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    if wav.ndim == 1:
        wav = wav[None, :]
    elif wav.ndim == 2:
        if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
            wav = wav.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape={wav.shape}.")

    return wav.astype(np.float32), int(sample_rate), None


def resample_channels(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    return np.stack(
        [librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr) for channel in wav],
        axis=0,
    ).astype(np.float32)


def to_uvr_stereo_mix(wav: np.ndarray) -> np.ndarray:
    if wav.shape[0] == 1:
        return np.repeat(wav, 2, axis=0).astype(np.float32)
    if wav.shape[0] >= 2:
        return wav[:2].astype(np.float32)
    raise ValueError(f"Unexpected waveform shape: {wav.shape}")


def ensure_channel_first_tensor(audio: np.ndarray) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio[None, :]
    return torch.from_numpy(np.ascontiguousarray(audio)).float()


def match_num_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1 and channels == 2:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] == 2 and channels == 1:
        return audio[:1]
    raise ValueError(f"Cannot match channel count from {audio.shape[0]} to {channels}.")


def trim_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] == length:
        return audio
    if audio.shape[-1] > length:
        return audio[..., :length]
    return np.pad(audio, ((0, 0), (0, length - audio.shape[-1])))


def derive_output_base(
    output_path: Union[str, Path], input_path: Optional[Path]
) -> Tuple[Path, str]:
    output = Path(output_path)
    if output.suffix:
        return output.with_suffix(""), output.suffix.lstrip(".")
    base_name = input_path.stem if input_path is not None else "output"
    return output / base_name, "wav"


def write_named_outputs(
    outputs: Dict[str, torch.Tensor],
    *,
    sample_rate: int,
    output_path: Optional[Union[str, Path]],
    input_path: Optional[Path],
) -> None:
    if output_path is None:
        return
    base_path, suffix = derive_output_base(output_path, input_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for name, tensor in outputs.items():
        target = base_path.parent / f"{base_path.name}_{name}.{suffix}"
        sf.write(target, tensor.detach().cpu().numpy().T, sample_rate)
