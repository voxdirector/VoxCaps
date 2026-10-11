"""SQUIM scoring adapted from the existing objective/subjective calculation."""

from __future__ import annotations

from pathlib import Path


class SQUIMScorer:
    def __init__(self, reference_audio: Path, device: str = "cpu"):
        import torch
        import torchaudio
        from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

        self.torch, self.torchaudio, self.device = torch, torchaudio, device
        self.reference = self.load(reference_audio)
        self.objective = SQUIM_OBJECTIVE.get_model().to(device).eval()
        self.subjective = SQUIM_SUBJECTIVE.get_model().to(device).eval()

    def load(self, path):
        waveform, rate = self.torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if rate != 16000:
            waveform = self.torchaudio.functional.resample(waveform, rate, 16000)
        if waveform.shape[-1] == 0:
            raise ValueError("empty_quality_input")
        return waveform.to(self.device)

    def __call__(self, audio: Path) -> dict:
        waveform = self.load(audio)
        with self.torch.inference_mode():
            stoi, pesq, si_sdr = self.objective(waveform)
            mos = self.subjective(waveform, self.reference)
        return {
            "quality_score": {
                "stoi": float(stoi.item()),
                "pesq": float(pesq.item()),
                "si_sdr": float(si_sdr.item()),
                "mos": float(mos.item()),
            },
            "method": "torchaudio_squim",
            "sample_rate": 16000,
        }


def score_audio(audio: Path, reference_audio: Path, device: str = "cpu") -> dict:
    return SQUIMScorer(reference_audio, device)(audio)
