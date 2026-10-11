"""Portable PCM-WAV segmentation of matching mixture and analysis tracks."""

from __future__ import annotations

import math
import wave
from pathlib import Path


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def segment_pair(mixture: Path, vocal: Path, start: float, end: float, output: Path) -> dict:
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
        raise ValueError("invalid_segment_interval")
    durations = [wav_duration(path) for path in (mixture, vocal)]
    if abs(durations[0] - durations[1]) > 0.02 or end > min(durations):
        raise ValueError("mixture_and_vocal_timelines_must_match")
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "mixture.wav", output / "vocal.wav"]
    if any(path.exists() for path in paths):
        raise ValueError("segment_outputs_already_exist")
    for source, destination in zip((mixture, vocal), paths):
        with wave.open(str(source), "rb") as reader:
            rate = reader.getframerate()
            begin, stop = round(start * rate), round(end * rate)
            if stop <= begin:
                raise ValueError("segment_shorter_than_one_frame")
            reader.setpos(begin)
            frames = reader.readframes(stop - begin)
            with wave.open(str(destination), "wb") as writer:
                writer.setparams(reader.getparams())
                writer.writeframes(frames)
    return {
        "audio": "mixture.wav",
        "analysis_audio": "vocal.wav",
        "source_start": start,
        "source_end": end,
    }
