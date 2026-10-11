"""Paper-level quality gates for cached SQUIM/DNSMOS measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilterConfig:
    min_duration: float = 1.0
    max_single_duration: float = 60.0
    max_multi_duration: float = 120.0
    quality_min: dict[str, float] = field(
        default_factory=lambda: {
            "pesq": 2.0,
            "stoi": 0.85,
            "si_sdr": 0.0,
            "mos": 2.5,
        }
    )

    def __post_init__(self):
        thresholds = [
            self.min_duration,
            self.max_single_duration,
            self.max_multi_duration,
            *self.quality_min.values(),
        ]
        if any(not math.isfinite(v) for v in thresholds):
            raise ValueError("filter_thresholds_must_be_finite")
        if not 0 < self.min_duration <= self.max_single_duration <= self.max_multi_duration:
            raise ValueError("invalid_duration_thresholds")
        if not self.quality_min or set(self.quality_min) - {"pesq", "stoi", "si_sdr", "mos"}:
            raise ValueError("invalid_quality_metric_names")


def filter_reasons(
    duration: float,
    scores: dict,
    speaker_count: int,
    *,
    targeted_short: bool = False,
    config: FilterConfig | None = None,
) -> list[str]:
    config = config or FilterConfig()
    reasons = []
    if not math.isfinite(duration) or duration <= 0:
        reasons.append("invalid_duration")
    else:
        if not targeted_short and duration < config.min_duration:
            reasons.append("duration_below_minimum")
        maximum = config.max_multi_duration if speaker_count > 1 else config.max_single_duration
        if duration > maximum:
            reasons.append("duration_above_maximum")
    scores = scores if isinstance(scores, dict) else {}
    for name, minimum in config.quality_min.items():
        value = scores.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            reasons.append(f"missing_or_invalid_quality:{name}")
        elif value < minimum:
            reasons.append(f"quality_below_minimum:{name}")
    return reasons
