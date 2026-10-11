# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import os
import sys
import torch
import numpy as np
from contextlib import contextmanager


def merge_vad(vad1: list, vad2: list):
    intervals = vad1 + vad2
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


@contextmanager
def silent_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def download_model_from_modelscope(model_id, model_revision=None, cache_dir=None):
    from modelscope.hub.snapshot_download import snapshot_download

    if cache_dir is None:
        cache_dir = snapshot_download(
            model_id,
            revision=model_revision,
        )
    else:
        cfg_file = os.path.join(cache_dir, model_id, "configuration.json")
        if not os.path.exists(cfg_file):
            cache_dir = snapshot_download(
                model_id,
                revision=model_revision,
                cache_dir=cache_dir,
            )
        else:
            cache_dir = os.path.join(cache_dir, model_id)
    return cache_dir


def circle_pad(x: torch.Tensor, target_len, dim=0):
    xlen = x.shape[dim]
    if xlen >= target_len:
        return x
    n = int(np.ceil(target_len / xlen))
    xcat = torch.cat([x for _ in range(n)], dim=dim)
    return torch.narrow(xcat, dim, 0, target_len)
