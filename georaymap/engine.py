"""Training and evaluation utilities: device, learning-rate schedule, EMA, evaluation, latency."""
import copy
import json
import os
import time

import numpy as np
import torch

from georaymap.metrics import MetricAccumulator


def get_device():
    # CUDA or CPU. Apple's MPS backend is skipped on purpose: its grid_sample has no "border"
    # padding mode, which the ray layer uses.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_lr_lambda(total_epochs, warmup_ratio=0.1):
    """Per-epoch factor: linear warm-up over the first 10% of epochs, then cosine decay."""
    warmup = max(1, int(total_epochs * warmup_ratio))

    def fn(epoch):
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return fn


class ModelEMA:
    """Exponential moving average of the model weights (decay 0.999, updated every step)."""

    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def evaluate(model, loader, device, progress=True):
    """Average metrics of `model` over `loader` (see georaymap.metrics)."""
    acc = MetricAccumulator(device)
    iterator = loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="Evaluating")
        except ImportError:
            pass
    for batch in iterator:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        acc.update(model(inputs), targets)
    return acc.result()


@torch.no_grad()
def benchmark_latency(model, sample_inputs, device, warmup=10, iters=50):
    """Wall-clock seconds per forward pass at batch size 1 (synchronised on CUDA)."""
    x = sample_inputs[:1].to(device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def report(name, metrics, latency_s=None, params=None, out_path=None, extra=None):
    """Print one results row (also in the paper's units) and append it to a JSONL file."""
    row = {"model": name, **{k: round(v, 6) for k, v in metrics.items()}}
    if latency_s is not None:
        row["latency_s"] = round(latency_s, 5)
    if params is not None:
        row["params_M"] = round(params / 1e6, 2)
    if extra:
        row.update(extra)
    print(json.dumps(row, indent=2))
    line = (f"{name}: NMSE {1e3 * metrics['NMSE']:.2f}e-3 | RMSE {1e2 * metrics['RMSE']:.3f}e-2 | "
            f"PSNR {metrics['PSNR']:.2f} dB | SSIM {metrics['SSIM']:.4f}")
    if latency_s is not None:
        line += f" | {1e3 * latency_s:.2f} ms per map"
    print(line)
    if out_path:
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    return row
