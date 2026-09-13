"""Evaluation metrics for test.py: the same computation as georaymap/metrics.py (batch-weighted
MSE/NMSE/MAE/RMSE, PSNR with peak value 1, SSIM with an 11x11 Gaussian window and C1 = 0.01^2,
C2 = 0.03^2), plus the pixel distance between the brightest pixels of prediction and target.

Inputs must be in [0, 1]. RadioUNet_c's targets already are (divided by 255, then threshold-
transformed). NMSE = MSE / mean(target^2) depends on the target normalisation and threshold, so
it is only comparable between models evaluated on the same targets.
"""
import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def build_ssim_window(window_size: int = 11, channel: int = 1) -> torch.Tensor:
    def gauss_1d(k, sigma=1.5):
        center = k // 2
        arr = torch.tensor([math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(k)])
        return arr / arr.sum()

    _1d = gauss_1d(window_size)
    _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    c = img1.size(1)
    pad = window.size(-1) // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=c)
    mu2 = F.conv2d(img2, window, padding=pad, groups=c)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def brightest_point_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean pixel distance between the brightest pixel of pred vs target, per-sample."""
    b, _, h, w = target.shape
    pred_flat = pred.view(b, -1)
    tgt_flat = target.view(b, -1)
    pred_idx = pred_flat.argmax(dim=1)
    tgt_idx = tgt_flat.argmax(dim=1)
    py, px = (pred_idx // w).float(), (pred_idx % w).float()
    ty, tx = (tgt_idx // w).float(), (tgt_idx % w).float()
    dist = torch.sqrt((py - ty) ** 2 + (px - tx) ** 2)
    return dist.mean()


class MetricAccumulator:
    """Accumulates per-batch metrics exactly like the RadioDiff comparison table."""

    def __init__(self, device, ssim_window: int = 11):
        self.window = build_ssim_window(ssim_window, 1).to(device)
        self.metrics = defaultdict(float)
        self.count = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, targets: torch.Tensor):
        """pred, targets: [B,1,H,W], values in [0, 1]."""
        bsz = targets.size(0)
        self.count += bsz
        mse = F.mse_loss(pred, targets)
        nmse = mse / (F.mse_loss(targets, torch.zeros_like(targets)) + 1e-12)
        mae = F.l1_loss(pred, targets)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))
        rmse = torch.sqrt(mse)
        ssim_val = ssim_torch(pred, targets, self.window)
        bpd = brightest_point_distance(pred, targets)
        for name, val in zip(
            ["MSE", "NMSE", "MAE", "PSNR", "RMSE", "SSIM", "BrightestPointDist"],
            [mse, nmse, mae, psnr, rmse, ssim_val, bpd],
        ):
            self.metrics[name] += val.item() * bsz

    def result(self):
        return {k: v / self.count for k, v in self.metrics.items()}
