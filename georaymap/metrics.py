"""Evaluation metrics used for every row of the paper's tables.

Per batch: MSE, NMSE = MSE / mean(target^2), MAE, RMSE, PSNR = 20 log10(1 / RMSE) (peak value 1)
and SSIM with an 11x11 Gaussian window (sigma 1.5, C1 = 0.01^2, C2 = 0.03^2); each is averaged
over batches weighted by batch size, as in RadioFlow's evaluation code. With batch size 1 (the
paper's protocol) every metric is a mean over maps. NMSE and PSNR change with the batch size
used for evaluation; SSIM does not.
"""
import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def build_ssim_window(window_size=11, channel=1):
    def gauss_1d(k, sigma=1.5):
        center = k // 2
        arr = torch.tensor([math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(k)])
        return arr / arr.sum()

    _1d = gauss_1d(window_size)
    _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_torch(img1, img2, window):
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


class MetricAccumulator:
    """Accumulates batch-size-weighted averages of the metrics above."""

    def __init__(self, device, ssim_window=11):
        self.window = build_ssim_window(ssim_window, 1).to(device)
        self.metrics = defaultdict(float)
        self.count = 0

    @torch.no_grad()
    def update(self, pred, targets):
        """pred, targets: [B,1,H,W] with values in [0, 1]."""
        bsz = targets.size(0)
        self.count += bsz
        mse = F.mse_loss(pred, targets)
        nmse = mse / (F.mse_loss(targets, torch.zeros_like(targets)) + 1e-12)
        mae = F.l1_loss(pred, targets)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))
        rmse = torch.sqrt(mse)
        ssim_val = ssim_torch(pred, targets, self.window)
        for name, val in zip(["MSE", "NMSE", "MAE", "PSNR", "RMSE", "SSIM"],
                             [mse, nmse, mae, psnr, rmse, ssim_val]):
            self.metrics[name] += val.item() * bsz

    def result(self):
        return {k: v / self.count for k, v in self.metrics.items()}
