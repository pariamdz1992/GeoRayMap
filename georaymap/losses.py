"""Training loss: L1 + edge_weight * gradient L1 + ssim_weight * (1 - SSIM)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from georaymap.metrics import build_ssim_window, ssim_torch


def gradient_loss(pred, target):
    """L1 distance between horizontal and vertical finite differences (edge loss)."""
    dpx = pred[..., :, 1:] - pred[..., :, :-1]
    dtx = target[..., :, 1:] - target[..., :, :-1]
    dpy = pred[..., 1:, :] - pred[..., :-1, :]
    dty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dpx, dtx) + F.l1_loss(dpy, dty)


class GeoRayMapLoss(nn.Module):
    """L1 + edge_weight * gradient_loss + ssim_weight * (1 - SSIM), SSIM as in georaymap.metrics."""

    def __init__(self, edge_weight=0.5, ssim_weight=0.2):
        super().__init__()
        self.edge_weight, self.ssim_weight = edge_weight, ssim_weight
        self.register_buffer("window", build_ssim_window(11, 1))

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target) + self.edge_weight * gradient_loss(pred, target)
        if self.ssim_weight > 0:
            with torch.autocast(device_type=pred.device.type, enabled=False):
                loss = loss + self.ssim_weight * (1.0 - ssim_torch(pred.float(), target.float(),
                                                                   self.window.float()))
        return loss
