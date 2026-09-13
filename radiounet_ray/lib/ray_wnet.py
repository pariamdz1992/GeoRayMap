"""RadioUNet (RadioWNet) + GeoRayMap ray channels: the backbone-transfer experiment of the paper.

The unmodified RadioWNet receives [buildings, Tx] plus the four parameter-free ray channels (6
inputs, RadioWNet's own branch for more than 3 inputs). Everything else (two-stage training, MSE
loss, Adam 1e-4, batch 15, StepLR) is RadioUNet's recipe in train.py, so the only difference from
the RadioUNet row is the extra input channels.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from georaymap.rays import RayFeatures  # noqa: E402

from lib.modules import RadioWNet  # noqa: E402


class RayWNet(nn.Module):
    def __init__(self, phase="firstU", n_samples=16):
        super().__init__()
        self.feat = RayFeatures(n_samples)
        self.wnet = RadioWNet(inputs=2 + self.feat.n_out, phase=phase)

    def forward(self, x):
        with torch.no_grad():                 # the layer has no parameters
            f = self.feat(x)
        return self.wnet(torch.cat([x, f], 1))
