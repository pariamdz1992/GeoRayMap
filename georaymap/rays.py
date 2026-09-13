"""The GeoRayMap layer: parameter-free ray marching from the transmitter.

For every pixel q of the map, the layer samples the binary building mask b at N points on the
straight segment from the transmitter p to q and reduces the samples to four channels:

    soft visibility    v(q) = 1 - max_k B(b, r_q(s_k))
    obstructed length  o(q) = |q - p| * (1/N) * sum_k B(b, r_q(s_k))      (scaled by 1/64)
    log-distance       log10(1 + |q - p|) / log10(1 + diagonal)
    distance           |q - p| / diagonal

with r_q(s) = p + s (q - p), s_k = k / N for k = 1..N, B bilinear interpolation, and distances in
pixels. v is fractional where the segment grazes a building edge, and since s_N = 1 includes q
itself, v(q) = 0 inside buildings. The samples of all pixels are taken with one F.grid_sample call
per chunk of samples, so
the layer has no parameters and no loop over pixels. It is differentiable, which the optional
learned-attenuation channel (an ablation in the paper) relies on.

The spacing between samples on a ray is |q - p| / N, so on long rays walls thinner than that can
be missed. N = 16 and N = 64 give the same accuracy on RadioMapSeer.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def tx_position(tx_map):
    """Transmitter pixel (row, col) of a one-hot transmitter channel [B,1,H,W] -> two [B] tensors."""
    B, _, H, W = tx_map.shape
    idx = tx_map.flatten(1).argmax(1)
    return (idx // W).to(tx_map.dtype), (idx % W).to(tx_map.dtype)


def ray_integrals(field, ty, tx, n_samples=16, chunk=0):
    """Sample `field` [B,C,H,W] at `n_samples` points on the segment Tx -> q for every pixel q.

    Returns (mean, max) over the samples, each [B,C,H,W]; mean * |q - Tx| is the line integral of
    the field along the ray. Uses one grid_sample call per `chunk` samples (0 = all samples in
    one call); memory grows as B * C * chunk * H * W.
    """
    B, C, H, W = field.shape
    dev, dt = field.device, field.dtype
    gy, gx = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt), indexing="ij")
    dy = gy[None] - ty[:, None, None]  # [B,H,W]
    dx = gx[None] - tx[:, None, None]
    s_all = torch.arange(1, n_samples + 1, device=dev, dtype=dt) / n_samples  # (0, 1]
    chunk = chunk or n_samples
    total, mx = None, None
    for s in s_all.split(chunk):
        n = s.numel()
        py = ty[:, None, None, None] + s[None, :, None, None] * dy[:, None]  # [B,n,H,W]
        px = tx[:, None, None, None] + s[None, :, None, None] * dx[:, None]
        grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], dim=-1)
        samp = F.grid_sample(field, grid.reshape(B, n * H, W, 2), mode="bilinear",
                             padding_mode="border", align_corners=True)
        samp = samp.reshape(B, C, n, H, W)
        total = samp.sum(2) if total is None else total + samp.sum(2)
        m = samp.amax(2)
        mx = m if mx is None else torch.maximum(mx, m)
    return total / n_samples, mx


class RayFeatures(nn.Module):
    """Ray-marching channels computed on the fly from the [buildings, Tx] inputs.

    Args:
        n_samples: samples per ray (N). The paper uses 16.
        learned_atten: add a fifth channel, the line integral of a learned attenuation field
            a = f(b) given by a small CNN. Ablation only; it did not help in the paper.
        hidden: width of that CNN.
        chunk: samples per grid_sample call. Trades memory for speed; the output is unchanged.
    """

    def __init__(self, n_samples=16, learned_atten=False, hidden=16, chunk=8):
        super().__init__()
        self.n_samples = n_samples
        self.chunk = chunk
        self.learned = learned_atten
        if learned_atten:
            self.atten = nn.Sequential(
                nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, 1, 1), nn.Softplus())

    @property
    def n_out(self):
        return 4 + (1 if self.learned else 0)

    @staticmethod
    def _binarize(ch):
        """Occupancy mask at half of each sample's peak value, so any input scaling works."""
        B = ch.size(0)
        peak = ch.flatten(1).amax(1).clamp_min(1e-6).view(B, 1, 1, 1)
        return (ch > 0.5 * peak).to(ch.dtype)

    def forward(self, inputs):
        """inputs: [B,C,H,W] with channel 0 = buildings and channel 1 = transmitter (one-hot). An
        optional channel 2 (cars, for dynamic maps) blocks rays like the buildings do."""
        bld, txm = inputs[:, :1], inputs[:, 1:2]
        B, _, H, W = bld.shape
        b = self._binarize(bld)                            # binary building mask
        if inputs.size(1) >= 3:
            b = torch.maximum(b, self._binarize(inputs[:, 2:3]))
        ty, tx = tx_position(txm)
        gy, gx = torch.meshgrid(torch.arange(H, device=inputs.device, dtype=inputs.dtype),
                                torch.arange(W, device=inputs.device, dtype=inputs.dtype),
                                indexing="ij")
        d = torch.sqrt((gy[None] - ty[:, None, None]) ** 2 +
                       (gx[None] - tx[:, None, None]) ** 2)[:, None]   # [B,1,H,W], pixels
        diag = math.sqrt(H * H + W * W)
        fields = [b]
        if self.learned:
            fields.append(self.atten(b))
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            mean, mx = ray_integrals(torch.cat(fields, 1).float(), ty.float(), tx.float(),
                                     self.n_samples, self.chunk)
        d = d.float()
        feats = [1.0 - mx[:, :1],                               # soft visibility
                 mean[:, :1] * d / 64.0,                        # obstructed length
                 torch.log10(1.0 + d) / math.log10(1.0 + diag),  # log-distance
                 d / diag]                                      # distance
        if self.learned:
            feats.append(mean[:, 1:2] * d / 64.0)               # learned line integral
        return torch.cat(feats, 1).to(inputs.dtype)
