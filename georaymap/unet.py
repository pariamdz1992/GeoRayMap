"""ConvNeXt U-Net backbone of GeoRayMap (also used alone, without ray channels, in the ablation).

ConvNeXt blocks (depthwise 7x7 convolution, LayerNorm, inverted MLP) in a U-Net with strided-
convolution downsampling, transposed-convolution upsampling and skip connections, and a sigmoid
output. With the default widths (64, 128, 256, 512) and depths (2, 2, 6, 2) it has 10.9 M
parameters.
"""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm for [B,C,H,W]."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # Statistics in fp32, as torch's own layer_norm does under autocast: a hand-written
        # mean/variance in fp16/bf16 loses precision and can overflow. No-op for fp32 inputs.
        x = x.float()
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pw2(self.act(self.pw1(h)))
        return x + self.gamma[:, None, None] * h


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm = LayerNorm2d(cin)
        self.conv = nn.Conv2d(cin, cout, 2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.fuse = nn.Conv2d(cout + cskip, cout, 1)

    def forward(self, x, skip):
        x = self.up(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class ConvNeXtUNet(nn.Module):
    def __init__(self, in_channels=2, dims=(64, 128, 256, 512),
                 depths=(2, 2, 6, 2), dec_depth=2):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (d, n) in enumerate(zip(dims, depths)):
            self.enc_stages.append(nn.Sequential(*[ConvNeXtBlock(d) for _ in range(n)]))
            if i < len(dims) - 1:
                self.downs.append(Down(d, dims[i + 1]))

        self.dec_stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(Up(dims[i], dims[i - 1], dims[i - 1]))
            self.dec_stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i - 1]) for _ in range(dec_depth)]))

        self.head = nn.Sequential(
            LayerNorm2d(dims[0]),
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[0], 1, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)
        for up, dec, skip in zip(self.ups, self.dec_stages, reversed(skips)):
            x = up(x, skip)
            x = dec(x)
        return torch.sigmoid(self.head(x))
