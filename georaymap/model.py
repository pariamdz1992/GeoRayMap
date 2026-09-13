"""GeoRayMap = ray-marching channels + ConvNeXt U-Net, and checkpoint loading."""
import torch
import torch.nn as nn

from georaymap.rays import RayFeatures
from georaymap.unet import ConvNeXtUNet

DEFAULT_DIMS = (64, 128, 256, 512)
DEFAULT_DEPTHS = (2, 2, 6, 2)
# Keys of a run config that define the architecture (see train.py and export_checkpoint.py).
ARCH_KEYS = ("n_samples", "learned_atten", "in_channels", "dims", "depths", "dec_depth")


class GeoRayMap(nn.Module):
    """Single-pass radio map estimator.

    Input: [B, 2, H, W] = [buildings, transmitter] as returned by georaymap.data.RadioMapSeer
    (3 channels when cars are included). Output: [B, 1, H, W] radio map in [0, 1].
    """

    def __init__(self, in_channels=2, n_samples=16, learned_atten=False,
                 dims=DEFAULT_DIMS, depths=DEFAULT_DEPTHS, dec_depth=2, chunk=8):
        super().__init__()
        self.feat = RayFeatures(n_samples, learned_atten, chunk=chunk)
        self.unet = ConvNeXtUNet(in_channels + self.feat.n_out, dims, depths, dec_depth)

    def forward(self, x):
        # Bring the raw inputs to [0, 1] whether they arrive scaled to 0..1 or 0..255.
        peak = x.flatten(1).amax(1).view(-1, 1, 1, 1)
        xn = torch.where(peak > 1.5, x / 255.0, x)
        f = self.feat(x)
        return self.unet(torch.cat([xn, f], 1))


def build_model(n_samples=16, learned_atten=False, in_channels=2, dims=DEFAULT_DIMS,
                depths=DEFAULT_DEPTHS, dec_depth=2, chunk=8):
    """GeoRayMap, or the same U-Net without ray channels when n_samples <= 0 (the ablation)."""
    dims, depths = tuple(dims), tuple(depths)
    if n_samples <= 0:
        return ConvNeXtUNet(in_channels, dims, depths, dec_depth)
    return GeoRayMap(in_channels, n_samples, learned_atten, dims, depths, dec_depth, chunk)


def arch_config(cfg):
    """The architecture part of a run config."""
    return {k: cfg[k] for k in ARCH_KEYS if k in cfg}


def load_checkpoint(path, config=None, map_location="cpu"):
    """Load a checkpoint and return (model in eval mode, config dict).

    Accepts a released checkpoint (a dict with "state_dict" and "config", written by
    export_checkpoint.py), or a raw state dict saved by train.py together with `config` (for
    example the config.json that train.py writes next to it) describing the architecture.
    """
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and "config" in obj:
        state, cfg = obj["state_dict"], dict(obj["config"])
    elif config is not None:
        state, cfg = obj, dict(config)
    else:
        raise ValueError(f"{path} is a raw state dict; pass the run's config (its config.json) "
                         f"so the architecture is known")
    model = build_model(**arch_config(cfg))
    model.load_state_dict(state)
    return model.eval(), cfg
