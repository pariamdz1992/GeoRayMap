"""Package a trained checkpoint for release: the EMA weights plus the config needed to load them.

    python export_checkpoint.py --ckpt checkpoints/georaymap_srm/best_ema.pt --out release/georaymap_srm_n16.pt
    # a raw state dict without config.json: give the architecture explicitly
    python export_checkpoint.py --ckpt run/best_ema.pt --n_samples 16 --out release/georaymap_srm_n16.pt

The output loads with georaymap.load_checkpoint() and evaluate.py. Its size and SHA-256 are printed.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import torch

from georaymap.model import arch_config, build_model, load_checkpoint


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="best_ema.pt (a raw state dict)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="default: config.json next to --ckpt")
    ap.add_argument("--n_samples", type=int, default=None, help="ray samples per pixel (0 = none)")
    ap.add_argument("--learned_atten", action="store_true")
    ap.add_argument("--task", choices=["srm", "drm"], default=None)
    ap.add_argument("--simulation", choices=["DPM", "IRT2"], default=None)
    ap.add_argument("--note", default="", help="free text stored with the weights, e.g. the paper row")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    elif args.n_samples is not None:
        cfg = {}
    else:
        sys.exit("ERROR: no config.json next to --ckpt; pass --n_samples (and --learned_atten if used)")
    for key in ("n_samples", "task", "simulation"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.learned_atten:
        cfg["learned_atten"] = True
    task = cfg.get("task", "srm")
    cfg = {"task": task, "simulation": "DPM", "learned_atten": False,
           "in_channels": 3 if task == "drm" else 2, "dims": [64, 128, 256, 512],
           "depths": [2, 2, 6, 2], "dec_depth": 2, **cfg}

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = build_model(**arch_config(cfg)).eval()
    model.load_state_dict(state)   # strict: fails loudly if the architecture does not match
    n_params = sum(p.numel() for p in model.parameters())
    out = {"format": "georaymap-v1", "config": cfg,
           "state_dict": {k: (v.float() if v.is_floating_point() else v).contiguous()
                          for k, v in model.state_dict().items()},
           "meta": {"source_run": os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))),
                    "note": args.note, "params_M": round(n_params / 1e6, 2),
                    "exported": datetime.date.today().isoformat()}}
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)

    # Round trip: the released file must give exactly the same output as the source weights.
    reloaded, _ = load_checkpoint(args.out)
    x = torch.zeros(1, cfg["in_channels"], 256, 256)
    x[0, 0, 96:160, 40:80] = 1.0
    x[0, 1, 128, 200] = 1.0
    with torch.no_grad():
        assert torch.equal(model(x), reloaded(x)), "round-trip check failed"
    print(json.dumps({"out": args.out, "MB": round(os.path.getsize(args.out) / 2 ** 20, 1),
                      "sha256": sha256(args.out), "config": cfg, "meta": out["meta"]}, indent=2))


if __name__ == "__main__":
    main()
