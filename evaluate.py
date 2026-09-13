"""Evaluate a GeoRayMap checkpoint on the RadioMapSeer test split (the paper's protocol: batch size 1).

    python evaluate.py --ckpt georaymap_srm_n16.pt --data_dir /path/to/RadioMapSeer/
    python evaluate.py --ckpt checkpoints/georaymap_srm/best_ema.pt --data_dir ...   # reads its config.json

--save_preds DIR writes one 8-bit <map>_<tx>.png per test map (prediction x 255).
--limit N scores only the first N test maps: a quick check, not a paper number.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from georaymap.data import RadioMapSeer
from georaymap.engine import benchmark_latency, count_params, evaluate, get_device, report
from georaymap.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="released .pt file, or best_ema.pt from train.py")
    ap.add_argument("--data_dir", required=True, help="RadioMapSeer root (the folder with png/ and gain/)")
    ap.add_argument("--config", default=None,
                    help="run config for a raw state dict (default: config.json next to --ckpt)")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="raw state dict without any config: ray samples it was trained with (0 = none)")
    ap.add_argument("--learned_atten", action="store_true",
                    help="raw state dict without any config: it has the learned-attenuation channel")
    ap.add_argument("--simulation", choices=["DPM", "IRT2"], default=None,
                    help="target simulator (default: the one the checkpoint was trained on)")
    ap.add_argument("--batch_size", type=int, default=1, help="1 = the paper's protocol")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N test maps")
    ap.add_argument("--save_preds", default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--name", default=None, help="row name in the results file")
    ap.add_argument("--results", default="results/local_runs.jsonl")
    args = ap.parse_args()
    device = get_device()

    cfg = None
    cfg_path = args.config or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    elif args.n_samples is not None:
        cfg = {"n_samples": args.n_samples, "learned_atten": args.learned_atten}
    try:
        model, cfg = load_checkpoint(args.ckpt, cfg)
    except ValueError as e:
        sys.exit(f"ERROR: {e}; or give the architecture with --n_samples (and --learned_atten)")
    model.to(device)
    simulation = args.simulation or cfg.get("simulation", "DPM")
    print(f"device={device}  n_samples={cfg.get('n_samples')}  learned_atten={cfg.get('learned_atten', False)}"
          f"  simulation={simulation}  parameters={count_params(model) / 1e6:.2f} M")

    ds = RadioMapSeer(args.data_dir, "test", simulation, cars=(cfg.get("task", "srm") == "drm"))
    if args.limit:
        ds = Subset(ds, range(args.limit))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.save_preds:
        from skimage import io
        os.makedirs(args.save_preds, exist_ok=True)
        with torch.no_grad():
            for inputs, _, names in loader:
                pred = model(inputs.to(device)).clamp(0, 1).cpu().numpy()
                for p, n in zip(pred, names):
                    io.imsave(os.path.join(args.save_preds, n + ".png"),
                              (p[0] * 255).round().astype(np.uint8), check_contrast=False)
        print(f"wrote {len(ds)} predictions to {args.save_preds}")

    metrics = evaluate(model, loader, device)
    latency = benchmark_latency(model, next(iter(loader))[0], device)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    if stem in ("best_ema", "final_ema"):
        stem = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))) + ("_final" if stem == "final_ema" else "")
    name = args.name or (stem + (f"_bs{args.batch_size}" if args.batch_size != 1 else "")
                         + (f"_first{args.limit}" if args.limit else ""))
    report(name, metrics, latency, count_params(model), out_path=args.results,
           extra={"batch_size": args.batch_size, "simulation": simulation, "n_maps": len(ds),
                  "device": torch.cuda.get_device_name() if device.type == "cuda" else device.type})


if __name__ == "__main__":
    main()
