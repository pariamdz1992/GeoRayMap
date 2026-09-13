"""Train GeoRayMap (or one of the paper's ablations) on RadioMapSeer.

    python train.py --data_dir /path/to/RadioMapSeer/                     # GeoRayMap (N = 16)
    python train.py --data_dir /path/to/RadioMapSeer/ --n_samples 0       # without ray channels
    python train.py --data_dir /path/to/RadioMapSeer/ --simulation IRT2   # ray-traced targets
    python train.py --data_dir /path/to/RadioMapSeer/ --smoke             # quick check, CPU is fine

The defaults are the paper's recipe: ConvNeXt U-Net (widths 64-512, depths 2,2,6,2), loss
L1 + 0.5 gradient + 0.2 (1 - SSIM), AdamW (lr 1e-3, weight decay 1e-4), linear warm-up over 10%
of the epochs then cosine decay, 60 epochs, batch 32, gradient clipping at 1, EMA 0.999; the EMA
weights with the lowest validation loss are kept. One run takes about 10 h on an 80 GB H100.
The kept checkpoint is then evaluated on the test split at batch size 1, the paper's protocol.
scripts/reproduce_paper.sh lists the command behind every row of the paper.
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from georaymap.data import RadioMapSeer
from georaymap.engine import (ModelEMA, benchmark_latency, build_lr_lambda, count_params,
                              evaluate, get_device, report)
from georaymap.losses import GeoRayMapLoss
from georaymap.model import build_model


def run_tag(args):
    """Every non-default setting appears in the tag, so no run can overwrite another's."""
    return (f"georaymap_{args.task}"
            + ("_learned" if args.learned_atten else "")
            + (f"_n{args.n_samples}" if args.n_samples != 16 else "")
            + (f"_ssim{args.ssim_weight}" if args.ssim_weight != 0.2 else "")
            + (f"_edge{args.edge_weight}" if args.edge_weight != 0.5 else "")
            + (f"_{args.simulation.lower()}" if args.simulation != "DPM" else "")
            + (f"_{args.precision}" if args.precision != "bf16" else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="RadioMapSeer root (the folder with png/ and gain/)")
    ap.add_argument("--task", choices=["srm", "drm"], default="srm",
                    help="srm = static maps (the paper); drm = maps with cars (supported, not evaluated)")
    ap.add_argument("--simulation", choices=["DPM", "IRT2"], default="DPM", help="target simulator")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32, help="32 fits an 80 GB H100")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--edge_weight", type=float, default=0.5)
    ap.add_argument("--ssim_weight", type=float, default=0.2)
    ap.add_argument("--n_samples", type=int, default=16, help="ray samples per pixel; 0 = no ray channels")
    ap.add_argument("--learned_atten", action="store_true", help="add the learned-attenuation channel")
    ap.add_argument("--chunk", type=int, default=8,
                    help="ray samples per grid_sample call (memory only; 0 = one call)")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="mixed-precision dtype on CUDA. The paper's DPM runs used fp16; two fp16 runs "
                         "diverged, hence bf16 (0.06 dB from fp16 on the N=64 model)")
    ap.add_argument("--dims", default="64,128,256,512")
    ap.add_argument("--depths", default="2,2,6,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--save_dir", default=None, help="default: checkpoints/<run tag>")
    ap.add_argument("--name", default=None, help="row name in the results file (default: run tag)")
    ap.add_argument("--results", default="results/local_runs.jsonl")
    ap.add_argument("--smoke", action="store_true", help="tiny model and data subset, 2 epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    if device.type != "cuda" and not args.smoke:
        sys.exit("ABORT: no CUDA device visible; a full run needs a GPU (use --smoke for a CPU check).")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("WARNING: this GPU does not support bfloat16; using --precision fp16")
        args.precision = "fp16"

    in_ch = 3 if args.task == "drm" else 2
    dims = tuple(int(x) for x in args.dims.split(","))
    depths = tuple(int(x) for x in args.depths.split(","))
    dec_depth = 2
    if args.smoke:
        args.epochs, args.batch_size, args.num_workers = 2, 4, 0
        dims, depths, dec_depth = (16, 32, 64, 128), (1, 1, 2, 1), 1

    tag = run_tag(args) + ("_smoke" if args.smoke else "")
    name = args.name or tag
    save_dir = args.save_dir or os.path.join("checkpoints", tag)
    if not args.smoke and os.path.exists(os.path.join(save_dir, "best_ema.pt")):
        sys.exit(f"ABORT: {save_dir}/best_ema.pt already exists; refusing to overwrite a finished run "
                 f"(pass --save_dir and --name to train another one)")

    cars = args.task == "drm"
    train_ds = RadioMapSeer(args.data_dir, "train", args.simulation, cars)
    val_ds = RadioMapSeer(args.data_dir, "val", args.simulation, cars)
    test_ds = RadioMapSeer(args.data_dir, "test", args.simulation, cars)
    if args.smoke:
        train_ds, val_ds, test_ds = Subset(train_ds, range(16)), Subset(val_ds, range(8)), Subset(test_ds, range(8))

    os.makedirs(save_dir, exist_ok=True)
    config = {"task": args.task, "simulation": args.simulation, "n_samples": args.n_samples,
              "learned_atten": args.learned_atten, "in_channels": in_ch, "dims": list(dims),
              "depths": list(depths), "dec_depth": dec_depth, "edge_weight": args.edge_weight,
              "ssim_weight": args.ssim_weight, "precision": args.precision, "epochs": args.epochs,
              "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
              "seed": args.seed}
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"device={device}  run={tag}  save_dir={save_dir}")
    print(json.dumps(config))

    model = build_model(args.n_samples, args.learned_atten, in_ch, dims, depths, dec_depth,
                        args.chunk).to(device)
    print(f"parameters: {count_params(model) / 1e6:.2f} M")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    criterion = GeoRayMapLoss(args.edge_weight, args.ssim_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, build_lr_lambda(args.epochs))
    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    # Loss scaling is only needed for fp16's narrow range.
    scaler = torch.amp.GradScaler(enabled=use_amp and args.precision == "fp16")
    ema = ModelEMA(model)

    best_val, t0 = float("inf"), time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss, nb, n_bad = 0.0, 0, 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(inputs)
                loss = criterion(pred, targets)
            if not torch.isfinite(loss):
                # Skip the batch (no update, no EMA step) so one bad batch cannot poison the weights.
                n_bad += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            ep_loss += loss.item()
            nb += 1
        sched.step()
        if n_bad:
            print(f"  WARNING epoch {epoch + 1}: skipped {n_bad} non-finite batches", flush=True)
        if n_bad > 0.05 * (nb + n_bad) or not all(torch.isfinite(p).all() for p in model.parameters()):
            sys.exit(f"ABORT: training diverged at epoch {epoch + 1} ({n_bad} non-finite batches); "
                     f"best_ema.pt holds the last good checkpoint. Rerun with --precision bf16 or fp32.")

        ema.ema_model.eval()
        with torch.no_grad():
            v_loss, vb = 0.0, 0
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                v_loss += criterion(ema.ema_model(inputs), targets).item()
                vb += 1
        v_loss /= max(1, vb)
        print(f"Epoch {epoch + 1}/{args.epochs}  train={ep_loss / max(1, nb):.5f}  val={v_loss:.5f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  elapsed={time.time() - t0:.0f}s", flush=True)
        if v_loss < best_val:
            best_val = v_loss
            torch.save(ema.ema_model.state_dict(), os.path.join(save_dir, "best_ema.pt"))
    torch.save(ema.ema_model.state_dict(), os.path.join(save_dir, "final_ema.pt"))

    # Test split, best-validation EMA weights, batch size 1 (the paper's protocol).
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_ema.pt"), map_location=device,
                                     weights_only=True))
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate(model, test_loader, device)
    latency = benchmark_latency(model, next(iter(test_loader))[0], device)
    report(name, metrics, latency, count_params(model), out_path=args.results,
           extra={"batch_size": 1, "simulation": args.simulation, "device": torch.cuda.get_device_name()
                  if device.type == "cuda" else device.type})


if __name__ == "__main__":
    main()
