# -*- coding: utf-8 -*-
"""RadioUNet (RadioWNet_c) evaluation and latency on the RadioMapSeer test split.

    export RADIOMAPSEER_DIR=/path/to/RadioMapSeer/     # trailing slash required
    python test.py                                     # RadioUNet (second U = the full W-Net)
    RU_RAY=1 python test.py                            # RadioUNet + ray channels
    RU_RAY=1 RU_CKPT=radiounet_ray_n16.pt python test.py   # a released checkpoint

Protocol of the paper: the full test split (maps 601-699 x 80 Tx = 7920 images, the same
seed-42 split as georaymap/data.py), batch size 1, targets thresholded at 0.2 and scaled to [0, 1].
Metrics come from lib/metrics.py (the same computation as georaymap/metrics.py). Latency is the
per-image model time measured with CUDA events. One CSV row is appended per run.
"""

from __future__ import print_function, division

import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from lib import loaders, modules
from lib.metrics import MetricAccumulator

DATASET_DIR = os.environ.get("RADIOMAPSEER_DIR")
if not DATASET_DIR:
    sys.exit("set RADIOMAPSEER_DIR=/path/to/RadioMapSeer/ (with the trailing slash)")
RESULTS_DIR = os.environ.get("RU_RESULTS", "./Results/RadioUNet_c_DPM_Thr2")
# RU_RAY=1 evaluates RadioUNet + GeoRayMap ray channels from "<RU_RESULTS>_ray" (see train.py).
RAY = os.environ.get("RU_RAY", "0").lower() in ("1", "true", "yes")
RAY_N = int(os.environ.get("RU_RAY_N", 16))
if RAY:
    RESULTS_DIR = RESULTS_DIR.rstrip("/") + "_ray"

# Which checkpoint / W-Net phase to evaluate. "secondU" is the full RadioUNet (W-Net)
# and is the number reported; "firstU" is the single-U ablation.
WNET_PHASE = os.environ.get("RU_WNET_PHASE", "secondU")
_default_ckpt = os.path.join(
    RESULTS_DIR,
    "Trained_Model_SecondU.pt" if WNET_PHASE == "secondU" else "Trained_Model_FirstU.pt",
)
CKPT = os.environ.get("RU_CKPT", _default_ckpt)

# Evaluation batch size 1: NMSE and PSNR are then per-image averages, and ms/sample is an
# unbatched latency. (Training stays at batch 15, the RadioUNet recipe.)
BATCH_SIZE = int(os.environ.get("RU_TEST_BATCH", 1))
WORKERS = int(os.environ.get("RU_WORKERS", 8))
STOP_BATCHES = int(os.environ.get("RU_TEST_STOP_BATCHES", -1))   # -1 = full test set
SIMULATION = os.environ.get("RU_SIMULATION", "DPM")
THRESH = float(os.environ.get("RU_THRESH", 0.2))

SPEED_CSV = os.environ.get("RU_SPEED_CSV", "speed_comparison_radiounet.csv")
SAVE_IMAGES = os.environ.get("RU_SAVE_IMAGES", "1").lower() in ("1", "true", "yes")
# Cap saved PNGs: writing 2x7920 images inflates wall-clock time. per_sample_ms_* (CUDA
# events, model only) is unaffected and is the speed number to quote.
SAVE_MAX = int(os.environ.get("RU_SAVE_MAX", 200))
OUT_DIR = os.environ.get("RU_OUT_DIR", os.path.join(RESULTS_DIR, "test_images"))

# RadioUNet_c's target is already in [0,1]; no rescale.
GAIN_SCALE = float(os.environ.get("RU_GAIN_SCALE", 1.0))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Radio_test = loaders.RadioUNet_c(phase="test", dir_dataset=DATASET_DIR,
                                 simulation=SIMULATION, thresh=THRESH)

print(f"device={device}  ckpt={CKPT}  wnet_phase={WNET_PHASE}")
print(f"simulation={SIMULATION}  thresh={THRESH}  gain_scale={GAIN_SCALE}  "
      f"test imgs={len(Radio_test)}  batch={BATCH_SIZE}")


def _sample_filename(dataset, global_idx):
    """The dataset's own '{map_index}_{tx_index}' name for a sequential (shuffle=False)
    index, matching RadioMapSeer's file naming. Mirrors RadioUNet_c.__getitem__.
    """
    idxr = global_idx // dataset.numTx
    idxc = global_idx - idxr * dataset.numTx
    map_ind = dataset.maps_inds[idxr + dataset.ind1] + 1
    return f"{map_ind}_{idxc}"


def _log_speed_comparison_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_speed_and_quality(model, ckpt_name):
    model.to(device)
    model.eval()

    # Both U outputs are scored: the forward pass computes both regardless of phase, so
    # this is free and gives the single-U ablation alongside the full W-Net number.
    acc_first = MetricAccumulator(device)
    acc_second = MetricAccumulator(device)
    per_sample_times = []
    n_samples = 0
    target_sq_sum = 0.0   # to report the NMSE normaliser explicitly
    in_channels = None    # measured from the real tensor, not asserted from config

    loader = DataLoader(Radio_test, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.time()

    with torch.no_grad():
        for bidx, (inputs, targets) in enumerate(loader):
            if STOP_BATCHES > 0 and bidx >= STOP_BATCHES:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if in_channels is None:
                # The loader must give geometry-only inputs [buildings, Tx]; fail loudly
                # rather than produce a row that is not comparable.
                in_channels = int(inputs.shape[1])
                if in_channels != 2:
                    raise RuntimeError(
                        f"expected 2 input channels [buildings, Tx] for the geometry-only "
                        f"RadioUNet_c regime, got {in_channels}. The run would not be "
                        f"comparable -- aborting rather than producing a misleading row."
                    )
                print(f"[verify] in_channels={in_channels} (geometry-only confirmed)  "
                      f"target min={targets.min().item():.4f} max={targets.max().item():.4f}")

            if device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                [out1, out2] = model(inputs)
                end_evt.record()
                torch.cuda.synchronize()
                batch_ms = start_evt.elapsed_time(end_evt)
            else:
                t0 = time.time()
                [out1, out2] = model(inputs)
                batch_ms = (time.time() - t0) * 1000

            bsz = targets.size(0)
            per_sample_times.append(batch_ms / bsz)
            n_samples += bsz

            tgt = (targets / GAIN_SCALE).clamp(0, 1)
            acc_first.update((out1 / GAIN_SCALE).clamp(0, 1), tgt)
            acc_second.update((out2 / GAIN_SCALE).clamp(0, 1), tgt)
            target_sq_sum += (tgt ** 2).mean().item() * bsz

            if SAVE_IMAGES and bidx * BATCH_SIZE < SAVE_MAX:
                pred_dir = os.path.join(OUT_DIR, "pred")
                gt_dir = os.path.join(OUT_DIR, "gt")
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)
                pred = out2 if WNET_PHASE == "secondU" else out1
                for j in range(bsz):
                    gidx = bidx * BATCH_SIZE + j
                    if gidx >= SAVE_MAX:
                        break
                    name = _sample_filename(Radio_test, gidx)
                    save_image((pred[j] / GAIN_SCALE).clamp(0, 1),
                               os.path.join(pred_dir, f"{name}.png"))
                    save_image((targets[j] / GAIN_SCALE).clamp(0, 1),
                               os.path.join(gt_dir, f"{name}.png"))

    wall_elapsed_s = time.time() - wall_start
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else 0.0)

    m_first = acc_first.result()
    m_second = acc_second.result()
    m_main = m_second if WNET_PHASE == "secondU" else m_first
    per_sample_times = np.array(per_sample_times)

    n_params = sum(p.numel() for p in model.parameters())

    # Provenance columns are derived from the live objects, never hardcoded.
    row = {
        "model": f"RadioUNet_c({WNET_PHASE})" + (f"+ray(N={RAY_N})" if RAY else ""),
        "ray_channels": RAY,
        "ckpt": ckpt_name,
        "loader": type(Radio_test).__name__,
        "in_channels": in_channels,                    # measured from the input tensor
        "geom_only": in_channels == 2,
        "thresh": Radio_test.thresh,                   # read back off the dataset object
        "target_space": f"thresholded@{Radio_test.thresh}" if Radio_test.thresh > 0 else "raw",
        "batch_size": BATCH_SIZE,
        "gain_scale": GAIN_SCALE,
        "n_samples": n_samples,
        "n_params": n_params,
        "per_sample_ms_mean": per_sample_times.mean() if len(per_sample_times) else float("nan"),
        "per_sample_ms_std": per_sample_times.std() if len(per_sample_times) else float("nan"),
        "images_per_sec": n_samples / wall_elapsed_s if wall_elapsed_s > 0 else float("nan"),
        "wall_time_s": wall_elapsed_s,
        "gpu_peak_mem_mb": peak_mem_mb,
        "MSE": m_main["MSE"],
        "NMSE": m_main["NMSE"],
        "MAE": m_main["MAE"],
        "RMSE": m_main["RMSE"],
        "PSNR": m_main["PSNR"],
        "SSIM": m_main["SSIM"],
        "BrightestPointDist_px": m_main["BrightestPointDist"],
        # ablation + provenance columns
        "firstU_RMSE": m_first["RMSE"],
        "firstU_PSNR": m_first["PSNR"],
        "firstU_SSIM": m_first["SSIM"],
        "secondU_RMSE": m_second["RMSE"],
        "secondU_PSNR": m_second["PSNR"],
        "secondU_SSIM": m_second["SSIM"],
        "nmse_normalizer_mean_target_sq": target_sq_sum / n_samples if n_samples else float("nan"),
    }

    print("\n--- results ---")
    for k, v in row.items():
        print(f"{k}: {v}")
    print("\nNOTE: quote per_sample_ms_mean for speed (CUDA events, model only). "
          "images_per_sec includes data loading and image saving.")

    # Check that the targets are in the thresholded space. Mean over the test images of
    # mean(target^2), measured on the full split: thresholded@0.2 ~0.054, raw gains ~0.113.
    norm = row["nmse_normalizer_mean_target_sq"]
    print(f"\n[verify] nmse_normalizer_mean_target_sq = {norm:.5f}")
    if 0.048 <= norm <= 0.062:
        print("[verify] OK -- matches the thresholded@0.2 target space (~0.054).")
    elif 0.100 <= norm <= 0.130:
        print("[verify] *** FAIL: this is the UNTHRESHOLDED space (~0.113). The threshold "
              "did not take effect; this row is NOT comparable. Check RU_THRESH. ***")
    else:
        print(f"[verify] *** WARNING: {norm:.5f} matches neither the thresholded (~0.054) "
              f"nor the unthresholded (~0.113) space (expected only for truncated runs); "
              f"check the dataset path and gain directory. ***")
    return row


if __name__ == "__main__":
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"checkpoint not found: {CKPT} (set RU_CKPT)")
    if RAY:
        from lib.ray_wnet import RayWNet
        model = RayWNet(phase=WNET_PHASE, n_samples=RAY_N)
    else:
        model = modules.RadioWNet(inputs=2, phase=WNET_PHASE)
    print(f"ray_channels={RAY}  ray_samples={RAY_N if RAY else '-'}  results_dir={RESULTS_DIR}")
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    row = evaluate_speed_and_quality(model, ckpt_name=os.path.basename(CKPT))
    _log_speed_comparison_row(SPEED_CSV, row)
    print(f"\nappended row to {SPEED_CSV}")
