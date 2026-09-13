# -*- coding: utf-8 -*-
"""RadioUNet (RadioWNet_c) training on RadioMapSeer, optionally with the GeoRayMap ray channels.

Script port of RadioUNet's official notebook `RadioWNet_c_DPM_Thr2.ipynb`
(github.com/RonLevie/RadioUNet), with the training loop, optimizer, scheduler and loss taken
from the notebook:

    batch_size 15, Adam(lr=1e-4), StepLR(step_size=30, gamma=0.1), MSE loss,
    50 epochs "firstU" then 50 epochs "secondU" (second U initialised from the first
    U's checkpoint), best-by-validation-loss weights kept.

RU_RAY=1 appends the four GeoRayMap ray channels to the inputs (lib/ray_wnet.py) and changes
nothing else. All settings are environment variables (see below). RadioUNet_c's inputs are
[buildings, Tx] only, the same information every model in the paper receives.

    export RADIOMAPSEER_DIR=/path/to/RadioMapSeer/     # trailing slash required
    python train.py                                    # RadioUNet
    RU_RAY=1 python train.py                           # RadioUNet + ray channels
"""

from __future__ import print_function, division

import copy
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lib import loaders, modules

# --- configuration ----------------------------------------------------------
# Trailing slash REQUIRED: the loader concatenates raw strings rather than using os.path.join.
DATASET_DIR = os.environ.get("RADIOMAPSEER_DIR")
if not DATASET_DIR:
    sys.exit("set RADIOMAPSEER_DIR=/path/to/RadioMapSeer/ (with the trailing slash)")
RESULTS_DIR = os.environ.get("RU_RESULTS", "./Results/RadioUNet_c_DPM_Thr2")
# RU_RAY=1 -> RadioUNet + GeoRayMap ray channels (lib/ray_wnet.py). Results go to a separate
# "<RU_RESULTS>_ray" directory so the baseline checkpoints can never be overwritten.
RAY = os.environ.get("RU_RAY", "0").lower() in ("1", "true", "yes")
RAY_N = int(os.environ.get("RU_RAY_N", 16))
if RAY:
    RESULTS_DIR = RESULTS_DIR.rstrip("/") + "_ray"

EPOCHS = int(os.environ.get("RU_EPOCHS", 50))
BATCH_SIZE = int(os.environ.get("RU_BATCH", 15))
LR = float(os.environ.get("RU_LR", 1e-4))
STEP_SIZE = int(os.environ.get("RU_STEP_SIZE", 30))
GAMMA = float(os.environ.get("RU_GAMMA", 0.1))
# The notebook used num_workers=1. Each __getitem__ reads 3 PNGs from disk, so on a
# 12-CPU GPU node that starves the GPU; raise it.
WORKERS = int(os.environ.get("RU_WORKERS", 8))

SIMULATION = os.environ.get("RU_SIMULATION", "DPM")
THRESH = float(os.environ.get("RU_THRESH", 0.2))      # 0.2 = noise floor, RadioUNet default

# Which stages to run, in order. "first,second" reproduces the full W-Net recipe.
STAGES = [s.strip() for s in os.environ.get("RU_STAGES", "first,second").split(",") if s.strip()]

FIRST_CKPT = os.path.join(RESULTS_DIR, "Trained_Model_FirstU.pt")
SECOND_CKPT = os.path.join(RESULTS_DIR, "Trained_Model_SecondU.pt")

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: the original notebook calls torch.set_default_tensor_type('torch.cuda.FloatTensor').
# Deliberately NOT reproduced: it makes the shuffling DataLoader's sampler expect a CUDA
# generator and crashes. Every tensor is moved to `device` explicitly instead, which is
# equivalent.
torch.set_default_dtype(torch.float32)

print(f"device={device}  dataset={DATASET_DIR}")
print(f"simulation={SIMULATION}  thresh={THRESH}  epochs={EPOCHS}  batch={BATCH_SIZE}  "
      f"lr={LR}  step={STEP_SIZE}  gamma={GAMMA}  workers={WORKERS}  stages={STAGES}")
print(f"ray_channels={RAY}  ray_samples={RAY_N if RAY else '-'}  results_dir={RESULTS_DIR}")

_loader_kwargs = dict(dir_dataset=DATASET_DIR, simulation=SIMULATION, thresh=THRESH)

# Standard split, seed-42 shuffled map order (the same test maps as every row of the paper):
#   train 0-500 (501 maps), val 501-600 (100 maps), test 601-699 (99 maps), x80 Tx each.
Radio_train = loaders.RadioUNet_c(phase="train", **_loader_kwargs)
Radio_val = loaders.RadioUNet_c(phase="val", **_loader_kwargs)

dataloaders = {
    "train": DataLoader(Radio_train, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=WORKERS, pin_memory=True),
    "val": DataLoader(Radio_val, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=WORKERS, pin_memory=True),
}
print(f"train imgs={len(Radio_train)}  val imgs={len(Radio_val)}")


# --- loss / logging (from the notebook) --------------------------------------
def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs1 = ["{}: {:4f}".format(k, metrics[k] / epoch_samples) for k in metrics.keys()]
    print("{}: {}".format(phase, ", ".join(outputs1)))


def _sanity_check_ranges(inputs, targets):
    """Print the value ranges once, as a guard against silent normalisation bugs.

    RadioUNet_c leaves buildings/Tx as uint8 and lets transforms.ToTensor() rescale them
    to [0,1], while the gain is divided by 255 in numpy and then threshold-transformed.
    Both should therefore be in [0,1], with a target maximum near 1.
    """
    print("[sanity] inputs  shape={} dtype={} min={:.4f} max={:.4f} mean={:.4f}".format(
        tuple(inputs.shape), inputs.dtype, inputs.min().item(), inputs.max().item(),
        inputs.mean().item()))
    print("[sanity] targets shape={} dtype={} min={:.4f} max={:.4f} mean={:.4f} "
          "mean_sq={:.6f}".format(
              tuple(targets.shape), targets.dtype, targets.min().item(),
              targets.max().item(), targets.mean().item(), (targets ** 2).mean().item()))
    print("[sanity] expected: inputs and targets both in [0,1]; target max ~1.0. "
          "mean_sq is the NMSE normaliser.")


def train_model(model, optimizer, scheduler, num_epochs, wnet_phase, tag):
    """The notebook's train_model, restricted to targetType='dense' (the only mode
    RadioUNet_c uses) and with per-epoch loss history returned for CSV logging.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    history = {"train": [], "val": []}
    printed_sanity = False

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        since = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                for param_group in optimizer.param_groups:
                    print("learning rate", param_group["lr"])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if not printed_sanity:
                    _sanity_check_ranges(inputs, targets)
                    printed_sanity = True

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    [outputs1, outputs2] = model(inputs)
                    pred = outputs1 if wnet_phase == "firstU" else outputs2
                    loss = calc_loss_dense(pred, targets, metrics)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples
            history[phase].append(epoch_loss)

            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # The notebook calls scheduler.step() at the *start* of the train phase, which
        # modern PyTorch warns about (scheduler before optimizer). Stepping at epoch end
        # is the standard order; with step_size=30 it shifts the single decay by one
        # epoch, which is immaterial.
        scheduler.step()

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("Best val loss: {:4f}".format(best_loss))
    model.load_state_dict(best_model_wts)

    np.savetxt(os.path.join(RESULTS_DIR, f"MSE_train_{tag}.csv"),
               np.array(history["train"]), delimiter=",")
    np.savetxt(os.path.join(RESULTS_DIR, f"MSE_val_{tag}.csv"),
               np.array(history["val"]), delimiter=",")
    return model, best_loss


def run_stage(stage):
    """stage: 'first' -> train U1 with U2 frozen; 'second' -> train U2 with U1 frozen."""
    if stage == "first":
        wnet_phase, out_ckpt, init_ckpt = "firstU", FIRST_CKPT, None
    elif stage == "second":
        wnet_phase, out_ckpt, init_ckpt = "secondU", SECOND_CKPT, FIRST_CKPT
    else:
        raise ValueError(f"unknown stage {stage!r}, expected 'first' or 'second'")

    print("\n" + "=" * 60)
    print(f"STAGE {stage} (RadioWNet phase={wnet_phase}) -> {out_ckpt}")
    print("=" * 60)

    # inputs=2 -> [buildings, Tx]; with RU_RAY=1, 2 + 4 ray channels.
    if RAY:
        from lib.ray_wnet import RayWNet
        model = RayWNet(phase=wnet_phase, n_samples=RAY_N)
    else:
        model = modules.RadioWNet(inputs=2, phase=wnet_phase)
    if init_ckpt is not None:
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(
                f"stage '{stage}' initialises from {init_ckpt}, which does not exist. "
                f"Run the 'first' stage before 'second' (RU_STAGES=first,second)."
            )
        print(f"initialising from {init_ckpt}")
        model.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    model, best_loss = train_model(model, optimizer, scheduler, EPOCHS, wnet_phase, tag=stage)
    torch.save(model.state_dict(), out_ckpt)
    print(f"saved {out_ckpt}  (best val MSE {best_loss:.6f})")


if __name__ == "__main__":
    t0 = time.time()
    for stage in STAGES:
        run_stage(stage)
    print(f"\nAll stages done in {(time.time() - t0) / 3600:.2f} h")
