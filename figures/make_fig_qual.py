"""Qualitative comparison (Fig. 2 of the paper): predictions and absolute-error maps.

    python figures/make_fig_qual.py --data_dir /path/to/RadioMapSeer/ \
        --ckpt georaymap_srm_n16.pt --nofeat_ckpt georaymap_srm_noray.pt \
        --baseline RadioUNet=/path/to/radiounet/pred --baseline RadioDiff=/path/to/radiodiff/pred
    python figures/make_fig_qual.py --render_only figures/out/fig_qual_panels.npz   # restyle only

Each --baseline is a folder of 8-bit <map>_<tx>.png test-set predictions. Code bases export
different value domains (RadioDiff's PNGs hold raw gains, RadioUNet's thresholded ones), so each
baseline is scored in both domains over all test maps it shares with the others and the better-
matching domain is used; the check is written to domain_check.txt. The panels are saved to an
.npz so the figure can be redrawn without data or GPU (--render_only).
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from georaymap.data import RadioMapSeer  # noqa: E402
from georaymap.model import load_checkpoint  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType, not Type 3
THRESH = 0.2
ERR_VMAX = 0.08   # colour scale of the error maps


def to_thresholded(v):
    """RadioMapSeer threshold transform, as in georaymap/data.py."""
    return np.clip((np.maximum(v, THRESH) - THRESH) / (1 - THRESH), 0, 1)


def psnr(p, t):
    return 10 * np.log10(1.0 / (float(((p - t) ** 2).mean()) + 1e-12))


def load_gray(path):
    from skimage import io
    a = np.asarray(io.imread(path)).astype(np.float32) / 255.0
    if a.ndim == 3:
        a = a[..., :3]
        if not (np.allclose(a[..., 0], a[..., 1], atol=2e-3) and np.allclose(a[..., 1], a[..., 2], atol=2e-3)):
            raise ValueError(f"{path} is a colour-mapped render and cannot be scored")
        a = a[..., 0]
    return a


def render(npz_path, out_dir):
    """Draw the figure from saved panels."""
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8})
    z = np.load(npz_path)
    bnames, picks = [str(b) for b in z["bnames"]], [str(p) for p in z["picks"]]
    cols = (["Ground truth"] + bnames + ["GeoRayMap (ours)"]
            + [f"{bnames[0]} error", "w/o ray error", "GeoRayMap error"])
    ncol = len(cols)
    fig = plt.figure(figsize=(7.0, 0.93 * len(picks) + 0.2))
    gs = fig.add_gridspec(len(picks), ncol + 1, width_ratios=[1] * ncol + [0.07],
                          wspace=0.05, hspace=0.05, left=0.004, right=0.912, top=0.915, bottom=0.035)
    for r in range(len(picks)):
        t, ours, nof = z[f"r{r}_gt"], z[f"r{r}_ours"], z[f"r{r}_nof"]
        base = [z[f"r{r}_b{i}"] for i in range(len(bnames))]
        panels = ([(t, False, None)] + [(b, False, psnr(b, t)) for b in base]
                  + [(ours, False, psnr(ours, t))]
                  + [(np.abs(base[0] - t), True, psnr(base[0], t)), (np.abs(nof - t), True, psnr(nof, t)),
                     (np.abs(ours - t), True, psnr(ours, t))])
        for c, (img, err, db) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap="inferno" if err else "viridis", vmin=0, vmax=ERR_VMAX if err else 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8, pad=2)
            if db is not None:
                ax.text(0.03, 0.04, f"{db:.1f} dB", transform=ax.transAxes, fontsize=8, color="white",
                        bbox=dict(facecolor="black", alpha=0.6, pad=0.8, lw=0))
            if err:
                err_im = im
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(err_im, cax=cax)
    cb.set_label("absolute error", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=8)
    for ext in ("pdf", "png"):
        fp = os.path.join(out_dir, f"fig_qual.{ext}")
        fig.savefig(fp, dpi=300)
        print("wrote", fp)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render_only", default=None, metavar="NPZ",
                    help="only redraw the figure from a saved fig_qual_panels.npz")
    ap.add_argument("--data_dir")
    ap.add_argument("--ckpt", help="GeoRayMap checkpoint (released .pt, or best_ema.pt with its config.json)")
    ap.add_argument("--nofeat_ckpt", help="the same network without ray channels")
    ap.add_argument("--baseline", action="append", default=[], help="NAME=DIR, repeatable")
    ap.add_argument("--pick", default="419_0,392_10", help="<map>_<tx> test samples, one row each")
    ap.add_argument("--out_dir", default="figures/out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.render_only:
        render(args.render_only, args.out_dir)
        return
    if not (args.data_dir and args.ckpt and args.nofeat_ckpt):
        sys.exit("ERROR: --data_dir, --ckpt and --nofeat_ckpt are required (unless --render_only)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(path, fallback):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path) as f:
                fallback = json.load(f)
        return load_checkpoint(path, fallback)[0].to(device)

    net = load(args.ckpt, {"n_samples": 16})
    nofeat = load(args.nofeat_ckpt, {"n_samples": 0})
    baselines = [b.split("=", 1) for b in args.baseline]
    bad = [f"  {n}={d}  ({'not a directory' if not os.path.isdir(d) else 'no .png files'})"
           for n, d in baselines
           if not os.path.isdir(d) or not any(f.endswith(".png") for f in os.listdir(d))]
    if not baselines or bad:
        sys.exit("ERROR: every --baseline must be NAME=DIR pointing at a folder of <map>_<tx>.png "
                 "predictions.\n" + ("\n".join(bad) if bad else "  (none given)"))

    ds = RadioMapSeer(args.data_dir, "test", "DPM")
    name_to_idx = {ds.sample_name(i): i for i in range(len(ds))}

    # domain check over every test sample present in all baseline folders
    common = [k for k in name_to_idx
              if all(os.path.exists(os.path.join(d, k + ".png")) for _, d in baselines)]
    print(f"{len(common)} test samples shared by all baseline folders")
    if not common:
        sys.exit("ERROR: no <map>_<tx>.png is present in every baseline folder and in the test split.")
    missing = [k for k in args.pick.split(",") if k not in common]
    if missing:
        sys.exit(f"ERROR: --pick {','.join(missing)} not available in every baseline folder. "
                 f"Available: {', '.join(common[:12])} ...")
    domain, report = {}, []
    for bname, bdir in baselines:
        asis, thr = [], []
        for k in common:
            t = ds[name_to_idx[k]][1].numpy()[0]
            v = load_gray(os.path.join(bdir, k + ".png"))
            asis.append(psnr(v, t))
            thr.append(psnr(to_thresholded(v), t))
        a, b = float(np.mean(asis)), float(np.mean(thr))
        domain[bname] = "thresholded" if b > a else "as-is"
        line = (f"{bname}: mean PSNR as-is {a:.2f} dB | after threshold transform {b:.2f} dB "
                f"-> using {domain[bname]}  (n={len(common)})")
        print(line)
        report.append(line)
    with open(os.path.join(args.out_dir, "domain_check.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")

    picks = args.pick.split(",")
    arrays = {"bnames": np.array([n for n, _ in baselines]), "picks": np.array(picks)}
    for r, k in enumerate(picks):
        x, t, _ = ds[name_to_idx[k]]
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            arrays[f"r{r}_ours"] = net(xb).cpu().numpy()[0, 0].astype(np.float32)
            arrays[f"r{r}_nof"] = nofeat(xb).cpu().numpy()[0, 0].astype(np.float32)
        arrays[f"r{r}_gt"] = t.numpy()[0].astype(np.float32)
        for i, (bname, bdir) in enumerate(baselines):
            v = load_gray(os.path.join(bdir, k + ".png"))
            arrays[f"r{r}_b{i}"] = (to_thresholded(v) if domain[bname] == "thresholded" else v).astype(np.float32)
    npz = os.path.join(args.out_dir, "fig_qual_panels.npz")
    np.savez_compressed(npz, **arrays)
    print("wrote", npz)
    render(npz, args.out_dir)


if __name__ == "__main__":
    main()
