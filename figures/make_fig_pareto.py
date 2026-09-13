"""Accuracy versus latency (Fig. 3 of the paper), drawn from results/paper_rows.csv.

    python figures/make_fig_pareto.py        # writes figures/out/fig_pareto.pdf and .png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 8
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42})
# Table 1 method -> (label, label offset in points, alignment).
BASELINES = {"RME-GAN": ("RME-GAN", (-4, 6), "left"),
             "RadioUNet": ("RadioUNet", (4, -9), "left"),
             "UVM-Net": ("UVM-Net", (4, 2), "left"),
             "RadioDiff (our reproduction)": ("RadioDiff", (-7, 5), "right"),
             "RadioFlow (our reproduction)": ("RadioFlow", (5, -11), "left")}


def main(out_dir=os.path.join(ROOT, "figures", "out")):
    with open(os.path.join(ROOT, "results", "paper_rows.csv")) as f:
        rows = list(csv.DictReader(f))
    table1 = {r["method"]: r for r in rows if r["table"] == "1"}
    no_ray = next(r for r in rows if r["table"] == "2" and r["block"] == "ConvNeXt U-Net / DPM"
                  and r["method"] == "without ray channels")

    def point(r):
        return float(r["latency_ms"]), float(r["nmse_e3"]) * 1e-3

    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    for method, (label, off, ha) in BASELINES.items():
        ms, nmse = point(table1[method])
        ax.scatter(ms, nmse, s=24, c="#c0392b", zorder=3)
        ax.annotate(label, (ms, nmse), xytext=off, textcoords="offset points", fontsize=FS, ha=ha)
    ms, nmse = point(no_ray)
    ax.scatter(ms, nmse, s=55, marker="*", facecolors="none", edgecolors="#1f5fa8", lw=0.8, zorder=4)
    ax.annotate("w/o ray channels", (ms, nmse), xytext=(6, 0), textcoords="offset points",
                fontsize=FS, color="#1f5fa8", va="center")
    ms, nmse = point(table1["GeoRayMap"])
    ax.scatter(ms, nmse, s=85, marker="*", c="#1f5fa8", zorder=5)
    ax.annotate("GeoRayMap (ours)", (ms, nmse), xytext=(6, -1), textcoords="offset points",
                fontsize=FS, color="#1f5fa8", weight="bold", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 7000)
    ax.set_ylim(0.0042, 0.019)
    xt = [2, 5, 10, 30, 100, 300, 1000, 3000]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.set_xticklabels([str(v) for v in xt])
    ax.xaxis.set_minor_locator(NullLocator())
    yt = [0.005, 0.0075, 0.01, 0.015]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_yticklabels(["0.005", "0.0075", "0.010", "0.015"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=FS)
    ax.set_xlabel("latency per map (ms, batch size 1, H100)", fontsize=FS)
    ax.set_ylabel("NMSE", fontsize=FS)
    ax.grid(alpha=0.3, lw=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    fig.tight_layout(pad=0.3)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"fig_pareto.{ext}")
        fig.savefig(path, dpi=300)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
