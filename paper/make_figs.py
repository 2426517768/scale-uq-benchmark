"""Data figures for the SCALE benchmark (PLOS ONE version).
Usage: python make_figs.py <analysis_dir> <out_figure_dir>
All values are computed from metrics_wide.csv (target macro-average over the
twelve target series, then mean across five seeds). Nothing is typed by hand.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# ---- palette for this paper (one colour = one configuration family) ----------
FAM = {
    "Statistical references": ("#3E9E8F", "#1F8475", "o"),
    "Neural predictive distributions": ("#4C84C4", "#2D66AB", "s"),
    "Calibration layers on MC Dropout": ("#E39B3A", "#CF7E12", "^"),
    "Augmentation and adversarial training": ("#9479B8", "#74589E", "D"),
}
REF_GRAY = "#8A8A8A"
FAMILY = {
    "ZeroGaussian": "Statistical references", "EWMAGaussian": "Statistical references",
    "GARCH-Norm": "Statistical references", "RollingEmpirical": "Statistical references",
    "GaussianTCN": "Neural predictive distributions", "MCDropout": "Neural predictive distributions",
    "DeepEnsembles": "Neural predictive distributions", "Evidential": "Neural predictive distributions",
    "Isotonic": "Calibration layers on MC Dropout", "SplitConformal": "Calibration layers on MC Dropout",
    "OnlineQuantile": "Calibration layers on MC Dropout",
    "RandomNoise": "Augmentation and adversarial training", "MixUp": "Augmentation and adversarial training",
    "FeaturePGD": "Augmentation and adversarial training", "AdvReg": "Augmentation and adversarial training",
    "SCALE-R": "Augmentation and adversarial training",
}
LABEL = {"ZeroGaussian": "Zero Gaussian", "EWMAGaussian": "EWMA Gaussian", "GARCH-Norm": "GARCH-Norm",
         "RollingEmpirical": "Rolling empirical", "GaussianTCN": "Gaussian TCN", "MCDropout": "MC Dropout",
         "DeepEnsembles": "Deep Ensembles", "Evidential": "Evidential", "Isotonic": "Isotonic",
         "SplitConformal": "Split conformal", "OnlineQuantile": "Online quantile", "RandomNoise": "Random noise",
         "MixUp": "MixUp", "FeaturePGD": "Feature PGD", "AdvReg": "Adv-Reg", "SCALE-R": "SCALE-R"}
SPLITS = [("historical", "2020–2022"), ("recent", "2023–2025")]

w = pd.read_csv(A / "metrics_wide.csv")
w = w[(w.market != "SP500ETF") & w.method.isin(FAMILY)]
qcols = [f"coverage_q_{q}" for q in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")]
seed = w.groupby(["split", "method", "seed"])[["ECE", "NWIS"] + qcols].mean()
mean = seed.groupby(["split", "method"]).mean()
sd = seed.groupby(["split", "method"]).std(ddof=1)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
    (OUT / f"{name}.png").unlink()
    plt.close(fig)


def panel_tag(ax, tag):
    ax.text(-0.13, 1.02, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


# ---- Fig 2: ECE versus NWIS (claims C1 and C3) ------------------------------
# label offsets in points, tuned per split so that no label covers a marker
OFFS = {
    "historical": {"EWMAGaussian": (8, -8), "GARCH-Norm": (10, -2), "RollingEmpirical": (10, 4),
                   "SCALE-R": (-14, 14), "Evidential": (12, 14), "Isotonic": (-12, -16), "SplitConformal": (8, 14),
                   "MCDropout": (62, -16), "OnlineQuantile": (14, -14), "DeepEnsembles": (-14, -12)},
    "recent": {"EWMAGaussian": (8, -8), "OnlineQuantile": (-6, 16), "RollingEmpirical": (-8, -14),
               "GARCH-Norm": (-4, -18), "DeepEnsembles": (14, -14), "MCDropout": (-34, 22),
               "Isotonic": (14, 18), "SplitConformal": (16, -16), "SCALE-R": (-16, 12), "Evidential": (14, 10)},
}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
for ax, (sp, title), tag in zip(axes, SPLITS, "AB"):
    for m, fam in FAMILY.items():
        c, cs, mk = FAM[fam]
        x, y = mean.loc[(sp, m), "ECE"], mean.loc[(sp, m), "NWIS"]
        hero = m == "EWMAGaussian"
        ax.errorbar(x, y, xerr=sd.loc[(sp, m), "ECE"], yerr=sd.loc[(sp, m), "NWIS"], fmt="none",
                    ecolor=c, alpha=0.35, elinewidth=0.6, capsize=0, zorder=2)
        ax.scatter(x, y, s=46 if hero else 26, marker=mk, color=cs if hero else c,
                   edgecolor="white", linewidth=0.6, zorder=3)
        if m in OFFS[sp]:
            dx, dy = OFFS[sp][m]
            ax.annotate(LABEL[m], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                        ha="left" if dx > 0 else "right", va="center",
                        fontweight="bold" if hero else "normal", color="#222222",
                        arrowprops=None if hero else dict(arrowstyle="-", lw=0.4, color="#9A9A9A", shrinkA=1, shrinkB=3))
    ax.set_xlabel("Quantile calibration error (ECE, lower is better)")
    ax.set_ylabel("Normalized weighted interval score (NWIS, lower is better)")
    ax.text(0.98, 0.98, f"Test period {title}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.margins(x=0.12 if sp == "historical" else 0.22, y=0.10)
    panel_tag(ax, tag)
handles = [Line2D([], [], marker=v[2], ls="", color=v[0], markeredgecolor="white", markersize=6, label=k)
           for k, v in FAM.items()]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995),
           handletextpad=0.3, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.5)
save(fig, "fig2")

# ---- Fig 3: coverage error by quantile level (claim C2) ---------------------
lev = np.arange(0.1, 0.91, 0.1)
LINES = [("MCDropout", "-", "o", 1.2), ("Isotonic", "--", "^", 1.0), ("SplitConformal", ":", "v", 1.2),
         ("OnlineQuantile", "-", "^", 1.8), ("EWMAGaussian", "-", "o", 1.2)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, (sp, title), tag in zip(axes, SPLITS, "AB"):
    ax.axhline(0, color=REF_GRAY, lw=0.8, ls="--", zorder=1)
    for m, ls, mk, lw in LINES:
        c, cs, _ = FAM[FAMILY[m]]
        col = cs if m == "OnlineQuantile" else c
        dev = mean.loc[(sp, m), qcols].values - lev
        ax.plot(lev, dev, ls=ls, marker=mk, ms=3.5, lw=lw, color=col, label=LABEL[m], zorder=3)
    ax.set_xticks(lev)
    ax.set_xticklabels([f"{v:.1f}" for v in lev])
    ax.set_xlabel("Nominal quantile level")
    if tag == "A":
        ax.set_ylabel("Empirical minus nominal coverage")
    ax.text(0.98, 0.03, f"Test period {title}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    panel_tag(ax, tag)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995), handlelength=2.4)
fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
save(fig, "fig3")

# numbers used in the figures, for the record
rec = mean[["ECE", "NWIS"]].round(4)
rec.to_csv(OUT / "fig2_fig3_source_values.csv")
print("done")
