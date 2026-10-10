"""Descriptive summaries and synchronized date-block bootstrap over a fixed market panel."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import json_write, source_id
from .distributions import LEVELS, QCOLS
from .metrics import risk_curve, score, wis_rows

FOCUS = ["nRMSE", "skill_vs_zero", "ECE", "PICP90", "MPIW90_pct", "NMPIW90", "NWIS", "UA_width", "UA_epistemic"]


def reporting_source(cfg):
    return source_id(cfg) if cfg.get("markets") else "SP500"


def load_prediction(root, split, seed, method):
    return pd.read_csv(Path(root) / "predictions" / split / f"seed_{seed}" / (method + ".csv.gz"))


def paired_bootstrap(root, cfg, split_id, a, b):
    left = pd.concat([load_prediction(root, split_id, s, a).assign(seed=s) for s in cfg["seeds"]])
    right = pd.concat([load_prediction(root, split_id, s, b).assign(seed=s) for s in cfg["seeds"]])
    source = reporting_source(cfg)
    left = left[left.market != source].copy()
    right = right[right.market != source].copy()
    keys = ["seed", "market", "target_date"]
    left, right = left.set_index(keys).sort_index(), right.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Methods do not share the same prediction panel.")
    if not np.allclose(left.target_pct, right.target_pct, rtol=0, atol=1e-7):
        raise ValueError("Paired methods have different realized targets.")
    markets = sorted(left.index.get_level_values("market").unique())
    dates = sorted(left.index.get_level_values("target_date").unique())
    grid = pd.MultiIndex.from_product([cfg["seeds"], dates, markets], names=["seed", "target_date", "market"])
    shape = (len(cfg["seeds"]), len(dates), len(markets))
    li = left.reorder_levels(["seed", "target_date", "market"]).reindex(grid)
    ri = right.reorder_levels(["seed", "target_date", "market"]).reindex(grid)
    y = li.target_pct.to_numpy().reshape(shape)
    valid = np.isfinite(y).astype(float)
    y = np.nan_to_num(y)
    arrays = []
    for frame in (li, ri):
        arrays.append({"mse": np.nan_to_num((frame.target_pct - frame["mean"]).to_numpy() ** 2).reshape(shape),
                       "wis": np.nan_to_num(wis_rows(frame)).reshape(shape),
                       "coverage": np.nan_to_num(((frame.target_pct >= frame["q_0.05"]) &
                                                  (frame.target_pct <= frame["q_0.95"])).to_numpy(float)).reshape(shape),
                       "quantile": np.stack([np.nan_to_num((frame.target_pct <= frame[f"q_{p:g}"]).to_numpy(float)).reshape(shape)
                                               for p in np.arange(1, 10) / 10], axis=-1)})
    def calculate(weights, seed_ids):
        denominator = np.einsum("d,sdm->sm", weights, valid)[seed_ids]
        ymean = np.einsum("d,sdm->sm", weights, y)[seed_ids] / np.maximum(denominator, 1)
        y2 = np.einsum("d,sdm->sm", weights, y*y)[seed_ids] / np.maximum(denominator, 1)
        sd = np.sqrt(np.maximum((y2 - ymean*ymean) * denominator / np.maximum(denominator-1, 1), 1e-10))
        result = []
        for ar in arrays:
            mse = np.einsum("d,sdm->sm", weights, ar["mse"])[seed_ids] / denominator
            wis = np.einsum("d,sdm->sm", weights, ar["wis"])[seed_ids] / denominator
            picp = np.einsum("d,sdm->sm", weights, ar["coverage"])[seed_ids] / denominator
            coverage = np.einsum("d,sdmq->smq", weights, ar["quantile"])[seed_ids] / denominator[..., None]
            result.append(np.array([(np.sqrt(mse)/sd).mean(), (wis/sd).mean(), picp.mean(),
                                     np.abs(coverage - np.arange(1, 10)/10).mean()]))
        return result[0] - result[1]
    point = calculate(np.ones(len(dates)), np.arange(len(cfg["seeds"])))
    rng = np.random.default_rng(98731)
    estimates = []
    length = cfg["analysis"]["block_days"]
    for _ in range(cfg["analysis"]["bootstrap_reps"]):
        starts = rng.integers(0, max(1, len(dates) - length + 1), int(np.ceil(len(dates) / length)))
        idx = np.concatenate([np.arange(s, min(s+length, len(dates))) for s in starts])[:len(dates)]
        counts = np.bincount(idx, minlength=len(dates))
        seed_ids = rng.integers(0, len(cfg["seeds"]), len(cfg["seeds"]))
        estimates.append(calculate(counts, seed_ids))
    interval = np.quantile(estimates, [.025, .975], axis=0)
    return [{"split": split_id, "A": a, "B": b, "metric": metric,
             "difference_A_minus_B": value, "ci_low": interval[0, j], "ci_high": interval[1, j],
             "bootstrap_reps": cfg["analysis"]["bootstrap_reps"], "block_calendar_observations": length,
             "scope": f"fixed {len(markets)}-market panel; shared date blocks and paired seed resampling; no population-of-markets inference"}
            for j, (metric, value) in enumerate(zip(["nRMSE", "NWIS", "PICP90", "ECE"], point))]


def summarize(root, cfg, complete_required=True):
    root = Path(root)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    paths = sorted((root / "predictions").glob("*/*/*.csv.gz"))
    if not paths:
        raise ValueError("No predictions; run experiments first.")
    rows, risks, cases, periods = [], [], [], []
    for path in paths:
        split, seed = path.parent.parent.name, int(path.parent.name.split("_")[-1])
        method = path.name.removesuffix(".csv.gz")
        df = pd.read_csv(path)
        for market, group in df.groupby("market"):
            base = {"split": split, "seed": seed, "method": method, "market": market,
                    "asset_class": group.asset_class.iloc[0]}
            rows.append({**base, **score(group)})
            risks.extend([{**base, **x} for x in risk_curve(group)])
            year = pd.to_datetime(group.target_date).dt.year
            for value in sorted(year.unique()):
                subset = group[year == value]
                periods.append({**base, "period": str(value), **score(subset)})
            for window in cfg["analysis"]["stress_windows"]:
                subset = group[(group.target_date >= window["start"]) & (group.target_date <= window["end"])]
                if len(subset) >= 10:
                    periods.append({**base, "period": window["id"], **score(subset)})
        if "OilETF" in df.market.values:
            march = df[(df.market == "OilETF") & (df.target_date >= "2020-03-01") & (df.target_date <= "2020-03-31")].copy()
            if len(march):
                march["split"], march["seed"], march["method"] = split, seed, method
                cases.append(march)
    wide = pd.DataFrame(rows)
    wide.to_csv(out / "metrics_wide.csv", index=False)
    wide.melt(id_vars=["split", "seed", "method", "market", "asset_class"], var_name="metric", value_name="value").to_csv(out / "metrics_long.csv", index=False)
    pd.DataFrame(risks).to_csv(out / "risk_coverage.csv", index=False)
    pd.DataFrame(periods).to_csv(out / "year_and_stress_metrics.csv", index=False)
    if cases:
        pd.concat(cases).to_csv(out / "oil_etf_march_2020.csv", index=False)
    source = reporting_source(cfg)
    ood = wide[wide.market != source]
    macro_seed = ood.groupby(["split", "seed", "method"])[FOCUS].mean().reset_index()
    macro_seed.to_csv(out / "macro_per_seed.csv", index=False)
    macro = macro_seed.groupby(["split", "method"])[FOCUS].agg(["mean", "std"])
    macro.columns = ["_".join(c) for c in macro.columns]
    macro.to_csv(out / "macro_mean_sd.csv")
    class_seed = ood.groupby(["split", "seed", "method", "asset_class"])[FOCUS].mean().reset_index()
    class_seed.to_csv(out / "asset_class_per_seed.csv", index=False)
    class_seed.groupby(["split", "method", "asset_class"])[FOCUS].agg(["mean", "std"]).to_csv(out / "asset_class_mean_sd.csv")
    if "GermanyGovtETF" in ood.market.values:
        ood[ood.market != "GermanyGovtETF"].groupby(["split", "seed", "method"])[FOCUS].mean().to_csv(out / "exclude_german_bond_proxy.csv")
    sensitivity = []
    for excluded in sorted(ood.asset_class.unique()):
        part = ood[ood.asset_class != excluded].groupby(["split", "seed", "method"])[FOCUS].mean().reset_index()
        part["excluded_asset_class"] = excluded
        sensitivity.append(part)
    pd.concat(sensitivity).to_csv(out / "leave_one_asset_class_out.csv", index=False)
    ci = []
    for split in cfg["splits"]:
        for a, b in [("DeepEnsembles", "MCDropout"), ("SCALE-R", "MCDropout"),
                     ("OnlineQuantile", "MCDropout"), ("MCDropout", "EWMAGaussian"),
                     ("MCDropout", "ZeroGaussian")]:
            if all((root / "predictions" / split["id"] / f"seed_{s}" / (m + ".csv.gz")).exists()
                   for s in cfg["seeds"] for m in (a, b)):
                print(f"Bootstrap {split['id']}: {a} vs {b}", flush=True)
                ci.extend(paired_bootstrap(root, cfg, split["id"], a, b))
    pd.DataFrame(ci).to_csv(out / "paired_block_bootstrap.csv", index=False)
    runtime = []
    for path in sorted((root / "models").glob("*/*/*/done.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        runtime.append({"split": path.parents[2].name, **{k:v for k,v in d.items() if k != "history"}})
    pd.DataFrame(runtime).to_csv(out / "training_runtime.csv", index=False)
    lines = ["# Fresh experimental results", "", "Generated only from the current run's daily predictions.",
             "", "OnlineQuantile uses revealed outcomes sequentially. SCALE-R is a specified reconstruction.", ""]
    for split in cfg["splits"]:
        lines.extend([f"## {split['id']}", "", "| Configuration | nRMSE | ECE | PICP90 | NWIS | UA width |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for (sid, method), row in macro.iterrows():
            if sid == split["id"]:
                vals = [f"{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}" for k in ("nRMSE", "ECE", "PICP90", "NWIS", "UA_width")]
                lines.append("| " + method + " | " + " | ".join(vals) + " |")
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    figures(wide, out, source)
    json_write(out / "analysis_status.json", {"prediction_files": len(paths), "completed": True,
               "source_market": source, "target_markets": sorted(ood.market.unique().tolist()),
               "scope": "new implementation; this report does not import old manuscript results"})
    print(f"Analysis written to {out}", flush=True)


def figures(wide, out, source_market="SP500"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split, frame in wide.groupby("split"):
        frame = frame[frame.market != source_market]
        for metric in ("nRMSE", "ECE"):
            table = frame.pivot_table(index="method", columns="market", values=metric)
            fig, ax = plt.subplots(figsize=(13, 9))
            image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(table.columns)), table.columns, rotation=65, ha="right")
            ax.set_yticks(range(len(table.index)), table.index)
            ax.set_title(f"{split}: {metric}, mean across seeds")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(out / f"{split}_{metric}_heatmap.png", dpi=160)
            plt.close(fig)
        macro = frame.groupby("method")[["PICP90", "NMPIW90"]].mean()
        fig, ax = plt.subplots(figsize=(11, 7))
        for method, row in macro.iterrows():
            ax.scatter(row.NMPIW90, row.PICP90)
            ax.annotate(method, (row.NMPIW90, row.PICP90), fontsize=8)
        ax.axhline(.9, color="black", linestyle="--")
        ax.set(xlabel="90% interval width / target SD", ylabel="90% empirical coverage", title=split)
        fig.tight_layout()
        fig.savefig(out / f"{split}_coverage_width.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 7))
        for method in ("MCDropout", "DeepEnsembles", "SCALE-R", "OnlineQuantile", "EWMAGaussian"):
            selected = frame[frame.method == method]
            if len(selected):
                cover = [selected[f"coverage_q_{p:g}"].mean() for p in LEVELS]
                ax.plot(LEVELS, cover, marker=".", label=method)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--")
        ax.set(xlabel="Nominal quantile", ylabel="Empirical one-sided coverage", title=split)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{split}_calibration.png", dpi=160)
        plt.close(fig)
