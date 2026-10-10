"""Independently verify a compact evidence ZIP; no training, Torch or network.

Only the seven exported configurations can be verified from daily arrays.
Sufficient statistics verify scores and intervals, not model correctness or
the historical point-in-time availability of the input data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def metric_values(data, weights, seed_ids, method):
    mask = data["valid"].astype(float)
    y = data["target_pct"]
    count = weights @ mask
    if (count <= 1).any():
        raise ValueError("Insufficient resampled observations")
    average = weights @ y / count
    second = weights @ (y * y) / count
    sd = np.sqrt(np.maximum((second - average**2) * count / (count - 1), 1e-10))
    mi = data["methods"].tolist().index(method)
    mse = np.einsum("d,sdm->sm", weights, data["mse"][mi])[seed_ids] / count
    wis = np.einsum("d,sdm->sm", weights, data["wis"][mi])[seed_ids] / count
    picp = np.einsum("d,sdm->sm", weights, data["coverage90"][mi])[seed_ids] / count
    hits = np.einsum("d,sdmq->smq", weights, data["quantile_hits"][mi])[seed_ids] / count[:, None]
    return {"nRMSE": np.sqrt(mse) / sd, "NWIS": wis / sd, "PICP90": picp,
            "ECE": np.abs(hits - np.arange(1, 10) / 10).mean(-1)}


def verify(path, output):
    rows = []
    summary_checks = []
    with zipfile.ZipFile(path) as archive:
        info = json.loads(archive.read("evidence_manifest.json"))
        for name, expected in info["component_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Component hash mismatch: " + name)
        cfg = json.loads(archive.read("manifest.json"))["config"]
        forecasts = len(cfg["splits"]) * len(cfg["seeds"]) * 22
        if info["full_prediction_file_count"] != forecasts or len(info["prediction_sha256"]) != forecasts:
            raise ValueError("Prediction file count mismatch")
        reported = pd.read_csv(io.BytesIO(archive.read("paired_block_bootstrap.csv")))
        wide = pd.read_csv(io.BytesIO(archive.read("metrics_wide.csv")))
        cache = {}
        for split in cfg["splits"]:
            sid = split["id"]
            with np.load(io.BytesIO(archive.read(sid + "_daily_sufficient_statistics.npz")), allow_pickle=False) as f:
                data = {k: f[k] for k in f.files}
            cache[sid] = data
            ns, nd, nm = len(data["seeds"]), len(data["dates"]), len(data["markets"])
            if data["seeds"].tolist() != cfg["seeds"] or nm != 12:
                raise ValueError("Evidence membership mismatch")
            for method in data["methods"]:
                values = metric_values(data, np.ones(nd), np.arange(ns), method)
                for si, seed in enumerate(data["seeds"]):
                    table = wide[(wide.split == sid) & (wide.method == method) & (wide.seed == seed)]
                    table = table.set_index("market").loc[data["markets"]]
                    for metric, array in values.items():
                        delta = float(np.max(np.abs(array[si] - table[metric].to_numpy())))
                        if delta > 1e-8:
                            raise ValueError(f"Daily summary mismatch: {sid}/{method}/{seed}/{metric}: {delta}")
                        summary_checks.append({"split": sid, "method": method, "seed": int(seed),
                                               "metric": metric, "max_absolute_difference": delta})
            print(f"Verified daily summaries: {sid}", flush=True)
        for (sid, a, b), table in reported.groupby(["split", "A", "B"], sort=False):
            data = cache[sid]
            ns, nd = len(data["seeds"]), len(data["dates"])
            def difference(weights, seeds):
                left, right = metric_values(data, weights, seeds, a), metric_values(data, weights, seeds, b)
                return np.array([left[m].mean() - right[m].mean() for m in ("nRMSE", "NWIS", "PICP90", "ECE")])
            point = difference(np.ones(nd), np.arange(ns))
            rng = np.random.default_rng(98731)
            estimates = []
            length = cfg["analysis"]["block_days"]
            for _ in range(cfg["analysis"]["bootstrap_reps"]):
                starts = rng.integers(0, max(1, nd-length+1), int(np.ceil(nd/length)))
                indices = np.concatenate([np.arange(s, min(s+length, nd)) for s in starts])[:nd]
                weights = np.bincount(indices, minlength=nd)
                seeds = rng.integers(0, ns, ns)
                estimates.append(difference(weights, seeds))
            bounds = np.quantile(estimates, [.025, .975], axis=0)
            for j, metric in enumerate(("nRMSE", "NWIS", "PICP90", "ECE")):
                original = table[table.metric == metric].iloc[0]
                recomputed = np.array([point[j], bounds[0,j], bounds[1,j]])
                expected = original[["difference_A_minus_B", "ci_low", "ci_high"]].to_numpy(float)
                delta = float(np.max(np.abs(recomputed-expected)))
                if delta > 1e-8:
                    raise ValueError(f"Bootstrap mismatch: {sid}/{a}/{b}/{metric}: {delta}")
                rows.append({"split": sid, "A": a, "B": b, "metric": metric,
                             "difference_A_minus_B": point[j], "ci_low": bounds[0,j], "ci_high": bounds[1,j],
                             "max_absolute_difference": delta})
            print(f"Verified paired intervals: {sid} {a} - {b}", flush=True)
        result = {"status": "passed", "daily_summary_checks": len(summary_checks), "bootstrap_rows_verified": len(rows),
                  "maximum_summary_difference": max(r["max_absolute_difference"] for r in summary_checks),
                  "maximum_bootstrap_difference": max(r["max_absolute_difference"] for r in rows),
                  "scope": "Seven configurations; fixed 12-target panel; original unadjusted 20-observation block protocol",
                  "scale_similarity_ablation_prediction_comparison": info.get("scale_similarity_ablation_prediction_comparison", []),
                  "limits": ["Hashes of 220 files are recorded by the exporter; full forecasts are not contained here.",
                             "This does not audit model checkpoints or prove absence of all implementation/data errors."]}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "bootstrap_independently_recomputed.csv", index=False)
    pd.DataFrame(summary_checks).to_csv(output / "daily_summary_checks.csv", index=False)
    (output / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=Path("evidence_verification"))
    args = p.parse_args()
    verify(args.input, args.output)
