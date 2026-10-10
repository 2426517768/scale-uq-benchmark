import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import environment, fingerprint, json_write, source_id
from .data import MarketData
from .distributions import (QCOLS, baseline, fit_garch, isotonic_fit, isotonic_quantiles,
                            online_quantiles, split_conformal_fit)


def freeze_run(cfg, raw_dir, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    now = fingerprint(cfg, raw_dir)
    for market in cfg["markets"]:
        meta_path = Path(raw_dir) / (market["id"] + ".json")
        if not meta_path.exists():
            raise RuntimeError("Raw provenance is missing. Use download or import before a formal run.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("synthetic") or meta.get("sha256") != now["data_sha256"][market["id"]]:
            raise RuntimeError("Formal runs require verified real-data snapshots; smoke fixtures are excluded.")
    manifest = root / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["fingerprint"] != now["fingerprint"]:
            raise RuntimeError("Code/config/data differs from this run. Choose a new --out directory; results cannot be mixed.")
        previous_env = json.loads((root / "environment.json").read_text(encoding="utf-8"))
        current_env = environment()
        if previous_env["versions"] != current_env["versions"] or previous_env["python"] != current_env["python"]:
            raise RuntimeError("Python/package versions changed. Restore environment_lock.txt or choose a new --out directory.")
    else:
        json_write(manifest, now)
        json_write(root / "environment.json", environment())
    return now


def frame_for(pred, data, market):
    n = len(data.y)
    df = pd.DataFrame({"market": [market["id"]] * n, "asset_class": [market["class"]] * n,
                       "origin_date": data.origin_dates, "target_date": data.target_dates,
                       "target_pct": data.target_pct, "origin_scale_pct": data.scale,
                       "mean": pred["mean"], "sd": pred["sd"], "epi_sd": pred["epi_sd"]})
    for j, col in enumerate(QCOLS):
        df[col] = pred["quantiles"][:, j]
    return df


def unscale(pred, scale):
    return {"mean": pred["mean"] * scale, "sd": pred["sd"] * scale,
            "epi_sd": pred["epi_sd"] * scale,
            "quantiles": pred["quantiles"] * scale[:, None]}


def write_predictions(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["market", "target_date"]).any():
        raise ValueError("Duplicate forecasts.")
    from .metrics import score
    for _, group in df.groupby("market"):
        score(group)  # Fail before committing nonfinite or crossing predictions.
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.10g")
    os.replace(tmp, path)


def run_experiments(cfg, raw_dir, prepared_dir, root, allow_cpu=False, seed_filter=None, split_filter=None):
    from .engine import (ABLATIONS, LEARNED, MAIN, ensemble_predict, predict_normalized,
                         select_device, synchronize, train)
    import torch
    from .data import verify_prepared
    verify_prepared(cfg, raw_dir, prepared_dir)
    root = Path(root)
    freeze_run(cfg, raw_dir, root)
    json_write(root / "prepared_manifest.json", json.loads((Path(prepared_dir)/"prepared_manifest.json").read_text(encoding="utf-8")))
    device = select_device(allow_cpu)
    torch.set_num_threads(cfg["training"]["threads"])
    hardware = {"device": str(device), "gpu":
               torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "cuda_runtime": torch.version.cuda, "torch_arch_list": torch.cuda.get_arch_list(),
               "deterministic_algorithms": True, "num_workers": 0}
    if (root / "hardware.json").exists() and json.loads((root/"hardware.json").read_text(encoding="utf-8")) != hardware:
        raise RuntimeError("GPU/runtime changed during a resumable run. Use the original hardware or a new --out directory.")
    json_write(root / "hardware.json", hardware)
    jobs = len(cfg["splits"]) * len(cfg["seeds"]) * (len(LEARNED) + cfg["training"]["ensemble_members"])
    print(f"Planned {jobs} trained-network jobs; one GPU task at a time. Completed jobs resume from checkpoints.", flush=True)
    for split in cfg["splits"]:
        if split_filter and split["id"] not in split_filter:
            continue
        path = Path(prepared_dir) / split["id"]
        source_train = MarketData(path / (source_id(cfg) + ".npz"), split, "train", cfg["window"])
        source_val = MarketData(path / (source_id(cfg) + ".npz"), split, "val", cfg["window"])
        stats_path = root / "source_statistics" / (split["id"] + ".json")
        if stats_path.exists():
            source_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        else:
            source_stats = {"target_sd": float(source_train.y.std(ddof=1)), "garch": fit_garch(source_train.y)}
            json_write(stats_path, source_stats)
        for seed in cfg["seeds"]:
            if seed_filter and seed not in seed_filter:
                continue
            base_dir = root / "models" / split["id"] / f"seed_{seed}"
            models = {}
            for method in LEARNED:
                models[method] = train(cfg, method, seed, source_train, source_val, base_dir / method, device)
            ensemble = [train(cfg, f"Ensemble-{i}", seed, source_train, source_val,
                              base_dir / f"Ensemble-{i}", device)
                        for i in range(cfg["training"]["ensemble_members"])]
            val = predict_normalized(models["GaussianTCN"], source_val, cfg, device, stochastic=True, seed=seed)
            iso = isotonic_fit(source_val.y, val["mean"], val["sd"])
            signed = split_conformal_fit(source_val.y, val["mean"], val["sd"])
            json_write(root / "calibration" / split["id"] / f"seed_{seed}.json",
                       {"fit_market": source_id(cfg), "fit_dates": [split["val_start"], split["val_end"]],
                        "isotonic": iso, "signed_standardized_quantiles": signed.tolist(),
                        "online_lr": cfg["analysis"]["online_lr"]})
            methods = MAIN + ABLATIONS
            todo = [m for m in methods if not (root / "predictions" / split["id"] / f"seed_{seed}" / (m + ".csv.gz")).exists()]
            all_frames = {m: [] for m in todo}
            inference = []
            for market in cfg["markets"]:
                if not todo:
                    break
                data = MarketData(path / (market["id"] + ".npz"), split, "test", cfg["window"])
                need_mc = any(m in todo for m in ("MCDropout", "Isotonic", "SplitConformal", "OnlineQuantile"))
                synchronize(device)
                mc_clock = time.perf_counter()
                mc = predict_normalized(models["GaussianTCN"], data, cfg, device, stochastic=True, seed=seed) if need_mc else None
                synchronize(device)
                mc_seconds = time.perf_counter() - mc_clock if need_mc else 0.
                for method in todo:
                    synchronize(device)
                    clock = time.perf_counter()
                    if method in ("ZeroGaussian", "RollingEmpirical", "EWMAGaussian", "GARCH-Norm"):
                        pred = baseline(data, method, source_stats)
                    else:
                        if method == "MCDropout":
                            p = mc
                        elif method in ("Isotonic", "SplitConformal", "OnlineQuantile"):
                            p = {k:v.copy() for k,v in mc.items()}
                            if method == "Isotonic":
                                q = isotonic_quantiles(p["mean"], p["sd"], iso)
                            elif method == "SplitConformal":
                                q = p["mean"][:, None] + p["sd"][:, None] * signed
                            else:
                                q, final_state = online_quantiles(data.y, p["mean"], p["sd"], signed, cfg["analysis"]["online_lr"])
                                json_write(root / "online_states" / split["id"] / f"seed_{seed}" / (market["id"] + ".json"),
                                           {"quantile_state": final_state.tolist(), "updates": len(data.y), "state_reset_per_market": True})
                            p["quantiles"] = q
                            p["sd"] = (q[:, QCOLS.index("q_0.95")] - q[:, QCOLS.index("q_0.05")]) / 3.2897072539
                        elif method == "DeepEnsembles":
                            p = ensemble_predict(ensemble, data, cfg, device, seed)
                        else:
                            p = predict_normalized(models[method], data, cfg, device, seed=seed)
                        pred = unscale(p, data.scale)
                    synchronize(device)
                    seconds = time.perf_counter() - clock
                    shared_seconds = mc_seconds if method in ("MCDropout", "Isotonic", "SplitConformal", "OnlineQuantile") else 0.
                    inference.append({"split": split["id"], "seed": seed, "method": method,
                                      "market": market["id"], "n": len(data.y), "seconds": seconds + shared_seconds,
                                      "shared_mc_component_seconds": shared_seconds,
                                      "scope": "independent-equivalent inference cost; shared MC component charged to each dependent configuration"})
                    all_frames[method].append(frame_for(pred, data, market))
                print(f"Forecasts: {split['id']} seed {seed} {market['id']} ({len(data.y)} sessions)", flush=True)
            if inference:
                p = root / "runtime" / split["id"] / f"seed_{seed}_batch_inference.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                merged = pd.DataFrame(inference)
                if p.exists():
                    merged = pd.concat([pd.read_csv(p), merged]).drop_duplicates(["split", "seed", "method", "market"], keep="last")
                temp = Path(str(p)+".tmp")
                merged.to_csv(temp, index=False)
                os.replace(temp,p)
            for method in todo:
                write_predictions(root / "predictions" / split["id"] / f"seed_{seed}" / (method + ".csv.gz"), all_frames[method])
            # Measure four main underlying prediction procedures with actual device synchronization.
            latency_path = root / "runtime" / split["id"] / f"seed_{seed}_latency.csv"
            if not latency_path.exists():
                one = copy.copy(source_val)
                for key in ("origin_index", "origin_dates", "target_dates", "y", "target_pct", "scale"):
                    setattr(one, key, getattr(one, key)[:1])
                latency_rows = []
                for method in ("GaussianTCN", "MCDropout", "DeepEnsembles", "Evidential", "SCALE-R"):
                    def call():
                        if method == "DeepEnsembles":
                            return ensemble_predict(ensemble, one, cfg, device, seed)
                        return predict_normalized(models["GaussianTCN" if method == "MCDropout" else method],
                                                  one, cfg, device, stochastic=method == "MCDropout", seed=seed)
                    for _ in range(5):
                        call()
                    timing = []
                    for _ in range(cfg["training"].get("latency_calls", 100)):
                        synchronize(device)
                        clock = time.perf_counter()
                        call()
                        synchronize(device)
                        timing.append(1000 * (time.perf_counter() - clock))
                    latency_rows.append({"method": method, "calls": len(timing), "median_ms": float(np.median(timing)),
                                         "p95_ms": float(np.quantile(timing, .95)), "scope": "one window to normalized prediction and quantiles; includes CPU transfer"})
                latency_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(latency_rows).to_csv(latency_path, index=False)
            del models, ensemble
            if device.type == "cuda":
                torch.cuda.empty_cache()


def verify_complete(cfg, root):
    from .engine import MAIN, ABLATIONS, LEARNED
    missing = []
    for split in cfg["splits"]:
        reference = None
        for seed in cfg["seeds"]:
            for job in LEARNED + [f"Ensemble-{i}" for i in range(cfg["training"]["ensemble_members"])]:
                directory = Path(root) / "models" / split["id"] / f"seed_{seed}" / job
                for name in ("done.json", "best.pt", "last.pt"):
                    if not (directory / name).exists():
                        missing.append(str(directory / name))
            for method in MAIN + ABLATIONS:
                path = Path(root) / "predictions" / split["id"] / f"seed_{seed}" / (method + ".csv.gz")
                if not path.exists():
                    missing.append(str(path))
                else:
                    d = pd.read_csv(path)
                    if set(d.market) != {m["id"] for m in cfg["markets"]}:
                        raise ValueError(f"Missing market forecasts in {path}")
                    if d.duplicated(["market", "target_date"]).any():
                        raise ValueError(f"Duplicate forecast keys in {path}")
                    indexed = d.set_index(["market", "target_date"]).sort_index()
                    if reference is None:
                        reference = indexed[["target_pct", "origin_date"]]
                    elif (not reference.index.equals(indexed.index)
                          or not np.allclose(reference.target_pct, indexed.target_pct, rtol=0, atol=1e-7)
                          or not reference.origin_date.equals(indexed.origin_date)):
                        raise ValueError(f"Methods/seeds have different realized evaluation panels: {path}")
                    from .metrics import score
                    for _, group in d.groupby("market"):
                        score(group)
    if missing:
        raise RuntimeError(f"Incomplete run: {len(missing)} required files are missing. Resume run. First: {missing[0]}")
    return {"forecast_files": len(cfg["splits"]) * len(cfg["seeds"]) * (len(MAIN) + len(ABLATIONS)),
            "main_configurations": len(MAIN), "diagnostic_configurations": len(ABLATIONS),
            "target_markets": len(cfg["markets"]) - 1, "seeds": len(cfg["seeds"]), "time_splits": len(cfg["splits"])}
