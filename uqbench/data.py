"""Own-market causal features, source-only scaling, and immutable price snapshots."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, json_write, source_id


def canonical(frame):
    d = frame.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
    if "date" in d:
        d.index = pd.to_datetime(d.pop("date"))
    else:
        d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d.index = d.index.normalize()
    d = d.sort_index()
    if d.index.duplicated().any():
        raise ValueError("Duplicate trading dates; resolve the source data before running.")
    if "close" not in d:
        raise ValueError("CSV must contain Date and adjusted Close.")
    for c in ("close", "open", "high", "low", "volume"):
        if c not in d:
            d[c] = 0.0 if c == "volume" else d["close"]
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if d["close"].isna().any() or (d["close"] <= 0).any():
        raise ValueError("Missing/nonpositive Close: no target filling or silent deletion is allowed.")
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0).clip(lower=0)
    if not np.isfinite(d[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("Nonfinite OHLCV values.")
    return d[["open", "high", "low", "close", "volume"]]


def download(cfg, raw_dir):
    from .download import DownloadRateLimited, MARKET_DELAY, fetch_history, wait_seconds
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = []
    failures = []
    requested = False
    for m in cfg["markets"]:
        p = raw_dir / (m["id"] + ".csv")
        meta_p = raw_dir / (m["id"] + ".json")
        if p.exists():
            if not meta_p.exists():
                raise ValueError(f"{p} lacks provenance. Use the import command for manual CSVs.")
            metadata = json.loads(meta_p.read_text(encoding="utf-8"))
            if metadata.get("synthetic"):
                raise ValueError("Synthetic smoke data cannot be used for a formal market run.")
            if digest(p) != metadata["sha256"]:
                raise ValueError(f"Snapshot changed: {p}. Create a new raw directory for a new snapshot.")
            d = canonical(pd.read_csv(p))
        else:
            if requested:
                wait_seconds(MARKET_DELAY)
            requested = True
            try:
                d = canonical(fetch_history(m, cfg, raw_dir))
            except Exception as error:
                failures.append({"market": m["id"], "ticker": m["ticker"], "error": str(error)})
                json_write(raw_dir / "download_report.json", report)
                json_write(raw_dir / "download_status.json", {"complete": False, "required": len(cfg["markets"]),
                           "verified_this_pass": len(report), "failures": failures})
                if isinstance(error, DownloadRateLimited):
                    raise
                print(f"FAILED {m['id']}: {error}. Trying the next market; this run remains incomplete.", flush=True)
                continue
            d.to_csv(p, index_label="Date", float_format="%.12g")
            metadata = {**m, "provider": "Yahoo Finance via yfinance", "adjustment": "auto_adjust=True",
                        "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(), "sha256": digest(p)}
            json_write(meta_p, metadata)
        report.append({**metadata, "rows": len(d), "first": str(d.index[0].date()),
                       "last": str(d.index[-1].date()), "large_moves_gt_50pct":
                       int((d.close.pct_change(fill_method=None).abs() > .5).sum())})
        print(f"Data: {m['id']} {len(d)} rows, {d.index[0].date()}--{d.index[-1].date()}", flush=True)
        json_write(raw_dir / "download_status.json", {"complete": False, "required": len(cfg["markets"]),
                   "verified_this_pass": len(report), "failures": failures})
    json_write(raw_dir / "download_report.json", report)
    json_write(raw_dir / "download_status.json", {"complete": not failures, "required": len(cfg["markets"]),
               "verified_this_pass": len(report), "failures": failures})
    if failures:
        raise RuntimeError("Missing required markets: " + ", ".join(x["market"] for x in failures)
                           + ". See data/raw/download_status.json. No missing market was substituted or omitted.")


def import_csv(cfg, market, src, raw_dir, provenance):
    m = next(x for x in cfg["markets"] if x["id"] == market)
    d = canonical(pd.read_csv(src))
    dest = Path(raw_dir) / (market + ".csv")
    if dest.exists():
        raise FileExistsError("Existing snapshots are immutable. Choose a new raw directory.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest, index_label="Date", float_format="%.12g")
    json_write(dest.with_suffix(".json"), {**m, "provider": provenance,
               "adjustment": "user asserts supplied Close/OHLC are adjusted",
               "sha256": digest(dest), "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat()})


def features(d, feature_mode="ohlcv"):
    c = d.close
    r = 100 * c.pct_change(fill_method=None)
    s = r.rolling(60, min_periods=60).std(ddof=1).clip(lower=.01)
    f = pd.DataFrame(index=d.index)
    for lag in range(10):
        f[f"ret_lag_{lag}"] = r.shift(lag) / s
    for w in (5, 10, 20, 40, 60):
        f[f"momentum_{w}"] = 100 * (c / c.shift(w) - 1) / (s * np.sqrt(w))
        f[f"vol_ratio_{w}"] = r.rolling(w).std(ddof=1) / s
    for w in (20, 60):
        f[f"downside_{w}"] = np.sqrt(r.clip(upper=0).pow(2).rolling(w).mean()) / s
        f[f"autocorr_{w}"] = r.rolling(w).corr(r.shift(1)).fillna(0)
    gains = r.clip(lower=0).rolling(14).mean()
    losses = -r.clip(upper=0).rolling(14).mean()
    f["rsi_14"] = (gains / (gains + losses)).fillna(.5) * 2 - 1
    f["drawdown_60"] = 100 * (c / c.rolling(60).max() - 1) / s
    f["skew_20"] = r.rolling(20).skew().fillna(0)
    f["kurt_20"] = r.rolling(20).kurt().fillna(0)
    if feature_mode == "ohlcv":
        f["range_scaled"] = 100 * (d.high - d.low) / c / s
        f["volume_change"] = np.log1p(d.volume).diff().fillna(0)
    elif feature_mode != "close_only":
        raise ValueError("Unknown feature mode: " + str(feature_mode))
    assert len(f.columns) == (28 if feature_mode == "close_only" else 30)
    f["scale"] = s
    f["return_pct"] = r
    f["target_pct"] = r.shift(-1)
    f["target_date"] = pd.Series(d.index, index=d.index).shift(-1)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def mask_dates(dates, start, end):
    return (dates >= start) & (dates <= end)


def prepare(cfg, raw_dir, prepared_dir):
    out = Path(prepared_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = cfg.get("feature_mode", "ohlcv")
    tables = {m["id"]: features(canonical(pd.read_csv(Path(raw_dir) / (m["id"] + ".csv"))), mode)
              for m in cfg["markets"]}
    feature_cols = list(tables[source_id(cfg)].columns[:-4])
    if len(feature_cols) != cfg.get("feature_count", 30):
        raise ValueError("Feature mode/count mismatch in config")
    counts = []
    for split in cfg["splits"]:
        source = tables[source_id(cfg)]
        source_target = source.target_date.dt.strftime("%Y-%m-%d").to_numpy()
        tm = mask_dates(source_target, split["train_start"], split["train_end"])
        if tm.sum() < 500:
            raise ValueError("Insufficient source training history.")
        train_values = source.loc[tm, feature_cols].to_numpy()
        low, high = np.quantile(train_values, [.005, .995], axis=0)
        clipped = np.clip(train_values, low, high)
        mu, sd = clipped.mean(0), np.maximum(clipped.std(0), 1e-5)
        normalizer = {"fit_market": source_id(cfg), "fit_target_dates":
                      [split["train_start"], split["train_end"]], "features": feature_cols,
                      "clip_low": low.tolist(), "clip_high": high.tolist(),
                      "mean": mu.tolist(), "sd": sd.tolist()}
        json_write(out / split["id"] / "normalizer.json", normalizer)
        for market, d in tables.items():
            vals = np.clip((np.clip(d[feature_cols].to_numpy(), low, high) - mu) / sd, -8, 8).astype("float32")
            origins = np.arange(cfg["window"] - 1, len(d))
            target_dates = d.target_date.dt.strftime("%Y-%m-%d").to_numpy()[origins]
            path = out / split["id"] / (market + ".npz")
            np.savez_compressed(path, features=vals, origin_index=origins,
                                origin_dates=d.index.strftime("%Y-%m-%d").to_numpy(dtype="U10")[origins],
                                target_dates=target_dates.astype("U10"),
                                y=(d.target_pct / d.scale).to_numpy()[origins].astype("float32"),
                                target_pct=d.target_pct.to_numpy()[origins],
                                scale=d.scale.to_numpy()[origins],
                                returns=d.return_pct.to_numpy(),
                                hist_scale=d.scale.to_numpy())
            row = {"split": split["id"], "market": market}
            for part in ("train", "val", "test"):
                row[part] = int(mask_dates(target_dates, split[part + "_start"], split[part + "_end"]).sum())
                if part == "test" or market == source_id(cfg):
                    observed = pd.DatetimeIndex(target_dates[mask_dates(target_dates, split[part+"_start"], split[part+"_end"])])
                    start, end = pd.Timestamp(split[part+"_start"]), pd.Timestamp(split[part+"_end"])
                    if (not len(observed) or observed[0] > start+pd.Timedelta(days=14)
                            or observed[-1] < end-pd.Timedelta(days=14)):
                        raise ValueError(f"History does not cover required endpoints: {market}, {split['id']}, {part}")
                    for year in range(start.year, end.year+1):
                        if int((observed.year == year).sum()) < 150:
                            raise ValueError(f"Too few observations in required year: {market}, {part}, {year}")
                    if len(observed)>1 and np.max(np.diff(observed.values).astype('timedelta64[D]').astype(int))>20:
                        raise ValueError(f"Suspicious trading-data gap over 20 calendar days: {market}, {part}")
            if row["test"] < 500 or (market == source_id(cfg) and row["val"] < 150):
                raise ValueError(f"Incomplete required period: {row}. Repair data before training; no market is silently omitted.")
            counts.append(row)
    pd.DataFrame(counts).to_csv(out / "sample_counts.csv", index=False)
    json_write(out / "feature_map.json", {"features": feature_cols, "feature_mode": mode, "target":
               "100*(protocol price/reference rate at next observed session / value at origin - 1)",
               "price_protocol": cfg.get("protocol", {"price_basis": "Yahoo auto_adjust=True"}),
               "target_scaling": "divide by origin's trailing-60-session return SD; minimum .01 percentage points",
               "macro_or_cross_market_predictors": False})
    json_write(out / "prepared_manifest.json", {
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "raw_sha256": {m["id"]: digest(Path(raw_dir)/(m["id"]+".csv")) for m in cfg["markets"]},
        "prepared_sha256": {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
                            if p.is_file() and p.name != "prepared_manifest.json"}})
    print("Prepared source-only transforms and all required market periods.", flush=True)


def verify_prepared(cfg, raw_dir, prepared_dir):
    out=Path(prepared_dir)
    path=out/"prepared_manifest.json"
    if not path.exists():
        raise RuntimeError("Run prepare before run; the prepared-data manifest is missing.")
    meta=json.loads(path.read_text(encoding="utf-8"))
    if meta["config_sha256"] != hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest():
        raise RuntimeError("Prepared data use a different config. Run prepare with this config.")
    for m in cfg["markets"]:
        if digest(Path(raw_dir)/(m["id"]+".csv")) != meta["raw_sha256"][m["id"]]:
            raise RuntimeError("Raw data changed since preparation. Use a new snapshot/run and prepare again.")
    for rel,sha in meta["prepared_sha256"].items():
        if digest(out/rel) != sha:
            raise RuntimeError(f"Prepared file changed: {rel}. Prepare again; do not mix datasets.")


class MarketData:
    def __init__(self, path, split, part, window):
        with np.load(path, allow_pickle=False) as z:
            self.features = z["features"]
            self.returns, self.hist_scale = z["returns"], z["hist_scale"]
            mask = mask_dates(z["target_dates"], split[part + "_start"], split[part + "_end"])
            for key in ("origin_index", "origin_dates", "target_dates", "y", "target_pct", "scale"):
                setattr(self, key, z[key][mask])
        self.window = window
        if not len(self.y):
            raise ValueError("Empty evaluation/training slice.")

    def windows(self, indices=None):
        ids = self.origin_index if indices is None else self.origin_index[np.asarray(indices)]
        return np.stack([self.features[j - self.window + 1:j + 1] for j in ids]).astype("float32")

    def history(self, j, length=250):
        k = self.origin_index[j]
        return self.returns[max(0, k - length + 1):k + 1]
