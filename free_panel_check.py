"""Independent candidate panel: Sina ETF prices and official ECB reference rates.

Does not modify config.json, data/raw, prepared data, checkpoints or formal results.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from free_data_check import coverage, normalize, request_wrapper, reuse_archive, sha, write_json

ROOT = Path(__file__).resolve().parent
ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ETF_CODES = {"SP500ETF": "SPY", "UKEquityETF": "EWU", "JapanEquityETF": "EWJ",
             "USAggregateBondETF": "AGG", "USTreasuryETF": "IEF", "GoldETF": "GLD",
             "OilETF": "USO", "CopperETF": "CPER"}
NEW_ETFS = {"SP500ETF", "UKEquityETF", "JapanEquityETF", "USAggregateBondETF"}
FX_IDS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")


def panel_config(original):
    cfg = json.loads(json.dumps(original))
    cfg["version"] = "free-panel-candidate-2.0"
    cfg["feature_mode"], cfg["feature_count"] = "close_only", 28
    cfg["protocol"] = {"status": "candidate_requires_data_review",
        "etf_return_basis": "split-adjusted price returns, excluding cash distributions",
        "fx_basis": "ECB daily euro reference rates and same-date cross rates",
        "fx_ohlcv": "no intraday OHLC or traded volume; OHLC placeholders equal reference rate; Volume=0",
        "transfer": "one US-listed SPY ETF source, twelve target series; US-listed regional ETFs are USD denominated",
        "source_index_replaced": True, "Germany_bond_replaced": True}
    replacements = {
        "SP500": ("SP500ETF", "SPY", "SPDR S&P 500 ETF Trust; USD market price, split-adjusted, cash distributions excluded"),
        "FTSE": ("UKEquityETF", "EWU", "iShares MSCI United Kingdom ETF; US-listed USD market price, not FTSE 100"),
        "Nikkei": ("JapanEquityETF", "EWJ", "iShares MSCI Japan ETF; US-listed USD market price, not Nikkei 225"),
        "GermanyGovtETF": ("USAggregateBondETF", "AGG", "iShares Core US Aggregate Bond ETF; not German government bonds")}
    for m in cfg["markets"]:
        if m["id"] in replacements:
            m["id"], m["ticker"], m["instrument"] = replacements[m["id"]]
        elif m["id"] in ETF_CODES:
            m["instrument"] += "; split-adjusted price returns, cash distributions excluded"
        elif m["id"] in FX_IDS:
            m["instrument"] += "; ECB reference/cross rate, not a market closing quote"
            m["ticker"] = "ECB:" + m["id"]
    return cfg


def ecb_cross_rates(payload):
    if len(payload) > 8_000_000:
        raise ValueError("ECB download exceeds 8 MB bound")
    if payload.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            files = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if files != ["eurofxref-hist.csv"] or z.getinfo(files[0]).file_size > 20_000_000:
                raise ValueError("Unexpected ECB archive members or expansion size")
            payload = z.read(files[0])
    raw = pd.read_csv(io.BytesIO(payload))
    required = ["Date", "USD", "GBP", "JPY", "AUD"]
    if not set(required).issubset(raw.columns):
        raise ValueError("Not the ECB historical euro-reference CSV")
    d = raw[required].copy()
    d["Date"] = pd.to_datetime(d.Date, errors="raise")
    if d.Date.isna().any() or d.Date.duplicated().any():
        raise ValueError("Missing/duplicate ECB dates")
    if not (d.Date == d.Date.dt.normalize()).all():
        raise ValueError("Unexpected intraday timestamps in ECB history")
    d = d.sort_values("Date")
    for c in required[1:]:
        d[c] = pd.to_numeric(d[c], errors="raise")
        if not np.isfinite(d[c]).all() or (d[c] <= 0).any():
            raise ValueError("Missing/nonpositive ECB rate for " + c + "; no filling")
    close = {"EURUSD": d.USD, "GBPUSD": d.USD / d.GBP,
             "USDJPY": d.JPY / d.USD, "AUDUSD": d.USD / d.AUD}
    return {mid: pd.DataFrame({"Date": d.Date, "Open": v, "High": v, "Low": v,
                               "Close": v, "Volume": 0.0}).reset_index(drop=True)
            for mid, v in close.items()}


def split_only_prices(raw, qfq, factors):
    """Validate Sina affine prices, then exclude additive cash adjustments explicitly.

    This is a new price-return protocol, not an estimate of total returns.
    Split factors are normalized to the final raw observation. Future common
    multiplicative scaling cancels exactly in price returns and close-only features.
    """
    d = normalize(raw)
    q = qfq.copy().rename(columns={"date": "Date", "open": "Open", "high": "High",
                                   "low": "Low", "close": "Close"})
    if "Date" not in q or not {"Open", "High", "Low", "Close"}.issubset(q.columns):
        raise ValueError("Missing qfq dates/OHLC")
    q["Date"] = pd.to_datetime(q.Date, errors="raise")
    q = q.sort_values("Date").reset_index(drop=True)
    if q.Date.duplicated().any() or not np.array_equal(d.Date.to_numpy(), q.Date.to_numpy()):
        raise ValueError("Raw and qfq trading dates differ")
    f = factors.rename(columns={"Date": "date"}).copy()
    if not {"date", "qfq_factor", "adjust"}.issubset(f):
        raise ValueError("Missing Sina factor fields")
    f["Date"] = pd.to_datetime(f["date"], errors="raise")
    f = f[["Date", "qfq_factor", "adjust"]].sort_values("Date")
    if f.Date.duplicated().any():
        raise ValueError("Duplicate factor dates")
    for c in ("qfq_factor", "adjust"):
        f[c] = pd.to_numeric(f[c], errors="raise")
    if not np.isfinite(f[["qfq_factor", "adjust"]]).all().all() or (f.qfq_factor <= 0).any():
        raise ValueError("Invalid corporate-action factors")
    aligned = pd.merge_asof(d[["Date"]], f, on="Date", direction="backward")
    if aligned[["qfq_factor", "adjust"]].isna().any().any():
        raise ValueError("No baseline factor for earliest raw date")
    cols = ["Open", "High", "Low", "Close"]
    expected = d[cols].to_numpy() * aligned.qfq_factor.to_numpy()[:, None] + aligned.adjust.to_numpy()[:, None]
    actual = q[cols].apply(pd.to_numeric, errors="raise").to_numpy()
    residual = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or residual > 0.002:
        raise ValueError(f"Affine factor verification failed: max residual {residual}")
    multipliers = aligned.qfq_factor.to_numpy() / aligned.qfq_factor.iloc[-1]
    output = d.copy()
    output[cols] = output[cols].to_numpy() * multipliers[:, None]
    bad_ohlc = ((output.High < output[["Open", "Close", "Low"]].max(axis=1)) |
                (output.Low > output[["Open", "Close", "High"]].min(axis=1)))
    action_changes = aligned.loc[aligned.qfq_factor.ne(aligned.qfq_factor.shift()), ["Date", "qfq_factor"]]
    detail = {"affine_qfq_max_abs_residual": residual,
        "price_basis": "split-only; additive cash adjustments excluded; not total return",
        "factor_changes": [{"date": str(r.Date.date()), "factor": float(r.qfq_factor)}
                            for r in action_changes.itertuples()],
        "bad_ohlc_dates": output.loc[bad_ohlc, "Date"].dt.strftime("%Y-%m-%d").tolist(),
        "manual_adjustment_review_required": True}
    return output, detail


def cached_response(out, mid, variant, symbol, instrument, ak, blocked, pause=time.sleep):
    p = out / "responses" / f"{mid}_{variant}.csv"
    mp = p.with_suffix(".json")
    if p.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("sha256") != sha(p) or meta.get("market") != mid or meta.get("variant") != variant:
            raise ValueError("Response cache identity/hash mismatch")
        if meta.get("provider") != "Sina via AKShare" or meta.get("provider_symbol") != symbol:
            raise ValueError("Cached ETF symbol/provider mismatch")
        return pd.read_csv(p), meta
    if blocked:
        raise RuntimeError("Sina access rejection; further requests stopped")
    print(f"GET Sina {symbol} {variant}", flush=True)
    pause(3)
    adjust = {"unadjusted": "", "qfq": "qfq", "qfq_factors": "qfq-factor"}[variant]
    try:
        raw = ak.stock_us_daily(symbol=symbol, adjust=adjust)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (403, 429) or any(x in str(e).lower() for x in ("captcha", "rate limit", "too many requests")):
            blocked.append(str(e))
        raise
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Empty Sina history")
    if isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.reset_index().rename(columns={raw.index.name or "index": "Date"})
        if "Date" not in raw:
            raw = raw.rename(columns={raw.columns[0]: "Date"})
    p.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(p, index=False, float_format="%.12g")
    meta = {"market": mid, "variant": variant, "provider": "Sina via AKShare",
            "provider_symbol": symbol, "requested_instrument": instrument,
            "sha256": sha(p), "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    write_json(mp, meta)
    return pd.read_csv(p), meta


def run_panel(cfg, out, ak, ecb_payload, pause=time.sleep):
    rows, blocked = [], []
    ecb_error, rates = None, {}
    try:
        rates = ecb_cross_rates(ecb_payload)
    except Exception as e:
        ecb_error = f"{type(e).__name__}: {e}"
    for m in cfg["markets"]:
        mid = m["id"]
        record = {"market": mid, "instrument": m["instrument"], "status": "failed",
                  "source": bool(m.get("source")), "promoted_to_formal": False}
        try:
            if mid in ETF_CODES:
                inputs, metas = {}, {}
                for variant in ("unadjusted", "qfq", "qfq_factors"):
                    inputs[variant], metas[variant] = cached_response(out, mid, variant, ETF_CODES[mid], m["instrument"], ak, blocked, pause)
                # Future cash events are excluded from returns; retain full response for affine validation.
                prices, detail = split_only_prices(inputs["unadjusted"], inputs["qfq"], inputs["qfq_factors"])
                record.update(provider="Sina via AKShare", provider_symbol=ETF_CODES[mid], **detail,
                              response_hashes={k: v["sha256"] for k, v in metas.items()})
            elif mid == "CSI300":
                p = out / "responses" / "CSI300_index.csv"
                meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
                if meta.get("sha256") != sha(p) or meta.get("market") != mid or meta.get("provider_symbol") != "sh000300":
                    raise ValueError("CSI300 cache identity/hash mismatch")
                prices = normalize(pd.read_csv(p))
                record.update(provider="Eastmoney via AKShare; previously acquired verified cache", response_hashes={"index": sha(p)})
            elif mid in FX_IDS:
                if ecb_error:
                    raise ValueError(ecb_error)
                prices = rates[mid]
                record.update(provider="European Central Bank official historical CSV", ecb_url=ECB_URL,
                    ecb_download_sha256=hashlib.sha256(ecb_payload).hexdigest(),
                    price_basis="daily reference/cross rate; not market close",
                    intraday_ohlcv_available=False,
                    formula={"EURUSD": "USD per EUR", "GBPUSD": "USD per EUR / GBP per EUR",
                             "USDJPY": "JPY per EUR / USD per EUR", "AUDUSD": "USD per EUR / AUD per EUR"}[mid])
            else:
                raise ValueError("Unsupported candidate market")
            cropped, detail = coverage(prices, cfg, m)
            record.update(detail)
            candidate = out / "candidates" / (mid + ".csv")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            cropped.to_csv(candidate, index=False, float_format="%.12g", date_format="%Y-%m-%d")
            record.update(candidate=str(candidate.relative_to(out)).replace("\\", "/"), candidate_sha256=sha(candidate),
                          status="candidate_review_required" if detail["coverage_ok"] else "insufficient_history")
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        rows.append(record)
        print(mid + ": " + record["status"] + " " + record.get("error", ""), flush=True)
        write_json(out / "audit_progress.json", {"complete": False, "records": rows})
    complete = len(rows) == 13 and all(r.get("coverage_ok") for r in rows)
    write_json(out / "audit.json", {"complete": True, "all_13_coverage_ok": complete,
                                    "formal_experiment_ready": False, "records": rows,
                                    "next_step": "review identity, adjustments, features and promote as a NEW protocol"})
    pd.DataFrame([{k: r.get(k) for k in ("market", "source", "status", "rows", "first", "last", "coverage_ok", "error")}
                  for r in rows]).to_csv(out / "coverage_summary.csv", index=False, encoding="utf-8-sig")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-config", default=str(ROOT / "config.json"))
    p.add_argument("--reuse-archive", default=str(ROOT / "FREE_DATA_RECOVERY.zip"))
    p.add_argument("--ecb-file", default=str(ROOT / "ecb_official_history.zip"))
    p.add_argument("--out", default=str(ROOT / "data" / "free_panel_audit"))
    args = p.parse_args()
    original = json.loads(Path(args.original_config).read_text(encoding="utf-8"))
    cfg = panel_config(original)
    signature = hashlib.sha256((json.dumps(cfg, sort_keys=True) + sha(__file__) + sha(ROOT / "free_data_check.py")).encode()).hexdigest()[:12]
    out = Path(args.out) / signature
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "reuse_status.json", reuse_archive(args.reuse_archive, out, original))
    write_json(out / "config_candidate.json", cfg)
    import requests
    import akshare as ak
    ecb = out / "responses" / "ecb_official_history.zip"
    emeta = ecb.with_suffix(".json")
    try:
        if ecb.exists():
            if json.loads(emeta.read_text(encoding="utf-8"))["sha256"] != sha(ecb):
                raise ValueError("ECB frozen cache hash changed")
            payload = ecb.read_bytes()
        else:
            local = Path(args.ecb_file)
            if local.exists():
                payload = local.read_bytes()
                source = "supplied ECB historical ZIP/CSV; source and rates still audited"
            else:
                print("GET official ECB historical ZIP", flush=True)
                response = requests.get(ECB_URL, timeout=(10, 30), stream=True)
                try:
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        data.extend(chunk)
                        if len(data) > 8_000_000:
                            raise ValueError("ECB download exceeds 8 MB")
                    payload = bytes(data)
                    source = ECB_URL
                finally:
                    response.close()
            ecb_cross_rates(payload)
            ecb.parent.mkdir(exist_ok=True)
            ecb.write_bytes(payload)
            write_json(emeta, {"source": source, "official_url": ECB_URL, "sha256": sha(ecb),
                              "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    except Exception as e:
        payload = b""
        write_json(out / "ecb_failure.json", {"error": f"{type(e).__name__}: {e}", "manual_fallback": ECB_URL})
    original_request = requests.sessions.Session.request
    requests.sessions.Session.request = request_wrapper(original_request)
    try:
        run_panel(cfg, out, ak, payload)
    finally:
        requests.sessions.Session.request = original_request
    write_json(out / "execution.json", {"python": sys.version, "platform": platform.platform(),
        "akshare": ak.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "script_sha256": sha(__file__), "formal_data_modified": False,
        "new_eastmoney_requests": False, "windows_proxy_config_changed": False})
    archive = ROOT / "FREE_PANEL_AUDIT.zip"
    temporary = Path(str(archive) + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f.relative_to(out))
        z.write(__file__, "collection_script.py")
        z.write(ROOT / "free_data_check.py", "collection_support.py")
    os.replace(temporary, archive)
    print(f"Send {archive.name} for review. No formal model training was started.", flush=True)


if __name__ == "__main__":
    main()
