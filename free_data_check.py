"""Free-source collection and coverage audit. Never writes formal data/raw or trains models."""
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
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INDEX_CODES = {"SP500": ("SPX", "标普500"), "FTSE": ("FTSE", "英国富时100"),
               "Nikkei": ("N225", "日经225")}
US_ETFS = {"USTreasuryETF": "IEF", "GoldETF": "GLD", "OilETF": "USO", "CopperETF": "CPER"}
FX_CODES = {"EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY", "AUDUSD": "AUDUSD"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp,path)


def reuse_archive(archive, out, cfg):
    """Reuse only fixed response filenames, with matching config, identity, and hashes; never execute archive code."""
    archive,out=Path(archive),Path(out)
    result={"archive_present":archive.exists(),"reused_tables":0,"errors":[]}
    if not archive.exists():
        return result
    try:
        with zipfile.ZipFile(archive) as z:
            if json.loads(z.read("config_snapshot.json")) != cfg:
                result["errors"].append("Archive config differs; no previous tables reused.")
                return result
            for market in cfg["markets"]:
                mid=market["id"]
                if mid in US_ETFS:
                    variants=("unadjusted","qfq","qfq_factors")
                    provider="Sina via AKShare"
                elif mid in INDEX_CODES or mid=="CSI300":
                    variants=("index",)
                    provider="Eastmoney via AKShare"
                elif mid in FX_CODES:
                    variants=("fx",)
                    provider="Eastmoney via AKShare"
                else:
                    continue
                for variant in variants:
                    csv_name=f"responses/{mid}_{variant}.csv"
                    meta_name=f"responses/{mid}_{variant}.json"
                    if csv_name not in z.namelist() or meta_name not in z.namelist():
                        continue
                    try:
                        meta=json.loads(z.read(meta_name))
                        payload=z.read(csv_name)
                        if (meta.get("market")!=mid or meta.get("variant")!=variant
                                or meta.get("provider")!=provider
                                or meta.get("requested_instrument")!=market["instrument"]
                                or hashlib.sha256(payload).hexdigest()!=meta.get("sha256")):
                            raise ValueError("Identity/provenance/hash mismatch.")
                        destination=out/csv_name
                        if destination.exists():
                            continue
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        destination.write_bytes(payload)
                        (out/meta_name).write_bytes(z.read(meta_name))
                        result["reused_tables"]+=1
                    except Exception as error:
                        result["errors"].append(csv_name+": "+str(error))
    except Exception as error:
        result["errors"].append(type(error).__name__+": "+str(error))
    return result


def request_wrapper(original, direct_eastmoney=False):
    def bounded(session, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"]=(10,30)
        host=urlparse(url).hostname
        if not direct_eastmoney or host not in ("push2.eastmoney.com","push2his.eastmoney.com"):
            return original(session,method,url,**kwargs)
        # A process-local transport choice to diagnose ProxyError; preserve system settings and TLS verification.
        trust_env,proxies=session.trust_env,session.proxies
        session.trust_env=False
        session.proxies={}
        kwargs["proxies"]={}
        if kwargs.get("verify") is None or kwargs.get("verify") is True:
            ca=os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
            if ca:
                kwargs["verify"]=ca
        try:
            return original(session,method,url,**kwargs)
        finally:
            session.trust_env,session.proxies=trust_env,proxies
    return bounded


def normalize(frame, expected_code=None):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Empty provider history.")
    d = frame.copy()
    if expected_code and "代码" in d:
        codes=set(d["代码"].dropna().astype(str))
        if codes != {expected_code}:
            raise ValueError(f"Instrument identity mismatch: expected {expected_code}, got {sorted(codes)}")
    aliases={"date":"Date", "日期":"Date", "open":"Open", "今开":"Open", "开盘":"Open",
             "close":"Close", "最新价":"Close", "收盘":"Close", "high":"High", "最高":"High",
             "low":"Low", "最低":"Low", "volume":"Volume", "成交量":"Volume"}
    d=d.rename(columns=aliases)
    if "Date" not in d:
        if isinstance(d.index,pd.DatetimeIndex) or d.index.name in ("date", "Date", "日期"):
            index_name=d.index.name or "index"
            d=d.reset_index().rename(columns={index_name:"Date"})
        else:
            raise ValueError("No actual trading dates in provider history.")
    if d.columns.duplicated().any():
        raise ValueError("Ambiguous duplicate columns after normalization.")
    if "Close" not in d:
        raise ValueError("No closing price in provider history.")
    dates=pd.to_datetime(d.Date,errors="raise")
    if dates.dt.tz is not None:
        dates=dates.dt.tz_localize(None)
    d["Date"]=dates.dt.normalize()
    d=d.sort_values("Date")
    if d.Date.duplicated().any():
        raise ValueError("Duplicate trading dates; no automatic deduplication.")
    d["Close"]=pd.to_numeric(d.Close,errors="raise")
    if not np.isfinite(d.Close).all() or (d.Close<=0).any():
        raise ValueError("Nonfinite/nonpositive Close; cannot be used by the existing return protocol.")
    for col in ("Open","High","Low","Volume"):
        if col not in d:
            d[col]=0. if col=="Volume" else d.Close
        d[col]=pd.to_numeric(d[col],errors="raise")
        if not np.isfinite(d[col]).all():
            raise ValueError(f"Invalid {col}; raw response saved for review.")
    return d[["Date","Open","High","Low","Close","Volume"]].reset_index(drop=True)


def coverage(frame, cfg, market):
    """Mirror required-period gates; candidates still require the original feature preparation."""
    d=frame[(frame.Date>=cfg["download_start"]) & (frame.Date<cfg["download_end_exclusive"])].copy()
    if len(d)<100:
        raise ValueError("Fewer than 100 rows in requested snapshot period.")
    issues=[]
    counts={}
    # Source fitting starts in 2010; target testing starts in 2020. Require calendar warm-up too.
    first_needed=min(pd.Timestamp(s["train_start"] if market.get("source") else s["test_start"]) for s in cfg["splits"])
    if int((d.Date<first_needed).sum()) < cfg["window"]+60:
        issues.append("Insufficient pre-period feature warm-up: need at least window+60 prior observations")
    for split in cfg["splits"]:
        for part in (("train","val","test") if market.get("source") else ("test",)):
            start,end=pd.Timestamp(split[part+"_start"]),pd.Timestamp(split[part+"_end"])
            dates=d.loc[(d.Date>=start)&(d.Date<=end),"Date"]
            key=f"{split['id']}_{part}"
            counts[key]=len(dates)
            if len(dates)<(500 if part in ("train","test") else 150):
                issues.append(f"{key}: insufficient period observations")
            if dates.empty or dates.iloc[0]>start+pd.Timedelta(days=14) or dates.iloc[-1]<end-pd.Timedelta(days=14):
                issues.append(f"{key}: required endpoints not covered")
            for year in range(start.year,end.year+1):
                if int((dates.dt.year==year).sum())<150:
                    issues.append(f"{key}: {year} has fewer than 150 observations")
            if len(dates)>1 and dates.diff().dropna().dt.days.max()>20:
                issues.append(f"{key}: gap exceeds 20 calendar days")
    detail={"rows":len(d),"first":d.Date.iloc[0].strftime("%Y-%m-%d"),"last":d.Date.iloc[-1].strftime("%Y-%m-%d"),
            "coverage_ok":not issues,"coverage_issues":issues,"period_counts":counts,
            "moves_gt_50pct":int((d.Close.pct_change(fill_method=None).abs()>.5).sum())}
    return d,detail


def available_global_names(ak):
    catalog=ak.index_global_spot_em()
    if not {"代码","名称"}.issubset(catalog.columns):
        raise ValueError("Global index catalog lacks expected identity fields.")
    result={}
    for code,_ in INDEX_CODES.values():
        rows=catalog[catalog["代码"].astype(str)==code]
        if len(rows)!=1:
            raise ValueError(f"Expected one global catalog entry for {code}; found {len(rows)}")
        result[code]=str(rows.iloc[0]["名称"])
    return result


def run_check(cfg, out, ak, pause=time.sleep):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    blocked_providers={}
    try:
        cached_indices=all((out/"responses"/(mid+"_index.csv")).exists() for mid in INDEX_CODES if any(m["id"]==mid for m in cfg["markets"]))
        global_names={code:name for code,name in INDEX_CODES.values()} if cached_indices else available_global_names(ak)
        catalog_error=None
    except Exception as e:
        global_names={}
        catalog_error=f"{type(e).__name__}: {e}"
    def collect(market, variant, provider, call, expected_code=None, reason="", candidate_kind="candidate"):
        relative=Path("responses")/(market["id"]+"_"+variant+".csv")
        path=out/relative
        meta_path=path.with_suffix(".json")
        base={"market":market["id"],"requested_instrument":market["instrument"],"variant":variant,
              "provider":provider,"status":"failed","reason":reason,"promoted_to_formal":False,
              "provider_symbol":expected_code or US_ETFS.get(market["id"]) or ("sh000300" if market["id"]=="CSI300" else None),
              "csv":str(relative).replace("\\","/")}
        try:
            if path.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("sha256")!=sha(path):
                    raise ValueError("Cached response hash changed; preserve this audit and start a new --out.")
                raw=pd.read_csv(path)
                base["retrieved_utc"]=meta["retrieved_utc"]
            else:
                if provider in blocked_providers:
                    raise RuntimeError("Further provider requests stopped after access rejection: "+blocked_providers[provider])
                print(f"GET {market['id']} {variant} from {provider}",flush=True)
                pause(3)
                raw=call()
                if raw is None or not isinstance(raw,pd.DataFrame) or raw.empty:
                    raise ValueError("Provider returned no table.")
                # Preserve unnormalized response and index-valued trading dates.
                if isinstance(raw.index,pd.DatetimeIndex):
                    raw=raw.reset_index()
                    raw=raw.rename(columns={raw.columns[0]:"Date"})
                path.parent.mkdir(parents=True,exist_ok=True)
                raw.to_csv(path,index=False,float_format="%.12g")
                base["retrieved_utc"]=pd.Timestamp.now(tz="UTC").isoformat()
                write_json(meta_path,{**base,"sha256":sha(path)})
            base["sha256"]=sha(path)
            if variant=="qfq_factors":
                if "Date" in raw and "date" not in raw:
                    raw=raw.rename(columns={"Date":"date"})
                if not {"date","qfq_factor","adjust"}.issubset(raw.columns):
                    raise ValueError("Adjustment-factor response lacks date/qfq_factor/adjust.")
                pd.to_datetime(raw["date"],errors="raise")
                factors=raw[["qfq_factor","adjust"]].apply(pd.to_numeric,errors="raise").to_numpy()
                if not np.isfinite(factors).all() or (factors[:,0]<=0).any():
                    raise ValueError("Nonfinite or nonpositive corporate-action factor.")
                base.update(status="factor_table_saved",rows=len(raw))
            else:
                normalized=normalize(raw,expected_code)
                cropped,detail=coverage(normalized,cfg,market)
                candidate=out/"candidates"/(market["id"]+"_"+variant+".csv")
                candidate.parent.mkdir(exist_ok=True)
                cropped.to_csv(candidate,index=False,float_format="%.12g",date_format="%Y-%m-%d")
                base.update(**detail, candidate=str(candidate.relative_to(out)).replace("\\","/"),
                            candidate_sha256=sha(candidate),
                            status=candidate_kind if detail["coverage_ok"] else "insufficient_history")
        except Exception as e:
            base["error"]=f"{type(e).__name__}: {e}"
            response=getattr(e,"response",None)
            message=str(e).lower()
            if (getattr(response,"status_code",None) in (403,429)
                    or any(x in message for x in ("http 403","http 429","too many requests","rate limit","captcha"))):
                blocked_providers.setdefault(provider,base["error"])
        rows.append(base)
        print(f"{market['id']} {variant}: {base['status']} {base.get('error','')}",flush=True)
        write_json(out/"audit_progress.json",{"complete":False,"records":rows})

    for m in cfg["markets"]:
        mid=m["id"]
        if mid in INDEX_CODES:
            code,default_name=INDEX_CODES[mid]
            name=global_names.get(code,default_name)
            collect(m,"index", "Eastmoney via AKShare",lambda name=name: ak.index_global_hist_em(symbol=name),code,
                    reason="Same price-index code; candidate close convention and volume availability require review.")
        elif mid=="CSI300":
            collect(m,"index", "Eastmoney via AKShare",lambda: ak.stock_zh_index_daily_em(
                symbol="sh000300",start_date=cfg["download_start"].replace("-",""),
                end_date=(pd.Timestamp(cfg["download_end_exclusive"])-pd.Timedelta(days=1)).strftime("%Y%m%d")),
                reason="CSI 300 price index, sh000300; no stock-price corporate-action adjustment implied.")
        elif mid in FX_CODES:
            code=FX_CODES[mid]
            collect(m,"fx", "Eastmoney via AKShare",lambda code=code: ak.forex_hist_em(symbol=code),code,
                    reason="Same currency quotation direction; provider daily fixing/close convention needs review.")
        elif mid in US_ETFS:
            symbol=US_ETFS[mid]
            for variant,adjust in (("unadjusted",""),("qfq","qfq"),("qfq_factors","qfq-factor")):
                collect(m,variant,"Sina via AKShare",lambda symbol=symbol,adjust=adjust: ak.stock_us_daily(symbol=symbol,adjust=adjust),
                        reason="Sina affine qfq is not assumed equivalent to Yahoo adjusted prices. Needs corporate-action/return-basis review.",
                        candidate_kind="adjustment_review_required")
        elif mid=="GermanyGovtETF":
            rows.append({"market":mid,"requested_instrument":m["instrument"],"variant":"manual",
                         "provider":"not verified","status":"manual_source_required","promoted_to_formal":False,
                         "reason":"No verified free full-history adapter for EXHA.DE. Do not substitute Bund yield, another ETF, or a monthly series."})
        else:
            rows.append({"market":mid,"variant":"manual","status":"unsupported_instrument","promoted_to_formal":False})
    write_json(out/"audit.json",{"complete":True,"formal_experiment_ready":False,
                                "global_catalog_error":catalog_error,"records":rows})
    table=[]
    for m in cfg["markets"]:
        records=[r for r in rows if r["market"]==m["id"]]
        prices=[r for r in records if r["variant"]!="qfq_factors"]
        usable=[r for r in prices if r.get("coverage_ok")]
        table.append({"market":m["id"],"ticker":m["ticker"],"instrument":m["instrument"],
                      "downloaded_variants":sum("sha256" in r for r in records),
                      "coverage_ok_variants":len(usable),"formal_ready":False,
                      "statuses":"; ".join(r["variant"]+":"+r["status"] for r in records),
                      "next_action":"review candidate convention/adjustment" if usable else "obtain same-instrument full history"})
    pd.DataFrame(table).to_csv(out/"coverage_summary.csv",index=False,encoding="utf-8-sig")
    write_json(out/"config_snapshot.json",cfg)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"config.json"))
    parser.add_argument("--out",default=str(ROOT/"data"/"free_data_audit"))
    parser.add_argument("--direct-eastmoney",action="store_true",help="Use direct connections only to Eastmoney API hosts to diagnose proxy failures.")
    parser.add_argument("--reuse-archive",default=str(ROOT/"FREE_DATA_AUDIT.zip"))
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    signature=hashlib.sha256((json.dumps(cfg,sort_keys=True)+sha(__file__)+str(args.direct_eastmoney)).encode()).hexdigest()
    out=Path(args.out)/signature[:12]
    out.mkdir(parents=True,exist_ok=True)
    reuse=reuse_archive(args.reuse_archive,out,cfg)
    write_json(out/"reuse_status.json",reuse)
    print(f"Verified/reused {reuse['reused_tables']} previous response tables.",flush=True)
    print("Eastmoney mode: "+("direct API connection, process-local" if args.direct_eastmoney else "existing environment routing"),flush=True)
    import akshare as ak
    import requests
    # Apply finite timeouts to AKShare's ordinary requests calls without replacing sessions/identities.
    original=requests.sessions.Session.request
    requests.sessions.Session.request=request_wrapper(original,args.direct_eastmoney)
    try:
        rows=run_check(cfg,out,ak)
    finally:
        requests.sessions.Session.request=original
    write_json(out/"execution.json",{"python":sys.version,"platform":platform.platform(),
               "akshare":getattr(ak,"__version__","unknown"),"pandas":pd.__version__,"numpy":np.__version__,
               "script_sha256":sha(__file__),"formal_data_modified":False,"network_source":"Eastmoney/Sina; no Yahoo"})
    execution=json.loads((out/"execution.json").read_text(encoding="utf-8"))
    execution.update(direct_eastmoney=args.direct_eastmoney,windows_proxy_config_changed=False)
    write_json(out/"execution.json",execution)
    archive=ROOT/("FREE_DATA_RECOVERY.zip" if args.direct_eastmoney else "FREE_DATA_AUDIT.zip")
    temp=Path(str(archive)+".tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                z.write(p,p.relative_to(out))
        z.write(__file__,"collection_script.py")
    os.replace(temp,archive)
    print(f"AUDIT COMPLETE: {archive}",flush=True)
    print(f"Send {archive.name} back for instrument/return-basis review. This is not a completed training run.",flush=True)
    print(f"{len(rows)} collection records; formal data/raw was not changed.",flush=True)


if __name__=="__main__":
    main()
