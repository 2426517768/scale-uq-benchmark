"""Freeze the reviewed 13-series panel as a separate price/reference-return protocol."""
import argparse
import hashlib
import json
from pathlib import Path

from free_data_check import sha, write_json


def freeze(audit_dir, root):
    audit_dir, root = Path(audit_dir), Path(root)
    cfg = json.loads((audit_dir / "config_candidate.json").read_text(encoding="utf-8"))
    audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
    expected = {m["id"] for m in cfg["markets"]}
    rows = {r["market"]: r for r in audit["records"]}
    if (len(expected) != 13 or set(rows) != expected or len(audit["records"]) != 13
            or not audit["all_13_coverage_ok"] or not all(r.get("coverage_ok") for r in rows.values())
            or cfg.get("feature_mode") != "close_only" or cfg.get("feature_count") != 28):
        raise ValueError("Panel coverage/identity/feature audit incomplete")
    cfg["version"] = "free-panel-frozen-2.0"
    cfg["protocol"]["status"] = "frozen_real_data; research outcomes not yet computed"
    cfg["protocol"]["quality_review"] = "OHLC anomalies retained in source audit; OHLCV are excluded from all predictors"
    raw = root / "data" / "free_panel_raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"synthetic": False, "series_count": 13, "source_count": 1, "target_count": 12,
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "audit_sha256": sha(audit_dir / "audit.json"), "files": {}}
    for market in cfg["markets"]:
        mid, record = market["id"], rows[market["id"]]
        relative = Path(record["candidate"])
        if relative.parts != ("candidates", mid + ".csv"):
            raise ValueError("Unexpected candidate filename")
        source = audit_dir / relative
        if sha(source) != record["candidate_sha256"]:
            raise ValueError("Candidate snapshot changed: " + mid)
        destination = raw / (mid + ".csv")
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError("Frozen raw snapshot differs; choose a new root")
        destination.write_bytes(source.read_bytes())
        meta = {**market, "synthetic": False, "sha256": sha(destination),
                "provider": record["provider"], "audit_record": record,
                "adjustment": cfg["protocol"]["fx_basis"] if market["class"] == "fx" else
                              ("CSI300 price index" if mid == "CSI300" else cfg["protocol"]["etf_return_basis"]),
                "frozen_utc": "2026-09-30", "raw_dataset": "free-panel-frozen-2.0"}
        mp = destination.with_suffix(".json")
        write_json(mp, meta)
        manifest["files"][destination.name], manifest["files"][mp.name] = sha(destination), sha(mp)
    write_json(raw / "snapshot_manifest.json", manifest)
    write_json(root / "config_free_panel.json", cfg)
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit_dir")
    p.add_argument("--root", default=str(Path(__file__).resolve().parent))
    a = p.parse_args()
    freeze(a.audit_dir, a.root)
