"""Offline identity, hash and period checks for the delivered real-data panel."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from free_data_check import coverage, normalize, sha


def verify(config, raw):
    config, raw = Path(config), Path(raw)
    cfg = json.loads(config.read_text(encoding="utf-8"))
    manifest = json.loads((raw / "snapshot_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic") or manifest.get("series_count") != 13:
        raise ValueError("Not the frozen real-data panel")
    if manifest["config_sha256"] != hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest():
        raise ValueError("Frozen protocol/config changed; preserve this run and define a new protocol")
    expected = {m["id"] + suffix for m in cfg["markets"] for suffix in (".csv", ".json")}
    if set(manifest["files"]) != expected:
        raise ValueError("Snapshot membership mismatch")
    for name, digest in manifest["files"].items():
        if sha(raw / name) != digest:
            raise ValueError("Frozen snapshot hash mismatch: " + name)
    for m in cfg["markets"]:
        p = raw / (m["id"] + ".csv")
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("synthetic") or any(meta.get(k) != m.get(k) for k in ("id", "ticker", "instrument", "source")):
            raise ValueError("Market identity mismatch: " + m["id"])
        frame = normalize(pd.read_csv(p))
        _, detail = coverage(frame, cfg, m)
        if not detail["coverage_ok"]:
            raise ValueError(str(detail))
    print("OFFLINE SNAPSHOT VERIFIED: 1 source + 12 targets, both temporal splits.", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(root / "config_free_panel.json"))
    p.add_argument("--raw", default=str(root / "data" / "free_panel_raw"))
    a = p.parse_args()
    verify(a.config, a.raw)
