import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))


def fingerprint(cfg, raw_dir):
    code = {str(p.relative_to(ROOT)): digest(p) for p in sorted(ROOT.glob("uqbench/*.py"))}
    code["run.py"] = digest(ROOT / "run.py")
    data = {m["id"]: digest(Path(raw_dir) / (m["id"] + ".csv")) for m in cfg["markets"]}
    payload = {"config": cfg, "code_sha256": code, "data_sha256": data}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def source_id(cfg):
    return next(m["id"] for m in cfg["markets"] if m.get("source"))


def environment():
    import importlib.metadata as md
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "yfinance", "matplotlib"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "versions": versions}
