"""Windows-friendly sequential entry point; all runs bind immutable data/config/code hashes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mpl_cache"))

import argparse
import json
import sys
import traceback
import zipfile
from pathlib import Path

from uqbench.common import ROOT, environment, json_write, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "doctor", "download", "import", "prepare", "run", "summarize", "verify", "all", "smoke", "pack"])
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--prepared", default=str(ROOT / "data" / "prepared"))
    parser.add_argument("--out", default=str(ROOT / "results" / "fresh_v1"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--split", nargs="+")
    parser.add_argument("--market")
    parser.add_argument("--file")
    parser.add_argument("--provenance")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "plan":
        print("16 main configurations + 6 SCALE diagnostics; 12 target markets + source.")
        print("2 temporal splits × 5 seeds; 180 learned-network jobs, 220 market-panel forecast files.")
        print("All daily predictions, quantiles, calibrators, logs, checkpoints and analyses are retained.")
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if args.command == "doctor":
        print(json.dumps(environment(), indent=2))
        from uqbench.engine import select_device
        import torch
        d = select_device(args.allow_cpu)
        print("Working device:", d, torch.cuda.get_device_name() if d.type == "cuda" else "CPU")
        print("CUDA runtime:", torch.version.cuda, "; PyTorch:", torch.__version__)
        return
    if args.command == "smoke":
        from uqbench.selftest import smoke
        smoke(args.allow_cpu, cfg)
        return
    if args.command == "pack":
        source = Path(args.out)
        if not (source / "RUN_COMPLETE.json").exists():
            raise RuntimeError("Full run is not complete. Resume it before packaging manuscript results.")
        if (source / "RUN.lock").exists():
            raise RuntimeError("Wait for the active experiment process to finish before packing.")
        from uqbench.experiment import freeze_run, verify_complete
        from uqbench.data import verify_prepared
        freeze_run(cfg, args.raw, source)
        verify_prepared(cfg, args.raw, args.prepared)
        verify_complete(cfg, source)
        archive = source.parent / (source.name + "_RESULTS.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for p in sorted(source.rglob("*")):
                if p.is_file() and p.name != "RUN.lock" and not p.name.endswith(".tmp"):
                    z.write(p, p.relative_to(source))
            for p in sorted(Path(args.raw).glob("*")):
                if p.is_file():
                    z.write(p, Path("raw_snapshot") / p.name)
            for p in sorted(Path(args.prepared).rglob("*.json")):
                z.write(p, Path("preprocessing") / p.relative_to(args.prepared))
            for p in sorted(ROOT.glob("uqbench/*.py")) + sorted(ROOT.glob("tests/*.py")):
                z.write(p, Path("reproduction_code") / p.relative_to(ROOT))
            for pattern in ("*.py", "*.md", "*.bat", "*.txt"):
                for p in sorted(ROOT.glob(pattern)):
                    z.write(p, Path("reproduction_code") / p.name)
            z.write(args.config, "reproduction_code/config.json")
        print("Send this archive back:", archive)
        return
    from uqbench.data import download, import_csv, prepare
    if args.command == "import":
        if not all([args.market, args.file, args.provenance]):
            raise ValueError("import needs --market, --file, and --provenance; adjusted prices required.")
        import_csv(cfg, args.market, args.file, args.raw, args.provenance)
        return
    if args.command == "download":
        download(cfg, args.raw)
    if args.command == "prepare":
        prepare(cfg, args.raw, args.prepared)
    if args.command in ("run", "all", "summarize", "verify"):
        from uqbench.experiment import freeze_run, run_experiments, verify_complete
        from uqbench.report import summarize
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=True)
        lock = root / "RUN.lock"
        if lock.exists():
            raise RuntimeError(f"A run lock exists: {lock}. If the previous process has stopped, delete only RUN.lock then resume.")
        lock.write_text(str(os.getpid()), encoding="ascii")
        try:
            if args.command == "all":
                download(cfg, args.raw)
                prepare(cfg, args.raw, args.prepared)
            freeze_run(cfg, args.raw, root)
            if args.command in ("run", "all"):
                run_experiments(cfg, args.raw, args.prepared, root, args.allow_cpu, args.seed, args.split)
            if args.command in ("summarize", "verify", "all"):
                complete = verify_complete(cfg, root)
                if args.command != "verify":
                    summarize(root, cfg)
                    json_write(root / "RUN_COMPLETE.json", {**complete, "synthetic": False,
                               "analysis_complete": True, "results_origin": "new experiments only"})
                print("Complete:", complete)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nSTOPPED. No missing market/method is treated as a completed experiment. Fix the error and run again to resume.", file=sys.stderr)
        sys.exit(1)
