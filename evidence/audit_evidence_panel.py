"""Check compact daily evidence against the frozen raw CSV panel; no training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def audit(evidence, raw, output):
    with zipfile.ZipFile(evidence) as z:
        manifest = json.loads(z.read("manifest.json"))
        cfg = manifest["config"]
        info = json.loads(z.read("evidence_manifest.json"))
        source = next(m["id"] for m in cfg["markets"] if m.get("source"))
        targets = sorted(m["id"] for m in cfg["markets"] if m["id"] != source)
        methods = ["MCDropout", "DeepEnsembles", "SCALE-R", "OnlineQuantile", "EWMAGaussian", "ZeroGaussian", "SCALE-nosim"]
        names = {f"predictions/{s['id']}/seed_{seed}/{method}.csv.gz"
                 for s in cfg["splits"] for seed in cfg["seeds"]
                 for method in ["GaussianTCN", "MCDropout", "DeepEnsembles", "Evidential", "RandomNoise", "FeaturePGD", "MixUp", "AdvReg", "SCALE-R", "Isotonic", "SplitConformal", "OnlineQuantile", "ZeroGaussian", "RollingEmpirical", "EWMAGaussian", "GARCH-Norm", "SCALE-noagg", "SCALE-noentropy", "SCALE-noadv", "SCALE-nosim", "SCALE-k8", "SCALE-k32"]}
        assert set(info["prediction_sha256"]) == names
        assert info["source_excluded"] == source
        raws = {}
        for market in cfg["markets"]:
            mid = market["id"]
            payload = (raw/(mid+".csv")).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == manifest["data_sha256"][mid]
            frame = pd.read_csv(io.BytesIO(payload))
            frame["return_pct"] = 100 * frame.Close.pct_change(fill_method=None)
            raws[mid] = frame
        period_checks = []
        samples = []
        for split in cfg["splits"]:
            sid = split["id"]
            with np.load(io.BytesIO(z.read(sid+"_daily_sufficient_statistics.npz")), allow_pickle=False) as f:
                a = {k:f[k] for k in f.files}
            assert a["methods"].tolist() == methods
            assert a["seeds"].tolist() == cfg["seeds"]
            assert a["markets"].tolist() == targets
            assert a["dates"].tolist() == sorted(set(a["dates"].tolist()))
            assert a["valid"].dtype == np.bool_
            shape = (len(methods), len(cfg["seeds"]), len(a["dates"]), len(targets))
            for key in ("mse", "wis", "coverage90", "quantile_hits"):
                assert a[key].shape == shape + ((9,) if key == "quantile_hits" else ())
                assert np.isfinite(a[key]).all()
            assert (a["mse"] >= 0).all() and (a["wis"] >= 0).all()
            assert np.isin(a["coverage90"], [0,1]).all()
            assert np.isin(a["quantile_hits"], [0,1]).all()
            assert (np.diff(a["quantile_hits"].astype(np.int8), axis=-1) >= 0).all()
            errors = []
            for j, mid in enumerate(targets):
                frame = raws[mid]
                sub = frame[frame.Date.between(split["test_start"], split["test_end"])]
                assert a["dates"][a["valid"][:,j]].tolist() == sub.Date.tolist()
                error = float(np.max(np.abs(a["target_pct"][a["valid"][:,j],j]-sub.return_pct.to_numpy())))
                assert error < 1e-8
                errors.append(error)
                samples.append({"split":sid,"market":mid,"observations":len(sub),"maximum_realized_return_difference_pct":error})
            i, j = methods.index("SCALE-R"), methods.index("SCALE-nosim")
            equality = {key:bool(np.array_equal(a[key][i], a[key][j])) for key in ("mse", "wis", "coverage90", "quantile_hits")}
            assert all(equality.values())
            period_checks.append({"split":sid,"union_observation_dates":len(a["dates"]),"target_observations_per_seed":int(a["valid"].sum()),
                                  "maximum_realized_return_difference_pct":max(errors),"SCALE_R_nosim_daily_statistics_exactly_equal":equality})
        comparisons = info["scale_similarity_ablation_prediction_comparison"]
        assert {(c["split"],c["seed"]) for c in comparisons} == {(s["id"],seed) for s in cfg["splits"] for seed in cfg["seeds"]}
        assert all(c["all_prediction_columns_exactly_equal"] and not c["unequal_columns"] for c in comparisons)
        result = {"status":"passed","evidence_sha256":hashlib.sha256(evidence.read_bytes()).hexdigest(),
                  "frozen_raw_hashes_checked":len(raws),"exporter_recorded_prediction_hashes":len(names),"period_checks":period_checks,
                  "exporter_recorded_full_column_comparisons":len(comparisons),
                  "exporter_recorded_comparison_rows_including_source":sum(c["rows_including_source"] for c in comparisons),
                  "scope":"Daily sufficient statistics independently checked against frozen realized returns; full quantile-column equality is reported by the exporter, not separately reconstructed here."}
    output.mkdir(parents=True,exist_ok=True)
    (output/"panel_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(samples).to_csv(output/"realized_returns_and_calendar_checks.csv",index=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",type=Path)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("evidence_verification"))
    args=parser.parse_args()
    audit(args.evidence,args.raw,args.output)
