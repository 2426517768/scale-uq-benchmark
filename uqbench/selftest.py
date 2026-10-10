"""Mandatory installed-hardware smoke run. Its synthetic outputs are never manuscript data."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, json_write, read_config, source_id
from .data import MarketData, prepare


def synthetic_prices(cfg, raw):
    raw=Path(raw)
    raw.mkdir(parents=True,exist_ok=True)
    dates=pd.bdate_range(cfg["download_start"],"2025-12-31")
    for i,m in enumerate(cfg["markets"]):
        rng=np.random.default_rng(333+i)
        close=100*np.exp(np.cumsum(rng.normal(0,.005,len(dates))))
        df=pd.DataFrame({"Date":dates,"Open":close,"High":close*1.002,"Low":close*.998,
                         "Close":close,"Volume":rng.integers(1000,5000,len(dates))})
        df.to_csv(raw/(m["id"]+".csv"),index=False)
        json_write(raw/(m["id"]+".json"),{"synthetic":True,"purpose":"hardware smoke only"})


def limit_data(data,n):
    result=copy.copy(data)
    for k in ("origin_index","origin_dates","target_dates","y","target_pct","scale"):
        setattr(result,k,getattr(data,k)[:n])
    return result


def smoke(allow_cpu=False, configuration=None):
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Core causality/metric tests failed.")
    from .engine import LEARNED, ensemble_predict, predict_normalized, select_device, train
    import torch
    device=select_device(allow_cpu)
    cfg=copy.deepcopy(configuration or read_config())
    cfg["training"].update(epochs=2,patience=2,batch_size=16,eval_batch_size=16,mc_samples=3,pgd_steps=1)
    cfg["scale"]["bank_size"]=32
    import hashlib
    code_hash=hashlib.sha256(("".join(digest(p) for p in sorted((ROOT/"uqbench").glob("*.py")))+json.dumps(cfg,sort_keys=True)).encode()).hexdigest()[:12]
    out=ROOT/"results"/"SYNTHETIC_SMOKE_NOT_FOR_PAPER"/code_hash
    synthetic_prices(cfg,out/"raw")
    prepare(cfg,out/"raw",out/"prepared")
    split=cfg["splits"][0]
    path=out/"prepared"/split["id"]
    td=limit_data(MarketData(path/(source_id(cfg)+".npz"),split,"train",cfg["window"]),32)
    vd=limit_data(MarketData(path/(source_id(cfg)+".npz"),split,"val",cfg["window"]),16)
    ed=limit_data(MarketData(path/"OilETF.npz",split,"test",cfg["window"]),16)
    trained={}
    for method in LEARNED:
        trained[method]=train(cfg,method,0,td,vd,out/"models"/method,device)
        p=predict_normalized(trained[method],ed,cfg,device)
        if not all(np.isfinite(p[k]).all() for k in ("mean","sd","quantiles")):
            raise FloatingPointError(f"Smoke forecast failed: {method}")
        if (np.diff(p["quantiles"],axis=1)<0).any():
            raise ValueError(f"Crossing smoke quantiles: {method}")
    members=[train(cfg,f"Ensemble-{i}",0,td,vd,out/"models"/f"Ensemble-{i}",device) for i in range(2)]
    ens=ensemble_predict(members,ed,cfg,device,0)
    mc=predict_normalized(trained["GaussianTCN"],ed,cfg,device,stochastic=True)
    assert np.isfinite(ens["sd"]).all() and np.isfinite(mc["sd"]).all()
    assert (mc["epi_sd"]>0).any()
    # A completed checkpoint must reload to the same prediction.
    reloaded=train(cfg,"GaussianTCN",0,td,vd,out/"models"/"GaussianTCN",device)
    p1=predict_normalized(reloaded,ed,cfg,device)
    p2=predict_normalized(trained["GaussianTCN"],ed,cfg,device)
    np.testing.assert_allclose(p1["mean"],p2["mean"],atol=1e-6)
    partial_cfg=copy.deepcopy(cfg)
    partial_cfg["training"]["epochs"]=1
    partial_dir=out/"resume_check"
    if not (partial_dir/"RESUME_CHECK_PASSED.json").exists():
        train(partial_cfg,"GaussianTCN",0,td,vd,partial_dir,device)
        (partial_dir/"done.json").unlink(missing_ok=True)
        resumed=train(cfg,"GaussianTCN",0,td,vd,partial_dir,device)
        pr=predict_normalized(resumed,ed,cfg,device)
        np.testing.assert_allclose(pr["mean"],p2["mean"],atol=1e-6)
        json_write(partial_dir/"RESUME_CHECK_PASSED.json",{"synthetic":True})
    json_write(out/"SMOKE_PASSED.json",{"synthetic":True,"core_tests":result.testsRun,
               "learned_variants_checked":len(LEARNED)+2,"device":str(device),
               "scope":"tiny data, 2 epochs; verifies execution paths, not research performance"})
    print("ALL SMOKE CHECKS PASSED. Synthetic outputs must not be used in the manuscript.",flush=True)
