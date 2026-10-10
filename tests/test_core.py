import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from uqbench.common import read_config
from uqbench.data import MarketData, canonical, features, prepare, verify_prepared
from uqbench.distributions import QCOLS, baseline, fit_garch, isotonic_fit, isotonic_quantiles, make_quantiles, online_quantiles, split_conformal_fit
from uqbench.metrics import score
from uqbench.report import paired_bootstrap, summarize


def mock_frame(seed=4, n=2500):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    q = make_quantiles(np.zeros(n), np.ones(n))
    d = pd.DataFrame({"target_pct": y, "mean": np.zeros(n), "epi_sd": np.zeros(n)})
    for j,c in enumerate(QCOLS):
        d[c] = q[:,j]
    return d


class CoreTests(unittest.TestCase):
    def test_source_only_scaler_label_split_and_snapshot_validation(self):
        cfg=read_config()
        cfg["markets"]=[m for m in cfg["markets"] if m["id"] in ("SP500", "OilETF")]
        dates=pd.bdate_range("2008-01-01","2025-12-31")
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/"raw"
            raw.mkdir()
            for i,m in enumerate(cfg["markets"]):
                rng=np.random.default_rng(23+i)
                close=100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
                pd.DataFrame({"Date":dates,"Close":close}).to_csv(raw/(m["id"]+".csv"),index=False)
            a,b=Path(temp)/"a",Path(temp)/"b"
            prepare(cfg,raw,a)
            verify_prepared(cfg,raw,a)
            s=cfg["splits"][0]
            td=MarketData(a/"historical"/"SP500.npz",s,"train",60)
            self.assertTrue((td.target_dates <= s["train_end"]).all())
            self.assertTrue((td.origin_dates < td.target_dates).all())
            oil=raw/"OilETF.csv"
            d=pd.read_csv(oil)
            d["Close"]=100*np.exp(np.arange(len(d))*.0003+np.sin(np.arange(len(d))*.07)*.01)
            d.to_csv(oil,index=False)
            source=raw/"SP500.csv"
            d=pd.read_csv(source)
            d.loc[d.Date>"2018-12-31","Close"]*=2
            d.to_csv(source,index=False)
            prepare(cfg,raw,b)
            self.assertEqual(json.loads((a/"historical"/"normalizer.json").read_text()),
                             json.loads((b/"historical"/"normalizer.json").read_text()))
            with self.assertRaises(RuntimeError):
                verify_prepared(cfg,raw,a)

    def test_causal_features_do_not_change_with_future_prices(self):
        rng = np.random.default_rng(19)
        dates = pd.bdate_range("2010-01-01", periods=450)
        prices = 100*np.exp(np.cumsum(rng.normal(0,.01,450)))
        raw = pd.DataFrame({"Date": dates, "Close": prices, "High": prices*1.01,
                            "Low": prices*.99, "Volume": rng.integers(100,500,450)})
        a = features(canonical(raw))
        raw.loc[300:, "Close"] *= 1.5
        raw.loc[300:, "High"] *= 1.5
        raw.loc[300:, "Low"] *= 1.5
        b = features(canonical(raw))
        before = a.index < dates[300]
        # Labels at the last pre-change origin may change; origin features must not.
        np.testing.assert_allclose(a.loc[before].iloc[:,:30], b.loc[before].iloc[:,:30], atol=1e-12)
        self.assertEqual(a.shape[1],34)

    def test_no_price_fill_or_duplicate_dates(self):
        with self.assertRaises(ValueError):
            canonical(pd.DataFrame({"Date":["2020-01-01","2020-01-01"],"Close":[1,2]}))
        with self.assertRaises(ValueError):
            canonical(pd.DataFrame({"Date":["2020-01-01","2020-01-02"],"Close":[1,-1]}))
        d=canonical(pd.DataFrame({"Date":["2020-01-01","2020-01-02"],"Close":[1,2],"High":[np.nan,3]}))
        self.assertEqual(d.high.iloc[0],1)

    def test_garch_fit_stationarity_and_finite_predictions(self):
        rng=np.random.default_rng(7)
        params=fit_garch(rng.normal(size=800))
        self.assertGreater(params["omega"],0)
        self.assertLess(params["alpha"]+params["beta"],1)
        class Fixture:
            y=np.zeros(10)
            scale=np.ones(10)
            origin_index=np.arange(400,410)
            returns=rng.normal(size=500)
            hist_scale=np.ones(500)
            def history(self,j,length=250):
                k=self.origin_index[j]
                return self.returns[k-length+1:k+1]
        d=Fixture()
        st={"target_sd":1.,"garch":params}
        for method in ("ZeroGaussian","RollingEmpirical","EWMAGaussian","GARCH-Norm"):
            a=baseline(d,method,st)
            d.returns[450:]=10000
            b=baseline(d,method,st)
            np.testing.assert_allclose(a["quantiles"],b["quantiles"])
            self.assertTrue(np.isfinite(a["quantiles"]).all())

    def test_calibration_and_zero_skill_reference(self):
        d = mock_frame(n=20000)
        s = score(d)
        self.assertLess(s["ECE"],.015)
        self.assertLess(abs(s["PICP90"]-.9),.015)
        self.assertAlmostEqual(s["skill_vs_zero"],0)
        self.assertTrue(np.isnan(s["UA_epistemic"]))

    def test_wider_intervals_have_cost(self):
        a = mock_frame(n=20000)
        b = a.copy()
        b[QCOLS] *= 4
        self.assertGreater(score(b)["WIS_pct"],score(a)["WIS_pct"])

    def test_online_emits_before_current_outcome(self):
        y = np.arange(50)/10
        initial = split_conformal_fit(np.linspace(-2,2,100),np.zeros(100),np.ones(100))
        a,_ = online_quantiles(y,np.zeros(50),np.ones(50),initial,.1)
        y[20] = -100
        b,_ = online_quantiles(y,np.zeros(50),np.ones(50),initial,.1)
        np.testing.assert_allclose(a[:21],b[:21])
        self.assertFalse(np.array_equal(a[21:],b[21:]))
        self.assertTrue((np.diff(b,axis=1)>=0).all())

    def test_posthoc_maps_are_finite_monotone(self):
        rng=np.random.default_rng(4)
        cal=isotonic_fit(rng.normal(size=300),np.zeros(300),np.ones(300))
        q=isotonic_quantiles(np.zeros(10),np.ones(10),cal)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue((np.diff(q,axis=1)>=0).all())

    def test_identical_methods_block_ci_is_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg={"seeds":[0,1],"splits":[{"id":"fixture"}],"analysis":{"bootstrap_reps":30,"block_days":5,"stress_windows":[]}}
            for seed in cfg["seeds"]:
                parts=[]
                for market in ("A","B"):
                    d=mock_frame(seed+9,n=80)
                    d["market"],d["asset_class"]=market,"equity"
                    d["target_date"]=pd.bdate_range("2020-01-01",periods=len(d)).strftime("%Y-%m-%d")
                    parts.append(d)
                p=Path(temp)/"predictions"/"fixture"/f"seed_{seed}"
                p.mkdir(parents=True)
                for name in ("sameA","sameB"):
                    pd.concat(parts).to_csv(p/(name+".csv.gz"),index=False,compression="gzip")
            results=paired_bootstrap(temp,cfg,"fixture","sameA","sameB")
            for row in results:
                self.assertAlmostEqual(row["difference_A_minus_B"],0)
                self.assertAlmostEqual(row["ci_low"],0)
                self.assertAlmostEqual(row["ci_high"],0)
            summarize(temp,cfg)
            self.assertTrue((Path(temp)/"analysis"/"tables.md").exists())


if __name__=="__main__":
    unittest.main()
