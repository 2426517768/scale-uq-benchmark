import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from uqbench.data import canonical, features
from uqbench.distributions import make_quantiles, QCOLS
from uqbench.metrics import score
from uqbench.report import summarize


class CloseOnlyTests(unittest.TestCase):
    def test_intraday_and_volume_do_not_enter_close_only_features(self):
        rng = np.random.default_rng(34)
        close = 100 * np.exp(np.cumsum(rng.normal(0, .01, 450)))
        raw = pd.DataFrame({"Date": pd.bdate_range("2015-01-01", periods=450),
                            "Close": close, "High": close, "Low": close, "Volume": 0})
        before = features(canonical(raw), "close_only")
        raw["High"], raw["Low"], raw["Volume"] = close * 2, close / 2, 999999
        after = features(canonical(raw), "close_only")
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.shape[1], 32)
        self.assertNotIn("range_scaled", before)
        self.assertNotIn("volume_change", before)

    def test_new_source_excluded_from_macro_and_figures(self):
        cfg = {"markets": [{"id": "SP500ETF", "source": True}, {"id": "Target"}],
               "seeds": [0], "splits": [{"id": "fixture"}],
               "analysis": {"stress_windows": [], "bootstrap_reps": 5, "block_days": 5}}
        rng = np.random.default_rng(5)
        y = rng.normal(size=80)
        target = pd.DataFrame({"market": "Target", "asset_class": "equity", "target_pct": y,
                               "mean": 0., "epi_sd": 0.,
                               "target_date": pd.bdate_range("2020-01-01", periods=80).strftime("%Y-%m-%d")})
        for j, name in enumerate(QCOLS):
            target[name] = make_quantiles(np.zeros(80), np.ones(80))[:, j]
        source = target.copy()
        source["market"], source["mean"] = "SP500ETF", 100.
        with tempfile.TemporaryDirectory() as tmp, patch("uqbench.report.figures") as figure:
            p = Path(tmp) / "predictions/fixture/seed_0"
            p.mkdir(parents=True)
            pd.concat([target, source]).to_csv(p / "GaussianTCN.csv.gz", index=False, compression="gzip")
            summarize(tmp, cfg)
            macro = pd.read_csv(Path(tmp) / "analysis/macro_per_seed.csv")
            self.assertAlmostEqual(macro.nRMSE.iloc[0], score(target)["nRMSE"])
            self.assertEqual(figure.call_args.args[2], "SP500ETF")


if __name__ == "__main__":
    unittest.main()
