import io
import json
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from free_panel_check import ROOT, ecb_cross_rates, panel_config, split_only_prices


class ReferenceRateTests(unittest.TestCase):
    def fixture(self):
        return b"Date,USD,GBP,JPY,AUD,\n2020-01-03,1.2,0.8,120,1.5,\n2020-01-02,1.1,0.9,110,1.6,\n"

    def test_currency_directions_and_same_date(self):
        r = ecb_cross_rates(self.fixture())
        self.assertAlmostEqual(r["GBPUSD"].Close.iloc[1], 1.5)
        self.assertAlmostEqual(r["USDJPY"].Close.iloc[1], 100.)
        self.assertAlmostEqual(r["AUDUSD"].Close.iloc[1], .8)
        self.assertEqual(r["EURUSD"].Date.iloc[0], pd.Timestamp("2020-01-02"))
        self.assertTrue((r["EURUSD"].High == r["EURUSD"].Close).all())
        self.assertTrue((r["EURUSD"].Volume == 0).all())

    def test_published_zip_format(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("eurofxref-hist.csv", self.fixture())
        self.assertEqual(len(ecb_cross_rates(b.getvalue())["EURUSD"]), 2)

    def test_missing_or_nonpositive_rejected(self):
        for replacement in (b"0", b"N/A", b"-1"):
            with self.assertRaises(ValueError):
                ecb_cross_rates(self.fixture().replace(b"1.5", replacement))

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            ecb_cross_rates(self.fixture().replace(b"2020-01-03", b"2020-01-02"))


class PriceBasisTests(unittest.TestCase):
    def prices(self):
        raw = pd.DataFrame({"date": ["2020-04-28", "2020-04-29"],
                            "open": [2., 16.], "high": [2., 16.], "low": [2., 16.],
                            "close": [2., 16.], "volume": [20., 30.]})
        factors = pd.DataFrame({"date": ["1970-01-01", "2020-04-29"],
                                "qfq_factor": [8., 1.], "adjust": [-1., 0.]})
        qfq = raw.copy()
        qfq[["open", "high", "low", "close"]] = [[15.] * 4, [16.] * 4]
        return raw, qfq, factors

    def test_split_removed_cash_not_added(self):
        result, detail = split_only_prices(*self.prices())
        np.testing.assert_array_equal(result.Close.to_numpy(), [16., 16.])
        self.assertEqual(result.Close.pct_change().iloc[1], 0.)
        self.assertEqual(detail["affine_qfq_max_abs_residual"], 0.)

    def test_future_cash_adjustments_do_not_change_price_returns(self):
        raw, q, f = self.prices()
        base, _ = split_only_prices(raw, q, f)
        f["adjust"] -= 10.
        q[["open", "high", "low", "close"]] -= 10.
        later, _ = split_only_prices(raw, q, f)
        np.testing.assert_array_equal(base.Close.to_numpy(), later.Close.to_numpy())

    def test_factor_error_and_date_error_rejected(self):
        raw, q, f = self.prices()
        q.loc[0, "close"] += .1
        with self.assertRaises(ValueError):
            split_only_prices(raw, q, f)
        raw, q, f = self.prices()
        q.loc[0, "date"] = "2020-04-27"
        with self.assertRaises(ValueError):
            split_only_prices(raw, q, f)

    def test_panel_identity_count_and_original_unchanged(self):
        original = json.loads((ROOT / "config.json").read_text())
        before = json.dumps(original, sort_keys=True)
        cfg = panel_config(original)
        self.assertEqual(len(cfg["markets"]), 13)
        self.assertEqual([m["id"] for m in cfg["markets"] if m.get("source")], ["SP500ETF"])
        self.assertNotIn("GermanyGovtETF", [m["id"] for m in cfg["markets"]])
        self.assertEqual(before, json.dumps(original, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
