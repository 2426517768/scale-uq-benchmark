"""Network-free regression tests; they do not assert that Yahoo is reachable."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from uqbench.data import download
from uqbench.download import DownloadRateLimited, fetch_history


class YFRateLimitError(Exception):
    pass


def fixture():
    return pd.DataFrame({"Close": 100+np.arange(120)*.1},
                        index=pd.bdate_range("2020-01-01", periods=120))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"download_start": "2008-01-01", "download_end_exclusive": "2026-01-01",
                    "markets": [{"id": "A", "ticker": "^A"}, {"id": "B", "ticker": "^B"}]}

    def test_rate_limit_is_preserved_and_waits_are_long(self):
        history = Mock(side_effect=[YFRateLimitError("Too Many Requests"), fixture()])
        yf = types.SimpleNamespace(Ticker=Mock(return_value=types.SimpleNamespace(history=history)))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"yfinance": yf}), \
                patch("uqbench.download.wait_seconds") as wait:
            result = fetch_history(self.cfg["markets"][0], self.cfg, temp)
            self.assertEqual(len(result), 120)
            wait.assert_called_once_with(300)
            self.assertEqual(history.call_count, 2)
            self.assertTrue(history.call_args.kwargs["raise_errors"])
            self.assertTrue(history.call_args.kwargs["auto_adjust"])
            self.assertFalse((Path(temp)/"yahoo_cooldown.json").exists())

    def test_persistent_limit_stops_entire_provider(self):
        history = Mock(side_effect=YFRateLimitError("Too Many Requests"))
        yf = types.SimpleNamespace(Ticker=Mock(return_value=types.SimpleNamespace(history=history)))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"yfinance": yf}), \
                patch("uqbench.download.wait_seconds"):
            with self.assertRaises(DownloadRateLimited):
                download(self.cfg, temp)
            self.assertEqual(history.call_count, 3)
            self.assertTrue(all(c.args[0] == "^A" for c in yf.Ticker.call_args_list))
            state = json.loads((Path(temp)/"download_status.json").read_text())
            self.assertFalse(state["complete"])
            self.assertEqual(state["failures"][0]["market"], "A")
            self.assertTrue((Path(temp)/"yahoo_cooldown.json").exists())
            self.assertFalse((Path(temp)/"A.csv").exists())

    def test_successful_snapshots_are_not_downloaded_again(self):
        history = Mock(return_value=fixture())
        yf = types.SimpleNamespace(Ticker=Mock(return_value=types.SimpleNamespace(history=history)))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"yfinance": yf}), \
                patch("uqbench.data.wait_seconds", create=True), patch("uqbench.download.wait_seconds"):
            download(self.cfg, temp)
            first = (Path(temp)/"A.csv").read_bytes()
            download(self.cfg, temp)
            self.assertEqual(history.call_count, 2)
            self.assertEqual(first, (Path(temp)/"A.csv").read_bytes())
            state=json.loads((Path(temp)/"download_status.json").read_text())
            self.assertTrue(state["complete"])
            self.assertEqual(state["verified_this_pass"], 2)

    def test_non_rate_failure_collects_missing_market(self):
        histories = {"^A": Mock(side_effect=ValueError("no history")), "^B": Mock(return_value=fixture())}
        yf=types.SimpleNamespace(Ticker=lambda symbol: types.SimpleNamespace(history=histories[symbol]))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"yfinance": yf}), \
                patch("uqbench.download.wait_seconds"):
            with self.assertRaisesRegex(RuntimeError, "Missing required markets: A"):
                download(self.cfg, temp)
            self.assertTrue((Path(temp)/"B.csv").exists())
            self.assertFalse((Path(temp)/"A.csv").exists())

    def test_saved_cooldown_survives_restart(self):
        history=Mock(return_value=fixture())
        yf=types.SimpleNamespace(Ticker=Mock(return_value=types.SimpleNamespace(history=history)))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"yfinance": yf}), \
                patch("uqbench.download.time.time", return_value=1000), patch("uqbench.download.wait_seconds") as wait:
            (Path(temp)/"yahoo_cooldown.json").write_text(json.dumps({"not_before_unix": 1200}))
            fetch_history(self.cfg["markets"][0], self.cfg, temp)
            wait.assert_called_once_with(200)

