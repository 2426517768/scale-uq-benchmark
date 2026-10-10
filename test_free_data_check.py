import json
import tempfile
import unittest
import zipfile
import hashlib
import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from free_data_check import INDEX_CODES, ROOT, coverage, normalize, run_check, reuse_archive, request_wrapper


def prices():
    dates=pd.bdate_range("2008-01-01","2025-12-31")
    close=100*np.exp(np.sin(np.arange(len(dates))*.07)*.05)
    return pd.DataFrame({"date":dates,"open":close,"high":close*1.01,"low":close*.99,"close":close,"volume":100})


def cn_prices(code):
    return prices().rename(columns={"date":"日期","open":"今开","high":"最高","low":"最低","close":"最新价","volume":"成交量"}).assign(代码=code)


class FakeAKShare:
    def __init__(self):
        self.calls=[]
    def index_global_spot_em(self):
        return pd.DataFrame({"代码":[x[0] for x in INDEX_CODES.values()],"名称":[x[1] for x in INDEX_CODES.values()]})
    def index_global_hist_em(self,symbol):
        self.calls.append(("index",symbol))
        code=next(code for code,name in INDEX_CODES.values() if name==symbol)
        return cn_prices(code)
    def stock_zh_index_daily_em(self,**kwargs):
        self.calls.append(("china",kwargs["symbol"]))
        return prices()
    def forex_hist_em(self,symbol):
        self.calls.append(("fx",symbol))
        return cn_prices(symbol)
    def stock_us_daily(self,symbol,adjust):
        self.calls.append(("us",symbol,adjust))
        if adjust=="qfq-factor":
            return pd.DataFrame({"date":["1900-01-01","2020-04-29"],"qfq_factor":[.125,1.],"adjust":[-.5,0.]})
        d=prices()
        if adjust=="qfq":
            d.loc[0,"close"]=-1 # Valid raw response, invalid existing positive-price protocol.
        return d


class FreeDataTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/"config.json").read_text())

    def test_chinese_normalization_and_identity(self):
        d=normalize(cn_prices("SPX"),"SPX")
        self.assertEqual(list(d.columns),["Date","Open","High","Low","Close","Volume"])
        with self.assertRaisesRegex(ValueError,"identity mismatch"):
            normalize(cn_prices("NDX"),"SPX")

    def test_named_datetime_index_is_a_date(self):
        d=prices().set_index("date")
        self.assertEqual(len(normalize(d)),len(d))
        with self.assertRaisesRegex(ValueError,"No actual trading dates"):
            normalize(pd.DataFrame({"close":[1,2,3]}))

    def test_full_and_short_history_are_distinguished(self):
        d=normalize(prices())
        _,full=coverage(d,self.cfg,self.cfg["markets"][0])
        self.assertTrue(full["coverage_ok"])
        _,short=coverage(d.iloc[-1000:],self.cfg,self.cfg["markets"][0])
        self.assertFalse(short["coverage_ok"])
        self.assertTrue(short["coverage_issues"])

    def test_large_gap_detected_and_bad_price_not_dropped(self):
        d=prices()
        with self.assertRaises(ValueError):
            normalize(d.assign(close=-1))
        d=normalize(d)
        d=d[~((d.Date>="2021-05-01")&(d.Date<="2021-07-01"))]
        _,detail=coverage(d,self.cfg,self.cfg["markets"][1])
        self.assertTrue(any("gap exceeds" in x for x in detail["coverage_issues"]))

    def test_audit_caches_responses_and_never_promotes_data(self):
        with tempfile.TemporaryDirectory() as temp:
            ak=FakeAKShare()
            rows=run_check(self.cfg,temp,ak,pause=Mock())
            n=len(ak.calls)
            self.assertEqual(len(rows),21) # 4 indices + 4 FX + 4 ETF × 3 variants + German manual.
            self.assertTrue(all(not r["promoted_to_formal"] for r in rows))
            self.assertTrue(all(r["status"]=="failed" for r in rows if r["variant"]=="qfq"))
            self.assertTrue(all(r["status"]=="factor_table_saved" for r in rows if r["variant"]=="qfq_factors"))
            self.assertEqual(next(r for r in rows if r["market"]=="GermanyGovtETF")["status"],"manual_source_required")
            run_check(self.cfg,temp,ak,pause=Mock())
            self.assertEqual(len(ak.calls),n)
            audit=json.loads((Path(temp)/"audit.json").read_text())
            self.assertFalse(audit["formal_experiment_ready"])
            self.assertTrue((Path(temp)/"responses"/"OilETF_qfq.csv").exists())
            self.assertFalse((Path(temp)/"raw").exists())
            summary=pd.read_csv(Path(temp)/"coverage_summary.csv")
            self.assertEqual(len(summary),13)
            self.assertFalse(summary.formal_ready.any())

    def test_failed_market_does_not_prevent_audit_completion(self):
        ak=FakeAKShare()
        ak.index_global_hist_em=Mock(side_effect=ConnectionError("unreachable"))
        with tempfile.TemporaryDirectory() as temp:
            rows=run_check(self.cfg,temp,ak,pause=Mock())
            self.assertEqual(len(rows),21)
            self.assertTrue(json.loads((Path(temp)/"audit.json").read_text())["complete"])
            self.assertTrue(any(r["market"]=="EURUSD" and r["status"]=="candidate" for r in rows))

    def test_access_rejection_stops_requests_to_same_provider(self):
        ak=FakeAKShare()
        ak.index_global_hist_em=Mock(side_effect=RuntimeError("HTTP 429 Too Many Requests"))
        ak.forex_hist_em=Mock(side_effect=AssertionError("Should not request a rejected provider"))
        with tempfile.TemporaryDirectory() as temp:
            rows=run_check(self.cfg,temp,ak,pause=Mock())
            self.assertEqual(ak.index_global_hist_em.call_count,1)
            self.assertEqual(ak.forex_hist_em.call_count,0)
            self.assertTrue(any(r["market"]=="GoldETF" and r["variant"]=="unadjusted" and r["status"]=="adjustment_review_required" for r in rows))

    def test_archive_reuse_verifies_identity_hash_and_config(self):
        m=next(m for m in self.cfg["markets"] if m["id"]=="GoldETF")
        payload=prices().to_csv(index=False).encode()
        meta={"market":"GoldETF","variant":"unadjusted","provider":"Sina via AKShare",
              "requested_instrument":m["instrument"],"sha256":hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/"prior.zip"
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("config_snapshot.json",json.dumps(self.cfg))
                z.writestr("responses/GoldETF_unadjusted.csv",payload)
                z.writestr("responses/GoldETF_unadjusted.json",json.dumps(meta))
                z.writestr("../../unexpected.txt","never extract arbitrary entries")
            target=Path(temp)/"out"
            r=reuse_archive(archive,target,self.cfg)
            self.assertEqual(r["reused_tables"],1)
            self.assertEqual((target/"responses/GoldETF_unadjusted.csv").read_bytes(),payload)
            self.assertFalse((Path(temp)/"unexpected.txt").exists())
            other={**self.cfg,"version":"different"}
            self.assertEqual(reuse_archive(archive,Path(temp)/"other",other)["reused_tables"],0)
            meta["sha256"]="bad"
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("config_snapshot.json",json.dumps(self.cfg))
                z.writestr("responses/GoldETF_unadjusted.csv",payload)
                z.writestr("responses/GoldETF_unadjusted.json",json.dumps(meta))
            r=reuse_archive(archive,Path(temp)/"bad",self.cfg)
            self.assertEqual(r["reused_tables"],0)
            self.assertTrue(r["errors"])

    def test_direct_route_scoped_to_eastmoney_and_restores_session(self):
        class Session:
            trust_env=True
            proxies={"https":"test-only-proxy"}
        session=Session()
        def original(s,method,url,**kwargs):
            return {"trust_env":s.trust_env,"session_proxies":s.proxies.copy(),"kwargs":kwargs}
        wrapped=request_wrapper(original,True)
        with patch.dict(os.environ,{"REQUESTS_CA_BUNDLE":"test-bundle.pem"}):
            seen=wrapped(session,"GET","https://push2his.eastmoney.com/api")
            self.assertFalse(seen["trust_env"])
            self.assertEqual(seen["session_proxies"],{})
            self.assertEqual(seen["kwargs"]["verify"],"test-bundle.pem")
            self.assertEqual(seen["kwargs"]["timeout"],(10,30))
            self.assertTrue(session.trust_env)
            self.assertEqual(session.proxies,{"https":"test-only-proxy"})
            for host in ("finance.sina.com.cn","push2his.eastmoney.com.unrelated.example"):
                other=wrapped(session,"GET","https://"+host+"/api")
                self.assertTrue(other["trust_env"])

    def test_failed_direct_request_restores_session(self):
        class Session:
            trust_env=True
            proxies={"https":"test-only-proxy"}
        session=Session()
        def error(s,method,url,**kwargs):
            raise ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            request_wrapper(error,True)(session,"GET","https://push2.eastmoney.com/api")
        self.assertTrue(session.trust_env)
        self.assertEqual(session.proxies,{"https":"test-only-proxy"})


if __name__=="__main__":
    unittest.main()
