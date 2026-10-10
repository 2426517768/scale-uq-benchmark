# Data

The panel `free-panel-frozen-2.0` contains thirteen daily series from 2 January 2008 to 31 December 2025: the source SPY and twelve targets. It was frozen on 30 September 2026.

## What is included

| File | Contents |
| --- | --- |
| `free_panel_raw/snapshot_manifest.json` | SHA256 of every series file and its metadata, and of the frozen configuration |
| `free_panel_raw/<series>.json` | For every series: identity, provider, provider symbol, price basis, split factors, row count, first and last date, test-period counts, and SHA256 of the CSV and of each raw provider response |
| `free_panel_raw/EURUSD.csv`, `GBPUSD.csv`, `USDJPY.csv`, `AUDUSD.csv` | Exchange-rate series derived from ECB reference rates |

## What is not included and why

The nine price series obtained through AKShare are not redistributed, because the terms of the underlying providers do not clearly permit redistribution:

| Series ID | Instrument | Provider |
| --- | --- | --- |
| SP500ETF | SPY | Sina Finance via AKShare |
| UKEquityETF | EWU | Sina Finance via AKShare |
| JapanEquityETF | EWJ | Sina Finance via AKShare |
| USTreasuryETF | IEF | Sina Finance via AKShare |
| USAggregateBondETF | AGG | Sina Finance via AKShare |
| GoldETF | GLD | Sina Finance via AKShare |
| OilETF | USO | Sina Finance via AKShare |
| CopperETF | CPER | Sina Finance via AKShare |
| CSI300 | CSI 300 price index (sh000300) | Eastmoney via AKShare |

All numbers reported in the paper can be checked without these files from `results/` and `evidence/` (see the main README). The files are needed only to rerun training or to run `evidence/audit_evidence_panel.py`.

## Price basis

ETF returns are split-adjusted price returns that exclude cash distributions. They are not total returns. Split factors are applied from the provider's forward-adjustment factors, and the additive cash adjustment is discarded. The factors used are listed in each series' JSON file. CSI 300 is the price index.

Exchange rates use the ECB historical file of euro reference rates (https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip), quoted as units of currency per euro. The authors transformed them as follows: EUR/USD is the USD rate; GBP/USD is USD per EUR divided by GBP per EUR; USD/JPY is JPY per EUR divided by USD per EUR; AUD/USD is USD per EUR divided by AUD per EUR. Only same-date rates are combined, and nothing is forward-filled. These are official reference fixings, not tradable quotes. The Open, High, and Low columns repeat the reference rate and Volume is 0. None of these columns enter the features, which use closing prices or reference rates only.

## Rebuilding the withheld series

The scripts below are the ones used to build the frozen panel. They need `akshare` and `requests` in addition to the packages in `requirements_free_panel.txt`.

1. `python free_data_check.py --direct-eastmoney` downloads the CSI 300 index and the ETF responses and writes `FREE_DATA_RECOVERY.zip`.
2. `python free_panel_check.py` reuses that archive, downloads the ECB file and any missing Sina responses, reconstructs split-only prices, checks coverage, and writes the candidate panel to `data/free_panel_audit/<signature>/`.
3. `python promote_free_panel.py data/free_panel_audit/<signature>` copies the candidate series into `data/free_panel_raw/`. It stops if an existing file differs.
4. `python verify_free_panel.py` checks every file against `snapshot_manifest.json`.

The underlying AKShare calls are `stock_us_daily(symbol, adjust)` with `adjust` set to `""`, `"qfq"`, and `"qfq-factor"` for the ETFs, and `stock_zh_index_daily_em(symbol="sh000300")` for CSI 300. Providers may revise their histories. If step 4 reports a hash mismatch, the rebuilt panel differs from the frozen one, and results obtained from it should be reported as a new panel version.
