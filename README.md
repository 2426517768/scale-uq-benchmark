# SCALE uncertainty quantification benchmark for daily financial returns

Code, frozen protocol, results, and verification files for the manuscript

> Ouyang W, Ren Z, Zheng X, Xu Z. Uncertainty quantification for daily financial returns under cross-asset transfer: A source-only benchmark. Submitted to PLOS ONE.

The benchmark trains every learned configuration on one source series (SPY) and evaluates it without fine-tuning on twelve target series (EWU, EWJ, CSI 300, IEF, AGG, GLD, USO, CPER, EUR/USD, GBP/USD, USD/JPY, and AUD/USD). Two chronological splits are used: train 2010–2018, validate 2019, and test 2020–2022; and train 2013–2021, validate 2022, and test 2023–2025. There are sixteen main configurations and six SCALE-R diagnostic variants, each run with five seeds. The full protocol is in `config_free_panel.json` (panel version `free-panel-frozen-2.0`) and is described in the Materials and methods section of the paper.

## Repository layout

| Path | Contents |
| --- | --- |
| `uqbench/` | Benchmark package: data preparation, features, models, training, metrics, and reports |
| `run.py` | Entry point (`doctor`, `smoke`, `prepare`, `run`, `verify`, `summarize`, `pack`) |
| `config_free_panel.json` | Frozen protocol used for all reported results |
| `config.json` | Earlier protocol, kept because the data-collection and test scripts read it |
| `free_data_check.py`, `free_panel_check.py`, `promote_free_panel.py`, `verify_free_panel.py` | Data collection, panel construction, freezing, and offline hash verification |
| `tests/`, `test_free_data_check.py`, `test_free_panel_check.py` | Unit tests |
| `data/` | Frozen panel metadata and hashes, and the four ECB exchange-rate series (see `data/README.md`) |
| `sample_counts_verified.csv` | Sample counts per series and period after preparation |
| `results/free_panel_v2/` | Run manifest, environment, hardware, runtime, and all analysis outputs |
| `evidence/` | Daily sufficient statistics for seven configurations and independent verification scripts |
| `paper/` | Scripts that produce Figs 2 and 3 and S1–S4 Tables from `results/`, and the S1–S4 Tables |
| `*.bat` | Windows setup and run scripts used for the reported run |

## Where each result in the paper comes from

All files below are in `results/free_panel_v2/analysis/` unless stated otherwise.

| Paper item | Source file |
| --- | --- |
| Table 1 (main results) and Table 3 (SCALE-R diagnostics) | `macro_mean_sd.csv` (per-seed values in `macro_per_seed.csv`) |
| Table 2 (predefined paired comparisons) | `paired_block_bootstrap.csv` |
| Table 4 (COVID-19 stress window) | `year_and_stress_metrics.csv`, rows with `period == "covid"` |
| Second predefined stress window (calendar year 2022, reported in the Results text) | `year_and_stress_metrics.csv`, rows with `period == "rates_2022"` |
| Per-target results and target counts | `metrics_wide.csv`, `metrics_long.csv` |
| Leave-one-asset-class-out sensitivity | `leave_one_asset_class_out.csv` |
| Figs 2 and 3 | `paper/make_figs.py` applied to `metrics_wide.csv` |
| S1–S4 Tables | `paper/make_stables.py` applied to the analysis folder |
| Training and inference cost | `training_runtime.csv`, `../runtime/` |

Fig 1 is a schematic of the design and has no underlying data.

## Reproducing the paper's tables and figures (no GPU, no data download)

Requires Python 3.12 with numpy, pandas, matplotlib, and openpyxl.

```bash
python paper/make_figs.py results/free_panel_v2/analysis/ out/
python paper/make_stables.py results/free_panel_v2/analysis/ out/
```

## Independent check of the reported scores and intervals

`evidence/REVIEW_EVIDENCE.zip` holds daily sufficient statistics for MC Dropout, Deep Ensembles, SCALE-R, Online quantile, EWMA Gaussian, Zero Gaussian, and the SCALE-R variant without the similarity penalty, together with the SHA256 hashes of all 220 prediction files. The script below recomputes the four main metrics and all 40 predefined block-bootstrap intervals from these arrays without the training code or a GPU.

```bash
python evidence/verify_review_evidence.py evidence/REVIEW_EVIDENCE.zip --output verification
```

The recorded output of this check is in `evidence/verification.json` (280 summary checks and 40 bootstrap rows, maximum absolute difference below 1e-14). `evidence/audit_evidence_panel.py` additionally checks the realized returns and calendars against the raw price files, so it needs the complete data panel described in `data/README.md`.

## Rerunning the full experiment

The reported run used Windows 11, Python 3.12.10, PyTorch 2.8.0 with CUDA 12.8, and one NVIDIA GeForce RTX 5060 Laptop GPU with deterministic algorithms enabled.

1. Rebuild the complete 13-series panel in `data/free_panel_raw/` as described in `data/README.md`.
2. Install the environment:
   ```bash
   python -m venv .venv
   .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   .venv/bin/python -m pip install -r requirements_free_panel.txt
   ```
3. Run the pipeline (this is what `RUN_FREE_PANEL_WINDOWS.bat` does):
   ```bash
   python run.py doctor --config config_free_panel.json
   python verify_free_panel.py
   python run.py smoke --config config_free_panel.json
   python run.py prepare --config config_free_panel.json --raw data/free_panel_raw --prepared data/free_panel_prepared
   python run.py run --config config_free_panel.json --raw data/free_panel_raw --prepared data/free_panel_prepared --out results/rerun
   python run.py verify --config config_free_panel.json --raw data/free_panel_raw --prepared data/free_panel_prepared --out results/rerun
   python run.py summarize --config config_free_panel.json --raw data/free_panel_raw --prepared data/free_panel_prepared --out results/rerun
   ```

Every run records the SHA256 of the configuration, the data files, and the `uqbench/` source files, and the pipeline refuses to mix results across changes to any of them. If the vendor price histories have been revised since September 2026, the rebuilt panel will not match the frozen hashes and a rerun defines a new panel version rather than reproducing this one.

## Licenses

- Code: MIT License (`LICENSE`).
- Results, evidence files, and supporting tables produced by the authors: CC BY 4.0 (`LICENSE-DATA`).
- Exchange-rate series in `data/free_panel_raw/` are derived from the euro foreign exchange reference rates of the European Central Bank (source: ECB). See `data/README.md` for how they were transformed.

## Citation

See `CITATION.cff`. Please cite the paper once it is published.
