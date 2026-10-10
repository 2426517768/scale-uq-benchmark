import numpy as np
from scipy.stats import spearmanr

from .distributions import LEVELS, QCOLS


def column(p):
    return f"q_{p:g}"


def wis_rows(frame):
    y = frame.target_pct.to_numpy()
    score = .5 * np.abs(y - frame[column(.5)].to_numpy())
    for alpha in (.5, .2, .1, .05):
        lo, hi = frame[column(alpha / 2)].to_numpy(), frame[column(1 - alpha / 2)].to_numpy()
        interval = hi - lo + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)
        score += alpha / 2 * interval
    return score / 4.5


def rank_correlation(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.ptp(np.asarray(a)[valid]) < 1e-10:
        return float("nan")
    return float(spearmanr(np.asarray(a)[valid], np.asarray(b)[valid]).statistic)


def score(frame):
    y, mean = frame.target_pct.to_numpy(), frame["mean"].to_numpy()
    q = frame[QCOLS].to_numpy()
    if len(y)<2 or not np.isfinite(y).all() or not np.isfinite(q).all() or not np.isfinite(mean).all():
        raise ValueError("Nonfinite predictions must be repaired, not dropped from evaluation.")
    if "sd" in frame and (not np.isfinite(frame.sd).all() or (frame.sd<0).any()):
        raise ValueError("Predictive standard deviations must be finite and nonnegative.")
    if (np.diff(q, axis=1) < -1e-7).any():
        raise ValueError("Crossing quantiles.")
    sd = max(float(np.std(y, ddof=1)), 1e-8)
    error = y - mean
    zero_mse = max(float(np.mean(y * y)), 1e-10)
    coverage = np.mean(y[:, None] <= q, axis=0)
    ids = [np.flatnonzero(np.isclose(LEVELS, p))[0] for p in np.arange(1, 10) / 10]
    lo, hi = frame[column(.05)].to_numpy(), frame[column(.95)].to_numpy()
    width = hi - lo
    out = {"n": len(y), "nRMSE": float(np.sqrt(np.mean(error ** 2)) / sd),
           "MAE_pct": float(np.mean(np.abs(error))), "skill_vs_zero": float(1 - np.mean(error ** 2) / zero_mse),
           "ECE": float(np.mean(np.abs(coverage[ids] - LEVELS[ids]))),
           "PICP90": float(np.mean((y >= lo) & (y <= hi))), "MPIW90_pct": float(width.mean()),
           "NMPIW90": float(width.mean() / sd), "WIS_pct": float(wis_rows(frame).mean()),
           "NWIS": float(wis_rows(frame).mean() / sd),
           "UA_width": rank_correlation(width, np.abs(error)),
           "UA_epistemic": rank_correlation(frame.epi_sd.to_numpy(), np.abs(error))}
    for p, c in zip(LEVELS, coverage):
        out[f"coverage_q_{p:g}"] = float(c)
    for alpha in (.5, .2, .05):
        lo, hi = frame[column(alpha / 2)].to_numpy(), frame[column(1 - alpha / 2)].to_numpy()
        out[f"PICP{int(round(100*(1-alpha)))}"] = float(np.mean((y >= lo) & (y <= hi)))
    return out


def risk_curve(frame):
    order = np.argsort((frame[column(.95)] - frame[column(.05)]).to_numpy(), kind="stable")
    e = (frame.target_pct - frame["mean"]).to_numpy() ** 2
    return [{"retained_fraction": f, "n": max(1, int(len(e) * f)),
             "RMSE_pct": float(np.sqrt(e[order[:max(1, int(len(e) * f))]].mean()))}
            for f in np.arange(1, 11) / 10]
