import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

LEVELS = np.array(sorted(set([.025, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, .975])))
QCOLS = [f"q_{p:g}" for p in LEVELS]


def make_quantiles(mean, var, df=None, tscale=None):
    if df is None:
        return mean[:, None] + np.sqrt(var)[:, None] * stats.norm.ppf(LEVELS)[None, :]
    return mean[:, None] + tscale[:, None] * stats.t.ppf(LEVELS[None, :], df[:, None])


def isotonic_fit(y, mu, sd):
    pit = stats.norm.cdf((y - mu) / np.maximum(sd, 1e-6))
    order = np.argsort(pit, kind="stable")
    rank = (np.arange(len(y)) + .5) / len(y)
    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pit[order], rank)
    return {"x": model.X_thresholds_.tolist(), "cdf": model.y_thresholds_.tolist()}


def isotonic_quantiles(mu, sd, calibration):
    p = np.clip(np.interp(LEVELS, calibration["cdf"], calibration["x"]), 1e-5, 1 - 1e-5)
    return mu[:, None] + sd[:, None] * stats.norm.ppf(p)


def split_conformal_fit(y, mu, sd):
    # Signed standardized residual quantile correction; does not promise simultaneous coverage.
    z = (y - mu) / np.maximum(sd, 1e-6)
    rank = np.clip(np.ceil((len(z) + 1) * LEVELS).astype(int), 1, len(z)) - 1
    return np.sort(z)[rank]


def online_quantiles(y, mu, sd, initial, lr):
    state = np.array(initial, float).copy()
    out = np.empty((len(y), len(LEVELS)))
    for i in range(len(y)):
        # Emit every quantile before this outcome is read.
        out[i] = mu[i] + sd[i] * state
        z = (y[i] - mu[i]) / max(sd[i], 1e-6)
        state = np.sort(state + lr * (LEVELS - (z <= state).astype(float)))
    return out, state


def fit_garch(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 100:
        raise ValueError("GARCH source fit needs at least 100 observations.")
    def parameters(theta):
        omega = np.exp(theta[0])
        a = .995 / (1 + np.exp(-theta[1]))
        b = (.995 - a) / (1 + np.exp(-theta[2]))
        return omega, a, b
    def loss(theta):
        w, a, b = parameters(theta)
        v = max(z.var(), .01)
        total = 0.
        for obs in z:
            total += .5 * (np.log(v) + obs * obs / v)
            v = max(w + a * obs * obs + b * v, 1e-8)
        return total / len(z)
    res = minimize(loss, [-3., -2.8, 2.2], method="L-BFGS-B", bounds=[(-12, 3), (-8, 8), (-8, 8)])
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError(f"Source GARCH optimization failed: {res.message}")
    return {"omega": float(parameters(res.x)[0]), "alpha": float(parameters(res.x)[1]),
            "beta": float(parameters(res.x)[2]), "initial_variance": float(z.var()),
            "converged": bool(res.success), "objective": float(res.fun)}


def baseline(data, name, source_stats):
    n = len(data.y)
    mu = np.zeros(n)
    sd = np.empty(n)
    q = np.empty((n, len(LEVELS)))
    if name == "GARCH-Norm":
        p = source_stats["garch"]
        v = max(p["initial_variance"], 1e-6)
        variance = np.empty(len(data.returns))
        for j, obs in enumerate(data.returns):
            z = obs / max(data.hist_scale[max(0, j - 1)], .01)
            v = max(p["omega"] + p["alpha"] * z * z + p["beta"] * v, 1e-8)
            variance[j] = v
        sd = np.sqrt(variance[data.origin_index]) * data.scale
        q = make_quantiles(mu, sd ** 2)
    else:
        for i in range(n):
            hist = data.history(i)
            if name == "ZeroGaussian":
                sd[i] = source_stats["target_sd"] * data.scale[i]
            elif name == "RollingEmpirical":
                mu[i], sd[i] = hist.mean(), max(hist.std(ddof=1), .01)
                q[i] = np.quantile(hist, LEVELS)
            elif name == "EWMAGaussian":
                v = max(hist[:min(20, len(hist))].var(), .0001)
                for r in hist:
                    v = .94 * v + .06 * r * r
                sd[i] = np.sqrt(v)
            else:
                raise ValueError(name)
        if name != "RollingEmpirical":
            q = make_quantiles(mu, sd ** 2)
    return {"mean": mu, "sd": sd, "epi_sd": np.full(n, np.nan), "quantiles": q}
