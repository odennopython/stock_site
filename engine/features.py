"""Turn the monthly panel into a market "state" for every month.

Each month gets a set of interpretable indicators (valuation, trend, stress,
rates, inflation, gold) plus the shape of the last 36 months of real price.
"""
import numpy as np
import pandas as pd

# Indicators shown to readers, with Japanese labels and units.
INDICATORS = {
    "cape": ("株価の割高さ（CAPE）", "倍"),
    "mom12": ("過去1年の実質リターン", "%"),
    "mom36": ("過去3年の実質リターン", "%"),
    "vol12": ("値動きの荒さ（年率）", "%"),
    "dd": ("5年高値からの下落", "%"),
    "rate": ("長期金利", "%"),
    "drate12": ("長期金利の1年変化", "pt"),
    "infl12": ("インフレ率", "%"),
    "real_rate": ("実質金利", "%"),
    "gold12": ("金の1年リターン", "%"),
    "dy": ("配当利回り", "%"),
}


def build(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    real = p["price"] / p["cpi"]
    lr = np.log(real)
    r1 = lr.diff()
    real_eps = p["eps"] / p["cpi"]

    f = pd.DataFrame(index=p.index)
    f["cape"] = real / real_eps.rolling(120).mean()
    f["mom1"] = r1 * 100
    f["mom3"] = lr.diff(3) * 100
    f["mom12"] = lr.diff(12) * 100
    f["mom36"] = lr.diff(36) * 100
    f["vol12"] = r1.rolling(12).std() * np.sqrt(12) * 100
    f["dd"] = (real / real.rolling(60, min_periods=1).max() - 1) * 100
    f["rate"] = p["rate"]
    f["drate12"] = p["rate"].diff(12)
    f["infl12"] = (p["cpi"] / p["cpi"].shift(12) - 1) * 100
    f["real_rate"] = f["rate"] - f["infl12"]
    f["gold12"] = np.log(p["gold"] / p["gold"].shift(12)) * 100
    f["dy"] = p["div"] / p["price"] * 100

    # Shape of the last 36 months of real price, relative to today (every 3 months).
    for k in range(3, 37, 3):
        f[f"path{k}"] = (lr.shift(k) - lr) * 100

    # Forward outcomes (targets, never used as inputs).
    f["fwd1"] = (lr.shift(-1) - lr) * 100
    f["fwd3"] = (lr.shift(-3) - lr) * 100
    f["fwd12"] = (lr.shift(-12) - lr) * 100
    fwd_min12 = pd.concat([lr.shift(-k) for k in range(1, 13)], axis=1).min(axis=1)
    f["fwd_dd12"] = (np.exp(fwd_min12 - lr) - 1).clip(upper=0) * 100
    fwd_min3 = pd.concat([lr.shift(-k) for k in range(1, 4)], axis=1).min(axis=1)
    f["fwd_dd3"] = (np.exp(fwd_min3 - lr) - 1).clip(upper=0) * 100
    # Last month for which each horizon is fully observed.
    n = len(f)
    f.loc[f.index[n - 12:], ["fwd12", "fwd_dd12"]] = np.nan
    f.loc[f.index[n - 3:], ["fwd3", "fwd_dd3"]] = np.nan
    f.loc[f.index[n - 1:], ["fwd1"]] = np.nan

    f["real_price"] = real
    f["nominal"] = p["price"]
    return f.dropna(subset=["cape", "mom36", "vol12", "infl12", "gold12", "path36"])


# Inputs to the state encoder.
STATE_COLS = (
    ["cape", "mom1", "mom3", "mom12", "mom36", "vol12", "dd", "rate", "drate12",
     "infl12", "real_rate", "gold12", "dy"]
    + [f"path{k}" for k in range(3, 37, 3)]
)


def standardize(f: pd.DataFrame, fit_rows=None) -> np.ndarray:
    """Robust z-scores (median / IQR), clipped so one crisis cannot dominate.

    fit_rows limits the statistics to the past, for walk-forward tests.
    """
    x = f[STATE_COLS].copy()
    x["cape"] = np.log(x["cape"])
    fit = x if fit_rows is None else x.iloc[:fit_rows]
    med = fit.median()
    iqr = (fit.quantile(0.75) - fit.quantile(0.25)).replace(0, 1)
    z = ((x - med) / (iqr / 1.349)).clip(-4, 4)
    return z.to_numpy(dtype=np.float64)
