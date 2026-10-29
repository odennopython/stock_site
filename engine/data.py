"""Load the open datasets and build one monthly panel (1871-).

Sources (github.com/datasets):
  s-and-p-500  Shiller: S&P composite, dividends, earnings, CPI, long rate
  cpi-us       CPI-U (fills CPI after Shiller's series stops)
  bond-yields-us-10y  10y Treasury (fills long rate after Shiller stops)
  gold-prices  monthly gold
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _monthly(df, col, name):
    s = df.set_index(pd.to_datetime(df["Date"]).dt.to_period("M"))[col]
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(name)


def load_panel() -> pd.DataFrame:
    sh = pd.read_csv(RAW / "sp500_shiller.csv")
    cpi = pd.read_csv(RAW / "cpi.csv")
    y10 = pd.read_csv(RAW / "us10y_m.csv")
    gold = pd.read_csv(RAW / "gold_m.csv")

    p = pd.concat(
        [
            _monthly(sh, "SP500", "price"),
            _monthly(sh, "Dividend", "div"),
            _monthly(sh, "Earnings", "eps"),
            _monthly(sh, "Consumer Price Index", "cpi_sh"),
            _monthly(sh, "Long Interest Rate", "rate_sh"),
        ],
        axis=1,
    ).replace(0.0, np.nan)

    # CPI: Shiller's series, extended with CPI-U scaled at the last overlap.
    cpi_u = _monthly(cpi, "Index", "cpi_u")
    p = p.join(cpi_u, how="left")
    last = p["cpi_sh"].last_valid_index()
    scale = p.loc[last, "cpi_sh"] / p.loc[last, "cpi_u"]
    p["cpi"] = p["cpi_sh"].fillna(p["cpi_u"] * scale).ffill()

    # Long rate: Shiller, then 10y Treasury.
    p = p.join(_monthly(y10, "Rate", "rate_t"), how="left")
    p["rate"] = p["rate_sh"].fillna(p["rate_t"]).ffill()

    p = p.join(_monthly(gold, "Price", "gold"), how="left")
    p["gold"] = p["gold"].ffill()

    # Earnings and dividends end in 2023; carry them forward in real terms.
    p["eps_known"] = p["eps"].notna()
    for c in ("eps", "div"):
        real = (p[c] / p["cpi"]).ffill()
        p[c] = real * p["cpi"]

    p = p[["price", "div", "eps", "eps_known", "cpi", "rate", "gold"]].dropna(
        subset=["price", "cpi", "rate"]
    )
    return p
