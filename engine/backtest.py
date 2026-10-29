"""Walk-forward test: would the analogs have helped, using only the past?

Every decade from 1921 the scaler and the encoder are refit on data before
that decade. Each month in the decade then looks for analogs only among
months whose 12-month outcome was already known.
"""
import json
from pathlib import Path

import numpy as np

from analogs import K, event_prob, logistic_fit, logistic_predict, neighbours
from data import load_panel
from encoder import fingerprints
from features import build, standardize

OUT = Path(__file__).resolve().parent.parent / "data" / "backtest.json"
RISK_COLS = ["vol12", "dd", "mom12", "mom3", "cape", "drate12", "infl12", "real_rate"]
EVENTS = {"crash3": ("fwd_dd3", -10.0), "bear12": ("fwd_dd12", -20.0)}


def crps(samples, y):
    s = np.asarray(samples)
    return np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))


def main():
    f = build(load_panel())
    n = len(f)
    dates = f.index
    fwd12 = f["fwd12"].to_numpy()
    targets = {k: f[c].to_numpy() for k, (c, _) in EVENTS.items()}
    thresholds = {k: th for k, (_, th) in EVENTS.items()}
    risk_x = f[RISK_COLS].to_numpy().copy()
    risk_x[:, RISK_COLS.index("cape")] = np.log(risk_x[:, RISK_COLS.index("cape")])

    start = int(np.searchsorted(dates.year, 1921))
    rows = []
    for d0 in range(start, n, 120):
        d1 = min(d0 + 120, n)
        z = standardize(f, fit_rows=d0)
        emb = fingerprints(z[:d0], z[:d1])
        for t in range(d0, d1):
            if np.isnan(fwd12[t]) and np.isnan(targets["crash3"][t]):
                continue
            pool = np.arange(0, t - 12 + 1)       # outcomes known by month t
            clim = fwd12[pool]
            r = {"t": t, "y12": fwd12[t]}
            for name, e in (("ae", emb), ("raw", z)):
                idx, _ = neighbours(e, t, pool)
                r[f"{name}_q"] = np.percentile(fwd12[idx], [10, 50, 90]).tolist()
                r[f"{name}_crps"] = crps(fwd12[idx], fwd12[t]) if not np.isnan(fwd12[t]) else np.nan
                for ev, y in targets.items():
                    base = np.mean(y[pool] <= thresholds[ev])
                    r[f"{name}_{ev}"] = event_prob(np.sum(y[idx] <= thresholds[ev]), K, base)
            r["clim_crps"] = crps(clim, fwd12[t]) if not np.isnan(fwd12[t]) else np.nan
            r["clim_q"] = np.percentile(clim, [10, 50, 90]).tolist()
            for ev, y in targets.items():
                yy = (y[pool] <= thresholds[ev]).astype(float)
                r[f"clim_{ev}"] = yy.mean()
                # Logistic risk model on the same past-only window (refit yearly).
                if t % 12 == 0 or "w_" + ev not in globals():
                    mu, sd = risk_x[pool].mean(0), risk_x[pool].std(0)
                    globals()["w_" + ev] = (logistic_fit((risk_x[pool] - mu) / sd, yy), mu, sd)
                w, mu, sd = globals()["w_" + ev]
                r[f"logit_{ev}"] = float(logistic_predict(w, ((risk_x[t] - mu) / sd)[None])[0])
                r[f"y_{ev}"] = np.nan if np.isnan(y[t]) else float(y[t] <= thresholds[ev])
            rows.append(r)

    res = summarize(rows)
    res["period"] = f"{dates[rows[0]['t']]} – {dates[rows[-1]['t']]}"
    res["months"] = len(rows)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


def summarize(rows):
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    y12 = col("y12")
    ok = ~np.isnan(y12)
    out = {"return12": {}, "events": {}}
    for m in ("clim", "raw", "ae"):
        q = np.array([r[f"{m}_q"] for r in rows])[ok]
        out["return12"][m] = {
            "crps": float(np.nanmean(col(f"{m}_crps")[ok])),
            "coverage_10_90": float(np.mean((y12[ok] >= q[:, 0]) & (y12[ok] <= q[:, 2]))),
            "rank_corr_median": float(spearman(q[:, 1], y12[ok])),
        }
    c0 = out["return12"]["clim"]["crps"]
    for m in ("raw", "ae"):
        out["return12"][m]["crps_skill_vs_clim"] = 1 - out["return12"][m]["crps"] / c0
    for ev in EVENTS:
        y = col(f"y_{ev}")
        ok_e = ~np.isnan(y)
        e = {"base_rate": float(y[ok_e].mean())}
        b0 = np.mean((col(f"clim_{ev}")[ok_e] - y[ok_e]) ** 2)
        for m in ("clim", "raw", "ae", "logit"):
            p = col(f"{m}_{ev}")[ok_e]
            b = np.mean((p - y[ok_e]) ** 2)
            e[m] = {"brier": float(b), "skill": float(1 - b / b0), "auc": float(auc(p, y[ok_e]))}
        blend = (col(f"ae_{ev}")[ok_e] + col(f"logit_{ev}")[ok_e]) / 2
        b = np.mean((blend - y[ok_e]) ** 2)
        e["blend"] = {"brier": float(b), "skill": float(1 - b / b0), "auc": float(auc(blend, y[ok_e]))}
        e["calibration"] = calibration(blend, y[ok_e])
        out["events"][ev] = e
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def calibration(p, y, bins=(0, 0.05, 0.1, 0.2, 0.3, 1.01)):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"bin": [lo, min(hi, 1.0)], "n": int(m.sum()),
                        "forecast": float(p[m].mean()), "observed": float(y[m].mean())})
    return out


if __name__ == "__main__":
    main()
