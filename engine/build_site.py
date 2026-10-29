"""Build site/data.json: today's analogs, their aftermath, and a time machine."""
import json
from pathlib import Path

import numpy as np

from analogs import episodes, neighbours
from data import load_panel
from encoder import fingerprints
from features import INDICATORS, build, standardize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data.json"
K_FAN = 120


def ym(p):
    return f"{p.year}-{p.month:02d}"


def main():
    panel = load_panel()
    f = build(panel)
    n = len(f)
    z = standardize(f)
    emb = fingerprints(z, z, seeds=(0, 1, 2, 3, 4))
    dates = f.index
    lr = np.log(f["real_price"].to_numpy())
    fwd12 = f["fwd12"].to_numpy()
    fwd_dd12 = f["fwd_dd12"].to_numpy()

    # Typical distance between two random months: 0% similarity.
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    d_typ = float(np.median(np.linalg.norm(emb[a] - emb[b], axis=1)))

    def score(d):
        return round(max(0.0, 1 - d / d_typ) * 100)

    def path(i, before=36, after=12):
        lo, hi = max(0, i - before), min(n - 1, i + after)
        return [[k - i, round(float(np.exp(lr[k] - lr[i]) * 100), 2)] for k in range(lo, hi + 1)]

    def ind(i):
        return {k: round(float(f[k].iloc[i]), 2) for k in INDICATORS}

    pct = {k: f[k].rank(pct=True).to_numpy() for k in INDICATORS}

    def why(t, s):
        """Indicators where both months sit on the same unusual side of history."""
        out = []
        for k in INDICATORS:
            pt, ps = pct[k][t], pct[k][s]
            if abs(pt - ps) < 0.12 and abs(pt - 0.5) > 0.3:
                out.append((abs(pt - 0.5), k))
        return [k for _, k in sorted(out, reverse=True)[:3]]

    def analog_block(t, n_ep=5):
        pool = np.arange(0, t - 12 + 1)           # aftermath already known
        pool = pool[np.abs(pool - t) >= 36]        # skip the recent past itself
        idx, dist = neighbours(emb, t, pool, k=300)
        eps = episodes(idx, dist, n=n_ep)
        fan_idx = idx[:K_FAN]
        return idx, eps, fan_idx

    # Today.
    t = n - 1
    _, eps, fan_idx = analog_block(t)
    now = {
        "month": ym(dates[t]),
        "nominal": round(float(f["nominal"].iloc[t]), 2),
        "indicators": ind(t),
        "percentiles": {k: round(float(pct[k][t]) * 100) for k in INDICATORS},
        "path": path(t, after=0),
    }
    analogs = []
    for i, d in eps:
        analogs.append({
            "month": ym(dates[i]),
            "score": score(d),
            "indicators": ind(i),
            "why": why(t, i),
            "path": path(i),
            "fwd12": round(float(fwd12[i]), 1),
            "fwd_dd12": round(float(fwd_dd12[i]), 1),
        })
    known = ~np.isnan(fwd12)
    fan = {
        "analog": np.round(np.sort(fwd12[fan_idx]), 1).tolist(),
        "all": np.round(np.percentile(fwd12[known], np.arange(0, 101, 2)), 1).tolist(),
        "analog_q": np.round(np.percentile(fwd12[fan_idx], [10, 25, 50, 75, 90]), 1).tolist(),
        "all_q": np.round(np.percentile(fwd12[known], [10, 25, 50, 75, 90]), 1).tolist(),
        "analog_down20": round(float(np.mean(fwd_dd12[fan_idx] <= -20)) * 100),
        "all_down20": round(float(np.nanmean(fwd_dd12[known] <= -20)) * 100),
    }

    # Time machine: for every month from 1900, its analogs among earlier months.
    machine = []
    start = int(np.searchsorted(dates.year, 1900))
    for t2 in range(start, n):
        idx, eps2, fan2 = analog_block(t2, n_ep=3)
        q = np.percentile(fwd12[fan2], [10, 50, 90])
        machine.append([
            ym(dates[t2]),
            [[ym(dates[i]), score(d)] for i, d in eps2],
            [round(float(x), 1) for x in q],
            None if np.isnan(fwd12[t2]) else round(float(fwd12[t2]), 1),
        ])

    # Long real price line (log scale on the page).
    history = [[ym(dates[i]), round(float(f["real_price"].iloc[i] / f["real_price"].iloc[-1] * 100), 3)]
               for i in range(0, n)]

    bt = json.loads((ROOT / "data" / "backtest.json").read_text())
    data = {
        "built": ym(dates[t]),
        "eps_last_known": ym(panel.index[panel["eps_known"]][-1]),
        "indicator_labels": INDICATORS,
        "now": now,
        "analogs": analogs,
        "fan": fan,
        "machine": machine,
        "history": history,
        "backtest": {
            "period": bt["period"],
            "months": bt["months"],
            "rank_corr": round(bt["return12"]["ae"]["rank_corr_median"], 2),
            "crps_skill": round(bt["return12"]["ae"]["crps_skill_vs_clim"], 3),
            "coverage": round(bt["return12"]["ae"]["coverage_10_90"] * 100),
            "clim_coverage": round(bt["return12"]["clim"]["coverage_10_90"] * 100),
            "bear_auc": round(bt["events"]["bear12"]["ae"]["auc"], 2),
            "crash_auc": round(bt["events"]["crash3"]["ae"]["auc"], 2),
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item()))
    print("now", now["month"], [(a["month"], a["score"], a["why"], a["fwd12"]) for a in analogs])
    print("fan", fan["analog_q"], fan["all_q"], fan["analog_down20"], fan["all_down20"])
    print("bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
