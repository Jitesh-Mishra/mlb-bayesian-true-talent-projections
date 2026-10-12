"""Rolling-origin backtest. For each target season Y: fit on seasons <= Y-1, project Y, score players with
200+ PA in both Y-1 and Y.

    python -m mlbproj.evaluate dev     # targets 2010-2014: sanity checks and the drift setting
    python -m mlbproj.evaluate eval    # targets 2015-2025 minus 2020: the backtest, logged in final_test_runs.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from .baselines import history, marcel, naive
from .data import STATS, league_rates, player_seasons
from .model import fit_project

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEV = [2010, 2011, 2012, 2013, 2014]
EVAL = [y for y in range(2015, 2026) if y != 2020]     # 2020 (60 games) skipped as a target, kept as an input
MIN_PA = 200
TAU = {"K%": 0.10, "BB%": 0.15, "HR%": 0.10}            # logit-scale drift, closest to 80% on dev targets (decisions.md D5)


TAU_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def run(years, tau=TAU, draws=1000, tune=1000, chains=4, tau_grid=(), log=print):
    ps = player_seasons()
    lg = league_rates(ps)
    out, diag = [], []
    for Y in years:
        T = Y - 1
        cur = ps[(ps["yearID"] == T) & (ps["PA"] >= MIN_PA)]["playerID"]
        nxt = ps[(ps["yearID"] == Y) & (ps["PA"] >= MIN_PA)].set_index("playerID")
        ids = sorted(set(cur) & set(nxt.index))
        h = history(ps, T).loc[ids]
        for stat, col in STATS.items():
            t0 = time.time()
            tg = pd.DataFrame({"playerID": ids, "PA_next": nxt.loc[ids, "PA"].to_numpy()})
            proj, idata = fit_project(ps, lg, stat, T, tg, tau=tau[stat], draws=draws, tune=tune,
                                       chains=chains, seed=Y, tau_grid=tau_grid)
            proj = proj.set_index("playerID").loc[ids]
            proj["actual"] = nxt.loc[ids, col] / nxt.loc[ids, "PA"]
            proj["naive"] = naive(h, stat).loc[ids]
            proj["marcel"] = marcel(h, stat, lg, T).loc[ids]
            proj["target"], proj["stat"], proj["age"] = Y, stat, h.loc[ids, "age"]
            out.append(proj.reset_index())
            rhat = az.rhat(idata)
            ess = az.ess(idata)
            diag.append({"target": Y, "stat": stat, "n_fit": idata.posterior["theta"].shape[-1],
                         "n_scored": len(ids), "max_rhat": float(max(rhat[v].max() for v in rhat)),
                         "min_ess_bulk": float(min(ess[v].min() for v in ess)),
                         "divergences": int(idata.sample_stats["diverging"].sum()),
                         "sigma": float(idata.posterior["sigma"].mean()), "seconds": round(time.time() - t0, 1)})
            log(f"{Y} {stat} n={len(ids)} rhat={diag[-1]['max_rhat']:.3f} {diag[-1]['seconds']}s")
    return pd.concat(out, ignore_index=True), pd.DataFrame(diag)


def wrmse(e, w):
    return float(np.sqrt(np.sum(w * e**2) / np.sum(w)))


def summarize(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stat, d in p.groupby("stat", sort=False):
        w = d["PA_next"]
        r = {m: wrmse(d[m] - d["actual"], w) for m in ["naive", "marcel", "bayes"]}
        cov = ((d["actual"] >= d["lo80"]) & (d["actual"] <= d["hi80"])).mean()
        rows.append({"stat": stat, "n": len(d), "seasons": d["target"].nunique(),
                     "rmse_naive": r["naive"], "rmse_marcel": r["marcel"], "rmse_bayes": r["bayes"],
                     "pct_vs_naive": 100 * (1 - r["bayes"] / r["naive"]),
                     "pct_vs_marcel": 100 * (1 - r["bayes"] / r["marcel"]),
                     "coverage80": float(cov)})
    return pd.DataFrame(rows)


def bootstrap_vs_marcel(p: pd.DataFrame, B=2000, seed=0) -> pd.DataFrame:
    """95% interval for the % RMSE change vs Marcel, resampling players (all their seasons together)."""
    rng = np.random.default_rng(seed)
    rows = []
    for stat, d in p.groupby("stat", sort=False):
        players = d["playerID"].unique()
        g = {k: v for k, v in d.groupby("playerID")}
        sq = {pid: (np.sum(v["PA_next"] * (v["bayes"] - v["actual"]) ** 2),
                    np.sum(v["PA_next"] * (v["marcel"] - v["actual"]) ** 2),
                    np.sum(v["PA_next"] * (v["naive"] - v["actual"]) ** 2), v["PA_next"].sum()) for pid, v in g.items()}
        arr = np.array([sq[pid] for pid in players])
        idx = rng.integers(0, len(players), (B, len(players)))
        s = arr[idx].sum(1)
        vm = 100 * (1 - np.sqrt(s[:, 0] / s[:, 1]))
        vn = 100 * (1 - np.sqrt(s[:, 0] / s[:, 2]))
        rows.append({"stat": stat, "vs_marcel_lo": np.percentile(vm, 2.5), "vs_marcel_hi": np.percentile(vm, 97.5),
                     "vs_naive_lo": np.percentile(vn, 2.5), "vs_naive_hi": np.percentile(vn, 97.5)})
    return pd.DataFrame(rows)


def drift_table(p: pd.DataFrame) -> pd.DataFrame:
    """Dev-season 80% coverage per stat for each candidate drift tau (column 0.0 = no drift)."""
    def cov(lo, hi):
        return p.assign(c=(p["actual"] >= p[lo]) & (p["actual"] <= p[hi])).groupby("stat", sort=False)["c"].mean()
    out = {0.0: cov("lo80", "hi80")} if all(TAU[s] == 0 for s in TAU) else {}
    out |= {tg: cov(f"lo80@{tg}", f"hi80@{tg}") for tg in TAU_GRID if f"lo80@{tg}" in p}
    return pd.DataFrame(out)


def main(phase: str):
    REPORTS.mkdir(exist_ok=True)
    years = DEV if phase == "dev" else EVAL
    if phase == "eval":
        lock = REPORTS / "final_test_runs.json"
        runs = json.loads(lock.read_text()) if lock.exists() else []
        runs.append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "tau": TAU})
        lock.write_text(json.dumps(runs, indent=1))
    p, diag = run(years, tau_grid=TAU_GRID if phase == "dev" else ())
    p.to_csv(REPORTS / f"projections_{phase}.csv", index=False, float_format="%.5f")
    diag.to_csv(REPORTS / f"diagnostics_{phase}.csv", index=False)
    s = summarize(p).merge(bootstrap_vs_marcel(p), on="stat")
    s.to_csv(REPORTS / f"results_{phase}.csv", index=False, float_format="%.4f")
    per = p.groupby(["stat", "target"]).apply(
        lambda d: pd.Series({m: wrmse(d[m] - d["actual"], d["PA_next"]) for m in ["naive", "marcel", "bayes"]}),
        include_groups=False)
    per.to_csv(REPORTS / f"per_season_{phase}.csv", float_format="%.5f")
    print(s.round(4).to_string(index=False))
    if phase == "dev":
        drift_table(p).to_csv(REPORTS / "drift_calibration_dev.csv", float_format="%.4f")
        print("dev 80% coverage by drift tau:\n", drift_table(p).round(3))
    print(diag[["max_rhat", "min_ess_bulk", "divergences"]].describe().loc[["min", "max"]])


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    if phase not in ("dev", "eval"):
        sys.exit("usage: python -m mlbproj.evaluate dev|eval   (eval is the backtest and is logged)")
    main(phase)
