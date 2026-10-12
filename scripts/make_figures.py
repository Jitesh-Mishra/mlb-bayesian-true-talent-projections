"""README figures, light + dark: results, shrinkage, posterior predictive check, calibration.

python scripts/make_figures.py   (after python -m mlbproj.evaluate eval)
Shrinkage and PPC refit the 2024 origin (projecting 2025) for each stat; everything else reads reports/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlbproj.baselines import history  # noqa: E402
from mlbproj.data import STATS, league_rates, player_seasons  # noqa: E402
from mlbproj.model import FIT_MIN_PA, fit_project  # noqa: E402

REP, OUT = ROOT / "reports", ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
THEMES = {"light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e9ed", raw="#afb8c1",
                        cols=["#2a78d6", "#eb6834", "#1baf7a"]),
          "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#21262d", raw="#6e7681",
                       cols=["#3987e5", "#d95926", "#199e70"])}
NAMES = {"K%": "Strikeout rate (K%)", "BB%": "Walk rate (BB%)", "HR%": "Home runs per PA"}
METHODS = [("naive", "Last season"), ("marcel", "Marcel"), ("bayes", "Hierarchical Bayes")]

res = pd.read_csv(REP / "results_eval.csv").set_index("stat")
proj = pd.read_csv(REP / "projections_eval.csv")
per = pd.read_csv(REP / "per_season_eval.csv")

# --- one refit at origin 2024 per stat, for shrinkage and the PPC
ps = player_seasons()
lg = league_rates(ps)
T = 2024
h = history(ps, T)
h = h[h["PA_0"] >= FIT_MIN_PA]
fits = {}
for stat, col in STATS.items():
    tg = pd.DataFrame({"playerID": h.index, "PA_next": 600})
    _, idata = fit_project(ps, lg, stat, T, tg, draws=1000, tune=1000, chains=2, seed=T)
    th = idata.posterior["theta"].stack(s=("chain", "draw")).to_numpy()
    p_T = expit(logit(lg.loc[T, stat]) + th)                      # talent in 2024 on the rate scale
    fits[stat] = dict(pa=h["PA_0"].to_numpy(), raw=(h[f"{col}_0"] / h["PA_0"]).to_numpy(),
                      post=p_T.mean(1), draws=p_T, lg=lg.loc[T, stat])


def style(ax, th, grid="y"):
    ax.set_facecolor(th["bg"]); ax.set_axisbelow(True); ax.grid(axis=grid, color=th["grid"], lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


for theme, th in THEMES.items():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": th["ink2"], "axes.labelcolor": th["ink"],
                         "xtick.color": th["ink2"], "ytick.color": th["ink2"], "text.color": th["ink"]})

    # 1. results: RMSE relative to last-season rate (=100)
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=th["bg"])
    x = np.arange(3)
    for j, (m, lab) in enumerate(METHODS):
        vals = [100 * res.loc[s, f"rmse_{m}"] / res.loc[s, "rmse_naive"] for s in STATS]
        bars = ax.bar(x + (j - 1) * 0.26, vals, 0.24, color=th["cols"][j], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9, color=th["ink"])
    ax.set_xticks(x); ax.set_xticklabels([NAMES[s] for s in STATS])
    ax.set_ylim(0, 128); ax.set_ylabel("PA-weighted RMSE\n(last season = 100)")
    n_seasons = int(res["seasons"].iloc[0])
    ax.set_title(f"Next-season projection error, {n_seasons} held-out seasons (2015-2025, 2020 skipped), "
                 f"{int(res['n'].iloc[0]):,} player-seasons. Lower is better.", loc="left", fontsize=10.5)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    style(ax, th)
    fig.tight_layout(); fig.savefig(OUT / f"results-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)

    # 2. shrinkage: raw 2024 rate -> posterior talent, by PA
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=th["bg"])
    keep = np.random.default_rng(1).choice(len(h), 220, replace=False)     # a readable sample of hitters
    for ax, (stat, f) in zip(axes, fits.items()):
        f = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim == 1 else v) for k, v in f.items()}
        for pa, r, q in zip(f["pa"], f["raw"], f["post"]):
            ax.plot([pa, pa], [r, q], color=th["raw"], lw=0.8, alpha=0.8)
        ax.scatter(f["pa"], f["raw"], s=9, color=th["raw"], label="2024 observed", zorder=2)
        ax.scatter(f["pa"], f["post"], s=9, color=th["cols"][2], label="posterior talent", zorder=3)
        ax.axhline(f["lg"], color=th["ink2"], ls="--", lw=1)
        ax.text(f["pa"].max(), f["lg"], " league", va="bottom", ha="right", fontsize=8.5, color=th["ink2"])
        ax.set_title(NAMES[stat], loc="left", fontsize=10.5)
        ax.set_xlabel("PA in 2024"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
        style(ax, th, "both")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.suptitle("Shrinkage: small samples are pulled toward the population, HR rate the hardest (2024, 220 random hitters with 100+ PA)",
                 x=0.01, ha="left", fontsize=11, color=th["ink"])
    fig.tight_layout(); fig.savefig(OUT / f"shrinkage-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)

    # 3. posterior predictive check: observed 2024 rate distribution vs 100 replicated seasons
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), facecolor=th["bg"])
    for ax, (stat, f) in zip(axes, fits.items()):
        bins = np.linspace(np.quantile(f["raw"], 0.002), np.quantile(f["raw"], 0.998), 30)
        S = f["draws"].shape[1]
        for s in rng.choice(S, 100, replace=False):
            rep = rng.binomial(f["pa"].astype(int), f["draws"][:, s]) / f["pa"]
            ax.hist(rep, bins, histtype="step", color=th["cols"][0], alpha=0.08, lw=1)
        ax.hist(f["raw"], bins, histtype="step", color=th["ink"], lw=2)
        ax.set_title(NAMES[stat], loc="left", fontsize=10.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.set_yticks([])
        style(ax, th, "x")
    fig.suptitle("Posterior predictive check: observed 2024 rates (black) vs 100 seasons simulated from the "
                 "posterior (blue)", x=0.01, ha="left", fontsize=11, color=th["ink"])
    fig.tight_layout(); fig.savefig(OUT / f"ppc-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)

    # 4. calibration: 80% interval coverage per stat and per season
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=th["bg"], gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    for j, s in enumerate(STATS):
        c, n = res.loc[s, "coverage80"], res.loc[s, "n"]
        se = np.sqrt(c * (1 - c) / n)
        ax.errorbar(j, 100 * c, yerr=196 * se, fmt="o", color=th["cols"][j], ms=8, capsize=4, lw=2)
        ax.text(j + 0.12, 100 * c, f"{100 * c:.1f}%", va="center", fontsize=9.5)
    ax.axhline(80, color=th["ink2"], ls="--", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(list(STATS)); ax.set_xlim(-0.5, 2.7); ax.set_ylim(70, 90)
    ax.set_title("80% intervals: share of actuals inside", loc="left", fontsize=10.5); style(ax, th)
    ax = axes[1]
    cov = proj.assign(c=(proj.actual >= proj.lo80) & (proj.actual <= proj.hi80)).groupby(["target", "stat"])["c"].mean().unstack()
    for j, s in enumerate(STATS):
        ax.plot(cov.index, 100 * cov[s], "o-", color=th["cols"][j], label=s, ms=4)
    ax.axhline(80, color=th["ink2"], ls="--", lw=1)
    ax.set_xticks(cov.index); ax.set_ylim(55, 100); ax.legend(frameon=False, ncol=3)
    ax.set_title("Coverage by target season", loc="left", fontsize=10.5); style(ax, th)
    fig.tight_layout(); fig.savefig(OUT / f"calibration-{theme}.png", dpi=150, facecolor=th["bg"]); plt.close(fig)

print("figures written to", OUT)
