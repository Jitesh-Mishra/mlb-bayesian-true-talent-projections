"""Hierarchical Bayesian true-talent model (PyMC), one fit per stat and origin season T.

For player i and season s in {T, T-1, T-2} (logit scale):
    eta_is   = league_s + theta_i - aging shift from age in s to age in T
    k_is     ~ Binomial(PA_is, invlogit(eta_is))      log-likelihood weighted 1 / 0.8 / 0.6 (Marcel's 5/4/3)
    theta_i  ~ Normal(mu + b_age * (age_T - 28) / 5, sigma)     population prior (centred)
Projection for T+1:
    p_i      = invlogit(league_T + theta_i + g(age_T) + drift),  drift ~ Normal(0, tau)
g is a delta-method aging curve fitted only on season pairs that end by T. tau (year-to-year talent drift
the 3-season window can't see) is set on development seasons, never on the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logit

from .baselines import history
from .data import STATS

LIK_WEIGHTS = (1.0, 0.8, 0.6)
FIT_MIN_PA = 100          # players in the fit population: 100+ PA in season T


def _rel_logit(x, pa, lg_rate, prior_n=50.0):
    """Player's logit rate relative to league; lightly smoothed so 0 HR doesn't give -inf."""
    r = (x + prior_n * lg_rate) / (pa + prior_n)
    return logit(r) - logit(lg_rate)


def aging_curve(ps: pd.DataFrame, lg: pd.DataFrame, stat: str, T: int, min_pa: int = 100):
    """Delta method on consecutive seasons (y, y+1) with y+1 <= T: mean change in league-relative logit rate
    from age a to a+1, weighted by harmonic-mean PA, smoothed with a weighted quadratic in age.
    Returns g(a) as a function, clamped to ages 21..38. Uses at most the 15 seasons before T."""
    c = STATS[stat]
    d = ps[ps["yearID"].between(T - 15, T)][["playerID", "yearID", "age", "PA", c]]
    d = d.assign(rl=_rel_logit(d[c], d["PA"], lg.loc[d["yearID"], stat].to_numpy()))
    nxt = d.assign(yearID=d["yearID"] - 1)
    pair = d.merge(nxt, on=["playerID", "yearID"], suffixes=("", "_n"))
    pair = pair[(pair["PA"] >= min_pa) & (pair["PA_n"] >= min_pa) & (pair["yearID"] + 1 <= T)]
    w = 2 / (1 / pair["PA"] + 1 / pair["PA_n"])
    coef = np.polyfit(pair["age"] - 28, pair["rl_n"] - pair["rl"], 2, w=np.sqrt(w))
    return lambda a: np.polyval(coef, np.clip(np.asarray(a, float), 21, 38) - 28)


def fit_project(ps, lg, stat, T, targets: pd.DataFrame, tau=0.0, draws=500, tune=500, chains=2, seed=0,
                tau_grid=()):
    """Fit on seasons <= T, project T+1 for `targets` (playerID, PA_next).
    Returns (projections DataFrame, InferenceData)."""
    c = STATS[stat]
    h = history(ps, T)
    h = h[h["PA_0"] >= FIT_MIN_PA]
    g = aging_curve(ps, lg, stat, T)
    n = len(h)
    age_T = h["age"].to_numpy(float)
    lg_logit = np.array([logit(lg.loc[T - k, stat]) for k in range(3)])
    # aging shift that carries season T-k up to age_T: sum of g over ages age_T-k .. age_T-1
    shift = np.stack([sum(g(age_T - j) for j in range(1, k + 1)) if k else np.zeros(n) for k in range(3)], 1)
    PA = np.stack([h[f"PA_{k}"].to_numpy(float) for k in range(3)], 1)
    X = np.stack([h[f"{c}_{k}"].to_numpy(float) for k in range(3)], 1)
    W = np.array(LIK_WEIGHTS)
    obs = PA > 0

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        b_age = pm.Normal("b_age", 0, 0.5)
        sigma = pm.HalfNormal("sigma", 1)
        # centred: every fitted player has 100+ PA, so the likelihood dominates and centred mixes better
        theta = pm.Normal("theta", mu + b_age * (age_T - 28) / 5, sigma, shape=n)
        eta = lg_logit[None, :] + theta[:, None] - shift
        ll = pm.logp(pm.Binomial.dist(n=PA[obs], logit_p=eta[obs]), X[obs])
        pm.Potential("weighted_lik", (np.broadcast_to(W, PA.shape)[obs] * ll).sum())
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False, cores=min(chains, 2),
                          compute_convergence_checks=False, target_accept=0.9)

    th = idata.posterior["theta"].stack(s=("chain", "draw")).to_numpy()      # (n, S)
    fit = pd.DataFrame({"playerID": h.index, "row": np.arange(n)}).set_index("playerID")
    t = targets[targets["playerID"].isin(fit.index)].copy()
    rows = fit.loc[t["playerID"], "row"].to_numpy()
    rng = np.random.default_rng(seed)
    eta0 = lg_logit[0] + th[rows] + g(age_T[rows])[:, None]          # talent in T+1, before drift
    n_next = t["PA_next"].to_numpy(int)[:, None]

    def predictive(tau_):
        eta = eta0 + rng.normal(0, tau_, eta0.shape) if tau_ > 0 else eta0
        p = expit(eta)
        y = rng.binomial(n_next, p) / n_next
        return p.mean(1), np.quantile(y, 0.1, axis=1), np.quantile(y, 0.9, axis=1)

    t["bayes"], t["lo80"], t["hi80"] = predictive(tau)
    for tg in tau_grid:                                               # dev only: intervals under other drifts
        _, t[f"lo80@{tg}"], t[f"hi80@{tg}"] = predictive(tg)
    t["raw_T"] = (X[rows, 0] / PA[rows, 0])
    t["talent_T"] = expit(lg_logit[0] + th[rows]).mean(1)     # shrunk rate in T, for the shrinkage plot
    t["PA_T"] = PA[rows, 0]
    return t, idata
