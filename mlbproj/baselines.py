"""Baselines: naive last-season rate and Marcel (Tango's 5/4/3 system), both projecting season T+1 from seasons up to T."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import STATS

WEIGHTS = (5.0, 4.0, 3.0)      # seasons T, T-1, T-2
REGRESS_PA = 1200.0            # PA of league-average performance added to every player


def history(ps: pd.DataFrame, T: int) -> pd.DataFrame:
    """Wide table: one row per player with PA and counts for T, T-1, T-2 (0 if absent) and age in T."""
    h = ps[ps["yearID"].between(T - 2, T)]
    cols = ["PA", *STATS.values()]
    wide = h.pivot_table(index="playerID", columns="yearID", values=cols, aggfunc="sum", fill_value=0)
    wide.columns = [f"{c}_{T - y}" for c, y in wide.columns]       # e.g. PA_0 = season T, SO_2 = season T-2
    for c in cols:
        for lag in range(3):
            if f"{c}_{lag}" not in wide:
                wide[f"{c}_{lag}"] = 0.0
    age = ps[ps["yearID"] == T].set_index("playerID")["age"]
    return wide.join(age, how="inner")                               # must have played in T


def naive(hist: pd.DataFrame, stat: str) -> pd.Series:
    return hist[f"{STATS[stat]}_0"] / hist["PA_0"]


def marcel_age_factor(age_next: pd.Series) -> pd.Series:
    """Marcel aging: +0.6% per year under 29, -0.3% per year over 29 (age in the projected season)."""
    return np.where(age_next < 29, 0.006 * (29 - age_next), -0.003 * (age_next - 29))


def marcel(hist: pd.DataFrame, stat: str, lg: pd.DataFrame, T: int) -> pd.Series:
    """Weighted 3-season rate regressed toward the league with 1,200 PA, then age-adjusted.
    League rate is the player's own PA-weighted mix of the league rates in the seasons he played.
    Age moves BB% and HR% up for young players and K% down (the inverse direction)."""
    c = STATS[stat]
    w_pa = sum(w * hist[f"PA_{k}"] for k, w in enumerate(WEIGHTS))
    w_x = sum(w * hist[f"{c}_{k}"] for k, w in enumerate(WEIGHTS))
    lg_rate = sum(w * hist[f"PA_{k}"] * lg.loc[T - k, stat] for k, w in enumerate(WEIGHTS)) / w_pa
    rate = (w_x + REGRESS_PA * lg_rate) / (w_pa + REGRESS_PA)
    adj = marcel_age_factor(hist["age"] + 1)
    return rate * (1 - adj) if stat == "K%" else rate * (1 + adj)
