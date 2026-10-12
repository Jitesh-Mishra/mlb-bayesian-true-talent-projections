"""Lahman Batting + People -> one row per player-season (stints merged) with PA, K, BB, HR and baseball age."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
BASE = "https://raw.githubusercontent.com/cdalzell/Lahman/master/data/"
STATS = {"K%": "SO", "BB%": "BB", "HR%": "HR"}


def _fetch(name: str) -> pd.DataFrame:
    """Download a Lahman .RData table once (cached under data/, git-ignored) and read it with pyreadr."""
    import pyreadr
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.RData"
    if not path.exists():
        urllib.request.urlretrieve(BASE + f"{name}.RData", path)
    obj = pyreadr.read_r(str(path))
    return obj[list(obj)[0]]


def player_seasons(min_year: int = 2000) -> pd.DataFrame:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bat, ppl = _fetch("Batting"), _fetch("People")
    bat = bat[bat["yearID"] >= min_year - 3]
    cols = ["AB", "BB", "HBP", "SF", "SH", "SO", "HR", "H"]
    bat = bat.copy()
    bat[cols] = bat[cols].astype(float).fillna(0)
    ps = bat.groupby(["playerID", "yearID"], as_index=False)[cols].sum()   # merge multi-team stints
    ps["PA"] = ps["AB"] + ps["BB"] + ps["HBP"] + ps["SF"] + ps["SH"]
    born = ppl[["playerID", "birthYear", "birthMonth"]].copy()
    ps = ps.merge(born, on="playerID", how="left")
    # baseball age = age on June 30 of the season
    ps["age"] = ps["yearID"] - ps["birthYear"] - (ps["birthMonth"].astype(float).fillna(1) >= 7).astype(int)
    ps["age"] = ps["age"].astype(float)
    ps = ps.drop(columns=["birthYear", "birthMonth"])
    return ps[ps["PA"] > 0].reset_index(drop=True)


def league_rates(ps: pd.DataFrame) -> pd.DataFrame:
    """PA-weighted league rate per season and stat (all batters)."""
    g = ps.groupby("yearID")[["PA", *STATS.values()]].sum()
    return pd.DataFrame({s: g[c] / g["PA"] for s, c in STATS.items()})


def validate(ps: pd.DataFrame) -> list[str]:
    """Data checks; returns a list of problems (empty = clean)."""
    bad = []
    if ps.duplicated(["playerID", "yearID"]).any():
        bad.append("duplicate player-season")
    for c in ["SO", "BB", "HR", "H"]:
        if (ps[c] > ps["PA"]).any():
            bad.append(f"{c} exceeds PA")
    if (ps["HR"] > ps["H"]).any():
        bad.append("HR exceeds H")
    if ps["age"].isna().mean() > 0.01:
        bad.append("more than 1% missing age")
    if ((ps["age"] < 15) | (ps["age"] > 50)).any():
        bad.append("implausible age")
    return bad
