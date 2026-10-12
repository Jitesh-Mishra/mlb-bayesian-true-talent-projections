"""Data validation on the real Lahman tables (downloaded once, cached in data/)."""
import pandas as pd
import pytest

from mlbproj.data import league_rates, player_seasons, validate


@pytest.fixture(scope="module")
def ps():
    return player_seasons()


def test_validation_passes(ps):
    assert validate(ps) == []


def test_one_row_per_player_season(ps):
    assert not ps.duplicated(["playerID", "yearID"]).any()


def test_stints_are_merged(ps):
    # Hunter Pence 2012: Phillies then Giants, 688 PA in total
    row = ps[(ps.playerID == "pencehu01") & (ps.yearID == 2012)]
    assert len(row) == 1 and int(row.PA.iloc[0]) == 688


def test_pa_formula(ps):
    assert (ps.PA == ps.AB + ps.BB + ps.HBP + ps.SF + ps.SH).all()


def test_league_rates_plausible(ps):
    lg = league_rates(ps).loc[2005:2025]
    assert lg["K%"].between(0.15, 0.26).all()
    assert lg["BB%"].between(0.06, 0.11).all()
    assert lg["HR%"].between(0.015, 0.045).all()
    assert lg.loc[2019, "K%"] > lg.loc[2010, "K%"]          # the strikeout era


def test_validate_catches_bad_rows():
    bad = pd.DataFrame({"playerID": ["a", "a"], "yearID": [2020, 2020], "PA": [10, 10], "SO": [11, 1],
                        "BB": [0, 0], "HR": [2, 0], "H": [1, 0], "age": [27.0, 27.0]})
    problems = validate(bad)
    assert "duplicate player-season" in problems and "SO exceeds PA" in problems and "HR exceeds H" in problems
