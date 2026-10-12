import numpy as np
import pandas as pd
import pytest

from mlbproj.baselines import REGRESS_PA, history, marcel, marcel_age_factor, naive


def toy():
    rows = []
    for y, pa, so in [(2018, 600, 120), (2019, 500, 100), (2020, 200, 40)]:
        rows.append({"playerID": "p", "yearID": y, "PA": pa, "SO": so, "BB": 50, "HR": 20, "H": 150, "age": 30 + y - 2020})
    rows.append({"playerID": "q", "yearID": 2020, "PA": 100, "SO": 40, "BB": 5, "HR": 1, "H": 20, "age": 24})
    lg = pd.DataFrame({"K%": [0.2] * 3, "BB%": [0.08] * 3, "HR%": [0.03] * 3}, index=[2018, 2019, 2020])
    return pd.DataFrame(rows), lg


def test_history_lags():
    ps, _ = toy()
    h = history(ps, 2020)
    assert h.loc["p", "PA_0"] == 200 and h.loc["p", "PA_2"] == 600 and h.loc["q", "PA_1"] == 0


def test_naive_is_last_season():
    ps, _ = toy()
    assert naive(history(ps, 2020), "K%")["q"] == pytest.approx(0.4)


def test_marcel_regresses_and_ages():
    ps, lg = toy()
    h = history(ps, 2020)
    # p is a league-average K hitter (20%) every year, so only aging moves him: age 31 next year
    assert marcel(h, "K%", lg, 2020)["p"] == pytest.approx(0.2 * (1 + 0.003 * 2))
    # q: 40% in 100 PA gets pulled most of the way to 20%
    w = 5 * 100
    raw = (5 * 40 + REGRESS_PA * 0.2) / (w + REGRESS_PA)
    assert marcel(h, "K%", lg, 2020)["q"] == pytest.approx(raw * (1 - 0.006 * 4))


def test_age_factor_sign():
    assert np.allclose(marcel_age_factor(pd.Series([25, 29, 35])), [0.024, 0.0, -0.018])
