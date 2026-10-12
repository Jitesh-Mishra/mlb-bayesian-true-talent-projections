"""Small simulated check: the hierarchical model recovers known talent better than raw rates and shrinks
small samples harder. Fast settings so it runs in CI."""
import numpy as np
import pandas as pd

from mlbproj.model import fit_project


def test_recovers_talent_and_shrinks():
    rng = np.random.default_rng(1)
    n, lg_k = 150, 0.2
    talent = 1 / (1 + np.exp(-(np.log(lg_k / (1 - lg_k)) + rng.normal(0, 0.35, n))))
    rows = []
    for i in range(n):
        pa = int(rng.integers(100, 650))
        for k, y in enumerate([2018, 2019, 2020]):
            p = pa if k == 2 else int(rng.integers(0, 650))
            if p:
                rows.append({"playerID": f"p{i}", "yearID": y, "PA": p, "SO": rng.binomial(p, talent[i]),
                             "BB": 0, "HR": 0, "H": 0, "age": 28.0 + y - 2020})
    ps = pd.DataFrame(rows)
    lg = pd.DataFrame({"K%": [lg_k] * 3, "BB%": [0.08] * 3, "HR%": [0.03] * 3}, index=[2018, 2019, 2020])
    tg = pd.DataFrame({"playerID": [f"p{i}" for i in range(n)], "PA_next": 500})
    out, idata = fit_project(ps, lg, "K%", 2020, tg, draws=300, tune=300, chains=2, seed=0)
    truth = pd.Series(talent, index=[f"p{i}" for i in range(n)]).loc[out.playerID].to_numpy()
    assert np.sqrt(np.mean((out.talent_T - truth) ** 2)) < np.sqrt(np.mean((out.raw_T - truth) ** 2))
    shrink = (out.raw_T - out.talent_T).abs() / (out.raw_T - lg_k).abs().clip(lower=1e-3)
    small, big = out.PA_T < 250, out.PA_T > 500
    assert shrink[small].median() > shrink[big].median()
    assert (out.lo80 < out.hi80).all()
    assert abs(float(idata.posterior["sigma"].mean()) - 0.35) < 0.12   # population spread recovered
