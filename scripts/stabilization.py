"""Implied stabilization point per stat: the PA at which a hitter's observed rate gets 50% weight against
the population (n = 1 / (sigma^2 * p * (1 - p)), the logit-scale prior spread converted to the rate scale).
Uses the mean posterior sigma across the backtest fits. python scripts/stabilization.py -> reports/stabilization.csv"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlbproj.data import league_rates, player_seasons  # noqa: E402

d = pd.read_csv(ROOT / "reports" / "diagnostics_eval.csv")
lg = league_rates(player_seasons()).loc[2014:2024]           # origin seasons of the backtest
rows = []
for stat, g in d.groupby("stat", sort=False):
    sigma, p = g["sigma"].mean(), lg[stat].mean()
    rows.append({"stat": stat, "sigma_logit": sigma, "league_rate": p, "stabilization_pa": 1 / (sigma**2 * p * (1 - p))})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "reports" / "stabilization.csv", index=False, float_format="%.4f")
print(out.round(3).to_string(index=False))
