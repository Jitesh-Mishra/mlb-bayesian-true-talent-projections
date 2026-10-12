# Decisions

**D1 Data: Lahman via cdalzell/Lahman, fetched at runtime**
`Batting.RData` and `People.RData` are read with pyreadr. They cover 1871-2025, with 128,598 batting rows. Seasons from 1997 on are kept: three seasons of history before 2000, and 15 years of aging pairs before the first target. Stints on different teams in one season are summed. Baseball age is age on June 30. The tables are CC BY-SA, so they are downloaded on first use and cached in `data/` (git-ignored) rather than committed.

**D2 Evaluation population and metric fixed before modelling**
Scored players have 200+ PA in both season T and T+1. Each stat's error is the RMSE between projected and actual rate, weighted by T+1 PA. Coverage is the share of actual T+1 rates inside the 80% interval, unweighted. The brief set these.

**D3 Development and backtest seasons**
Targets 2010-2014 are for development: sanity checks, the parameterisation choice, and the drift term (D5). Targets 2015-2025 are the backtest, run once after everything was frozen. The run is logged in `reports/final_test_runs.json` with the drift settings it used. 2020 (60 games) is skipped as a target, because a 60-game rate is a different quantity, but it stays as an input for 2021.

**D4 Model shape**
- One model per stat, on the logit scale, relative to that season's league rate. This takes out league-wide trends such as the rise in strikeouts.
- Seasons T, T-1 and T-2 enter as binomial likelihoods weighted 1 / 0.8 / 0.6, the same 5/4/3 ratio as Marcel. The weighting stands in for talent changing over time. A random-walk talent process is the principled version and is listed as next.
- The aging curve is not sampled. It is a delta-method quadratic fitted on consecutive-season pairs that end by T, so it uses no future data.
- Parameterisation is **centred**. The non-centred version mixed badly (R-hat 1.06 and bulk ESS 38 on `mu` for K% at origin 2012), because every fitted player has 100+ PA and the likelihood dominates. The centred version gave R-hat 1.00 and ESS above 1,400 on the same fit.
- Fit population: 100+ PA in season T, which keeps out pitchers batting.

**D5 Talent drift term, chosen on dev only**
On the dev targets with no drift, 80% intervals covered 68.8% (K%), 71.5% (BB%) and 79.0% (HR%). The intervals describe talent as of season T, but real talent moves between seasons. A logit-scale drift `Normal(0, tau)` is added to the projection. The rule was set before looking: pick, per stat, the tau on the grid 0.05 to 0.30 whose dev coverage is closest to 80%. That gave K% 0.10 (77.6%), BB% 0.15 (81.1%) and HR% 0.10 (80.3%). The grid is in `reports/drift_calibration_dev.csv`. Point projections are the posterior predictive mean including drift.

**D6 Marcel implementation details**
- Regression is 1,200 PA of league average for every stat, the published Marcel rule.
- The league rate is the player's own PA-weighted mix of the league rates in the seasons he played.
- The age factor applies to the age in the projected season: +0.006 a year under 29, −0.003 a year over 29. It multiplies BB% and HR% by (1 + factor) and K% by (1 − factor), because getting older adds strikeouts.
- There is no reliability or playing-time component; only rates are compared.

**D7 Honest reading of the backtest**
The development seasons showed a 4.3% edge over Marcel on K%. The backtest shows 1.3%, with a 95% interval that includes zero (player-level bootstrap). BB% and HR are ties. The headline is "matches Marcel, beats naive by 8% to 14%, and adds calibrated intervals", not "beats Marcel".
