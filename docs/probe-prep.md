# Probe prep (answers come only from what this run did)

Before an interview, run `pytest -q` and open `reports/results_eval.csv`, `reports/per_season_eval.csv` and `reports/stabilization.csv`.

## How it works, in 5 lines
1. `mlbproj/data.py` turns Lahman Batting + People into one row per player-season: stints merged, PA = AB + BB + HBP + SF + SH, age on June 30. `validate()` and pytest check it.
2. `mlbproj/baselines.py` computes last-season rates and Marcel (5/4/3, 1,200 PA of league, age factor).
3. `mlbproj/model.py` fits one PyMC model per stat and season. Talent is `Normal(mu + b·age, sigma)` on the logit scale, and three seasons enter as binomial evidence weighted 1 / 0.8 / 0.6, offset by each season's league rate. The projection adds an aging curve and a drift term, then binomial noise at actual PA for the 80% interval.
4. `mlbproj/evaluate.py` runs a rolling-origin backtest. It fits on seasons before the target and scores hitters with 200+ PA in both seasons, using PA-weighted RMSE and coverage. Development targets are 2010-2014; the backtest is 2015-2025, run once, with 2020 skipped as a target.
5. Diagnostics are logged for every fit. Across all 30 backtest fits: max R-hat 1.010, min bulk ESS 2,104, 0 divergences.

## Numbers to know
- Backtest, 10 seasons, 2,528 player-seasons per stat.
- RMSE vs last season: K% −7.8%, BB% −13.4%, HR −14.2%.
- RMSE vs Marcel: K% −1.3% (CI −2.8% to +0.1%), BB% +0.3%, HR +0.9%. A tie.
- 80% coverage: 76.9% / 80.9% / 77.4%.
- Talent sd (logit): 0.337 / 0.332 / 0.453. Implied stabilization: 51 / 119 / 164 PA.

## Probes

**Q: So you didn't beat Marcel?**
- 15s: No, it ties Marcel within about 1% on all three stats, and I report it that way. On the dev seasons it looked 4% better on K%, but that shrank to 1.3% on the backtest, with a CI that includes zero. Marcel is a strong baseline because it already encodes the right idea. What the Bayesian model adds is calibrated intervals and learned shrinkage per stat, not a lower RMSE.

**Q: Why hierarchical instead of just regressing to the mean?**
- Marcel uses 1,200 PA of regression for every stat. The model learns how much talent varies across hitters, per stat. K% varies a lot relative to its noise, so it stabilizes in about 50 PA; HR needs about 165. Those are close to the published stabilization points (about 60 / 120 / 170), which is a good sanity check that the model learned something real.

**Q: Your intervals were 69% on dev. What happened?**
- The posterior describes talent as of last season, and talent changes year to year. I added a drift term and chose its size per stat on the dev seasons only, with a rule fixed in advance (closest to 80%). On the backtest that gave 77% / 81% / 77%. K% and HR still run 3 points light, so the next step is drift that depends on age.

**Q: How do you know there's no leakage?**
- Every fit uses only seasons up to T: the likelihood, the aging curve (pairs ending by T) and the league rate. The drift term was set on 2010-2014 and frozen. The backtest ran once, logged in `final_test_runs.json`.

**Q: Why centred parameterisation? Everyone says non-centred.**
- Non-centred helps when data per group is weak. Every player here has 100+ PA, so the data dominates, and non-centred mixed badly: R-hat 1.06 and ESS 38 on `mu`. Centred gave R-hat 1.00 and ESS above 1,400.

**Q: Why are HR-rate results so noisy by season?**
- Neither method knows next year's league rate, and the HR environment jumped around: +20% in 2019, −12% in 2022. The model's error relative to Marcel swings from 9% better to 7% worse across seasons. The fix is to model the league environment as a latent season effect, or use batted-ball inputs.

**Q: Why the 1 / 0.8 / 0.6 weights? That's not very Bayesian.**
- It's a power likelihood, a stand-in for talent changing over time, and it keeps the comparison with Marcel apples to apples. The principled version is a random-walk talent process, which would learn the decay. It's the next model.

**Q: What would you do with Orioles data?**
- Replace outcomes with inputs that stabilize faster: exit velocity and barrel rate for HR, whiff and chase rate for K% and BB%, as priors or covariates. Add park and level adjustments, and the same model gives minor-league translations with honest uncertainty.

**Q: What's the PPC telling you?**
- Simulated seasons match the observed shape, including the HR skew. K% is slightly flatter than the simulations, because one fixed talent over three seasons is too rigid. That's consistent with the coverage gap.
