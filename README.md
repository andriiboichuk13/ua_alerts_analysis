# Air raid alerts in Ukraine — time series analysis and short-term forecasting

For every oblast and every hour, the model predicts the probability that the oblast will be under an air raid alert
at some point within the next **1, 3 or 6 hours**, using only information available at the forecast time.

![Forecasts vs actual alerts](reports/figures/m08_predictions_h3.png)

## Results (in short)

Evaluated on the **last 5 weeks before 4 Sep 2026** (held-out test set, 4 rolling folds), after tuning on 4 validation folds
spread over Feb–Jun 2026. Brier skill is measured against a per-oblast hour-of-day baseline (higher is better).

| h = 3 h | all hours | **quiet hours only** (will a *new* alert start?) |
|---|---|---|
| XGBoost — Brier skill, test | 0.36 | **0.19** |
| XGBoost — Brier skill, validation | 0.28 | **0.09** |
| XGBoost — mean per-oblast ROC AUC, test / validation | 0.83 / 0.83 | **0.76 / 0.72** |

* **Final model: XGBoost** (expanding training window, tuned). LightGBM is equivalent; a logistic autoregression is a close,
  interpretable runner-up and the best model in the quiet western oblasts. All of them clearly beat the baselines and the ETS family,
  with the same ranking on the validation and the test folds.
* **Most of the headline skill comes from predicting that a running alert continues** (if an alert is active now, the target is 1 in
  99.9 % of cases). For new alerts the model gives a useful but limited early warning — mainly from alerts already running in other oblasts.
* **Text data did not help.** All Air Force Telegram posts (Oct 2025 – Sep 2026, ~31 000 messages: drone tracks, launches,
  aircraft take-offs) were scraped and tested as features, with the feature set chosen on validation folds only. They did not
  improve the forecasts — the information is largely already in the regional alert state, and the channel's activity doubled over
  the year. Aircraft take-off warnings can give hours of lead time but occur only a handful of times; MiG-31K take-offs coincide with the alert.
* Seasonality is weak: the daily cycle is large in relative terms in the rear (up to 7× between night and day) but explains only a few
  percent of the variance; no weekly or monthly seasonality was found.

## Second track: up-to-date forecasts for Kyiv, Lviv and Odesa

The main track has to stop on 4 Sep 2026. A second track (`notebooks/city_track.ipynb`, `ua_alerts/city_track.py`) continues **until
today** for cities whose official Telegram channels post every alert and all-clear: Kyiv (`@kyivcityofficial`), Lviv (`@dczloda`, Lvivskyi
raion) and Odesa (`@odesacityofficial`, Odeskyi raion). The messages are parsed into alert intervals and **validated against the official data**
while it exists (recall 0.98–1.00, ≥ 98 % of alert starts matched within 15 min). Same features, models and backtest; the test period
(5 Sep – 10 Oct 2026) lies entirely in the new yellow/red system. No official channel posting alerts was found for Kharkiv.

| forecast made 2026-10-11 01:00 Kyiv | alert now | P(alert within 1 h) | 3 h | 6 h |
|---|---|---|---|---|
| Kyiv | yes | 0.99 | 0.99 | 1.00 |
| Odesa | no | 0.17 | 0.46 | 0.71 |
| Lviv | no | 0.04 | 0.06 | 0.07 |

* The models stay calibrated after the switch (they follow the current state), but early warning of new alerts is weaker than in the
  main track (per-city ROC AUC on quiet hours 0.56–0.69): three cities give a much poorer "regional view" than 24 oblasts.
* The channels revealed that the **official feed was already incomplete in its last ~2 weeks** (see below).

## Data — and why the evaluation stops on 4 September 2026

Source: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT),
snapshot from 2026-10-10 committed in `data/raw/`. Finding a reliable, *current* source was the hardest part of the project:

* The **official** data (raion level) is reliable but **stopped on 2026-09-07**, because its source — the official Telegram channel
  `@air_alert_ua` — stopped posting.
* The **volunteer** data (eTryvoga) is up to date but **unreliable since 2024**: measured minute by minute against the official data,
  it covers only 31 % of official alert time in 2026, splits long alerts into short pieces, and contains spurious alerts in two oblasts
  (`notebooks/eda.ipynb`, section 1).
* Other sources were checked and rejected: APIs fed by the same stopped channel, regional administration channels (alerts are posted
  only for Lviv and Kyiv, not for the front line), alerts.in.ua (history only with a manually reviewed API token).
* On **4 Sep 2026** Ukraine introduced a two-level (yellow / red) alert system, so alerts after that date mean something different anyway.
* **Caveat found later (city track):** in its last ~2 weeks the official feed was already **missing alerts** — Kyiv alerts posted by the
  city administration and confirmed by the volunteer data are absent, and nationally the official feed records fewer alert starts while the
  volunteer data records more (volunteer/official ratio 0.29–0.35 all summer, then 0.40 / 0.44 / 0.60). The labels of the main track's last
  test fold (26 Aug – 3 Sep) are therefore probably incomplete.

**Decision:** use the official data from **1 Nov 2025** (raion-level alerting) to **4 Sep 2026 00:00 Kyiv** — the latest period in which
reliable data exists and "alert" has a stable meaning. An oblast counts as under alert if any oblast- or raion-level alert is active in it.

## Method

* **Features** (`ua_alerts/features.py`): the oblast's current state and recent history, regional state (national and per macro-region
  share of oblasts under alert, new alerts in the last hours), Kyiv hour of day, oblast identity.
* **Models** (`ua_alerts/models.py`): persistence, hour-of-day rate, ETS(A,N,N) / (A,N,A) / (A,A,A) / (A,Ad,A) and AutoETS (per-oblast
  AICc selection), logistic autoregression, LightGBM, XGBoost — the last three also with Air Force text features.
* **Text features** (`ua_alerts/telegram.py`, `text_features.py`, `text_experiment.py`): public posts of the Air Force channel `@kpszsu`
  scraped from the web preview; per-oblast mentions and national counts by message type, using only messages published before the forecast time.
* **Validation** (`ua_alerts/backtest.py`, `ua_alerts/tuning.py`): rolling-origin backtest with expanding and sliding (12-week) training
  windows; training rows whose target window overlaps the evaluated fold are purged; hyperparameters are tuned on the validation folds
  only (time-based — random k-fold would leak through autocorrelated neighbouring hours); the test folds are used once.
* **No leakage**: `tests/test_leakage.py` rebuilds all features (alert and text) from data truncated at the forecast time and checks they
  are identical; the test was verified to fail on deliberately injected look-aheads in both.

## Repository

```
data/raw/                 dataset snapshot (+ LICENSE, README with snapshot date); telegram/kpszsu.jsonl - Air Force posts
notebooks/eda.ipynb       data sources and quality, trend, seasonality (STL / MSTL, ACF, periodogram), regional dependence
notebooks/model.ipynb     training and validation: tuning, test results, onset analysis, ETS experiments, text features, per-oblast results
notebooks/city_track.ipynb  second track: Kyiv / Lviv / Odesa from official city channels, validation, backtest, latest forecast
ua_alerts/                data loading, features, models, tuning, backtest, evaluation plots, predict
reports/figures/          all figures;  reports/tuned_params.json, tuning_results.csv
tests/test_leakage.py     leakage test
```

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                                   # install dependencies
uv run python -m pytest tests             # leakage test
uv run python -m ua_alerts.eda            # EDA figures (headless)
uv run python -m ua_alerts.tuning         # hyperparameter search on validation folds (~5 min)
uv run python -m ua_alerts.backtest       # full backtest, all models (~10 min)
uv run python -m ua_alerts.text_experiment  # text-feature selection on validation folds, then test (~5 min)
uv run python -m ua_alerts.telegram kpszsu --from-id 45500 --to-id 76500   # re-scrape Air Force posts
uv run python -m ua_alerts.telegram kyivcityofficial --from-id 15500 --to-id <latest id>   # city channels (also dczloda, odesacityofficial)
uv run python -m ua_alerts.city_track     # city track: validation, backtest, latest forecast
uv run python -m ua_alerts.predict        # final forecast at 2026-09-04 00:00 Kyiv
uv run python -m ua_alerts.predict --at "2026-08-20 03:00"   # forecast for any hour in the modelling window
```

The notebooks run end to end from `notebooks/` (`uv run jupyter notebook`); `model.ipynb` reads the cached backtest predictions
(`backtest.run_all(rerun=True)` retrains everything).

## Limitations and next steps

* The test period ends on 4 Sep 2026; under the yellow/red system the model would have to be re-trained on new data.
* The model predicts the *official alert*, not the threat itself. Results for the western oblasts rest on very few alerts.
* Early warning of **new** alerts is limited. Text mention counts were not enough; a next step would be to model drone *routes*
  (direction, speed, distance to each oblast) from the Air Force posts and to normalise for the channel's changing activity.
* Rare, long-horizon warnings (e.g. public warnings of a massive attack "within 48 hours") were considered but not used: a few events per
  year cannot be learned from, and a 48-hour window does not match a 1–6-hour target — they would suit a daily risk-level forecast.

---
Built with Claude Code as the primary engineering tool; the full AI conversation log is submitted separately.
