"""Forecasters. Each has fit(train_panel, h) and predict(test_panel) -> P(alert within the next h hours)."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ua_alerts.features import FEATURES

NUMERIC = [c for c in FEATURES if c not in ("hour", "region_id")]


def _times(panel):
    return panel.index.get_level_values("time")


def _regions(panel):
    return panel.index.get_level_values("region")


class Persistence:
    """Naive forecast: an alert active now continues, no alert now stays quiet."""
    name = "persistence"

    def fit(self, train, h):
        return self

    def predict(self, test):
        return test["active_now"].to_numpy(dtype=float)


class HourOfDayRate:
    """Climatology: training frequency of y_h per (oblast, Kyiv hour)."""
    name = "hour-of-day rate"

    def fit(self, train, h):
        self.table = train.groupby(["region_id", "hour"], observed=True)[f"y_{h}"].mean()
        self.fallback = train.groupby("region_id", observed=True)[f"y_{h}"].mean()
        return self

    def predict(self, test):
        keys = pd.MultiIndex.from_arrays([test["region_id"], test["hour"]])
        p = self.table.reindex(keys).to_numpy(dtype=float, copy=True)
        missing = np.isnan(p)
        p[missing] = self.fallback.reindex(test["region_id"][missing]).to_numpy()
        return p


# --- Exponential smoothing -------------------------------------------------------------------------------------
# Additive ETS on the hourly 0/1 alert indicator of each oblast. Multiplicative error/seasonality is impossible:
# the series contains zeros. Seasonal period m = 24 h.
ETS_SPECS = {
    "A,N,N": dict(trend=None, damped_trend=False, seasonal=None),
    "A,N,A": dict(trend=None, damped_trend=False, seasonal="add"),
    "A,A,A": dict(trend="add", damped_trend=False, seasonal="add"),
    "A,Ad,A": dict(trend="add", damped_trend=True, seasonal="add"),
}
M = 24


class ETSCache:
    """Shares fitted ETS parameters and filtered states between ETS models and horizons of one backtest run."""

    def __init__(self, indicator: pd.DataFrame):
        self.indicator = indicator          # wide: hour x region, 1 if any alert in the hour
        self.fits, self.states = {}, {}

    def fit(self, region, spec, start, end):
        """Parameters estimated on indicator[start .. end] (inclusive) - data known before the test fold."""
        key = (region, spec, start, end)
        if key not in self.fits:
            y = self.indicator.loc[start:end, region].to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ExponentialSmoothing(y, seasonal_periods=M if ETS_SPECS[spec]["seasonal"] else None,
                                           initialization_method="heuristic", **ETS_SPECS[spec]).fit()
            self.fits[key] = res
        return self.fits[key]

    def forecasts(self, region, spec, start, end, max_h):
        """h-step forecasts (h = 1..max_h) made at every hour t >= start, using fixed parameters from fit(),
        so the state at t depends only on observations <= t. Returns array [n_hours, max_h]."""
        key = (region, spec, start, end)
        if key not in self.states:
            p = self.fit(region, spec, start, end).params
            s = ETS_SPECS[spec]
            y = self.indicator.loc[start:, region]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ExponentialSmoothing(
                    y.to_numpy(), seasonal_periods=M if s["seasonal"] else None, initialization_method="known",
                    initial_level=p["initial_level"], initial_trend=p["initial_trend"] if s["trend"] else None,
                    initial_seasonal=p["initial_seasons"] if s["seasonal"] else None, **s,
                ).fit(smoothing_level=p["smoothing_level"],
                      smoothing_trend=p["smoothing_trend"] if s["trend"] else None,
                      smoothing_seasonal=p["smoothing_seasonal"] if s["seasonal"] else None,
                      damping_trend=p["damping_trend"] if s["damped_trend"] else None, optimized=False)
            n = len(y)
            phi = p["damping_trend"] if s["damped_trend"] else 1.0
            level = res.level
            trend = res.trend if s["trend"] else np.zeros(n)
            season = res.season if s["seasonal"] else None
            out = np.empty((n, max_h))
            idx = np.arange(n)
            for j in range(1, max_h + 1):
                f = level + sum(phi ** k for k in range(1, j + 1)) * trend
                if season is not None:
                    src = idx - M + j                                      # s_{t+j-m}, j <= m
                    f = f + np.where(src >= 0, season[np.clip(src, 0, None)], 0.0)
                out[:, j - 1] = f
            self.states[key] = pd.DataFrame(out, index=y.index)
        return self.states[key]


class ETSRate:
    """ETS forecast of the hourly alert indicator, turned into P(any alert within h hours):
    score = max of the 1..h step forecasts, mapped to a probability by a logistic calibration fitted on training rows.
    spec = one of ETS_SPECS, or "auto" = per-oblast choice of the spec with the lowest AICc on training data."""

    def __init__(self, cache: ETSCache, spec: str):
        self.cache, self.spec = cache, spec
        self.name = "AutoETS" if spec == "auto" else f"ETS({spec})"

    def _spec_for(self, region):
        if self.spec != "auto":
            return self.spec
        return min(ETS_SPECS, key=lambda s: self.cache.fit(region, s, self.start, self.end).aicc)

    def _scores(self, panel):
        times, regions = _times(panel), _regions(panel)
        out = np.empty(len(panel))
        for region in np.unique(regions):
            m = regions == region
            f = self.cache.forecasts(region, self._spec_for(region), self.start, self.end, self.h)
            out[m] = f.reindex(times[m]).to_numpy()[:, : self.h].max(axis=1)
        return out

    def fit(self, train, h):
        self.h = h
        # The indicator up to the last hour of the last training target window is known before the test fold.
        self.start, self.end = _times(train).min(), _times(train).max() + pd.Timedelta(hours=h)
        self.chosen = {r: self._spec_for(r) for r in np.unique(_regions(train))}
        # Calibrate per oblast: the same ETS score means very different probabilities in Lviv and in Donetsk
        # (a single pooled calibration made quiet western oblasts look like ~12 % - see model notebook, section 5).
        scores, y, regions = self._scores(train), train[f"y_{h}"].to_numpy(), _regions(train)
        self.calib = {}
        for region in np.unique(regions):
            m = regions == region
            self.calib[region] = (LogisticRegression().fit(scores[m, None], y[m]) if 0 < y[m].mean() < 1
                                  else float(y[m].mean()))
        return self

    def predict(self, test):
        scores, regions = self._scores(test), _regions(test)
        out = np.empty(len(test))
        for region in np.unique(regions):
            m = regions == region
            c = self.calib[region]
            out[m] = c if isinstance(c, float) else c.predict_proba(scores[m, None])[:, 1]
        return out


# --- Feature-based models ---------------------------------------------------------------------------------------
class LogisticAR:
    """Logistic autoregression: lagged own state, regional state, hour of day and oblast."""
    name = "logistic AR"

    def fit(self, train, h):
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["hour", "region_id"]),
        ])
        self.model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000))
        self.model.fit(train[FEATURES], train[f"y_{h}"])
        return self

    def predict(self, test):
        return self.model.predict_proba(test[FEATURES])[:, 1]


# Both boosters get comparable, fixed hyperparameters chosen before looking at test results.
class LightGBM:
    name = "LightGBM"

    def fit(self, train, h):
        self.model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0,
        )
        self.model.fit(train[FEATURES], train[f"y_{h}"], categorical_feature=["region_id"])
        return self

    def predict(self, test):
        return self.model.predict_proba(test[FEATURES])[:, 1]


class XGBoost:
    name = "XGBoost"

    def fit(self, train, h):
        self.model = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=6, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
            random_state=0, n_jobs=-1,
        )
        self.model.fit(train[FEATURES], train[f"y_{h}"])
        return self

    def predict(self, test):
        return self.model.predict_proba(test[FEATURES])[:, 1]


def all_models(indicator: pd.DataFrame) -> list:
    cache = ETSCache(indicator)
    return [Persistence(), HourOfDayRate(),
            *[ETSRate(cache, s) for s in ETS_SPECS], ETSRate(cache, "auto"),
            LogisticAR(), LightGBM(), XGBoost()]
