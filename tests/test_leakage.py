"""Features at forecast time T must be identical whether or not data after T exists.

Run: uv run python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from ua_alerts import data, features, telegram


def test_features_do_not_use_future_data():
    off = data.load_official()
    msgs = telegram.load("kpszsu")
    end = data.regime_end()
    full = features.build_panel(data.hourly_state(off, end=end), messages=msgs)
    rng = np.random.default_rng(0)
    hours = full.index.get_level_values("time").unique()
    hours = hours[(hours >= pd.Timestamp("2026-05-01", tz="UTC")) & (hours < end - pd.Timedelta(hours=2))]
    for t in rng.choice(hours, size=3, replace=False):
        t = pd.Timestamp(t)
        forecast_time = t + pd.Timedelta(hours=1)
        known = off[off["started_at"] < forecast_time]          # alerts still running at T are clipped at T
        known_msgs = msgs[msgs["time"] < forecast_time]          # only messages published before T
        partial = features.build_panel(data.hourly_state(known, end=forecast_time), messages=known_msgs)
        cols = features.FEATURES + features.TEXT_FEATURES
        a = full.loc[t, cols].sort_index()
        b = partial.loc[t, cols].sort_index()
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
