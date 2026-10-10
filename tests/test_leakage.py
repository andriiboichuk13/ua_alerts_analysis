"""Features at forecast time T must be identical whether or not data after T exists.

Run: uv run python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from ua_alerts import data, features


def test_features_do_not_use_future_data():
    off = data.load_official()
    end = data.regime_end()
    full = features.build_panel(data.hourly_state(off, end=end))
    rng = np.random.default_rng(0)
    hours = full.index.get_level_values("time").unique()
    hours = hours[(hours >= pd.Timestamp("2026-05-01", tz="UTC")) & (hours < end - pd.Timedelta(hours=2))]
    for t in rng.choice(hours, size=3, replace=False):
        t = pd.Timestamp(t)
        forecast_time = t + pd.Timedelta(hours=1)
        known = off[off["started_at"] < forecast_time]          # alerts still running at T are clipped at T
        partial = features.build_panel(data.hourly_state(known, end=forecast_time))
        a = full.loc[t, features.FEATURES].sort_index()
        b = partial.loc[t, features.FEATURES].sort_index()
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
