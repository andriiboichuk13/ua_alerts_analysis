"""Feature and target construction for P(alert in oblast within the next h hours).

Row (t, region) = forecast made at the END of hour t (time T = t + 1h). Features use only hours <= t;
target y_h = 1 if the region has any alert during hours t+1 .. t+h.
"""
import numpy as np
import pandas as pd

from ua_alerts import config


def _run_length(mask: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive hours (up to and including t) for which mask is True; 0 where False."""
    m = mask.to_numpy()
    out = np.zeros(m.shape, dtype=np.float32)
    for i in range(m.shape[0]):
        out[i] = np.where(m[i], (out[i - 1] if i else 0) + 1, 0)
    return pd.DataFrame(out, index=mask.index, columns=mask.columns)


def build_panel(state: dict[str, pd.DataFrame], horizons: list[int] = config.HORIZONS) -> pd.DataFrame:
    """Long panel indexed by (time, region) with feature columns and target columns y_{h}."""
    cov, starts, active_end = state["coverage"], state["starts"], state["active_end"]
    any_alert = (cov > 0).astype(np.float32)
    regions = cov.columns
    f: dict[str, pd.DataFrame] = {}

    # Own state and recent history
    f["active_now"] = active_end
    f["cov_0"] = cov
    f["cov_1"] = cov.shift(1)
    for w in (3, 6, 24, 168):
        f[f"cov_mean_{w}h"] = cov.rolling(w, min_periods=1).mean()
    for w in (1, 6, 24):
        f[f"starts_{w}h"] = starts.rolling(w, min_periods=1).sum()
    f["hours_in_alert"] = _run_length(any_alert > 0) * active_end      # length of the current alert episode
    f["hours_quiet"] = _run_length(any_alert == 0)                       # time since the last alert
    # Same hour of day as the first target hour (t+1), on previous days
    f["cov_same_hour_1d"] = cov.shift(23)
    f["cov_same_hour_7d"] = sum(cov.shift(24 * k - 1) for k in range(1, 8)) / 7

    # Regional state (same value for every region at a given time)
    def broadcast(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(s.to_numpy()[:, None], len(regions), axis=1), index=cov.index, columns=regions)

    f["national_active"] = broadcast(active_end.mean(axis=1))
    f["national_starts_1h"] = broadcast((starts > 0).sum(axis=1))
    f["national_starts_3h"] = broadcast((starts.rolling(3, min_periods=1).sum() > 0).sum(axis=1))
    for name, members in config.MACRO_REGIONS.items():
        cols = [c for c in members if c in regions]
        f[f"{name}_active"] = broadcast(active_end[cols].mean(axis=1))
        f[f"{name}_starts_3h"] = broadcast((starts[cols].rolling(3, min_periods=1).sum() > 0).sum(axis=1))

    # Targets: any alert in hours t+1 .. t+h
    for h in horizons:
        fwd = any_alert[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
        f[f"y_{h}"] = fwd

    panel = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    panel.index.names = ["time", "region"]

    local_hour = (panel.index.get_level_values("time") + pd.Timedelta(hours=1)).tz_convert(config.LOCAL_TZ).hour
    panel["hour"] = np.asarray(local_hour, dtype=np.int16)               # Kyiv hour of the first target hour
    panel["region_id"] = pd.Categorical(panel.index.get_level_values("region"), categories=list(regions))
    return panel


FEATURES = [
    "active_now", "cov_0", "cov_1", "cov_mean_3h", "cov_mean_6h", "cov_mean_24h", "cov_mean_168h",
    "starts_1h", "starts_6h", "starts_24h", "hours_in_alert", "hours_quiet", "cov_same_hour_1d", "cov_same_hour_7d",
    "national_active", "national_starts_1h", "national_starts_3h",
    *[f"{m}_active" for m in config.MACRO_REGIONS], *[f"{m}_starts_3h" for m in config.MACRO_REGIONS],
    "hour", "region_id",
]
