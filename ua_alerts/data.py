"""Loading, cleaning and gridding of air raid alert data."""
import urllib.request

import numpy as np
import pandas as pd

from ua_alerts import config


def download(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in (config.VOLUNTEER_FILE, config.OFFICIAL_FILE):
        path = config.RAW_DIR / name
        if force or not path.exists():
            urllib.request.urlretrieve(config.DATASET_URL.format(name), path)


def _clean(df: pd.DataFrame, max_hours: float) -> pd.DataFrame:
    df = df[~df["region"].isin(config.PERMANENT_REGIONS)]
    df = df[df["finished_at"] > df["started_at"]].copy()
    cap = df["started_at"] + pd.Timedelta(hours=max_hours)
    df["capped"] = df["finished_at"] > cap
    df["finished_at"] = df["finished_at"].where(~df["capped"], cap)
    return df.sort_values("started_at").reset_index(drop=True)


def load_volunteer() -> pd.DataFrame:
    """Oblast-level alerts (eTryvoga), columns: region, started_at, finished_at, naive, capped."""
    df = pd.read_csv(config.RAW_DIR / config.VOLUNTEER_FILE, parse_dates=["started_at", "finished_at"])
    return _clean(df, config.MAX_ALERT_HOURS["volunteer"])


def load_official(levels: list[str] | None = config.OFFICIAL_LEVELS) -> pd.DataFrame:
    """Official alerts with oblast renamed to region; exact duplicate rows removed.

    levels: which alert levels to keep (None = all, including hromada).
    """
    df = pd.read_csv(config.RAW_DIR / config.OFFICIAL_FILE, parse_dates=["started_at", "finished_at"])
    df = df.drop_duplicates().rename(columns={"oblast": "region"})
    if levels is not None:
        df = df[df["level"].isin(levels)]
    return _clean(df, config.MAX_ALERT_HOURS["official"])


def regime_end() -> pd.Timestamp:
    return pd.Timestamp(config.REGIME_END).tz_convert("UTC")


def data_cutoff(df: pd.DataFrame) -> pd.Timestamp:
    """Last full hour we can trust: alerts still running at download time are not in the file."""
    return df["started_at"].max().floor("h")


def hourly_grid(df: pd.DataFrame, end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coverage, starts), both indexed by hour (UTC, left-closed) with one column per region.

    coverage: fraction of the hour during which the region had at least one active alert
              (overlapping alerts, e.g. several raions, are unioned, not summed).
    starts:   number of alerts that started in the hour.
    """
    start = df["started_at"].min().floor("h")
    end = end if end is not None else data_cutoff(df)
    n_min = int((end - start).total_seconds() // 60)
    hours = pd.date_range(start, end, freq="h", inclusive="left")

    coverage, starts = {}, {}
    for region, g in df.groupby("region"):
        s = ((g["started_at"] - start).dt.total_seconds() // 60).astype(int).clip(0, n_min).to_numpy()
        e = ((g["finished_at"] - start).dt.total_seconds() // 60).astype(int).clip(0, n_min).to_numpy()
        delta = np.zeros(n_min + 1, dtype=np.int32)
        np.add.at(delta, s, 1)
        np.add.at(delta, e, -1)
        active = (np.cumsum(delta)[:n_min] > 0).astype(np.float32)
        coverage[region] = active[: len(hours) * 60].reshape(len(hours), 60).mean(axis=1)
        starts[region] = np.bincount(s[s < len(hours) * 60] // 60, minlength=len(hours))[: len(hours)]

    return pd.DataFrame(coverage, index=hours), pd.DataFrame(starts, index=hours)
