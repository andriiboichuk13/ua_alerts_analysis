"""Second track: up-to-date forecasts for cities whose official Telegram channels post every alert and all-clear.

The official dataset stops on 2026-09-07; these channels continue to today, including the yellow/red alert system
introduced on 2026-09-04. Pipeline: messages -> alert intervals -> validation against the official data (until it
stops) -> the same features / models / rolling backtest -> latest forecast.

Run: uv run python -m ua_alerts.city_track     (backtest + latest forecast; scrape first with ua_alerts.telegram)
"""
import re

import numpy as np
import pandas as pd

from ua_alerts import backtest, config, data, features, models, predict, telegram

# start / end: message patterns; unit: the message must also mention this unit (channels covering several raions);
# official: (oblast, raion) in the official dataset used for validation - oblast-level alerts plus alerts of that raion.
CITIES = {
    "Kyiv": dict(channel="kyivcityofficial", start=r"У Києві оголошена", end=r"^\W*Відбій повітряної тривоги",
                 unit=None, official=("Kyiv City", None)),
    "Lviv": dict(channel="dczloda", start=r"(?i)^\W*увага!.*повітряна тривога!",
                 end=r"(?i)^\W*увага!.*відбій повітряної тривоги",
                 unit=r"Львівськ", official=("Lvivska oblast", "Lvivskyi raion")),
    "Odesa": dict(channel="odesacityofficial", start=r"Одеський район\s*[-—–]\s*повітряна тривога",
                  end=r"Одеський район\s*[-—–]\s*відбій", unit=None, official=("Odeska oblast", "Odeskyi raion")),
}
MAX_ALERT_HOURS = config.MAX_ALERT_HOURS["official"]
CITY_FEATURES = [f for f in features.FEATURES
                 if not any(f.startswith(m + "_") for m in config.MACRO_REGIONS)]   # regional state = the tracked cities
PREDICTIONS = config.PROCESSED_DIR / "city_backtest_predictions.parquet"


def _level(text: str) -> str:
    if re.search(r"червон|🔴.*ракетн|Ракетна загроза", text):
        return "red"
    if re.search(r"жовт|🟡|дронова", text, re.I):
        return "yellow"
    return "alert"                                            # before 4 Sep 2026: a single alert level


def parse_intervals(city: str, data_end: pd.Timestamp) -> pd.DataFrame:
    """Alert intervals of one city from its channel. A start while already active and an all-clear while quiet are
    ignored; an alert still running at data_end is cut there; alerts longer than MAX_ALERT_HOURS are capped."""
    c = CITIES[city]
    msgs = telegram.load(c["channel"])
    msgs = msgs[msgs["time"] < data_end]
    rows, open_at, level = [], None, None
    for t, text in zip(msgs["time"], msgs["text"]):
        head = text.strip()[:300]
        if c["unit"] and not re.search(c["unit"], head):
            continue
        if re.search(c["end"], head, re.S):
            if open_at is not None:
                rows.append((open_at, t, level))
                open_at = None
        elif re.search(c["start"], head, re.S):
            if open_at is None:
                open_at, level = t, _level(head)
    if open_at is not None:
        rows.append((open_at, data_end, level))
    df = pd.DataFrame(rows, columns=["started_at", "finished_at", "alert_level"]).assign(region=city)
    cap = df["started_at"] + pd.Timedelta(hours=MAX_ALERT_HOURS)
    df["capped"] = df["finished_at"] > cap
    df["finished_at"] = df["finished_at"].where(~df["capped"], cap)
    return df


def data_end() -> pd.Timestamp:
    """End of the data: the latest message over all channels, floored to the hour. All channels were scraped after
    that moment, so a channel without messages until then was genuinely quiet (no alert started or ended)."""
    return max(telegram.load(c["channel"])["time"].max() for c in CITIES.values()).floor("h")


def load_intervals() -> pd.DataFrame:
    end = data_end()
    return pd.concat([parse_intervals(city, end) for city in CITIES], ignore_index=True)


def official_intervals() -> pd.DataFrame:
    """Official alerts for the same units: oblast-level alerts plus alerts of the city's raion."""
    off = data.load_official()
    parts = []
    for city, c in CITIES.items():
        oblast, raion = c["official"]
        m = (off["region"] == oblast) & ((off["level"] == "oblast") | (off["raion"] == raion) if raion else True)
        parts.append(off[m].assign(region=city))
    return pd.concat(parts, ignore_index=True)


def validate(channel: pd.DataFrame, official: pd.DataFrame, start: str = config.REGIME_START) -> pd.DataFrame:
    """Minute-level agreement per city and month, from `start` until the official data ends:
    precision = share of channel alert time also under official alert; recall = share of official alert time covered;
    plus the share of official alert starts with a channel start within 15 minutes."""
    start = pd.Timestamp(start).tz_convert("UTC")
    end = data.data_cutoff(data.load_official())
    n = int((end - start).total_seconds() // 60)
    months = pd.date_range(start, periods=n, freq="min").tz_convert(config.LOCAL_TZ).strftime("%Y-%m").to_numpy()

    def active(df):
        s = ((df["started_at"] - start).dt.total_seconds() // 60).astype(int).clip(0, n).to_numpy()
        e = ((df["finished_at"] - start).dt.total_seconds() // 60).astype(int).clip(0, n).to_numpy()
        d = np.zeros(n + 1, np.int32)
        np.add.at(d, s, 1)
        np.add.at(d, e, -1)
        return np.cumsum(d)[:n] > 0

    rows = []
    for city in CITIES:
        ch, of = channel[channel.region == city], official[official.region == city]
        a, b = active(ch), active(of)
        t = pd.DataFrame({"month": months, "ch": a, "of": b, "both": a & b}).groupby("month").sum()
        ch_starts = ch["started_at"].sort_values()
        of_starts = of[(of["started_at"] >= start) & (of["started_at"] < end)]["started_at"].sort_values()
        near = pd.merge_asof(of_starts.to_frame("t"), ch_starts.to_frame("c"), left_on="t", right_on="c",
                             direction="nearest", tolerance=pd.Timedelta("15min"))
        for month, r in t.iterrows():
            rows.append({"city": city, "month": month, "channel_hours": r.ch / 60, "official_hours": r.of / 60,
                         "precision": r.both / max(r.ch, 1), "recall": r.both / max(r.of, 1)})
        rows.append({"city": city, "month": "all", "channel_hours": t.ch.sum() / 60, "official_hours": t.of.sum() / 60,
                     "precision": t.both.sum() / max(t.ch.sum(), 1), "recall": t.both.sum() / max(t.of.sum(), 1),
                     "starts_matched_15min": near["c"].notna().mean()})
    return pd.DataFrame(rows)


def load_inputs():
    intervals = load_intervals()
    state = data.hourly_state(intervals, end=data_end())
    return features.build_panel(state), (state["coverage"] > 0).astype(float), intervals


def city_models(indicator: pd.DataFrame) -> list:
    params = backtest.load_tuned_params()          # tuned on the main track; no re-tuning on the city data
    cache = models.ETSCache(indicator)
    return [models.Persistence(), models.HourOfDayRate(),
            *[models.ETSRate(cache, s) for s in models.ETS_SPECS], models.ETSRate(cache, "auto"),
            *[cls(features=CITY_FEATURES, **params.get(cls.base_name, {}))
              for cls in (models.LogisticAR, models.LightGBM, models.XGBoost)]]


def run_backtest(rerun: bool = False) -> pd.DataFrame:
    if PREDICTIONS.exists() and not rerun:
        return pd.read_parquet(PREDICTIONS)
    panel, indicator, _ = load_inputs()
    end = data_end()
    preds = pd.concat([backtest.run(s, "expanding", panel, indicator, city_models(indicator), end=end)
                       for s in ("validation", "test")])
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PREDICTIONS)
    return preds


def latest_forecast(panel: pd.DataFrame) -> pd.DataFrame:
    return predict.forecast(panel=panel, feature_list=CITY_FEATURES, permanent=False)


def main() -> None:
    panel, indicator, intervals = load_inputs()
    print(f"city track data until {data_end():%Y-%m-%d %H:%M} UTC; alerts per city:",
          intervals.groupby("region").size().to_dict())
    print(validate(intervals, official_intervals()).query("month == 'all'").round(3).to_string(index=False))
    preds = run_backtest(rerun=True)
    board = backtest.leaderboard(preds)
    print(board[(board.h == 3)].sort_values(["set", "brier"]).round(4).to_string(index=False))
    out = latest_forecast(panel)
    print(f"\nForecast at {out.attrs['forecast_time'].tz_convert(config.LOCAL_TZ):%Y-%m-%d %H:%M} Kyiv:\n",
          out.round(3).to_string())


if __name__ == "__main__":
    main()


# --- helpers for notebooks/city_track.ipynb -------------------------------------------------------------------------

def official_feed_check(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Was the official feed complete in its last weeks?
    (1) weekly national alert starts, official vs volunteer;
    (2) Kyiv channel alerts 20 Aug - 6 Sep: overlap with an official alert vs with a volunteer alert."""
    off, vol = data.load_official(), data.load_volunteer()
    week = lambda d: d[(d["started_at"] >= "2026-07-06") & (d["started_at"] < "2026-09-07")].groupby(
        d["started_at"].dt.tz_convert(config.LOCAL_TZ).dt.tz_localize(None).dt.to_period("W").dt.start_time.dt.strftime("%m-%d")).size()
    weekly = pd.DataFrame({"official_starts": week(off), "volunteer_starts": week(vol)})
    weekly["volunteer / official"] = (weekly["volunteer_starts"] / weekly["official_starts"]).round(2)
    ch = intervals[(intervals.region == "Kyiv") & (intervals.started_at >= "2026-08-20") & (intervals.started_at < "2026-09-07")]
    of = official_intervals().query("region == 'Kyiv'")
    vk = vol[vol.region == "Kyiv City"]
    overlaps = lambda df: ch.apply(lambda r: ((df.started_at < r.finished_at) & (df.finished_at > r.started_at)).any(), axis=1)
    witness = pd.crosstab(overlaps(of).rename("official has it"), overlaps(vk).rename("volunteer has it"), margins=True)
    return weekly, witness


def plot_regime(coverage: pd.DataFrame) -> None:
    """Daily share of time under alert per city, with the start of the yellow/red system."""
    import matplotlib.pyplot as plt

    from ua_alerts.eda import BLUE, INK2, save
    daily = coverage.tz_convert(config.LOCAL_TZ).resample("D").mean().loc["2026-06-01":] * 100
    fig, axes = plt.subplots(len(daily.columns), 1, figsize=(11, 2.2 * len(daily.columns)), sharex=True)
    for ax, city in zip(axes, daily.columns):
        ax.plot(daily.index, daily[city], color="#b7d3f6", linewidth=0.8)
        ax.plot(daily.index, daily[city].rolling(7, center=True).mean(), color=BLUE)
        ax.axvline(pd.Timestamp(config.REGIME_END), color=INK2, linestyle=":", linewidth=1)
        ax.set_title(f"{city}: % of the day under alert (7-day mean); dotted: yellow/red system from 4 Sep", fontsize=10)
    save(fig, "c01_regime")
