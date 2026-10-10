"""Forecast P(air raid alert in the oblast within the next 1 / 3 / 6 hours) with the final model.

The model (XGBoost, tuned hyperparameters) is fitted on all rows whose target window ended before the forecast time,
so nothing after the forecast time is used. Luhansk oblast and Crimea are permanent alerts (probability 1).

Run:
  uv run python -m ua_alerts.predict                         # latest possible forecast: 2026-09-04 00:00 Kyiv
  uv run python -m ua_alerts.predict --at "2026-08-20 03:00" # any full hour (Kyiv time) inside the modelling window
"""
import argparse

import pandas as pd

from ua_alerts import backtest, config, data, features, models

OUT_DIR = config.ROOT / "reports" / "forecasts"


def forecast(at: str | None = None, panel: pd.DataFrame | None = None, feature_list: list[str] = features.FEATURES,
             permanent: bool = True) -> pd.DataFrame:
    if panel is None:
        panel, _ = backtest.load_inputs()
    times = panel.index.get_level_values("time")
    if at is None:
        t = times.max()                                     # last observed hour; forecast time T = t + 1 h
    else:
        t = pd.Timestamp(at, tz=config.LOCAL_TZ).tz_convert("UTC") - pd.Timedelta(hours=1)
        if not (pd.Timestamp(config.REGIME_START) <= t <= times.max()):
            raise ValueError(f"forecast time must be within the modelling window, up to {times.max() + pd.Timedelta(hours=1)}")
    forecast_time = t + pd.Timedelta(hours=1)

    params = backtest.load_tuned_params().get("XGBoost", {})
    rows = panel.loc[t]
    out = pd.DataFrame({"alert_now": rows["active_now"].astype(int)}, index=rows.index)
    for h in config.HORIZONS:
        train = panel[(times >= pd.Timestamp(config.REGIME_START)) & (times + pd.Timedelta(hours=h) < forecast_time)
                      & panel[f"y_{h}"].notna()]
        model = models.XGBoost(features=feature_list, **params).fit(train, h)
        out[f"p_{h}h"] = model.predict(panel.loc[[t]].reset_index("time", drop=True)
                                       .set_index(pd.MultiIndex.from_product([[t], rows.index], names=["time", "region"])))
    for region in config.PERMANENT_REGIONS if permanent else []:
        out.loc[region] = [1] + [1.0] * len(config.HORIZONS)
    out.index.name = "region"
    out.attrs["forecast_time"] = forecast_time
    return out.sort_values("p_3h", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--at", help="forecast time, full hour in Kyiv time, e.g. '2026-08-20 03:00'")
    args = parser.parse_args()
    out = forecast(args.at)
    when = out.attrs["forecast_time"].tz_convert(config.LOCAL_TZ)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"forecast_{when:%Y%m%d_%H%M}.csv"
    out.round(4).to_csv(path)
    print(f"Forecast made at {when:%Y-%m-%d %H:%M} Kyiv time (data up to this moment only)\n")
    print(out.round(3).to_string())
    print(f"\nsaved to {path.relative_to(config.ROOT)}")
    if when >= pd.Timestamp(config.REGIME_END):
        print("note: from 4 Sep 2026 the yellow/red alert system applies; the model was trained on the previous system.")


if __name__ == "__main__":
    main()
