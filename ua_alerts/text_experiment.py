"""Experiment: do Air Force (@kpszsu) Telegram messages improve the forecasts?

1. aviation_events - strategic-aviation and MiG-31K messages vs the national alert share (lead time).
2. mention_table   - quiet oblast-hours (test set): new-alert rate with / without a mention of the oblast in the
                     hour before the forecast, compared with what the alert-only XGBoost predicted.
3. variants        - XGBoost with different text-feature sets; the variant is SELECTED on the validation folds,
                     and only the selected one is compared with the alert-only model on the test folds.

Run: uv run python -m ua_alerts.text_experiment   (writes reports/text_variants.csv, ~5 min)
"""
import pandas as pd

from ua_alerts import backtest, config, data, features, models, telegram, text_features

RESULTS = config.ROOT / "reports" / "text_variants.csv"


def category_counts(msgs: pd.DataFrame) -> pd.DataFrame:
    out = {c: msgs["text"].str.contains(p, regex=True) for c, p in text_features.CATEGORIES.items()}
    out["MiG-31K take-off"] = msgs["text"].str.contains(r"МіГ-?31", regex=True)
    out["strategic bombers"] = msgs["text"].str.contains(r"Ту-?95|Ту-?160|Ту-?22|стратегічн", regex=True)
    return pd.DataFrame(out).sum().rename("messages").to_frame()


def aviation_events(msgs: pd.DataFrame) -> pd.DataFrame:
    """National share of oblasts under alert when a strategic-aviation message was posted, and in the next 12 h."""
    st = data.hourly_state(data.load_official(), end=data.regime_end())
    nat = st["active_end"].mean(axis=1)
    sub = msgs[msgs["text"].str.contains(r"Ту-?95|Ту-?160|Ту-?22|стратегічн", regex=True)]
    rows = []
    for r in sub.itertuples():
        t0 = r.time.floor("h")
        if t0 > nat.index.max():
            continue
        nxt = nat.loc[t0:t0 + pd.Timedelta(hours=12)]
        rows.append({"posted (Kyiv)": r.time.tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                     "share under alert now": round(nat.loc[t0], 2), "max share next 12 h": round(nxt.max(), 2),
                     "hours to max": round((nxt.idxmax() - r.time).total_seconds() / 3600, 1),
                     "message": r.text[:90].replace("\n", " ")})
    return pd.DataFrame(rows)


def mention_table(preds: pd.DataFrame, panel: pd.DataFrame, h: int = 3) -> pd.DataFrame:
    """Quiet oblast-hours in the test set: actual new-alert rate vs alert-only XGBoost prediction, by macro-region
    and whether the oblast was mentioned in the hour before the forecast."""
    p = preds[(preds.set == "test") & (preds.window == "expanding") & (preds.h == h)].reset_index()
    idx = pd.MultiIndex.from_frame(p[["time", "region"]])
    p["active_now"] = panel["active_now"].reindex(idx).to_numpy()
    p["mentioned"] = panel["tg_mentions_1h"].reindex(idx).to_numpy() > 0
    p = p[p.active_now == 0]
    macro = {r: m for m, rs in config.MACRO_REGIONS.items() for r in rs}
    p["macro-region"] = p["region"].map(macro)
    return (p.groupby(["macro-region", "mentioned"])
            .agg(hours=("y", "size"), actual_new_alert_rate=("y", "mean"), xgboost_predicted=("XGBoost", "mean"))
            .round(3))


def variants(panel: pd.DataFrame, h: int = config.TUNING_HORIZON) -> pd.DataFrame:
    panel = panel.copy()
    for w in ("1h", "3h"):
        panel[f"tg_mentioned_{w}"] = (panel[f"tg_mentions_{w}"] > 0).astype(float).where(panel[f"tg_mentions_{w}"].notna())
    f = features.FEATURES
    sets = {"no text": f, "all text": f + features.TEXT_FEATURES,
            "mention counts": f + ["tg_mentions_1h", "tg_mentions_3h"],
            "mention yes/no": f + ["tg_mentioned_1h", "tg_mentioned_3h"]}
    params = backtest.load_tuned_params().get("XGBoost", {})

    def run(fold_set, names):
        rows = []
        for name in names:
            for k, (start, end) in enumerate(backtest.folds(fold_set)):
                train, test = backtest.split(panel, h, start, end)
                p = models.XGBoost(features=sets[name], **params).fit(train, h).predict(test)
                y, quiet = test[f"y_{h}"].to_numpy(), test["active_now"].to_numpy() == 0
                rows.append({"set": fold_set, "variant": name, "fold": k, "brier_all": backtest.metrics(y, p)["brier"],
                             "brier_quiet": backtest.metrics(y[quiet], p[quiet])["brier"]})
        return pd.DataFrame(rows)

    val = run("validation", list(sets))
    best = val.groupby("variant")["brier_quiet"].mean().idxmin()
    test = run("test", ["no text", best] if best != "no text" else ["no text"])
    out = pd.concat([val, test]).groupby(["set", "variant"])[["brier_all", "brier_quiet"]].mean().reset_index()
    out["selected_on_validation"] = out["variant"] == best
    return out


def main() -> None:
    panel, _ = backtest.load_inputs()
    res = variants(panel)
    res.to_csv(RESULTS, index=False)
    print(res.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
