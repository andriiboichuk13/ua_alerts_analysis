"""Rolling-origin backtest.

Fold sets:
  validation - VALIDATION_FOLDS spread over Feb-Jun 2026; used for hyperparameter tuning and seasonal robustness;
  test       - the last TEST_DAYS before the regime end in N_FOLDS consecutive folds; used once, for final scores.
Training-window schemes:
  expanding - train on everything from REGIME_START up to the fold;
  sliding   - train only on the SLIDING_TRAIN_DAYS before the fold (adapts faster to level shifts).

Run: uv run python -m ua_alerts.backtest
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ua_alerts import config, data, features, models, telegram

PREDICTIONS = config.PROCESSED_DIR / "backtest_predictions.parquet"
TUNED_PARAMS = config.ROOT / "reports" / "tuned_params.json"
WINDOWS = {"expanding": None, "sliding": config.SLIDING_TRAIN_DAYS}


def folds(fold_set: str = "test", end: pd.Timestamp | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) of each fold in UTC. end: end of the data for the test folds (default: the regime end)."""
    if fold_set == "validation":
        kyiv = lambda d: pd.Timestamp(d, tz=config.LOCAL_TZ).tz_convert("UTC")
        return [(kyiv(a), kyiv(b)) for a, b in config.VALIDATION_FOLDS]
    end = end if end is not None else data.regime_end()
    edges = pd.date_range(end - pd.Timedelta(days=config.TEST_DAYS), end, periods=config.N_FOLDS + 1)
    return list(zip(edges[:-1], edges[1:]))


def split(panel: pd.DataFrame, h: int, test_start: pd.Timestamp, test_end: pd.Timestamp,
          train_days: int | None = None):
    """Training rows: the whole target window (t+1 .. t+h) ends before test_start (purging), starting at REGIME_START
    or, for a sliding window, train_days before test_start. Test rows: t in [test_start, test_end), complete target."""
    t = panel.index.get_level_values("time")
    start = pd.Timestamp(config.REGIME_START)
    if train_days is not None:
        start = max(start, test_start - pd.Timedelta(days=train_days))
    target = panel[f"y_{h}"].notna()
    train = panel[(t >= start) & (t + pd.Timedelta(hours=h) < test_start) & target]
    test = panel[(t >= test_start) & (t < test_end) & target]
    return train, test


def load_inputs():
    """(panel with alert and Air Force text features, hourly 0/1 alert indicator)."""
    state = data.hourly_state(data.load_official(), end=data.regime_end())
    return features.build_panel(state, messages=telegram.load("kpszsu")), (state["coverage"] > 0).astype(float)


def load_tuned_params() -> dict:
    return json.loads(TUNED_PARAMS.read_text()) if TUNED_PARAMS.exists() else {}


def run(fold_set: str, window: str, panel: pd.DataFrame, indicator: pd.DataFrame, model_list: list,
        horizons: list[int] = config.HORIZONS, verbose: bool = True, end: pd.Timestamp | None = None) -> pd.DataFrame:
    preds = []
    for k, (start, end) in enumerate(folds(fold_set, end)):
        for h in horizons:
            train, test = split(panel, h, start, end, WINDOWS[window])
            out = test[[f"y_{h}"]].rename(columns={f"y_{h}": "y"}).assign(set=fold_set, window=window, h=h, fold=k)
            for model in model_list:
                out[model.name] = model.fit(train, h).predict(test)
            preds.append(out)
            if verbose:
                print(f"{fold_set:10s} {window:9s} fold={k} h={h} train={len(train):,} "
                      f"[{train.index.get_level_values('time').min():%Y-%m-%d} ..] test={len(test):,} "
                      f"[{start:%m-%d %H:%M} .. {end:%m-%d %H:%M})")
    return pd.concat(preds)


def run_all(rerun: bool = False) -> pd.DataFrame:
    """Predictions of all models (tuned hyperparameters) for both fold sets and window schemes; cached."""
    if PREDICTIONS.exists() and not rerun:
        return pd.read_parquet(PREDICTIONS)
    panel, indicator = load_inputs()
    params = load_tuned_params()
    preds = pd.concat([run(s, w, panel, indicator, models.all_models(indicator, params))
                       for s in ("validation", "test") for w in WINDOWS])
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PREDICTIONS)
    return preds


META = ("y", "set", "window", "h", "fold", "time", "region", "state")


def model_names(preds: pd.DataFrame) -> list[str]:
    return [c for c in preds.columns if c not in META]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
        "base_rate": y.mean(),
        "n": len(y),
    }


def summarize(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Metrics per group in `by` and model (long format)."""
    rows = []
    for keys, g in preds.reset_index().groupby(by, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in model_names(preds):
            rows.append({**dict(zip(by, keys)), "model": m, **metrics(g["y"].to_numpy(), g[m].to_numpy())})
    return pd.DataFrame(rows)


def leaderboard(preds: pd.DataFrame, by: tuple[str, ...] = ("set", "window", "h")) -> pd.DataFrame:
    """Pooled metrics plus the mean of per-oblast ROC AUCs (pooled AUC is inflated by front-vs-rear differences)
    and the Brier skill vs the hour-of-day rate."""
    by = list(by)
    pooled = summarize(preds, by)
    per_oblast = summarize(preds, by + ["region"])
    macro = per_oblast.groupby(by + ["model"])["roc_auc"].mean().rename("roc_auc_mean_per_oblast")
    out = pooled.merge(macro.reset_index(), on=by + ["model"])
    ref = out[out.model == "hour-of-day rate"].set_index(by)["brier"]
    out["brier_skill"] = 1 - out["brier"] / out.set_index(by).index.map(ref).to_numpy()
    return out.drop(columns=["n"])


def main() -> None:
    preds = run_all(rerun=True)
    board = leaderboard(preds)
    print(board[(board.h == 3) & (board.window == "expanding")].sort_values(["set", "brier"]).round(4)
          .to_string(index=False))


if __name__ == "__main__":
    main()
