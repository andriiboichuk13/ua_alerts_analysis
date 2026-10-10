"""Rolling-origin backtest over the last TEST_DAYS before the regime end.

Two training-window schemes:
  expanding - train on everything from REGIME_START up to the test fold;
  sliding   - train only on the SLIDING_TRAIN_DAYS before the test fold (adapts faster to level shifts).

Run: uv run python -m ua_alerts.backtest
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ua_alerts import config, data, features, models

PREDICTIONS = config.PROCESSED_DIR / "backtest_predictions.parquet"
WINDOWS = {"expanding": None, "sliding": config.SLIDING_TRAIN_DAYS}


def folds() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """N_FOLDS consecutive test windows covering the last TEST_DAYS before the regime end (UTC)."""
    end = data.regime_end()
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
    state = data.hourly_state(data.load_official(), end=data.regime_end())
    return features.build_panel(state), (state["coverage"] > 0).astype(float)


def run(window: str, panel: pd.DataFrame, indicator: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    model_list = models.all_models(indicator)
    preds = []
    for k, (start, end) in enumerate(folds()):
        for h in config.HORIZONS:
            train, test = split(panel, h, start, end, WINDOWS[window])
            out = test[[f"y_{h}"]].rename(columns={f"y_{h}": "y"}).assign(h=h, fold=k, window=window)
            for model in model_list:
                out[model.name] = model.fit(train, h).predict(test)
            preds.append(out)
            if verbose:
                print(f"{window:9s} fold={k} h={h} train={len(train):,} "
                      f"[{train.index.get_level_values('time').min():%m-%d} ..] test={len(test):,} "
                      f"[{start:%m-%d %H:%M} .. {end:%m-%d %H:%M})")
    return pd.concat(preds)


def run_all(rerun: bool = False) -> pd.DataFrame:
    """Predictions for both window schemes; cached in data/processed."""
    if PREDICTIONS.exists() and not rerun:
        return pd.read_parquet(PREDICTIONS)
    panel, indicator = load_inputs()
    preds = pd.concat([run(w, panel, indicator) for w in WINDOWS])
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PREDICTIONS)
    return preds


def model_names(preds: pd.DataFrame) -> list[str]:
    return [c for c in preds.columns if c not in ("y", "h", "fold", "window", "time", "region")]


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


def leaderboard(preds: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics plus the mean of per-oblast ROC AUCs (pooled AUC is inflated by front-vs-rear differences)."""
    pooled = summarize(preds, ["window", "h"])
    per_oblast = summarize(preds, ["window", "h", "region"])
    macro = per_oblast.groupby(["window", "h", "model"])["roc_auc"].mean().rename("roc_auc_mean_per_oblast")
    out = pooled.merge(macro.reset_index(), on=["window", "h", "model"])
    ref = out[out.model == "hour-of-day rate"].set_index(["window", "h"])["brier"]
    out["brier_skill"] = 1 - out["brier"] / out.set_index(["window", "h"]).index.map(ref).to_numpy()
    return out.drop(columns=["n"])


def main() -> None:
    preds = run_all(rerun=True)
    board = leaderboard(preds)
    print(board[board.h == 3].sort_values(["window", "brier"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
