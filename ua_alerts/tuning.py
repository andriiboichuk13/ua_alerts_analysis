"""Hyperparameter search on the validation folds only (time-based; random k-fold would leak through
autocorrelated neighbouring hours). Selection criterion: mean Brier score over the validation folds at
TUNING_HORIZON, expanding window. The test folds are not touched.

Run: uv run python -m ua_alerts.tuning
"""
import itertools
import json

import pandas as pd

from ua_alerts import backtest, config, models

GRIDS = {
    "logistic AR": {"C": [0.01, 0.1, 1.0, 10.0]},
    "LightGBM": {"num_leaves": [15, 31, 63], "min_child_samples": [50, 200],
                 "learning_rate,n_estimators": [(0.03, 400), (0.1, 150)]},
    "XGBoost": {"max_depth": [4, 6, 8], "min_child_weight": [5, 20],
                "learning_rate,n_estimators": [(0.03, 400), (0.1, 150)]},
}
MODEL_CLASSES = {"logistic AR": models.LogisticAR, "LightGBM": models.LightGBM, "XGBoost": models.XGBoost}
RESULTS = config.ROOT / "reports" / "tuning_results.csv"


def configurations(grid: dict) -> list[dict]:
    out = []
    for values in itertools.product(*grid.values()):
        params = {}
        for key, value in zip(grid, values):
            if "," in key:
                params.update(dict(zip(key.split(","), value)))
            else:
                params[key] = value
        out.append(params)
    return out


def search(panel: pd.DataFrame, h: int = config.TUNING_HORIZON) -> pd.DataFrame:
    rows = []
    splits = [backtest.split(panel, h, start, end) for start, end in backtest.folds("validation")]
    for name, grid in GRIDS.items():
        for params in configurations(grid):
            scores = []
            for k, (train, val) in enumerate(splits):
                p = MODEL_CLASSES[name](**params).fit(train, h).predict(val)
                scores.append(backtest.metrics(val[f"y_{h}"].to_numpy(), p)["brier"])
            rows.append({"model": name, "params": json.dumps(params), "mean_brier": sum(scores) / len(scores),
                         **{f"fold_{k + 1}": s for k, s in enumerate(scores)}})
            print(f"{name:12s} {params} -> {rows[-1]['mean_brier']:.5f}")
    return pd.DataFrame(rows)


def best_params(results: pd.DataFrame) -> dict:
    best = results.loc[results.groupby("model")["mean_brier"].idxmin()]
    return {r.model: json.loads(r.params) for r in best.itertuples()}


def main() -> None:
    panel, _ = backtest.load_inputs()
    results = search(panel)
    results.to_csv(RESULTS, index=False)
    params = best_params(results)
    backtest.TUNED_PARAMS.write_text(json.dumps(params, indent=2))
    print("\nselected:", json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
