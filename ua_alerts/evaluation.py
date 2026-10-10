"""Plots and tables for the training/validation notebook (notebooks/model.ipynb)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from ua_alerts import backtest, config, models
from ua_alerts.eda import BLUE, DIV, GRID, INK2, SEQ, save, short

# Fixed categorical order (never cycled) for the models shown in multi-series charts.
SERIES = {"hour-of-day rate": "#eb6834", "AutoETS": "#1baf7a", "logistic AR": "#eda100",
          "LightGBM": "#2a78d6", "XGBoost": "#6250d6"}
GROUP = {"persistence": "baseline", "hour-of-day rate": "baseline", "logistic AR": "statistical", "LightGBM": "ML",
         "XGBoost": "ML"}


def model_group(name: str) -> str:
    if name.endswith(" + Air Force"):
        return GROUP[name.removesuffix(" + Air Force")] + " + text"
    return GROUP.get(name, "ETS")


def plot_folds() -> None:
    """Timeline of training and evaluation windows: validation folds (tuning) and test folds, both window schemes."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    regime_start = pd.Timestamp(config.REGIME_START)
    y, ys, labels = 0.0, [], []
    for fold_set, color in (("validation", "#1baf7a"), ("test", BLUE)):
        for window, days in backtest.WINDOWS.items():
            for k, (start, end) in enumerate(backtest.folds(fold_set)):
                tr_start = regime_start if days is None else max(regime_start, start - pd.Timedelta(days=days))
                ax.barh(y, start - tr_start, left=tr_start, color="#d9e8fb", height=0.6)
                ax.barh(y, end - start, left=start, color=color, height=0.6)
                ys.append(y)
                labels.append(f"{fold_set} · {window} · {k + 1}")
                y += 1
            y += 0.4
        y += 0.6
    ax.set_yticks(ys, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Training (light) and evaluation windows: validation (green) and test (blue)")
    ax.grid(axis="y", visible=False)
    save(fig, "m01_folds")


def leaderboard_table(board: pd.DataFrame, h: int, fold_set: str = "test") -> pd.DataFrame:
    t = board[(board.h == h) & (board.set == fold_set)].pivot_table(index="model", columns="window",
                                        values=["brier", "brier_skill", "log_loss", "roc_auc", "roc_auc_mean_per_oblast"])
    t = t.sort_values(("brier", "expanding"))
    t.insert(0, "group", [model_group(m) for m in t.index])
    return t.round(4)


def plot_skill_by_horizon(board: pd.DataFrame, fold_set: str = "test") -> None:
    """Brier skill vs the hour-of-day rate, per model, horizon and window scheme."""
    board = board[board.set == fold_set]
    order = board[(board.h == 3) & (board.window == "expanding")].sort_values("brier_skill").model.tolist()
    fig, axes = plt.subplots(1, len(config.HORIZONS), figsize=(13, 4), sharey=True)
    for ax, h in zip(axes, config.HORIZONS):
        b = board[board.h == h].pivot(index="model", columns="window", values="brier_skill").loc[order]
        yy = np.arange(len(order))
        ax.barh(yy + 0.2, b["expanding"], height=0.38, color=BLUE, label="expanding")
        ax.barh(yy - 0.2, b["sliding"], height=0.38, color="#86b6ef", label="sliding (12 weeks)")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(yy, order)
        ax.set_title(f"h = {h} h")
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Brier skill score vs hour-of-day rate (higher is better)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "m02_skill_by_horizon")


def _select(preds: pd.DataFrame, h: int, window: str, fold_set: str = "test") -> pd.DataFrame:
    return preds[(preds.h == h) & (preds.window == window) & (preds.set == fold_set)]


def per_oblast(preds: pd.DataFrame, h: int, window: str) -> pd.DataFrame:
    return backtest.summarize(_select(preds, h, window), ["region"])


def plot_per_oblast(preds: pd.DataFrame, h: int, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heatmaps per oblast x model: Brier skill vs hour-of-day rate, and ROC AUC. Oblasts sorted by base rate."""
    po = per_oblast(preds, h, window)
    base = po.groupby("region")["base_rate"].first().sort_values()
    brier = po.pivot(index="region", columns="model", values="brier").loc[base.index]
    skill = 1 - brier.div(brier["hour-of-day rate"], axis=0)
    auc = po.pivot(index="region", columns="model", values="roc_auc").loc[base.index]
    cols = [c for c in backtest.model_names(preds) if c != "hour-of-day rate"]
    labels = [f"{short(r)} ({base[r]:.0%})" for r in base.index]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    im = axes[0].imshow(skill[cols].clip(-1, 1).values, aspect="auto", cmap=DIV.reversed(),
                        norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    axes[0].set_title(f"Brier skill vs hour-of-day rate (h = {h} h, {window})")
    fig.colorbar(im, ax=axes[0], shrink=0.6)
    im = axes[1].imshow(auc[cols].values, aspect="auto", cmap=SEQ, vmin=0.5, vmax=1)
    axes[1].set_title(f"ROC AUC per oblast (h = {h} h, {window})")
    fig.colorbar(im, ax=axes[1], shrink=0.6)
    for ax in axes:
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
        ax.grid(False)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylabel("oblast (test base rate)")
    save(fig, f"m03_per_oblast_h{h}_{window}")
    return skill, auc


def plot_roc(preds: pd.DataFrame, h: int, window: str) -> None:
    """ROC curves pooled and per macro-region for the main models."""
    p = _select(preds, h, window).reset_index()
    panels = {"all oblasts": p, **{k: p[p.region.isin(v)] for k, v in config.MACRO_REGIONS.items()}}
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.9), sharey=True)
    for ax, (name, g) in zip(axes, panels.items()):
        for m, color in SERIES.items():
            fpr, tpr, _ = roc_curve(g["y"], g[m])
            ax.plot(fpr, tpr, color=color, linewidth=1.6, label=m)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        ax.set_title(f"{name} (base {g['y'].mean():.0%})", fontsize=10)
        ax.set_xlabel("false positive rate")
        ax.set_aspect("equal")
    axes[0].set_ylabel("true positive rate")
    axes[-1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle(f"ROC curves, h = {h} h, {window} window", x=0.01, ha="left", fontweight="bold")
    save(fig, f"m04_roc_h{h}_{window}")


def plot_calibration(preds: pd.DataFrame, h: int, window: str) -> None:
    p = _select(preds, h, window)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for m, color in SERIES.items():
        frac, mean_pred = calibration_curve(p["y"], p[m], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, color=color, marker="o", markersize=4, linewidth=1.6, label=m)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration, h = {h} h, {window}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, f"m05_calibration_h{h}_{window}")


def plot_fold_stability(board_folds: pd.DataFrame, h: int, window: str) -> None:
    """Brier skill vs hour-of-day rate per fold: 4 validation folds (Feb-Jun) and 4 test folds (Aug)."""
    b = board_folds[(board_folds.h == h) & (board_folds.window == window)].copy()
    ref = b[b.model == "hour-of-day rate"].set_index(["set", "fold"])["brier"]
    b["skill"] = 1 - b["brier"] / b.set_index(["set", "fold"]).index.map(ref).to_numpy()
    labels = [f"val {a[5:]}" for a, _ in config.VALIDATION_FOLDS] + \
             [f"test {s:%m-%d}" for s, _ in backtest.folds("test")]
    fig, ax = plt.subplots(figsize=(11, 4))
    for m, color in SERIES.items():
        if m == "hour-of-day rate":
            continue
        s = pd.concat([b[(b.model == m) & (b.set == "validation")].sort_values("fold"),
                       b[(b.model == m) & (b.set == "test")].sort_values("fold")])
        ax.plot(range(len(s)), s["skill"], color=color, marker="o", markersize=5, label=m)
    ax.axvline(len(config.VALIDATION_FOLDS) - 0.5, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Brier skill vs hour-of-day rate")
    ax.set_title(f"Stability across seasons and folds, h = {h} h, {window} window")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, f"m06_folds_h{h}_{window}")


def ets_selection(indicator: pd.DataFrame, start, end) -> pd.DataFrame:
    """AICc of each ETS spec per oblast on one training window, and the AutoETS choice."""
    cache = models.ETSCache(indicator)
    rows = {}
    for region in indicator.columns:
        aicc = {s: cache.fit(region, s, start, end).aicc for s in models.ETS_SPECS}
        best = min(aicc, key=aicc.get)
        fit = cache.fit(region, best, start, end).params
        rows[region] = {**{f"AICc {s}": v for s, v in aicc.items()}, "chosen": best,
                        "alpha": fit["smoothing_level"],
                        "gamma (seasonal)": fit["smoothing_seasonal"] if models.ETS_SPECS[best]["seasonal"] else np.nan}
    t = pd.DataFrame(rows).T
    aicc_cols = [c for c in t.columns if c.startswith("AICc")]
    t[aicc_cols] = t[aicc_cols].astype(float).sub(t[aicc_cols].astype(float).min(axis=1), axis=0)
    return t.rename(columns={c: c.replace("AICc", "ΔAICc") for c in aicc_cols})


def feature_importance(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Gain importance of both boosters fitted on the last expanding training window."""
    start, end = backtest.folds()[-1]
    train, _ = backtest.split(panel, h, start, end)
    params = backtest.load_tuned_params()
    out = {}
    for m in (models.LightGBM(**params.get("LightGBM", {})), models.XGBoost(**params.get("XGBoost", {}))):
        m.fit(train, h)
        if m.name == "LightGBM":
            imp = pd.Series(m.model.booster_.feature_importance("gain"), index=m.model.feature_name_)
        else:
            imp = pd.Series(m.model.get_booster().get_score(importance_type="total_gain"))
        out[m.name] = imp / imp.sum()
    return pd.DataFrame(out).fillna(0).sort_values("LightGBM", ascending=False)


def plot_feature_importance(imp: pd.DataFrame, h: int) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    yy = np.arange(len(top))
    ax.barh(yy + 0.2, top["LightGBM"], height=0.38, color=SERIES["LightGBM"], label="LightGBM")
    ax.barh(yy - 0.2, top["XGBoost"], height=0.38, color=SERIES["XGBoost"], label="XGBoost")
    ax.set_yticks(yy, top.index)
    ax.set_xlabel("share of total gain")
    ax.set_title(f"Feature importance (gain), h = {h} h")
    ax.legend(frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, f"m07_importance_h{h}")


def plot_predictions(preds: pd.DataFrame, coverage: pd.DataFrame, regions: list[str], start: str, days: int = 7,
                     h: int = 3, window: str = "expanding", name: str = "m08_predictions") -> None:
    """Forecasts over time vs actual alerts. Shaded: hours with an alert in the oblast.
    Lines: P(alert within the next h hours) issued at each hour (plotted at the forecast time T)."""
    p = _select(preds, h, window).reset_index()
    t0 = pd.Timestamp(start, tz=config.LOCAL_TZ)
    t1 = t0 + pd.Timedelta(days=days)
    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 2.1 * len(regions)), sharex=True)
    for ax, region in zip(axes, regions):
        cov = coverage[region].tz_convert(config.LOCAL_TZ).loc[t0:t1]
        ax.fill_between(cov.index, 0, (cov > 0).astype(float), step="post", color="#e4e3df", linewidth=0,
                        label="alert active (actual)")
        g = p[p.region == region].copy()
        g["T"] = (g["time"] + pd.Timedelta(hours=1)).dt.tz_convert(config.LOCAL_TZ)
        g = g[(g["T"] >= t0) & (g["T"] <= t1)].sort_values("T")
        for m in ("hour-of-day rate", "logistic AR", "XGBoost"):
            ax.plot(g["T"], g[m], color=SERIES[m], linewidth=1.3, label=m)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P", rotation=0)
        ax.set_title(f"{short(region)} (test base rate {p[p.region == region]['y'].mean():.0%})", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.15))
    fig.suptitle(f"P(alert within {h} h) vs actual alerts, {t0:%d %b} - {t1:%d %b %Y} (Kyiv time)",
                 x=0.01, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, f"{name}_h{h}")


def season_table(board: pd.DataFrame, h: int, window: str = "expanding") -> pd.DataFrame:
    """Brier, Brier skill and mean per-oblast AUC per model on the validation set (Feb-Jun) and the test set (Aug)."""
    b = board[(board.h == h) & (board.window == window)]
    t = b.pivot_table(index="model", columns="set", values=["brier", "brier_skill", "roc_auc_mean_per_oblast"])
    return t.sort_values(("brier", "test")).round(4)


def onset_table(preds: pd.DataFrame, panel: pd.DataFrame, h: int, window: str = "expanding") -> pd.DataFrame:
    """Metrics split by the state at the forecast moment. If an alert is active now, 'alert within h hours' is
    almost always true (the current alert continues into the first target hour), so those rows are trivial;
    rows that are quiet now measure the real task: will a NEW alert start within h hours?"""
    p = preds[(preds.window == window) & (preds.h == h)].reset_index()
    p["state"] = np.where(panel["active_now"].reindex(pd.MultiIndex.from_frame(p[["time", "region"]])).to_numpy() > 0,
                          "alert now", "quiet now")
    pooled = backtest.summarize(p, ["set", "state"])
    per = backtest.summarize(p, ["set", "state", "region"]).groupby(["set", "state", "model"])["roc_auc"].mean()
    pooled["roc_auc_mean_per_oblast"] = pooled.set_index(["set", "state", "model"]).index.map(per).to_numpy()
    ref = pooled[pooled.model == "hour-of-day rate"].set_index(["set", "state"])["brier"]
    pooled["brier_skill"] = 1 - pooled["brier"] / pooled.set_index(["set", "state"]).index.map(ref).to_numpy()
    return pooled
