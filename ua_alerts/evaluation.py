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
    return GROUP.get(name, "ETS")


def plot_folds() -> None:
    """Timeline of training and test windows for both window schemes."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    regime_start = pd.Timestamp(config.REGIME_START)
    y, ys, labels = 0.0, [], []
    for window, days in backtest.WINDOWS.items():
        for k, (start, end) in enumerate(backtest.folds()):
            tr_start = regime_start if days is None else max(regime_start, start - pd.Timedelta(days=days))
            ax.barh(y, start - tr_start, left=tr_start, color="#b7d3f6", height=0.6)
            ax.barh(y, end - start, left=start, color=BLUE, height=0.6)
            ys.append(y)
            labels.append(f"{window} · fold {k + 1}")
            y += 1
        y += 0.5
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_title("Rolling-origin backtest: training (light) and test (dark) windows")
    ax.grid(axis="y", visible=False)
    save(fig, "m01_folds")


def leaderboard_table(board: pd.DataFrame, h: int) -> pd.DataFrame:
    t = board[board.h == h].pivot_table(index="model", columns="window",
                                        values=["brier", "brier_skill", "log_loss", "roc_auc", "roc_auc_mean_per_oblast"])
    t = t.sort_values(("brier", "expanding"))
    t.insert(0, "group", [model_group(m) for m in t.index])
    return t.round(4)


def plot_skill_by_horizon(board: pd.DataFrame) -> None:
    """Brier skill vs the hour-of-day rate, per model, horizon and window scheme."""
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


def per_oblast(preds: pd.DataFrame, h: int, window: str) -> pd.DataFrame:
    return backtest.summarize(preds[(preds.h == h) & (preds.window == window)], ["region"])


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
    p = preds[(preds.h == h) & (preds.window == window)].reset_index()
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
    p = preds[(preds.h == h) & (preds.window == window)]
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
    b = board_folds[(board_folds.h == h) & (board_folds.window == window)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for m, color in SERIES.items():
        s = b[b.model == m].sort_values("fold")
        ax.plot(s["fold"] + 1, s["brier"], color=color, marker="o", markersize=5, label=m)
    ax.set_xticks(range(1, config.N_FOLDS + 1))
    ax.set_xlabel("fold (test window)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(f"Stability across folds, h = {h} h, {window}")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
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
    out = {}
    for m in (models.LightGBM(), models.XGBoost()):
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
