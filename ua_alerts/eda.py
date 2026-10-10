"""Exploratory data analysis helpers. Used by notebooks/eda.ipynb; figures are also saved to reports/figures.

Run headless: uv run python -m ua_alerts.eda
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf

from ua_alerts import config, data

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1baf7a", "#f0efec", "#eb6834"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "font.size": 9, "figure.dpi": 110,
})


def regime(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the modelling window (raion-level alerting, before the yellow/red system)."""
    return df[(df.index >= pd.Timestamp(config.REGIME_START)) & (df.index < pd.Timestamp(config.REGIME_END))]


def short(name: str) -> str:
    return name.replace(" oblast", "")


def save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    if "inline" in matplotlib.get_backend():
        plt.show()
    else:
        plt.close(fig)


def plot_trend(cov: pd.DataFrame) -> pd.Series:
    """Daily national intensity: average share of oblasts under alert, in %."""
    daily = cov.mean(axis=1).tz_convert(config.LOCAL_TZ).resample("D").mean() * 100
    daily = daily.iloc[1:-1]  # drop partial first/last day
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(daily.index, daily, color="#b7d3f6", linewidth=0.8, label="daily")
    ax.plot(daily.index, daily.rolling(30, center=True).mean(), color=BLUE, label="30-day mean")
    ax.axvspan(pd.Timestamp(config.REGIME_START), pd.Timestamp(config.REGIME_END), color=GRID, alpha=0.5,
               linewidth=0, label="modelling window")
    ax.set_title("Share of oblast-hours under alert, % (national, daily)")
    ax.set_ylabel("% of oblast-hours")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "01_trend")
    return daily


def plot_oblast_month(cov: pd.DataFrame) -> None:
    monthly = cov.tz_convert(config.LOCAL_TZ).resample("MS").mean().iloc[:-1] * 100
    order = cov.mean().sort_values(ascending=False).index
    m = monthly[order].T
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ, interpolation="nearest")
    ax.set_yticks(range(len(order)), [short(r) for r in order])
    ticks = [i for i, d in enumerate(m.columns) if d.month in (1, 7)]
    ax.set_xticks(ticks, [m.columns[i].strftime("%Y-%m") for i in ticks])
    ax.grid(False)
    ax.set_title("Monthly % of time under alert, by oblast")
    fig.colorbar(im, ax=ax, label="% of time", shrink=0.8)
    save(fig, "02_oblast_month")


def plot_hour_weekday(cov: pd.DataFrame) -> None:
    local = cov.mean(axis=1).tz_convert(config.LOCAL_TZ) * 100
    recent = regime(local)
    grid = recent.groupby([recent.index.dayofweek, recent.index.hour]).mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    im = axes[0].imshow(grid.values, aspect="auto", cmap=SEQ)
    axes[0].set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].set_xlabel("hour (Kyiv time)")
    axes[0].grid(False)
    axes[0].set_title("Modelling window: % of oblast-hours under alert")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    periods = {"2022-03 – 2023-02": ("2022-03", "2023-02"), "2024": ("2024-01", "2024-12"),
               "modelling window": (str(recent.index.min().date()), str(recent.index.max().date()))}
    for (label, (a, b)), color in zip(periods.items(), [AQUA, ORANGE, BLUE]):
        s = local[a:b]
        prof = s.groupby(s.index.hour).mean()
        axes[1].plot(prof.index, prof / prof.mean(), color=color, label=label)
    axes[1].axhline(1, color=INK2, linewidth=0.8)
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].set_xlabel("hour (Kyiv time)")
    axes[1].set_title("Daily profile, relative to period mean")
    axes[1].legend(frameon=False)
    save(fig, "03_hour_weekday")


def plot_stl(daily: pd.Series, cov: pd.DataFrame) -> dict:
    stats = {}
    s = daily.asfreq("D").interpolate()
    res = STL(s, period=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for ax, (name, comp) in zip(axes, [("observed", res.observed), ("trend", res.trend),
                                         ("weekly seasonal", res.seasonal), ("residual", res.resid)]):
        ax.plot(comp.index, comp, color=BLUE, linewidth=0.9 if name != "trend" else 2)
        ax.set_title(name, fontsize=10)
    fig.suptitle("STL decomposition of daily national alert share (period = 7 days)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_stl_daily")
    var = lambda x: np.var(x)
    stats["weekly_strength"] = max(0, 1 - var(res.resid) / var(res.resid + res.seasonal))
    stats["trend_strength"] = max(0, 1 - var(res.resid) / var(res.resid + res.trend))

    hourly = cov.mean(axis=1) * 100
    hourly = regime(hourly)
    mres = MSTL(hourly, periods=(24, 168), stl_kwargs={"robust": True}).fit()
    seas = mres.seasonal
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for ax, (name, comp) in zip(axes, [("observed", mres.observed), ("trend", mres.trend),
                                         ("daily seasonal (24h)", seas["seasonal_24"]),
                                         ("weekly seasonal (168h)", seas["seasonal_168"])]):
        ax.plot(comp.index, comp, color=BLUE, linewidth=0.6 if name == "observed" else 1.2)
        ax.set_title(name, fontsize=10)
    fig.suptitle("MSTL of hourly national alert share, modelling window (periods 24h, 168h)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "05_mstl_hourly")
    r = mres.resid
    stats["daily_strength_hourly"] = max(0, 1 - var(r) / var(r + seas["seasonal_24"]))
    stats["weekly_strength_hourly"] = max(0, 1 - var(r) / var(r + seas["seasonal_168"]))
    return stats


def plot_acf_periodogram(daily: pd.Series) -> None:
    s = daily.asfreq("D").interpolate()
    detrended = s - s.rolling(61, center=True, min_periods=31).mean()
    detrended = detrended.dropna()
    lags = 60
    a = acf(detrended, nlags=lags)
    conf = 1.96 / np.sqrt(len(detrended))

    freqs = np.fft.rfftfreq(len(detrended), d=1)[1:]
    power = np.abs(np.fft.rfft(detrended - detrended.mean()))[1:] ** 2
    periods = 1 / freqs
    keep = (periods >= 2) & (periods <= 120)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    axes[0].vlines(range(lags + 1), 0, a, color=BLUE, linewidth=2)
    axes[0].axhspan(-conf, conf, color="#cde2fb", alpha=0.6, linewidth=0)
    for p in (7, 14, 30):
        axes[0].axvline(p, color=INK2, linewidth=0.6, linestyle=":")
    axes[0].set_title("ACF of detrended daily series (lags in days)")
    axes[0].set_xlabel("lag (days)")
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[1].plot(periods[keep], power[keep], color=BLUE, linewidth=1.2)
    axes[1].set_xscale("log")
    for p in (7, 14, 30):
        axes[1].axvline(p, color=INK2, linewidth=0.6, linestyle=":")
        axes[1].text(p, axes[1].get_ylim()[1] * 0.95, f"{p}d", color=INK2, ha="center")
    axes[1].set_title("Periodogram of detrended daily series")
    axes[1].set_xlabel("period (days, log)")
    axes[1].set_ylabel("power")
    save(fig, "06_acf_periodogram")


def plot_lift(cov: pd.DataFrame) -> pd.DataFrame:
    """lift[X, Y] = P(X active at t+1h | Y active at t) / P(X active), modelling window."""
    act = (regime(cov) > 0).astype(float)
    nxt = act.shift(-1).iloc[:-1]
    now = act.iloc[:-1]
    base = nxt.mean()
    joint = nxt.T.values @ now.values / len(now)  # P(X_{t+1}, Y_t)
    cond = joint / now.mean().values[None, :]
    lift = pd.DataFrame(cond / base.values[:, None], index=act.columns, columns=act.columns)
    order = cov.mean().sort_values().index
    lift = lift.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(lift.values, cmap=DIV, norm=TwoSlopeNorm(vcenter=1, vmin=0, vmax=min(lift.values.max(), 12)))
    labels = [short(r) for r in order]
    ax.set_xticks(range(len(order)), labels, rotation=90)
    ax.set_yticks(range(len(order)), labels)
    ax.set_xlabel("Y: oblast under alert at hour t")
    ax.set_ylabel("X: oblast under alert at hour t+1")
    ax.grid(False)
    ax.set_title("Lift  P(X at t+1 | Y at t) / P(X), modelling window")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "07_region_lift")
    return lift


def plot_onset_lift(cov: pd.DataFrame) -> pd.DataFrame:
    """onset lift[X, Y] = P(X starts at t+1 | X quiet at t, Y active at t) / P(X starts at t+1 | X quiet at t).

    Conditioning on X being quiet removes X's own persistence, so this measures whether an alert
    in Y *precedes* a new alert in X, rather than both simply being active at the same time.
    """
    act = (regime(cov) > 0).astype(float)
    now, nxt = act.iloc[:-1].values, act.shift(-1).iloc[:-1].values
    quiet = 1 - now
    onset = nxt * quiet
    base = onset.sum(0) / quiet.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = (onset.T @ now) / (quiet.T @ now)  # [X, Y]
    ratio = cond / base[:, None]
    np.fill_diagonal(ratio, np.nan)
    lift = pd.DataFrame(ratio, index=act.columns, columns=act.columns)
    order = cov.mean().sort_values().index
    lift = lift.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(lift.values, cmap=DIV, norm=TwoSlopeNorm(vcenter=1, vmin=0, vmax=np.nanmax(lift.values)))
    labels = [short(r) for r in order]
    ax.set_xticks(range(len(order)), labels, rotation=90)
    ax.set_yticks(range(len(order)), labels)
    ax.set_xlabel("Y: oblast under alert at hour t")
    ax.set_ylabel("X: oblast quiet at t, alert starts at t+1")
    ax.grid(False)
    ax.set_title("Onset lift  P(X starts | X quiet, Y active) / P(X starts | X quiet)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "08_onset_lift")
    return lift


def plot_daily_strength_by_oblast(cov: pd.DataFrame) -> pd.DataFrame:
    """Daily seasonality per oblast from a fixed hour-of-day profile (Kyiv time), modelling window.

    eta2:        share of hourly variance explained by the hour-of-day mean profile.
    peak_trough: ratio of the highest to the lowest hourly mean - the relative amplitude of the cycle.
    (Per-oblast STL is not used: its time-varying seasonal component absorbs individual bursty alert
    episodes, which made e.g. Zakarpattia look 'most seasonal' despite a flat profile.)
    """
    local = regime(cov).tz_convert(config.LOCAL_TZ)
    hour = local.index.hour
    rows = {}
    for region, s in local.items():
        profile = s.groupby(hour).mean()
        rows[region] = {"eta2": s.groupby(hour).transform("mean").var() / s.var(),
                        "peak_trough": profile.max() / profile.min(), "peak_hour": profile.idxmax()}
    out = pd.DataFrame(rows).T.sort_values("peak_trough")
    labels = [short(r) for r in out.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(labels, out["peak_trough"], color=BLUE, height=0.7)
    axes[0].axvline(1, color=INK2, linewidth=0.8)
    axes[0].set_title("Peak / trough ratio of hourly alert share")
    axes[1].barh(labels, out["eta2"] * 100, color=BLUE, height=0.7)
    axes[1].set_title("Variance explained by hour of day, %")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    save(fig, "05b_daily_strength_by_oblast")
    return out


def source_reliability(official: pd.DataFrame, volunteer: pd.DataFrame) -> pd.DataFrame:
    """Minute-level agreement of volunteer vs official alerts per year, summed over oblasts.

    precision: share of volunteer alert-minutes during which official also had an alert.
    recall:    share of official alert-minutes that the volunteer data also covers.
    """
    start, end = official["started_at"].min().floor("h"), data.data_cutoff(official)
    n = int((end - start).total_seconds() // 60)
    years = pd.date_range(start, periods=n, freq="min").year.to_numpy()

    def active(df):
        s = ((df["started_at"] - start).dt.total_seconds() // 60).astype(int).clip(0, n).to_numpy()
        e = ((df["finished_at"] - start).dt.total_seconds() // 60).astype(int).clip(0, n).to_numpy()
        d = np.zeros(n + 1, np.int32)
        np.add.at(d, s, 1)
        np.add.at(d, e, -1)
        return np.cumsum(d)[:n] > 0

    rows = []
    for region in sorted(set(official["region"]) & set(volunteer["region"])):
        v, o = active(volunteer[volunteer["region"] == region]), active(official[official["region"] == region])
        rows.append(pd.DataFrame({"year": years, "vol": v, "off": o, "both": v & o}).groupby("year").sum())
    t = sum(rows)
    return pd.DataFrame({"volunteer_hours": t["vol"] / 60, "official_hours": t["off"] / 60,
                         "precision": t["both"] / t["vol"], "recall": t["both"] / t["off"]}).round(2)


def plot_source_comparison(cov_o: pd.DataFrame, cov_v: pd.DataFrame) -> pd.Series:
    idx = cov_v.index.intersection(cov_o.index)
    cols = cov_v.columns.intersection(cov_o.columns)
    m = pd.DataFrame({
        "volunteer": cov_v.loc[idx, cols].sum(axis=1),
        "official": cov_o.loc[idx, cols].sum(axis=1),
    }).resample("MS").sum().iloc[1:-1]
    ratio = m["volunteer"] / m["official"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(m.index, m["official"], color=ORANGE, label="official (oblast + raion)")
    axes[0].plot(m.index, m["volunteer"], color=BLUE, label="volunteer (eTryvoga)")
    axes[0].set_title("Monthly oblast-hours under alert: volunteer vs official")
    axes[0].legend(frameon=False)
    axes[1].plot(ratio.index, ratio, color=INK2)
    axes[1].set_ylim(0, 1.2)
    axes[1].axvline(pd.Timestamp("2025-12-01", tz="UTC"), color=INK2, linestyle=":", linewidth=0.8)
    axes[1].set_title("ratio volunteer / official", fontsize=10)
    save(fig, "09_source_comparison")
    return ratio


def load_grids() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly coverage grids: official (target source, cut at the regime end) and volunteer (comparison only)."""
    off = data.load_official()
    cov, _ = data.hourly_grid(off, end=data.regime_end())
    cov_v, _ = data.hourly_grid(data.load_volunteer())
    return cov, cov_v


def main() -> None:
    matplotlib.use("Agg")
    cov, cov_v = load_grids()
    print(f"Official grid: {cov.index.min()} .. {cov.index.max()} (UTC), {cov.shape[1]} oblasts")

    ratio = plot_source_comparison(cov, cov_v)
    daily = plot_trend(cov)
    plot_oblast_month(cov)
    plot_hour_weekday(cov)
    stats = plot_stl(daily, cov)
    by_oblast = plot_daily_strength_by_oblast(cov)
    plot_acf_periodogram(daily)
    plot_lift(cov)
    onset = plot_onset_lift(cov)

    print("\nSeasonality strength (0 = none, 1 = all variance):")
    for k, val in stats.items():
        print(f"  {k}: {val:.2f}")
    print("\nDaily seasonality by oblast:\n" + by_oblast.astype(float).round(3).to_string())
    print("\nMean % oblast-hours under alert by year:\n" + daily.groupby(daily.index.year).mean().round(1).to_string())
    print("\nVolunteer/official ratio, last 6 months:\n" + ratio.tail(6).round(2).to_string())
    print("\nTop region pairs by onset lift:")
    print(onset.stack().sort_values(ascending=False).head(10).round(2).to_string())


if __name__ == "__main__":
    main()
