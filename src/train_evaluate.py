"""Train on 2016, test on all of 2017 (a full unseen year), compare against simple baselines.

Run:  python src/train_evaluate.py
Writes metrics and plots to results/.
"""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import CATEGORICAL, FEATURES, build_table

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42

LGB_PARAMS = dict(
    learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0,
)


# ----------------------------------------------------------------------------- metrics
def cv_rmse(y, p):
    """ASHRAE Guideline 14 'coefficient of variation of RMSE', in %. < 30% hourly = acceptable model."""
    return 100 * np.sqrt(np.mean((y - p) ** 2)) / np.mean(y)


def nmbe(y, p):
    """ASHRAE Guideline 14 'normalised mean bias error', in %. |NMBE| < 10% hourly = acceptable."""
    return 100 * np.sum(y - p) / (len(y) * np.mean(y))


def per_building_scores(test: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for b, g in test.groupby("building_id"):
        y, p = g["load"].to_numpy(), g[pred_col].to_numpy()
        peak = y >= np.quantile(y, 0.95)  # the 5% highest-load hours of the year
        rows.append(dict(
            building_id=b,
            cv_rmse=cv_rmse(y, p),
            nmbe=nmbe(y, p),
            peak_mape=100 * np.mean(np.abs(y[peak] - p[peak]) / y[peak]),
        ))
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> dict:
    meets = (scores["cv_rmse"] < 30) & (scores["nmbe"].abs() < 10)
    return dict(
        median_cv_rmse=round(float(scores["cv_rmse"].median()), 2),
        median_abs_nmbe=round(float(scores["nmbe"].abs().median()), 2),
        median_peak_hour_mape=round(float(scores["peak_mape"].median()), 2),
        pct_buildings_meeting_ashrae14=round(100 * float(meets.mean()), 1),
    )


# ----------------------------------------------------------------------------- main
def main():
    print("building feature table ...")
    df = build_table()
    # score every model on exactly the same hours: drop rows where any baseline input is missing
    df = df.dropna(subset=["y", "lag_24", "lag_48", "lag_168", "same_hour_mean_7d", "airTemperature"])

    year = df["timestamp"].dt.year
    train_full = df[year == 2016]
    test = df[year == 2017].copy()
    # inside 2016: Jan-Aug to fit, Sep-Dec to choose the number of trees (early stopping)
    is_val = train_full["timestamp"].dt.month >= 9
    fit, val = train_full[~is_val], train_full[is_val]
    print(f"buildings: {df['building_id'].nunique()} | train rows: {len(train_full):,} | test rows: {len(test):,}")

    # ---- baselines (predict normalised load, convert back to kWh-equivalent with the scale)
    test["pred_persistence"] = test["lag_24"] * test["scale"]          # "same as yesterday"
    test["pred_week_mean"] = test["same_hour_mean_7d"] * test["scale"]  # "average of last 7 days, same hour"

    num = [f for f in FEATURES if f not in CATEGORICAL]
    sc = StandardScaler().fit(train_full[num].fillna(0))
    ridge = Ridge(alpha=1.0).fit(sc.transform(train_full[num].fillna(0)), train_full["y"])
    test["pred_ridge"] = ridge.predict(sc.transform(test[num].fillna(0))) * test["scale"]

    # ---- LightGBM point forecast
    dfit = lgb.Dataset(fit[FEATURES], fit["y"], categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(val[FEATURES], val["y"], categorical_feature=CATEGORICAL, reference=dfit)
    probe = lgb.train(LGB_PARAMS, dfit, 3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = probe.best_iteration
    print("best number of trees:", n_trees)
    dall = lgb.Dataset(train_full[FEATURES], train_full["y"], categorical_feature=CATEGORICAL)
    model = lgb.train(LGB_PARAMS, dall, n_trees)
    test["pred_lgbm"] = model.predict(test[FEATURES]) * test["scale"]

    # ---- LightGBM quantile models -> 80% prediction interval (P10 .. P90)
    for q in (0.1, 0.9):
        qm = lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": q}, dall, n_trees)
        test[f"pred_q{int(q*100)}"] = qm.predict(test[FEATURES]) * test["scale"]
    inside = (test["load"] >= test["pred_q10"]) & (test["load"] <= test["pred_q90"])
    coverage = float(inside.mean())

    # ---- scores
    models = {
        "Persistence (same hour yesterday)": "pred_persistence",
        "7-day same-hour average": "pred_week_mean",
        "Ridge regression (same features)": "pred_ridge",
        "LightGBM (global model)": "pred_lgbm",
    }
    all_scores, summary = {}, {}
    for name, col in models.items():
        s = per_building_scores(test, col)
        all_scores[name] = s
        summary[name] = summarise(s)
    base = summary["Persistence (same hour yesterday)"]["median_cv_rmse"]
    best = summary["LightGBM (global model)"]["median_cv_rmse"]
    out = dict(
        n_buildings=int(df["building_id"].nunique()),
        n_train_rows=int(len(train_full)),
        n_test_rows=int(len(test)),
        train_period="2016 (first week used only for lag features)",
        test_period="2017-01-01 to 2017-12-31",
        n_trees=int(n_trees),
        models=summary,
        lgbm_cv_rmse_reduction_vs_persistence_pct=round(100 * (base - best) / base, 1),
        p10_p90_interval_coverage_pct=round(100 * coverage, 1),
    )
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    table = pd.DataFrame(summary).T
    table.to_csv(RESULTS / "metrics_table.csv")

    make_plots(test, model, all_scores)


# ----------------------------------------------------------------------------- plots
def make_plots(test, model, all_scores):
    # 1. what the model relies on
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
    imp = 100 * imp / imp.sum()
    fig, ax = plt.subplots(figsize=(7, 7))
    imp.plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("share of total split gain (%)")
    ax.set_title("What drives the cooling-load forecast")
    fig.tight_layout(); fig.savefig(RESULTS / "feature_importance.png", dpi=150); plt.close(fig)

    # 2. CV(RMSE) distribution per model
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [s["cv_rmse"].clip(upper=150) for s in all_scores.values()]
    ax.boxplot(data, orientation="horizontal", tick_labels=list(all_scores.keys()), showfliers=False)
    ax.axvline(30, color="red", ls="--", label="ASHRAE Guideline 14 hourly limit (30%)")
    ax.set_xlabel("CV(RMSE) per building on 2017 (%)  - lower is better")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(RESULTS / "cv_rmse_by_model.png", dpi=150); plt.close(fig)

    # 3. one summer week, three buildings of different types
    week = test[(test["timestamp"] >= "2017-07-10") & (test["timestamp"] < "2017-07-17")]
    picks = []
    for use in ["Education", "Office", "Lodging/residential"]:
        cand = week[week["use"] == use]["building_id"].unique()
        if len(cand):
            picks.append(sorted(cand)[0])
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3 * len(picks)), sharex=True)
    for ax, b in zip(np.atleast_1d(axes), picks):
        g = week[week["building_id"] == b]
        ax.fill_between(g["timestamp"], g["pred_q10"], g["pred_q90"], color="#3b6ea5", alpha=0.2, label="P10-P90 band")
        ax.plot(g["timestamp"], g["load"], color="black", lw=1.2, label="actual")
        ax.plot(g["timestamp"], g["pred_lgbm"], color="#3b6ea5", lw=1.2, label="day-ahead forecast")
        ax.set_title(b, fontsize=10); ax.set_ylabel("chilled water")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Day-ahead cooling-load forecast, one week of July 2017 (unseen test year)")
    fig.tight_layout(); fig.savefig(RESULTS / "forecast_example_week.png", dpi=150); plt.close(fig)

    # 4. load vs outdoor temperature (the physics the model learned)
    sample = test.sample(60000, random_state=SEED)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(sample["airTemperature"], sample["load"] / sample["scale"], s=1, alpha=0.15, color="grey", label="actual")
    ax.scatter(sample["airTemperature"], sample["pred_lgbm"] / sample["scale"], s=1, alpha=0.15, color="#3b6ea5", label="forecast")
    ax.set_xlabel("outdoor air temperature (°C)"); ax.set_ylabel("load / building's 2016 mean")
    ax.set_title("Cooling load rises with outdoor temperature"); ax.legend(markerscale=10)
    fig.tight_layout(); fig.savefig(RESULTS / "load_vs_temperature.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
