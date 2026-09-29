"""
Parameter-Optimised SARIMAX for Bike Sharing Hourly Dataset
============================================================
Target : cnt (total bike rentals per hour)
Exogenous : temp, hum, windspeed, weathersit, holiday, workingday, season
Seasonal period : 24 (hourly data -> daily seasonality)
"""

import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# -------------------------------------------------------------
# 1.  LOAD & PREPARE DATA
# -------------------------------------------------------------

def load_data(filepath: str = "hour.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # ── Robust date parsing ───────────────────────────────────
    # Detect format from the first value (handles DD-MM-YYYY and YYYY-MM-DD)
    sample = str(df["dteday"].iloc[0]).strip()
    parts  = sample.split("-")
    if len(parts[0]) == 4:
        fmt = "%Y-%m-%d"          # e.g. 2011-01-01
    else:
        fmt = "%d-%m-%Y"          # e.g. 01-01-2011

    log.info(f"Detected date format: {fmt}  (sample: {sample})")
    df["dteday"] = pd.to_datetime(df["dteday"], format=fmt)

    # Build a proper hourly datetime index
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"].astype(int), unit="h")
    df = df.set_index("datetime").sort_index()

    log.info(f"Data loaded: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} rows)")
    return df


def select_features(df: pd.DataFrame):
    exog_cols = ["temp", "hum", "windspeed", "weathersit",
                 "holiday", "workingday", "season"]
    target = df["cnt"]
    exog   = df[exog_cols]
    return target, exog


# -------------------------------------------------------------
# 2.  STATIONARITY CHECK
# -------------------------------------------------------------

def check_stationarity(series: pd.Series, name: str = "Series") -> int:
    """ADF test; returns suggested d (0 or 1)."""
    result = adfuller(series.dropna(), autolag="AIC")
    p_val  = result[1]
    log.info(f"[ADF] {name}: p-value = {p_val:.4f} -> "
             f"{'Stationary' if p_val < 0.05 else 'Non-stationary'}")
    return 0 if p_val < 0.05 else 1


# -------------------------------------------------------------
# 3.  PARAMETER GRID SEARCH
# -------------------------------------------------------------

def grid_search_sarimax(
    endog,
    exog,
    p_values = [0, 1],
    d_values = [0, 1],
    q_values = [0, 1],
    P_values = [0, 1],
    D_values = [0],
    Q_values = [0, 1],
    s        = 24,
    top_n    = 5,
):
    pdq   = list(itertools.product(p_values, d_values, q_values))
    PDQs  = list(itertools.product(P_values, D_values, Q_values, [s]))
    total = len(pdq) * len(PDQs)
    results = []

    log.info(f"Grid search: {total} combinations over (p,d,q) x (P,D,Q,{s})")

    for i, (order, seasonal_order) in enumerate(itertools.product(pdq, PDQs), 1):
        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=30)
            results.append({
                "order":          order,
                "seasonal_order": seasonal_order,
                "AIC":            fit.aic,
                "BIC":            fit.bic,
            })
            log.info(f"  [{i}/{total}] SARIMAX{order}{seasonal_order}  AIC={fit.aic:.1f}")
        except Exception as e:
            log.info(f"  [{i}/{total}] SARIMAX{order}{seasonal_order}  FAILED: {e}")

    results_df = (
        pd.DataFrame(results)
        .sort_values("AIC")
        .reset_index(drop=True)
    )
    log.info("\nTop configurations by AIC:")
    log.info(results_df.head(top_n).to_string(index=False))
    return results_df


# -------------------------------------------------------------
# 4.  TRAIN / TEST SPLIT
# -------------------------------------------------------------

def train_test_split_ts(endog, exog, test_hours: int = 24 * 7):
    train_y = endog.iloc[:-test_hours]
    test_y  = endog.iloc[-test_hours:]
    train_x = exog.iloc[:-test_hours]
    test_x  = exog.iloc[-test_hours:]
    log.info(f"Train: {len(train_y)} rows | Test: {len(test_y)} rows")
    return train_y, test_y, train_x, test_x


# -------------------------------------------------------------
# 5.  FIT BEST MODEL & EVALUATE
# -------------------------------------------------------------

def fit_best_model(train_y, train_x, test_y, test_x,
                   best_order, best_seasonal_order):
    log.info(f"\nFitting best model: SARIMAX{best_order}{best_seasonal_order}")
    model = SARIMAX(
        train_y,
        exog=train_x,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    print(fit.summary())

    forecast = fit.forecast(steps=len(test_y), exog=test_x)
    forecast = forecast.clip(lower=0)        # no negative rentals
    forecast.index = test_y.index            # align index explicitly

    mae  = mean_absolute_error(test_y, forecast)
    rmse = np.sqrt(mean_squared_error(test_y, forecast))

    # MAPE: skip hours where actual == 0 to avoid division by zero
    nonzero = test_y > 0
    mape = np.mean(
        np.abs((test_y[nonzero] - forecast[nonzero]) / test_y[nonzero])
    ) * 100

    log.info(f"\n-- Test-set metrics --")
    log.info(f"  MAE  : {mae:.2f}")
    log.info(f"  RMSE : {rmse:.2f}")
    log.info(f"  MAPE : {mape:.2f}%  (computed on {nonzero.sum()} non-zero hours)")

    return fit, forecast, {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# -------------------------------------------------------------
# 6.  PLOTS
# -------------------------------------------------------------

def plot_diagnostics(fit):
    fig = fit.plot_diagnostics(figsize=(14, 8))
    fig.suptitle("SARIMAX Model Diagnostics", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("sarimax_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved: sarimax_diagnostics.png")


def plot_forecast(train_y, test_y, forecast, metrics):
    mape_str = f"{metrics['MAPE']:.1f}%" if not np.isnan(metrics['MAPE']) else "N/A"

    fig, ax = plt.subplots(figsize=(18, 5))

    # Only plot actual vs forecast over the test window
    ax.plot(test_y.index, test_y.values,
            label="Actual", color="#2CA02C", lw=1.8)
    ax.plot(forecast.index, forecast.values,
            label="Forecast", color="#D62728", lw=1.8, linestyle="--")
    ax.fill_between(test_y.index, test_y.values, forecast.values,
                    alpha=0.12, color="#D62728", label="Error band")

    ax.set_title(
        f"Actual vs Forecast  |  "
        f"{test_y.index[0].strftime('%d %b %Y')} — {test_y.index[-1].strftime('%d %b %Y')}  |  "
        f"MAE={metrics['MAE']:.1f}  RMSE={metrics['RMSE']:.1f}  MAPE={mape_str}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Bike Rentals (cnt)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("sarimax_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: sarimax_forecast.png")


def plot_acf_pacf(series, lags=48):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(),  lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.savefig("acf_pacf.png", dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved: acf_pacf.png")


# -------------------------------------------------------------
# 7.  MAIN PIPELINE
# -------------------------------------------------------------

def main(
    filepath:        str   = "hour.csv",
    run_grid_search: bool  = True,
    manual_order:    tuple = (1, 0, 1),
    manual_seasonal: tuple = (1, 0, 1, 24),
    subsample:       int   = 500,    # rows used for grid search
    test_hours:      int   = 24 * 7, # 1-week hold-out
):
    # Load
    df            = load_data(filepath)
    target, exog  = select_features(df)

    # Stationarity
    suggested_d = check_stationarity(target, "cnt")
    plot_acf_pacf(target)

    # Split
    train_y, test_y, train_x, test_x = train_test_split_ts(
        target, exog, test_hours=test_hours
    )

    # Grid search
    if run_grid_search:
        gs_y = train_y.iloc[-subsample:] if subsample else train_y
        gs_x = train_x.iloc[-subsample:] if subsample else train_x

        results_df = grid_search_sarimax(
            endog    = gs_y,
            exog     = gs_x,
            p_values = [0, 1],
            d_values = [suggested_d],
            q_values = [0, 1],
            P_values = [0, 1],
            D_values = [0],
            Q_values = [0, 1],
            s        = 24,
        )
        results_df.to_csv("grid_search_results.csv", index=False)
        log.info("Saved: grid_search_results.csv")

        best_order    = results_df.iloc[0]["order"]
        best_seasonal = results_df.iloc[0]["seasonal_order"]
    else:
        best_order    = manual_order
        best_seasonal = manual_seasonal

    # Fit & evaluate
    fit, forecast, metrics = fit_best_model(
        train_y, train_x, test_y, test_x,
        best_order, best_seasonal,
    )

    # Plots
    plot_diagnostics(fit)
    plot_forecast(train_y, test_y, forecast, metrics)

    return fit, forecast, metrics


# -------------------------------------------------------------
if __name__ == "__main__":
    # On Windows use raw strings for paths: r"final\hour.csv"
    fit, forecast, metrics = main(
        filepath        = r"hour.csv",   # change to r"final\hour.csv" if needed
        run_grid_search = True,
        subsample       = 500,
        test_hours      = 24 * 7,
    )