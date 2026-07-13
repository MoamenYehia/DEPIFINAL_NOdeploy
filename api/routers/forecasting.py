"""
Forecasting endpoints (Milestone 2 — Engineer 1).

POST /api/forecast/train      — Train Prophet + log to MLflow
GET  /api/forecast            — Forward forecast for N days
GET  /api/forecast/anomalies  — Flag historical days that deviate from forecast
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.config import settings

router = APIRouter()

# Singleton — one forecaster per process lifetime
_forecaster = None


def _get_forecaster():
    global _forecaster
    if _forecaster is None:
        from ml_engine.forecasting import SalesForecaster
        _forecaster = SalesForecaster()
    return _forecaster


def _ensure_daily_df() -> pd.DataFrame:
    """Load the daily sales CSV, building it first if missing."""
    daily_path = settings.PROCESSED_DIR / "daily_sales_forecast_data.csv"
    if daily_path.exists():
        return pd.read_csv(daily_path, parse_dates=["date"])

    clean_path = settings.PROCESSED_DIR / "cleaned_master_df.parquet"
    if not clean_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No data found. POST /api/pipeline/run first.",
        )
    from data.pipeline import build_daily_forecasting_df
    df = pd.read_parquet(clean_path)
    daily = build_daily_forecasting_df(df)
    daily.to_csv(daily_path, index=False)
    return daily


def _ensure_model_loaded():
    """Load persisted model if the in-memory singleton has no model yet."""
    fc = _get_forecaster()
    if fc.model is None:
        pkl = settings.PROCESSED_DIR / "forecaster.pkl"
        if pkl.exists():
            fc.load_model()
        else:
            raise HTTPException(
                status_code=400,
                detail="Model not trained. POST /api/forecast/train first.",
            )
    return fc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/forecast/train")
async def train_forecast(
    seasonality_mode: str = Query(
        "multiplicative",
        description="Prophet seasonality mode: 'multiplicative' or 'additive'.",
    ),
    changepoint_prior_scale: float = Query(
        0.05,
        description="Controls trend flexibility. Higher = more flexible.",
    ),
):
    """
    Train a Prophet model on the daily sales data and register it in MLflow.
    Saves the model to disk so /api/forecast and /api/insights can use it immediately.
    """
    try:
        daily_df    = _ensure_daily_df()
        forecaster  = _get_forecaster()
        metrics     = forecaster.train(
            daily_df,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        forecaster.save_model()
        return {"status": "trained", **metrics}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/forecast")
async def get_forecast(
    periods: int = Query(30, ge=1, le=365, description="Forecast horizon in days."),
):
    """
    Return a Prophet forecast for the next *periods* days.

    Response fields per row: ``ds`` (date), ``yhat`` (predicted sales),
    ``yhat_lower``, ``yhat_upper`` (80 % confidence interval).
    """
    try:
        forecaster   = _ensure_model_loaded()
        forecast_df  = forecaster.predict(periods=periods)
        forecast_df["ds"] = forecast_df["ds"].astype(str)
        return {
            "periods":  periods,
            "forecast": forecast_df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/forecast/anomalies")
async def get_forecast_anomalies(
    std_threshold: float = Query(
        2.0,
        description=(
            "Number of standard deviations from the forecast to flag as anomaly. "
            "Lower = more sensitive."
        ),
    ),
):
    """
    Compare historical actual sales against the Prophet forecast.
    Returns days that deviate more than *std_threshold* σ.
    """
    try:
        forecaster = _ensure_model_loaded()
        daily_df   = _ensure_daily_df()

        # Ensure forecast covers the historical range
        if forecaster._forecast_df is None:
            forecaster.predict(periods=0)

        anomalies = forecaster.detect_trend_anomalies(daily_df, std_threshold=std_threshold)

        return {
            "anomaly_count": len(anomalies),
            "threshold":     std_threshold,
            "anomalies":     anomalies,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
