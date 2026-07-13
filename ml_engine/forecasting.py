"""
Milestone 2 — Engineer 1: Time-Series & Demand Forecasting
Prophet model with full MLflow experiment tracking.

Run standalone:  python ml_engine/forecasting.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.prophet
import numpy as np
import pandas as pd
from prophet import Prophet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import settings


class SalesForecaster:
    """
    Wraps Prophet for Olist daily sales forecasting.
    Every training run is logged to MLflow (params, metrics, model artefact).
    """

    def __init__(self, experiment_name: Optional[str] = None):
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        self.model: Optional[Prophet] = None
        self._forecast_df: Optional[pd.DataFrame] = None
        self._run_id: Optional[str] = None

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        *,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
    ) -> dict:
        """
        Train Prophet on a daily DataFrame and log everything to MLflow.

        Args:
            df: DataFrame with columns ``date`` and ``total_sales``.

        Returns:
            Dict containing ``run_id``, ``mae``, ``rmse``, ``mape``.
        """
        prophet_df = (
            df.rename(columns={"date": "ds", "total_sales": "y"})[["ds", "y"]]
            .copy()
        )
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.dropna().sort_values("ds").reset_index(drop=True)

        # Chronological 80/20 train-test split
        split = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split]
        test_df  = prophet_df.iloc[split:]

        params = {
            "seasonality_mode":          seasonality_mode,
            "changepoint_prior_scale":   changepoint_prior_scale,
            "seasonality_prior_scale":   seasonality_prior_scale,
            "holidays_prior_scale":      holidays_prior_scale,
            "yearly_seasonality":        yearly_seasonality,
            "weekly_seasonality":        weekly_seasonality,
            "train_rows":                len(train_df),
            "test_rows":                 len(test_df),
        }

        with mlflow.start_run() as run:
            self._run_id = run.info.run_id
            mlflow.log_params(params)

            self.model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
            )
            self.model.fit(train_df)

            metrics = self._evaluate(test_df)
            mlflow.log_metrics(metrics)

            mlflow.prophet.log_model(
                self.model,
                artifact_path="prophet_model",
                registered_model_name="olist_sales_forecaster",
            )

        print(
            f"MLflow run {self._run_id[:8]}… | "
            f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%"
        )
        return {"run_id": self._run_id, **metrics}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, periods: int = 30, freq: str = "D") -> pd.DataFrame:
        """
        Generate a forward forecast for *periods* days.

        Returns DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
        """
        if self.model is None:
            raise RuntimeError("Call train() or load_model() before predict().")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self._forecast_df = self.model.predict(future)
        return self._forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    def detect_trend_anomalies(
        self,
        actual_df: pd.DataFrame,
        std_threshold: float = 2.0,
    ) -> list[dict]:
        """
        Flag days where actual sales deviate more than *std_threshold* σ from the forecast.

        Args:
            actual_df: DataFrame with ``date`` and ``total_sales`` columns.
            std_threshold: Sensitivity — lower catches more anomalies.

        Returns:
            List of anomaly dicts with keys: ds, y, yhat, deviation_pct, direction.
        """
        if self._forecast_df is None:
            self.predict(periods=0)

        forecast = self._forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"])

        actual = actual_df.rename(columns={"date": "ds", "total_sales": "y"}).copy()
        actual["ds"] = pd.to_datetime(actual["ds"])

        merged = actual.merge(forecast, on="ds", how="inner")
        merged["residual"] = merged["y"] - merged["yhat"]
        std = merged["residual"].std()
        if std == 0:
            return []

        flagged = merged[np.abs(merged["residual"]) > std_threshold * std].copy()
        flagged["deviation_pct"] = (
            (flagged["y"] - flagged["yhat"]) / flagged["yhat"].abs() * 100
        ).round(2)
        flagged["direction"] = flagged["residual"].apply(
            lambda x: "spike" if x > 0 else "drop"
        )

        records = flagged[["ds", "y", "yhat", "deviation_pct", "direction"]].to_dict(
            orient="records"
        )
        # Convert Timestamps to ISO strings for JSON safety
        for r in records:
            if hasattr(r["ds"], "isoformat"):
                r["ds"] = r["ds"].isoformat()
        return records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Pickle the trained model to *path* (defaults to PROCESSED_DIR/forecaster.pkl)."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        p = path or (settings.PROCESSED_DIR / "forecaster.pkl")
        with open(p, "wb") as fh:
            pickle.dump(self.model, fh)
        print(f"Model saved -> {p}")
        return p

    def load_model(self, path: Optional[Path] = None) -> "SalesForecaster":
        """Load a pickled model from *path*."""
        p = path or (settings.PROCESSED_DIR / "forecaster.pkl")
        with open(p, "rb") as fh:
            self.model = pickle.load(fh)
        print(f"Model loaded from {p}")
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, test_df: pd.DataFrame) -> dict:
        """Compute MAE, RMSE, MAPE on the held-out test set."""
        future = self.model.make_future_dataframe(periods=len(test_df), freq="D")
        forecast = self.model.predict(future)
        predicted = forecast.tail(len(test_df))["yhat"].values
        actual    = test_df["y"].values

        mae  = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    daily_path = settings.PROCESSED_DIR / "daily_sales_forecast_data.csv"
    if not daily_path.exists():
        print("Daily data not found — running pipeline first …")
        from data.pipeline import run_pipeline, build_daily_forecasting_df
        import pandas as pd
        from data.cleaning import load_or_build_clean_data
        clean_df = load_or_build_clean_data()
        daily_df = build_daily_forecasting_df(clean_df)
        daily_df.to_csv(daily_path, index=False)
    else:
        daily_df = pd.read_csv(daily_path, parse_dates=["date"])

    forecaster = SalesForecaster()
    metrics = forecaster.train(daily_df)
    forecaster.save_model()

    forecast = forecaster.predict(periods=30)
    print(forecast.tail())

    anomalies = forecaster.detect_trend_anomalies(daily_df)
    print(f"\nAnomalies detected: {len(anomalies)}")
    for a in anomalies[:3]:
        print(f"  {a['ds'][:10]}  {a['direction']:5s}  {a['deviation_pct']:+.1f}%")
