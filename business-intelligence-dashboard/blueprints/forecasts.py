import pandas as pd
from flask import Blueprint, render_template, request
from ui_config import REGIONS, TIME_RANGES, get_current_user

# Import your actual ML engine and config
import sys
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.config import settings
from ml_engine.forecasting import SalesForecaster

forecasts_bp = Blueprint("forecasts", __name__)

# Lazy singleton — load the Prophet model once per process, not once per request.
_forecaster = None


def get_forecaster():
    global _forecaster
    if _forecaster is not None:
        return _forecaster

    pkl_path = settings.PROCESSED_DIR / "forecaster.pkl"
    if not pkl_path.exists():
        raise RuntimeError(
            "No trained forecaster found. POST /api/forecast/train first, "
            "or run ml_engine/forecasting.py standalone."
        )

    fc = SalesForecaster()
    fc.load_model()
    _forecaster = fc
    return _forecaster


@forecasts_bp.route("/forecasts")
def index():
    time_range = request.args.get("time_range", "30d")
    region = request.args.get("region", "all")

    # 1. Parse requested periods
    periods_map = {"7d": 7, "30d": 30, "90d": 90}
    periods = periods_map.get(time_range, 30)

    # 2. Load historical data
    data_path = settings.PROCESSED_DIR / "daily_sales_forecast_data.csv"
    daily_df = pd.read_csv(data_path, parse_dates=["date"])

    # 3. Generate forecast
    forecaster = get_forecaster()
    forecast_df = forecaster.predict(periods=periods)

    # 4. Correctly align actuals and forecast with an empty-data safety check
    history_days = 60
    forecast_start_date = forecast_df["ds"].min()

    # Get actuals up until (but not including) the forecast start, limited to last 60 days
    recent_actual = daily_df[daily_df["date"] < forecast_start_date].tail(history_days)

    # SAFETY CHECK: If recent_actual is empty, fallback gracefully
    if not recent_actual.empty:
        # Create continuous labels and data lines
        labels = recent_actual["date"].dt.strftime("%b %d").tolist() + \
                 forecast_df["ds"].dt.strftime("%b %d").tolist()

        # Actuals line stops exactly when the forecast starts
        actual_line = recent_actual["total_sales"].round(2).tolist() + [None] * len(forecast_df)

        # Forecast line: Pad with None for historical length, connect to last actual data point
        forecast_line = [None] * (len(recent_actual) - 1) + \
                        [recent_actual["total_sales"].iloc[-1]] + \
                        forecast_df["yhat"].round(2).tolist()[1:]
    else:
        # Fallback if no history exists before the prediction
        labels = forecast_df["ds"].dt.strftime("%b %d").tolist()
        actual_line = [None] * len(forecast_df)
        forecast_line = forecast_df["yhat"].round(2).tolist()

    real_forecast_chart = {
        "labels": labels,
        "actual": actual_line,
        "forecast": forecast_line
    }

    # 5. Build real summary
    total_forecasted = forecast_df["yhat"].sum()
    real_summary = {
        "total_forecasted_sales": f"R${total_forecasted:,.2f}",
        "expected_growth": "N/A",
        "top_performing_region": "Data Pending",
        "lowest_performing_region": "Data Pending"
    }

    context = {
        "active_page": "forecasts",
        "forecast_chart": real_forecast_chart,
        "forecast_summary": real_summary,
        "regions": REGIONS,
        "time_ranges": TIME_RANGES,
        "selected_time_range": time_range,
        "selected_region": region,
        "user": get_current_user(),
    }

    return render_template("forecasts.html", **context)