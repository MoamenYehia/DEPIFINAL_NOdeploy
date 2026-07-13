import pandas as pd
from functools import lru_cache
from flask import Blueprint, render_template
from ui_config import get_current_user
from ml_engine.anomaly_detection import OrderAnomalyDetector
from ml_engine.forecasting import SalesForecaster
from core.config import settings

dashboard_bp = Blueprint("dashboard", __name__)

# 1. Lazy singletons — nothing loads until the first request that needs it,
#    and both have a retrain fallback so a missing/version-mismatched pickle
#    doesn't crash the whole app at import time.
_detector = None
_forecaster = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = OrderAnomalyDetector()
        try:
            _detector.load()
        except Exception as exc:
            print(f"[dashboard] anomaly model load failed ({exc}); retraining locally...")
            _detector.retrain()
            _detector.save()
    return _detector


def get_forecaster():
    global _forecaster
    if _forecaster is None:
        _forecaster = SalesForecaster()
        pkl_path = settings.PROCESSED_DIR / "forecaster.pkl"
        if pkl_path.exists():
            try:
                _forecaster.load_model()
            except Exception as exc:
                print(f"[dashboard] forecaster load failed ({exc}); train a fresh model via "
                      f"POST /api/forecast/train before this page will work.")
                raise
        else:
            raise RuntimeError(
                "No trained forecaster found. POST /api/forecast/train first."
            )
    return _forecaster


# 2. Cached data load (unchanged)
@lru_cache(maxsize=1)
def load_cached_data():
    return pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
def index():
    detector = get_detector()
    forecaster = get_forecaster()

    df = load_cached_data()

    # Get stats
    anomaly_stats = detector.summary(df)
    forecast_df = forecaster.predict(periods=30)

    # Calculate Real Sales Growth
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    monthly_revenue = df.groupby(df['order_purchase_timestamp'].dt.to_period('M'))['payment_value'].sum()

    growth_str = "N/A"
    arrow = ""
    growth_val = 0.0

    if len(monthly_revenue) >= 2:
        current_month = monthly_revenue.iloc[-1]
        prev_month = monthly_revenue.iloc[-2]
        growth_val = ((current_month - prev_month) / prev_month) * 100
        arrow = "↑" if growth_val >= 0 else "↓"
        growth_str = f"{abs(growth_val):.1f}%"

    # Build KPI context
    kpis = {
        "total_revenue": {"display": f"R${df['payment_value'].sum():,.0f}", "change": "0"},
        "sales_growth": {
            "display": growth_str,
            "change": growth_str,
            "arrow": arrow,
            "sparkline": [10, 20, 15, 25]
        },
        "inventory_status": {"display": "0", "label": "items low"},
        "active_alerts": {"value": str(anomaly_stats["anomaly_count"]), "label": "View all alerts"}
    }

    # Populate Charts
    forecast_chart = {
        "labels": forecast_df["ds"].dt.strftime("%b %d").tolist(),
        "actual": [],
        "forecast": forecast_df["yhat"].round(2).tolist()
    }

    # Format Alerts safely
    raw_anomalies = detector.detect_anomalies(df, top_n=5)
    recent_alerts = []
    for a in raw_anomalies:
        recent_alerts.append({
            "message": f"Anomaly in {a.get('product_category', 'Unknown')}: Order {a.get('order_id', 'N/A')[:8]}",
            "date": "Recent",
            "severity": "Warning"
        })

    # 3. Safe check for anomalies to prevent page crash
    top_insight = {
        "title": "System Status",
        "summary": "No anomalies detected currently.",
        "recommendation": "Everything looks normal.",
        "confidence": 100
    }

    if raw_anomalies and len(raw_anomalies) > 0:
        top_anomaly = raw_anomalies[0]
        top_insight = {
            "title": f"Anomaly: {top_anomaly.get('product_category', 'Unknown')}",
            "summary": f"Detected order {top_anomaly.get('order_id', 'N/A')[:8]} with unusual metrics.",
            "recommendation": "Review anomaly report for investigation.",
            "confidence": 85
        }

    context = {
        "active_page": "dashboard",
        "kpis": kpis,
        "sales_chart": forecast_chart,
        "recent_alerts": recent_alerts,
        "growth": growth_val,
        "top_insight": top_insight,
        "user": get_current_user(),
    }

    return render_template("dashboard.html", **context)