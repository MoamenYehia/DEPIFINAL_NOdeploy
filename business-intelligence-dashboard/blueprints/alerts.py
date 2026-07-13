from flask import Blueprint, render_template, request
import pandas as pd
from core.config import settings
from ml_engine.anomaly_detection import OrderAnomalyDetector
from ui_config import REGIONS, TIME_RANGES, get_current_user

alerts_bp = Blueprint("alerts", __name__)

# Lazy singleton — load/retrain the Isolation Forest once per process.
_detector = None


def get_detector():
    global _detector
    if _detector is not None and _detector.model is not None:
        return _detector

    _detector = OrderAnomalyDetector()
    try:
        _detector.load()
        if _detector.model is None or _detector.scaler is None:
            raise RuntimeError("load() completed but model/scaler are still None.")
    except Exception as exc:
        print(f"[alerts] anomaly model load failed ({exc}); retraining locally...")
        _detector.retrain()
        _detector.save()

    return _detector


@alerts_bp.route("/alerts")
def index():
    severity = request.args.get("severity", "All")
    status = request.args.get("status", "All")
    date_range = request.args.get("date_range", "All Time")

    # 1. Load data + cached detector
    data_path = settings.PROCESSED_DIR / "cleaned_master_df.parquet"
    df = pd.read_parquet(data_path)
    detector = get_detector()

    # 2. Get real anomalies
    raw_anomalies = detector.detect_anomalies(df, top_n=50)

    # 3. Format into the structure the template expects
    alerts = []
    for i, a in enumerate(raw_anomalies):
        alerts.append({
            "id": i,
            "title": f"Anomalous order {a.get('order_id', 'N/A')[:8]}",
            "date": "Recent",  # Placeholder for the anomaly date logic
            "severity": "Critical" if a["anomaly_score"] < -0.2 else "Warning",
            "status": "Open",
            "detail": f"Cat: {a['product_category']}, Payment: R${a['payment_value']}, Score: {a['anomaly_score']}"
        })

    # 4. Apply filters
    if severity != "All":
        alerts = [a for a in alerts if a["severity"] == severity]
    if status != "All":
        alerts = [a for a in alerts if a["status"] == status]

    context = {
        "active_page": "alerts",
        "alerts": alerts,
        "severities": ["All", "Critical", "Warning"],
        "statuses": ["All", "Open"],
        "date_ranges": ["All Time"],
        "selected_severity": severity,
        "selected_status": status,
        "selected_date_range": date_range,
        "user": get_current_user(),
    }
    return render_template("alerts.html", **context)