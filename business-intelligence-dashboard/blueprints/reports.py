import pandas as pd
import os
import io
from datetime import datetime
from flask import Blueprint, render_template, send_file
from core.config import settings

reports_bp = Blueprint("reports", __name__)

@reports_bp.route("/reports")
def index():
    # Load dataset
    df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")
    
    # 1. Calculate Metrics
    file_path = settings.PROCESSED_DIR / "cleaned_master_df.parquet"
    last_mod = os.path.getmtime(file_path)
    
    # Calculate coverage (e.g., non-null values in key columns)
    coverage_score = (df.notnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    report_summary = {
        "total_reports": 3, 
        "scheduled_exports": 0,
        "last_refresh": datetime.fromtimestamp(last_mod).strftime("%b %d, %I:%M %p"),
        "coverage": f"{coverage_score:.0f}%"
    }
    
    # 2. Define Dynamic Report Cards
    report_cards = [
        {"title": "Revenue by Region", "description": "Monthly revenue split across all regions.", "status": "Ready"},
        {"title": "Forecast Accuracy", "description": "How close the latest forecast was to observed sales.", "status": "Ready"},
        {"title": "Open Alerts Snapshot", "description": "Current unresolved anomalies by severity.", "status": "Needs review"}
    ]
    
    return render_template("reports.html", report_summary=report_summary, report_cards=report_cards)

@reports_bp.route("/reports/export")
def export_reports():
    # Load your master data
    df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")
    
    # Create a CSV in memory
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype="text/csv",
        download_name="bi_dashboard_export.csv",
        as_attachment=True
    )