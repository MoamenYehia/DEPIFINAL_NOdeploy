"""
Order-level anomaly endpoints (Milestone 2 — Engineer 2, Isolation Forest).

POST /api/anomalies/retrain   — Refit Isolation Forest in the current environment
GET  /api/anomalies/orders    — Top-N most anomalous individual orders
GET  /api/anomalies/summary   — Aggregate normal-vs-anomaly comparison stats

Distinct from /api/forecast/anomalies, which detects TIME-SERIES (daily sales)
trend anomalies via Prophet. This router detects POINT anomalies — individual
orders that are operationally unusual.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.config import settings

router = APIRouter()

_detector = None


def _get_detector():
    """Lazy singleton. Loads saved artifacts; falls back to retrain on failure."""
    global _detector
    if _detector is None:
        from ml_engine.anomaly_detection import OrderAnomalyDetector

        _detector = OrderAnomalyDetector()
        try:
            _detector.load()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not load anomaly model ({exc}). "
                    "POST /api/anomalies/retrain first."
                ),
            )
    return _detector


def _check_data():
    if not (settings.PROCESSED_DIR / "cleaned_master_df.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="cleaned_master_df.parquet not found. POST /api/pipeline/run first.",
        )


@router.post("/anomalies/retrain")
async def retrain_anomaly_model(
    contamination: float = Query(
        0.02, ge=0.001, le=0.2,
        description="Expected fraction of anomalies. Higher = flags more orders.",
    ),
):
    """
    Refit the Isolation Forest from cleaned_master_df.parquet in the current
    sklearn version. Recommended on first setup to avoid version-mismatch issues
    with the artifacts trained in Google Colab.
    """
    global _detector
    try:
        _check_data()
        from ml_engine.anomaly_detection import OrderAnomalyDetector

        _detector = OrderAnomalyDetector()
        result = _detector.retrain(contamination=contamination)
        _detector.save()
        return {"status": "retrained", **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/anomalies/orders")
async def get_order_anomalies(
    top_n: int = Query(10, ge=1, le=100, description="Number of top anomalies to return."),
):
    """Return the *top_n* most anomalous individual orders, ranked by severity."""
    try:
        _check_data()
        detector = _get_detector()
        anomalies = detector.detect_anomalies(top_n=top_n)
        return {"anomaly_count": len(anomalies), "anomalies": anomalies}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/anomalies/summary")
async def get_anomaly_summary():
    """Aggregate comparison of anomalous vs normal orders (for dashboard charts)."""
    try:
        _check_data()
        detector = _get_detector()
        return detector.summary()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
