"""
AI Insights endpoints (Milestone 3).

POST /api/insights/index  — Index cleaned data into ChromaDB
GET  /api/insights        — Full pipeline: forecast → anomalies → RAG → LLM
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.config import settings

router = APIRouter()

# Singletons
_rag_pipeline  = None
_vector_store  = None


def _get_rag():
    global _rag_pipeline, _vector_store
    if _rag_pipeline is None:
        from rag_service.pipeline import BIRAGPipeline
        from rag_service.vector_store import BusinessVectorStore

        _vector_store  = BusinessVectorStore()
        _rag_pipeline  = BIRAGPipeline(vector_store=_vector_store)
    return _rag_pipeline, _vector_store


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/insights/index")
async def index_business_data():
    """
    Generate narrative text summaries from cleaned_master_df.parquet
    and upsert them into the ChromaDB vector store.

    Call this once after /api/pipeline/run before using /api/insights.
    """
    try:
        clean_path = settings.PROCESSED_DIR / "cleaned_master_df.parquet"
        if not clean_path.exists():
            raise HTTPException(
                status_code=400,
                detail="cleaned_master_df.parquet not found. POST /api/pipeline/run first.",
            )

        df = pd.read_parquet(clean_path)
        _, vs = _get_rag()
        count = vs.index_business_data(df)

        return {"status": "indexed", "documents_indexed": count}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/insights/orders")
async def get_order_insights(
    top_n: int = Query(
        3, ge=1, le=10,
        description="Number of top anomalous orders to explain with the LLM.",
    ),
):
    """
    AI explanations for ORDER-LEVEL anomalies (Isolation Forest → RAG → LLM).

    For each of the top-N most anomalous orders, retrieves business context
    from ChromaDB and asks the LLM why it is unusual and what to do.

    Prerequisites:
        POST /api/pipeline/run
        POST /api/anomalies/retrain   (or rely on saved artifacts)
        POST /api/insights/index
    """
    try:
        rag, vs = _get_rag()

        if vs.document_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. POST /api/insights/index first.",
            )

        from ml_engine.anomaly_detection import OrderAnomalyDetector

        detector = OrderAnomalyDetector()
        try:
            detector.load()
        except Exception:
            detector.retrain()
            detector.save()

        anomalies = detector.detect_anomalies(top_n=top_n)
        if not anomalies:
            return {"status": "no_anomalies", "insights": []}

        insights = rag.generate_insights_batch(anomalies)
        return {
            "status":              "success",
            "insights_generated":  len(insights),
            "llm_provider":        settings.LLM_PROVIDER.value,
            "insights":            insights,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/insights")
async def get_insights(
    std_threshold: float = Query(
        2.0,
        description="Anomaly detection sensitivity (σ from forecast). Lower = more anomalies.",
    ),
    max_anomalies: int = Query(
        3, ge=1, le=10,
        description="Maximum number of anomalies to explain with the LLM.",
    ),
):
    """
    Full intelligence pipeline (the core GET /api/insights endpoint):

    1. Detect anomalies using the trained Prophet forecast
    2. Rank by absolute deviation
    3. For each top anomaly → retrieve ChromaDB context → call LLM
    4. Return structured JSON with diagnosis, impact level, and recommendations

    Prerequisites (in order):
        POST /api/pipeline/run
        POST /api/forecast/train
        POST /api/insights/index
    """
    try:
        rag, vs = _get_rag()

        if vs.document_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. POST /api/insights/index first.",
            )

        # Reuse forecasting router helpers to avoid code duplication
        from api.routers.forecasting import _ensure_daily_df, _ensure_model_loaded

        forecaster = _ensure_model_loaded()
        daily_df   = _ensure_daily_df()

        if forecaster._forecast_df is None:
            forecaster.predict(periods=0)

        anomalies = forecaster.detect_trend_anomalies(daily_df, std_threshold=std_threshold)

        if not anomalies:
            return {
                "status":   "no_anomalies",
                "message":  "No significant deviations detected at the current threshold.",
                "insights": [],
            }

        # Sort by absolute deviation, take top N
        top = sorted(anomalies, key=lambda x: abs(x.get("deviation_pct", 0)), reverse=True)
        top = top[:max_anomalies]

        insights = rag.generate_insights_batch(top)

        return {
            "status":                    "success",
            "total_anomalies_detected":  len(anomalies),
            "insights_generated":        len(insights),
            "llm_provider":              settings.LLM_PROVIDER.value,
            "insights":                  insights,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
