import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import threading
from flask import Blueprint, render_template, request
import pandas as pd
import traceback
from core.config import settings
from ml_engine.anomaly_detection import OrderAnomalyDetector
from rag_service.pipeline import BIRAGPipeline
from rag_service.vector_store import BusinessVectorStore
from ui_config import get_current_user

insights_bp = Blueprint("insights", __name__)

# ------------------------------------------------------------------
# Model / pipeline singletons (load once per process)
# ------------------------------------------------------------------
_detector = None
_vector_store = None
_rag_pipeline = None
_detector_lock = threading.Lock()
_rag_lock = threading.Lock()


def get_detector():
    global _detector
    if _detector is not None and _detector.model is not None:
        return _detector

    with _detector_lock:
        if _detector is not None and _detector.model is not None:
            return _detector

        _detector = OrderAnomalyDetector()
        try:
            _detector.load()
            if _detector.model is None or _detector.scaler is None:
                raise RuntimeError("load() completed but model/scaler are still None.")
        except Exception as exc:
            print(f"[insights] load() failed ({exc}); retraining locally...")
            _detector.retrain()
            _detector.save()

        if _detector.model is None or _detector.scaler is None:
            raise RuntimeError("Detector still has no model after load()+retrain().")
    return _detector


def get_rag():
    global _vector_store, _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline

    with _rag_lock:
        if _rag_pipeline is not None:
            return _rag_pipeline

        _vector_store = BusinessVectorStore()
        if _vector_store.document_count == 0:
            print("[insights] vector store empty — indexing now...")
            df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")
            _vector_store.index_business_data(df)
        _rag_pipeline = BIRAGPipeline(vector_store=_vector_store)
    return _rag_pipeline


# ------------------------------------------------------------------
# Insight caching — avoid hitting Groq on every single page view.
# ------------------------------------------------------------------
CACHE_TTL_SECONDS = 15 * 60  # regenerate at most every 15 minutes

_anomalies_cache = {"data": None, "timestamp": 0}
_index_cache = {"data": None, "timestamp": 0}
_detail_cache = {}  # insight_id (str) -> {"data": ..., "timestamp": ...}
_cache_lock = threading.RLock()

def _get_anomalies(df, detector, top_n=50, force_refresh=False):
    """Cheap (no LLM call) — cached separately from the RAG-generated insights
    so /insights and /insights/<id> stay consistent with each other."""
    now = time.time()
    if (not force_refresh
            and _anomalies_cache["data"] is not None
            and now - _anomalies_cache["timestamp"] < CACHE_TTL_SECONDS):
        return _anomalies_cache["data"]

    with _cache_lock:
        now = time.time()
        if (not force_refresh
                and _anomalies_cache["data"] is not None
                and now - _anomalies_cache["timestamp"] < CACHE_TTL_SECONDS):
            return _anomalies_cache["data"]

        anomalies = detector.detect_anomalies(df, top_n=top_n)
        _anomalies_cache["data"] = anomalies
        _anomalies_cache["timestamp"] = now
    return anomalies


def _get_index_insights(df, detector, rag, force_refresh=False):
    """The expensive path — top 5 anomalies through the LLM. Cached with a TTL
    so switching between pages doesn't re-trigger 5 Groq calls every time."""
    now = time.time()
    if (not force_refresh
            and _index_cache["data"] is not None
            and now - _index_cache["timestamp"] < CACHE_TTL_SECONDS):
        return _index_cache["data"]

    with _cache_lock:
        now = time.time()
        if (not force_refresh
                and _index_cache["data"] is not None
                and now - _index_cache["timestamp"] < CACHE_TTL_SECONDS):
            return _index_cache["data"]

        anomalies = _get_anomalies(df, detector, top_n=50, force_refresh=force_refresh)
        top5 = anomalies[:5]
        insights = rag.generate_insights_batch(top5)
        _index_cache["data"] = insights
        _index_cache["timestamp"] = now
    return insights


def _get_detail_insight(insight_id, df, detector, rag, force_refresh=False):
    now = time.time()
    cached = _detail_cache.get(insight_id)
    if not force_refresh and cached and now - cached["timestamp"] < CACHE_TTL_SECONDS:
        return cached["data"]

    with _cache_lock:
        now = time.time()
        cached = _detail_cache.get(insight_id)
        if not force_refresh and cached and now - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["data"]

        anomalies = _get_anomalies(df, detector, top_n=50, force_refresh=force_refresh)
        insight_data = rag.generate_insight(anomalies[int(insight_id)])
        _detail_cache[insight_id] = {"data": insight_data, "timestamp": now}
    return insight_data


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@insights_bp.route("/insights")
def index():
    try:
        force_refresh = request.args.get("refresh") == "1"

        df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")
        detector = get_detector()
        rag = get_rag()

        insights = _get_index_insights(df, detector, rag, force_refresh=force_refresh)

        context = {
            "active_page": "insights",
            "insights": [
                {
                    "id": str(i),
                    "title": f"Anomaly: {a['anomaly'].get('product_category', 'Unknown')}",
                    "summary": a['recommendation'].get('diagnosis', 'No analysis available.'),
                    "confidence": int(a['recommendation'].get('confidence', 0) * 100)
                }
                for i, a in enumerate(insights)
            ],
            "user": get_current_user(),
        }
        return render_template("insights.html", **context)
    except Exception as e:
        print(traceback.format_exc())
        return f"Error: {str(e)}", 500


@insights_bp.route("/insights/<insight_id>")
def detail(insight_id):
    try:
        force_refresh = request.args.get("refresh") == "1"

        df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")
        detector = get_detector()
        rag = get_rag()

        insight_data = _get_detail_insight(insight_id, df, detector, rag, force_refresh=force_refresh)

        context = {
            "active_page": "insights",
            "insight": {
                "id": insight_id,
                "title": f"Anomaly in {insight_data['anomaly'].get('product_category', 'Unknown')}",
                "summary": insight_data['recommendation'].get('diagnosis', 'No diagnosis provided.'),
                "recommendation": ", ".join(insight_data['recommendation'].get('recommendations', [])),
                "confidence": int(insight_data['recommendation'].get('confidence', 0) * 100),
                "key_factors": insight_data.get('context_used', [])[:3],
                "chart": {"labels": ["Current", "Previous"], "data": [100, 200]}
            },
            "user": get_current_user(),
        }
        return render_template("insight_details.html", **context)
    except Exception as e:
        print(traceback.format_exc())
        return f"Error: {str(e)}", 500