"""
FastAPI application entry point.

Start server:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import anomalies, data_ingest, forecasting, insights


def _parse_cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI BI Agent API — started")
    yield
    print("AI BI Agent API — stopped")


app = FastAPI(
    title="AI Business Intelligence Agent",
    description=(
        "Autonomous BI system with predictive sales forecasting (Prophet + MLflow), "
        "anomaly detection, and RAG-powered strategic recommendations "
        "(LangChain + ChromaDB + Groq | Ollama | Azure OpenAI)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_ingest.router, prefix="/api", tags=["Data Ingestion"])
app.include_router(forecasting.router, prefix="/api", tags=["Forecasting (M2)"])
app.include_router(anomalies.router,   prefix="/api", tags=["Anomaly Detection (M2)"])
app.include_router(insights.router,    prefix="/api", tags=["AI Insights (M3)"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "service": "AI BI Agent API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    from core.config import settings

    data_dir = Path(settings.DATA_DIR)
    processed_dir = Path(settings.PROCESSED_DIR)
    vector_db_dir = Path(settings.VECTOR_DB_DIR)
    anomaly_dir = Path(settings.ANOMALY_DIR)

    forecast_model_ready = (processed_dir / "forecaster.pkl").exists()
    forecast_data_ready = (processed_dir / "cleaned_master_df.parquet").exists() and (
        processed_dir / "daily_sales_forecast_data.csv"
    ).exists()
    anomaly_ready = all(
        (anomaly_dir / filename).exists()
        for filename in ("isolation_forest.pkl", "scaler.pkl", "label_encoders.pkl")
    )

    llm_ready = True
    if settings.LLM_PROVIDER.value == "groq":
        llm_ready = bool(settings.GROQ_API_KEY)
    elif settings.LLM_PROVIDER.value == "azure":
        llm_ready = bool(settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT)
    elif settings.LLM_PROVIDER.value == "ollama":
        llm_ready = bool(settings.OLLAMA_BASE_URL)

    return {
        "status":        "healthy",
        "llm_provider":  settings.LLM_PROVIDER.value,
        "data_dir":      {"path": str(data_dir), "exists": data_dir.exists()},
        "processed_dir": {"path": str(processed_dir), "exists": processed_dir.exists()},
        "vector_db_dir": {"path": str(vector_db_dir), "exists": vector_db_dir.exists()},
        "readiness": {
            "llm_configured": llm_ready,
            "forecast_data_ready": forecast_data_ready,
            "forecast_model_ready": forecast_model_ready,
            "anomaly_model_ready": anomaly_ready,
            "vector_store_ready": vector_db_dir.exists(),
        },
    }
