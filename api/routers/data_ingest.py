"""
Data ingestion endpoints.

POST /api/upload           — Upload a replacement CSV to Main_dataset/
POST /api/pipeline/run     — Execute merge + clean + GX validation
GET  /api/pipeline/status  — Check which output files exist
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.config import settings

router = APIRouter()

_VALID_TABLES = {
    "olist_orders_dataset",
    "olist_order_items_dataset",
    "olist_products_dataset",
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation",
}


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    table_name: str = Query(
        ...,
        description=(
            "Target Olist table name without .csv extension. "
            "E.g. 'olist_orders_dataset'"
        ),
    ),
):
    """
    Upload a CSV file to Main_dataset/, overwriting the existing table.
    Use this to refresh individual datasets without re-running the full pipeline.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")
    if table_name not in _VALID_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown table '{table_name}'. Valid tables: {sorted(_VALID_TABLES)}",
        )

    dest = settings.DATA_DIR / f"{table_name}.csv"
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    return {
        "status":       "uploaded",
        "table":        table_name,
        "destination":  str(dest),
        "size_bytes":   dest.stat().st_size,
    }


@router.post("/pipeline/run")
async def run_pipeline(
    force_rebuild: bool = Query(
        False,
        description="Re-merge and re-clean even if cleaned_master_df.parquet already exists.",
    ),
):
    """
    Full pipeline:
        1. Merge all 9 Olist CSVs → master table
        2. Clean (remove cancelled orders, fill nulls, type-cast)
        3. Run Great Expectations validation
        4. Save cleaned_master_df.parquet + daily_sales_forecast_data.csv
    """
    try:
        import pandas as pd

        from data.cleaning import clean_master_table, validate_with_gx
        from data.pipeline import build_daily_forecasting_df, merge_master_table

        out = settings.PROCESSED_DIR
        target = out / "cleaned_master_df.parquet"

        if target.exists() and not force_rebuild:
            daily_target = out / "daily_sales_forecast_data.csv"
            return {
                "status": "skipped",
                "reason": "cleaned data already exists; pass force_rebuild=true to regenerate it",
                "master_rows": None,
                "daily_rows": None,
                "validation": None,
                "files": {
                    "cleaned_master_df.parquet": target.exists(),
                    "daily_sales_forecast_data.csv": daily_target.exists(),
                },
            }

        master_df = merge_master_table()
        cleaned   = clean_master_table(master_df)
        daily_df  = build_daily_forecasting_df(cleaned)

        out.mkdir(parents=True, exist_ok=True)

        # Non-destructive: preserve the existing (validated) master before overwrite
        if target.exists():
            backup = out / "cleaned_master_df.backup.parquet"
            if not backup.exists():
                shutil.copy2(target, backup)

        cleaned.to_parquet(target, index=False)
        daily_df.to_csv(out  / "daily_sales_forecast_data.csv", index=False)

        validation = validate_with_gx(cleaned)

        return {
            "status":       "success",
            "master_rows":  len(cleaned),
            "daily_rows":   len(daily_df),
            "validation":   validation,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/pipeline/status")
async def pipeline_status():
    """Return which pipeline artefacts are present on disk."""
    files = {
        "cleaned_master_df.parquet":      (settings.PROCESSED_DIR / "cleaned_master_df.parquet").exists(),
        "daily_sales_forecast_data.csv":  (settings.PROCESSED_DIR / "daily_sales_forecast_data.csv").exists(),
        "forecaster.pkl":                 (settings.PROCESSED_DIR / "forecaster.pkl").exists(),
        "raw_orders.csv":                 (settings.DATA_DIR / "olist_orders_dataset.csv").exists(),
    }
    return {
        "pipeline_ready": all(files.values()),
        "files":          files,
    }
