"""
Data cleaning and Great Expectations validation.
Cleaning steps match those documented in 'Data Cleaning + EDA/README .txt'.

Run standalone:  python data/cleaning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import settings


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_master_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the merged master table.

    Steps (from README):
        1. Remove duplicate rows
        2. Remove cancelled orders
        3. Remove rows with null product_id
        4. Fill missing monetary values with 0, review_score with median
        5. Cast date columns to datetime, monetary columns to float
        6. Add delivery_days column
        7. Filter out negative / >365 delivery_days
    """
    df = df.copy()
    before = len(df)

    # 1. Duplicates
    df.drop_duplicates(inplace=True)

    # 2. Keep only valid/completed order statuses (drops cancelled, unavailable,
    #    invoiced, created, etc.). Aligns with the GX order_status expectation and
    #    the team's approach of analysing completed transactions only.
    if "order_status" in df.columns:
        valid_statuses = ["delivered", "shipped", "processing", "approved"]
        df = df[df["order_status"].isin(valid_statuses)]

    # 3. Rows without a product can't be analysed
    if "product_id" in df.columns:
        df = df[df["product_id"].notna()]

    # 4. Date casting
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "shipping_limit_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 5. Monetary casting + fill
    for col in ("payment_value", "price", "freight_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(0.0)

    # 6. Review score fill
    if "review_score" in df.columns:
        median_score = df["review_score"].median()
        df["review_score"] = df["review_score"].fillna(median_score)

    # 7. delivery_days
    if "delivery_days" not in df.columns:
        ts_col = "order_purchase_timestamp"
        del_col = "order_delivered_customer_date"
        if ts_col in df.columns and del_col in df.columns:
            df["delivery_days"] = (df[del_col] - df[ts_col]).dt.days

    # 8. Filter logically invalid delivery times
    if "delivery_days" in df.columns:
        mask = df["delivery_days"].isna() | (
            (df["delivery_days"] >= 0) & (df["delivery_days"] <= 365)
        )
        df = df[mask]

    after = len(df)
    print(f"Cleaning complete: {before:,} -> {after:,} rows  ({before - after:,} removed)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_with_gx(df: pd.DataFrame) -> dict:
    """
    Run the Great Expectations suite used in Milestone 1.
    Returns a dict with 'success', 'statistics', and per-expectation 'results'.
    """
    try:
        import great_expectations as gx

        ctx = gx.get_context(mode="ephemeral")
        ds   = ctx.data_sources.add_pandas("pandas_src")
        asset = ds.add_dataframe_asset("df_asset")
        batch_def = asset.add_batch_definition_whole_dataframe("batch")
        batch = batch_def.get_batch(batch_parameters={"dataframe": df})

        suite = ctx.suites.add(gx.ExpectationSuite(name="olist_suite"))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id"))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id"))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="payment_value", min_value=0.0))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="review_score", min_value=1.0, max_value=5.0))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="order_status",
                value_set=["delivered", "shipped", "processing", "approved"]))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="delivery_days", min_value=0.0, max_value=365.0))

        val_def = ctx.validation_definitions.add(
            gx.ValidationDefinition(
                name="olist_validation", data=batch_def, suite=suite))
        result = val_def.run(batch_parameters={"dataframe": df})

        return {
            "success": bool(result.success),
            "statistics": result.statistics,
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_or_build_clean_data(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Return cleaned master DataFrame.
    Loads from the existing parquet produced by the team unless *force_rebuild* is True.
    """
    clean_path = settings.PROCESSED_DIR / "cleaned_master_df.parquet"

    if clean_path.exists() and not force_rebuild:
        print(f"Loading clean data from {clean_path}")
        return pd.read_parquet(clean_path)

    from data.pipeline import run_pipeline
    master_df, _ = run_pipeline()
    cleaned = clean_master_table(master_df)
    cleaned.to_parquet(clean_path, index=False)
    print(f"Saved cleaned data -> {clean_path}")
    return cleaned


if __name__ == "__main__":
    df = load_or_build_clean_data(force_rebuild=True)
    result = validate_with_gx(df)
    print("GX validation:", result)
