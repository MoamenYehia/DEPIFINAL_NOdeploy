"""
Data merging pipeline — converts the 9 raw Olist CSVs into a unified master table.
Logic extracted and productionised from notebooks/1.ipynb.

Run standalone:  python data/pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import settings


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: pd.Series, lon1: pd.Series,
                   lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Vectorised great-circle distance in kilometres."""
    R = 6_371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_datasets(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all 9 Olist CSVs from *data_dir* (defaults to settings.DATA_DIR)."""
    d = data_dir or settings.DATA_DIR
    print(f"Loading raw datasets from {d} …")
    return {
        "orders":       pd.read_csv(d / "olist_orders_dataset.csv"),
        "order_items":  pd.read_csv(d / "olist_order_items_dataset.csv"),
        "products":     pd.read_csv(d / "olist_products_dataset.csv"),
        "customers":    pd.read_csv(d / "olist_customers_dataset.csv"),
        "geolocation":  pd.read_csv(d / "olist_geolocation_dataset.csv"),
        "payments":     pd.read_csv(d / "olist_order_payments_dataset.csv"),
        "reviews":      pd.read_csv(d / "olist_order_reviews_dataset.csv"),
        "sellers":      pd.read_csv(d / "olist_sellers_dataset.csv"),
        "translations": pd.read_csv(d / "product_category_name_translation.csv"),
    }


def merge_master_table(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Execute the full JOIN pipeline following the ER diagram.

    Merge path:
        orders → customers → order_items → products (English) → sellers
               → payments (aggregated) → reviews → customer geo → seller geo

    Returns the master DataFrame (~43 columns).
    """
    raw = load_raw_datasets(data_dir)

    orders       = raw["orders"]
    order_items  = raw["order_items"]
    products     = raw["products"]
    customers    = raw["customers"]
    geolocation  = raw["geolocation"]
    payments     = raw["payments"]
    reviews      = raw["reviews"]
    sellers      = raw["sellers"]
    translations = raw["translations"]

    # 1. Aggregate geolocation: one centroid per zip-code prefix
    geo = (
        geolocation
        .groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .reset_index()
    )

    # 2. Aggregate payments: one row per order to avoid fan-out.
    #    Keep payment_type as the DOMINANT single type (not a comma-join) so the
    #    label encoder used downstream still sees a known category, and retain
    #    payment_sequential (max) which the anomaly detector requires.
    payments_agg = (
        payments
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_sequential=("payment_sequential", "max"),
            payment_type=("payment_type", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )

    # 3. Map product categories to English
    products = products.copy()
    cat_map = translations.set_index("product_category_name")["product_category_name_english"]
    products["product_category"] = (
        products["product_category_name"].map(cat_map).fillna(products["product_category_name"])
    )
    products.drop(columns=["product_category_name"], inplace=True)

    # 4. Step-by-step merge (ER diagram order)
    df = orders.copy()
    df = df.merge(customers,     on="customer_id",  how="left")
    df = df.merge(order_items,   on="order_id",     how="left")
    df = df.merge(products,      on="product_id",   how="left")
    df = df.merge(sellers,       on="seller_id",    how="left")
    df = df.merge(payments_agg,  on="order_id",     how="left")
    df = df.merge(
        reviews[["order_id", "review_id", "review_score",
                 "review_comment_message", "review_creation_date"]],
        on="order_id", how="left",
    )

    # 5. Attach customer geolocation
    df = df.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "lat": "customer_lat", "lng": "customer_lng",
        }),
        on="customer_zip_code_prefix", how="left",
    )

    # 6. Attach seller geolocation
    df = df.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "lat": "seller_lat", "lng": "seller_lng",
        }),
        on="seller_zip_code_prefix", how="left",
    )

    # 7. Parse all date columns
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "shipping_limit_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 8. Derived features
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["is_delayed"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype("Int8")
    df["logistics_distance_km"] = _haversine_km(
        df["seller_lat"], df["seller_lng"],
        df["customer_lat"], df["customer_lng"],
    )
    df["purchase_year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    print(f"Master table: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def build_daily_forecasting_df(
    master_df: pd.DataFrame,
    trim_sparse_edges: bool = True,
) -> pd.DataFrame:
    """
    Aggregate the master table to daily granularity for Prophet.

    Args:
        trim_sparse_edges: Drop the leading ramp-up and trailing data-collection
            cutoff (days at the very start/end with < 20% of median daily orders).
            This is standard for the Olist dataset — its first weeks (late 2016)
            and final weeks (late Aug 2018) are incomplete and otherwise inflate
            forecast error massively.

    Returns DataFrame with columns:
        date, total_sales, total_orders, unique_customers, avg_freight
    """
    daily = (
        master_df
        .groupby(master_df["order_purchase_timestamp"].dt.normalize())
        .agg(
            total_sales=("payment_value", "sum"),
            total_orders=("order_id", "nunique"),
            unique_customers=("customer_unique_id", "nunique"),
            avg_freight=("freight_value", "mean"),
        )
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    if trim_sparse_edges and len(daily) > 30:
        threshold = daily["total_orders"].median() * 0.2
        dense = daily.index[daily["total_orders"] >= threshold]
        if len(dense) > 0:
            daily = daily.loc[dense.min():dense.max()].reset_index(drop=True)

    return daily


def run_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline entry point: load raw → merge → save master + daily tables.

    Returns (master_df, daily_df).
    """
    out = output_dir or settings.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    master_df = merge_master_table(data_dir)
    daily_df  = build_daily_forecasting_df(master_df)

    master_df.to_parquet(out / "olist_master_dataset.parquet", index=False)
    daily_df.to_csv(out / "daily_sales_forecast_data.csv", index=False)

    print(f"Saved -> {out / 'olist_master_dataset.parquet'}")
    print(f"Saved -> {out / 'daily_sales_forecast_data.csv'}")
    return master_df, daily_df


if __name__ == "__main__":
    run_pipeline()
