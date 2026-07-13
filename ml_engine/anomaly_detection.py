"""
Milestone 2 — Engineer 2: Anomaly & Outlier Detection.

Production wrapper around the Isolation Forest model built in
'anomly_detection/Isolation_forest.ipynb'.

This detects ORDER-LEVEL (point) anomalies — individual orders that are
operationally unusual (very high payment, heavy products, slow delivery,
expensive freight, poor reviews). This complements the DATE-LEVEL
(time-series) anomaly detection in ``forecasting.py``.

The deterministic feature pipeline below replicates the notebook exactly so
the saved scaler / model receive features in the expected order.

Run standalone:  python ml_engine/anomaly_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import settings

# ---------------------------------------------------------------------------
# Feature contract — the EXACT 29 features (in order) the scaler/model expect.
# Order is taken from the notebook's df_model construction sequence.
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "order_status", "customer_state", "order_item_id", "price", "freight_value",
    "product_name_lenght", "product_description_lenght", "product_photos_qty",
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    "product_category", "seller_state", "payment_sequential", "payment_type",
    "payment_installments", "payment_value", "review_score", "delivery_days",
    "purchase_year", "purchase_month", "purchase_day", "purchase_weekday",
    "purchase_hour", "distance_km", "total_cost", "payment_per_installment",
    "freight_ratio",
]

CATEGORICAL_COLS = ["order_status", "customer_state", "product_category",
                    "seller_state", "payment_type"]

# Business-friendly columns surfaced in anomaly reports (kept from the raw df)
_REPORT_COLS = ["order_id", "customer_state", "seller_state", "product_category",
                "payment_value", "freight_value", "delivery_days", "review_score",
                "product_weight_g"]

# Model hyper-parameters (match the notebook)
_N_ESTIMATORS = 100
_CONTAMINATION = 0.02
_RANDOM_STATE = 42


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (matches the notebook formula)."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class OrderAnomalyDetector:
    """
    Isolation-Forest order-level anomaly detector.

    Two modes:
        - ``load()``    — reuse Engineer 2's saved artifacts (fast, but tied to
                          the sklearn version they trained with).
        - ``retrain()`` — refit from the cleaned master table in the current
                          environment (recommended — removes version-mismatch risk
                          and is fully deterministic via random_state=42).
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders: dict = {}

    # ------------------------------------------------------------------
    # Feature engineering (deterministic — mirrors the notebook)
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame, *, fit_encoders: bool) -> pd.DataFrame:
        """
        Transform a raw cleaned_master_df into the 29-column model matrix.

        Args:
            df: Raw cleaned master table (44 cols).
            fit_encoders: If True, fit fresh LabelEncoders/scaler (retrain mode).
                          If False, use the loaded encoders (inference mode).
        """
        from sklearn.preprocessing import LabelEncoder

        d = df.copy()

        # --- date parts from purchase timestamp ---
        ts = pd.to_datetime(d["order_purchase_timestamp"], errors="coerce")
        d["purchase_year"] = ts.dt.year
        d["purchase_month"] = ts.dt.month
        d["purchase_day"] = ts.dt.day
        d["purchase_weekday"] = ts.dt.weekday
        d["purchase_hour"] = ts.dt.hour

        # --- seller↔customer distance ---
        d["distance_km"] = _haversine_km(
            d["customer_lat"], d["customer_lng"], d["seller_lat"], d["seller_lng"]
        )

        # --- numeric fill (median) for the base feature columns ---
        base_numeric = [c for c in FEATURE_ORDER if c not in CATEGORICAL_COLS
                        and c not in ("total_cost", "payment_per_installment", "freight_ratio")]
        for col in base_numeric:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
                d[col] = d[col].fillna(d[col].median())

        # --- categorical fill + encode ---
        for col in CATEGORICAL_COLS:
            d[col] = d[col].astype(str).fillna("Unknown").replace("nan", "Unknown")
            if fit_encoders:
                le = LabelEncoder()
                d[col] = le.fit_transform(d[col])
                self.label_encoders[col] = le
            else:
                d[col] = self._safe_encode(col, d[col])

        # --- derived features (after fill, so no NaN) ---
        d["payment_installments"] = d["payment_installments"].replace(0, 1)
        d["total_cost"] = d["price"] + d["freight_value"]
        d["payment_per_installment"] = d["payment_value"] / d["payment_installments"]
        d["freight_ratio"] = d["freight_value"] / (d["price"] + 1)

        # --- exact column order the scaler/model expect ---
        return d[FEATURE_ORDER].copy()

    def _safe_encode(self, col: str, series: pd.Series) -> pd.Series:
        """Encode with a fitted LabelEncoder, mapping unseen labels to class 0."""
        le = self.label_encoders[col]
        known = set(le.classes_)
        fallback = le.classes_[0]
        safe = series.where(series.isin(known), fallback)
        return pd.Series(le.transform(safe), index=series.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> "OrderAnomalyDetector":
        """Load Engineer 2's saved artifacts from settings.ANOMALY_DIR."""
        d = settings.ANOMALY_DIR
        model_path = d / "isolation_forest.pkl"
        scaler_path = d / "scaler.pkl"
        encoders_path = d / "label_encoders.pkl"

        missing = [str(path) for path in (model_path, scaler_path, encoders_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing anomaly detection artifacts: " + ", ".join(missing)
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        print(f"Loaded Isolation Forest artifacts from {d}")
        return self

    def save(self) -> None:
        """Persist the current model/scaler/encoders back to settings.ANOMALY_DIR."""
        d = settings.ANOMALY_DIR
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, d / "isolation_forest.pkl")
        joblib.dump(self.scaler, d / "scaler.pkl")
        joblib.dump(self.label_encoders, d / "label_encoders.pkl")
        print(f"Saved Isolation Forest artifacts -> {d}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def retrain(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        contamination: float = _CONTAMINATION,
        log_to_mlflow: bool = True,
    ) -> dict:
        """
        Refit the Isolation Forest from the cleaned master table in the current
        environment. Recommended to remove the sklearn version-mismatch risk.

        Returns dict with anomaly_count and anomaly_percentage.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        if df is None:
            df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")

        X_df = self._build_features(df, fit_encoders=True)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_df)

        self.model = IsolationForest(
            n_estimators=_N_ESTIMATORS,
            contamination=contamination,
            random_state=_RANDOM_STATE,
        )
        self.model.fit(X)

        preds = self.model.predict(X)
        anomaly_count = int((preds == -1).sum())
        result = {
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_count / len(preds), 4),
            "total_orders": len(preds),
        }

        if log_to_mlflow:
            self._log_mlflow(contamination, result)

        print(f"Retrained: {anomaly_count:,} anomalies "
              f"({result['anomaly_percentage']:.2%}) of {len(preds):,} orders")
        return result

    def _log_mlflow(self, contamination: float, metrics: dict) -> None:
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("olist_anomaly_detection")
            with mlflow.start_run():
                mlflow.log_param("model", "IsolationForest")
                mlflow.log_param("n_estimators", _N_ESTIMATORS)
                mlflow.log_param("contamination", contamination)
                mlflow.log_metric("anomaly_count", metrics["anomaly_count"])
                mlflow.log_metric("anomaly_percentage", metrics["anomaly_percentage"])
                mlflow.sklearn.log_model(self.model, name="isolation_forest")
        except Exception as exc:  # MLflow is optional per the brief
            print(f"[MLflow skipped] {exc}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row in *df*.

        Returns the original df with two added columns:
            anomaly       : 1 (normal) / -1 (anomaly)
            anomaly_score : lower = more anomalous (Isolation Forest decision fn)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call load() or retrain() before score().")

        X_df = self._build_features(df, fit_encoders=False)
        X = self.scaler.transform(X_df)

        out = df.copy()
        out["anomaly"] = self.model.predict(X)
        out["anomaly_score"] = self.model.decision_function(X)
        return out

    def detect_anomalies(
        self,
        df: Optional[pd.DataFrame] = None,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Return the *top_n* most anomalous orders, ranked by severity
        (most negative anomaly_score first), as JSON-ready dicts tagged for RAG.
        """
        if df is None:
            df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")

        scored = self.score(df)
        anomalies = scored[scored["anomaly"] == -1].copy()
        anomalies = anomalies.sort_values("anomaly_score").head(top_n)

        records = []
        for _, row in anomalies.iterrows():
            rec = {"type": "order_anomaly", "anomaly_score": round(float(row["anomaly_score"]), 4)}
            for col in _REPORT_COLS:
                if col in row:
                    val = row[col]
                    rec[col] = (round(float(val), 2)
                                if isinstance(val, (int, float, np.floating)) else str(val))
            records.append(rec)
        return records

    def summary(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Aggregate stats comparing anomalous vs normal orders (for the dashboard)."""
        if df is None:
            df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")

        scored = self.score(df)
        anom = scored[scored["anomaly"] == -1]
        norm = scored[scored["anomaly"] == 1]

        def _stats(frame, col):
            return round(float(frame[col].mean()), 2) if col in frame else None

        return {
            "total_orders": len(scored),
            "anomaly_count": len(anom),
            "anomaly_percentage": round(len(anom) / len(scored), 4),
            "comparison": {
                "payment_value":  {"normal": _stats(norm, "payment_value"),  "anomaly": _stats(anom, "payment_value")},
                "freight_value":  {"normal": _stats(norm, "freight_value"),  "anomaly": _stats(anom, "freight_value")},
                "delivery_days":  {"normal": _stats(norm, "delivery_days"),  "anomaly": _stats(anom, "delivery_days")},
                "product_weight_g": {"normal": _stats(norm, "product_weight_g"), "anomaly": _stats(anom, "product_weight_g")},
                "review_score":   {"normal": _stats(norm, "review_score"),   "anomaly": _stats(anom, "review_score")},
            },
            "top_categories": anom["product_category"].value_counts().head(10).to_dict(),
            "top_customer_states": anom["customer_state"].value_counts().head(10).to_dict(),
        }


if __name__ == "__main__":
    df = pd.read_parquet(settings.PROCESSED_DIR / "cleaned_master_df.parquet")

    detector = OrderAnomalyDetector()
    # Retrain in the current environment (recommended — fixes version mismatch)
    detector.retrain(df)
    detector.save()

    top = detector.detect_anomalies(df, top_n=5)
    print(f"\nTop 5 anomalous orders:")
    for a in top:
        print(f"  {a['order_id'][:10]}  R${a['payment_value']:>9,.2f}  "
              f"freight R${a['freight_value']:>6,.2f}  "
              f"{a['delivery_days']:>3.0f}d  score={a['anomaly_score']}")
