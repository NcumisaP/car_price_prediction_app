# app.py 

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Ensure classes used in the pickle are importable
import xgboost  # required for unpickling XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------
# Custom transformer from training
# -----------------------------
class LogFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_cols=("km_driven", "engine", "max_power")):
        self.add_cols = list(add_cols)

    def fit(self, X, y=None):
        self._present_cols_ = [c for c in self.add_cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self._present_cols_:
            X[f"log_{c}"] = np.log1p(pd.to_numeric(X[c], errors="coerce").fillna(0))
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = []
        out = list(input_features)
        for c in getattr(self, "_present_cols_", []):
            out.append(f"log_{c}")
        return np.array(out, dtype=object)

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Car Price: Drivers & Prediction", layout="centered")
st.title("Car Price: Drivers & Prediction")
st.caption("Upload a CSV for batch scoring or use the Predict tab for a single estimate.")

# Show versions (helpful for marking)
try:
    import sklearn, numpy, pandas
    st.caption(
        f"Python {sys.version.split()[0]} | numpy {numpy.__version__} | "
        f"pandas {pandas.__version__} | sklearn {sklearn.__version__} | xgboost {xgboost.__version__}"
    )
except Exception:
    pass

# -----------------------------
# Load pipeline (joblib)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(pkl_path: Path):
    try:
        return joblib.load(pkl_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

MODEL_PATH = Path(__file__).parent / "car_price_pipeline.pkl"
pipe = load_pipeline(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# Expected raw columns (must match training)
NUMERIC_RAW = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CATEG_RAW  = ["brand", "seller_type", "fuel_type", "transmission_type"]
RAW_COLS   = NUMERIC_RAW + CATEG_RAW

st.markdown(
    "<small>Expected columns: "
    + ", ".join([f"<code>{c}</code>" for c in RAW_COLS])
    + "</small>",
    unsafe_allow_html=True,
)

# -----------------------------
# Predictor 
# -----------------------------
def predict_price(df_raw: pd.DataFrame) -> np.ndarray:
    """Model predicts log_price; convert back to price."""
    log_preds = pipe.predict(df_raw)
    return np.expm1(log_preds)

# -----------------------------
# UI
# -----------------------------
tab_predict, tab_batch, tab_insights = st.tabs(["🔮 Predict", "📦 Batch", "🔎 Insights"])

# === Predict (single) ===
with tab_predict:
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)
    with col1:
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=40, value=6)
        km_driven   = st.number_input("Kilometers Driven", min_value=0, value=45000, step=100)
        mileage     = st.number_input("Mileage (km/l)", min_value=0.0, value=18.0, step=0.1)
    with col2:
        engine      = st.number_input("Engine (cc)", min_value=500, max_value=7000, value=1197, step=1)
        max_power   = st.number_input("Max Power (bhp)", min_value=20.0, max_value=600.0, value=82.0, step=0.5)
        seats       = st.number_input("Seats", min_value=2, max_value=10, value=5)

    brand = st.text_input("Brand", value="Maruti")
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("Predict Price"):
        row = pd.DataFrame([{
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "brand": brand,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
        }])
        try:
            price = predict_price(row)[0]
            st.success(f"Estimated Price: **₹{price:,.0f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# === Batch (CSV) ===
with tab_batch:
    st.subheader("Batch Scoring (CSV)")
    st.caption("CSV must have the same column names as above.")
    csv = st.file_uploader("Upload CSV", type=["csv"])
    if csv is not None:
        try:
            df = pd.read_csv(csv)
            missing = [c for c in RAW_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing columns in CSV: {missing}")
            else:
                preds = predict_price(df[RAW_COLS])
                out = df.copy()
                out["predicted_price"] = preds
                st.write("Preview:", out.head(20))
                st.download_button(
                    "Download predictions",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="car_price_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# === Insights (automatic, Booster-based) ===
with tab_insights:
    st.subheader("Feature Importance")
    st.caption("Computed from XGBoost Booster (gain). No CSV required.")

    def get_estimator(p):
        if hasattr(p, "named_steps"):
            # try common keys; else last step
            for k in ("model", "regressor", "final_estimator"):
                if k in p.named_steps:
                    return p.named_steps[k]
            return p.named_steps[list(p.named_steps.keys())[-1]]
        return p

    def get_transformed_feature_names(p):
        """
        Try to recover transformed feature names after (add_log_feats -> prep).
        Falls back to f0..fN when names are unavailable.
        """
        try:
            add = p.named_steps.get("add_log_feats", None)
            prep = p.named_steps.get("prep", None)
            if add is None or prep is None:
                raise RuntimeError("No add/prep steps")

            # Create a minimal single-row frame with required RAW_COLS
            # Use safe defaults; OHE name generation uses fitted categories (already in prep)
            dummy = pd.DataFrame([{c: 0 for c in RAW_COLS}])
            # put some strings for categoricals so OHE name rendering is okay
            for c in ["brand", "seller_type", "fuel_type", "transmission_type"]:
                if c in dummy.columns:
                    dummy[c] = "dummy"

            after_add = add.transform(dummy)
            input_feats = list(after_add.columns)
            names = prep.get_feature_names_out(input_features=input_feats)
            return list(names)
        except Exception:
            # fallback: unknown feature names; they will be f0..fN
            try:
                est = get_estimator(p)
                n = getattr(est, "n_features_in_", None)
                if n is None:
                    # last-resort: run a tiny transform to infer width
                    add = p.named_steps.get("add_log_feats", None)
                    prep = p.named_steps.get("prep", None)
                    dummy = pd.DataFrame([{c: 0 for c in RAW_COLS}])
                    for c in ["brand", "seller_type", "fuel_type", "transmission_type"]:
                        if c in dummy.columns:
                            dummy[c] = "dummy"
                    X = prep.transform(add.transform(dummy))
                    n = X.shape[1]
                return [f"f{i}" for i in range(int(n))]
            except Exception:
                return None

    try:
        est = get_estimator(pipe)
        # Prefer model attribute if available
        fi_attr = getattr(est, "feature_importances_", None)

        # Booster-based importance (robust, recommended)
        booster = est.get_booster()
        gain_dict = booster.get_score(importance_type="gain")  # e.g., {"f0": 0.12, "f1": 0.05, ...}

        # Build a dense vector aligned to feature order
        # XGBoost indices are 0..n-1, exposed as "f0","f1",...
        if hasattr(est, "n_features_in_") and est.n_features_in_ is not None:
            n_feats = int(est.n_features_in_)
        else:
            # fallback: deduce from names
            names_fallback = get_transformed_feature_names(pipe) or []
            n_feats = len(names_fallback) if names_fallback else max([int(k[1:]) for k in gain_dict.keys()]) + 1

        importances = np.zeros(n_feats, dtype=float)
        for k, v in gain_dict.items():
            try:
                idx = int(k[1:])  # strip leading 'f'
                if 0 <= idx < n_feats:
                    importances[idx] = float(v)
            except Exception:
                continue

        # If completely zero (rare), fall back to attribute importances
        if not np.any(importances) and fi_attr is not None:
            importances = np.asarray(fi_attr, dtype=float)

        # Normalize for readability
        if importances.sum() > 0:
            importances = importances / importances.sum()

        # Map to feature names if we can
        feat_names = get_transformed_feature_names(pipe)
        if feat_names and len(feat_names) == len(importances):
            labels = feat_names
        else:
            labels = [f"f{i}" for i in range(len(importances))]

        fi_df = pd.DataFrame({"feature": labels, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).head(20)

        st.dataframe(fi_df, use_container_width=True)
        st.bar_chart(fi_df.set_index("feature"))
        st.caption("Importance = average gain contributed by splits using each feature (top 20).")

    except Exception as e:
        st.error(f"Could not compute feature importance automatically: {e}")
        st.info("You can still upload a small sample CSV (200–1000 rows) below for permutation importance.")
        samp = st.file_uploader("Upload sample CSV", type=["csv"], key="pi_fallback")
        if samp is not None:
            try:
                df_s = pd.read_csv(samp)
                missing = [c for c in RAW_COLS if c not in df_s.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    from sklearn.inspection import permutation_importance
                    with st.spinner("Computing permutation importances..."):
                        Xs = df_s[RAW_COLS]
                        base = pipe.predict(Xs)  # unsupervised PI baseline
                        res = permutation_importance(pipe, Xs, base, n_repeats=8, random_state=42, scoring=None)
                    imp = res.importances_mean
                    names = RAW_COLS[:len(imp)]
                    imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
                    st.dataframe(imp_df.head(20), use_container_width=True)
                    st.bar_chart(imp_df.set_index("feature").head(20))
            except Exception as e2:
                st.error(f"Permutation importance failed: {e2}")

