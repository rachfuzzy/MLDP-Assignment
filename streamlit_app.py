# streamlit_app.py
# Streamlit web app for Healthcare Cost Forecasting (Regression)
# Predicts annual_medical_cost using a trained Gradient Boosting model (joblib)

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------
# Page setup (UI)
# -----------------------
st.set_page_config(
    page_title="Healthcare Cost Forecaster",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"  # cleaner UI (sidebar hidden by default)
)

st.title("💳 Healthcare Cost Forecaster")
st.caption(
    "Estimate a patient's **annual medical cost** using a trained machine learning model "
    "(Gradient Boosting Regressor)."
)

# -----------------------
# Paths (edit if needed)
# -----------------------
MODEL_PATH = Path("healthcare_cost_best_gb_model.pkl")     # your tuned GB model file
FEATURE_COLS_PATH = Path("feature_columns.json")          # saved one-hot columns list
DATASET_PATH = Path("healthcare_cost_dataset.csv")        # optional: to auto-load categories

# -----------------------
# Helpers
# -----------------------
def load_feature_columns():
    """
    Load training one-hot columns from feature_columns.json.
    If not found, try to infer from dataset csv (optional).
    """
    if FEATURE_COLS_PATH.exists():
        return json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))

    if DATASET_PATH.exists():
        df_ref = pd.read_csv(DATASET_PATH)
        col_y = "annual_medical_cost"
        X_ref = df_ref.drop(columns=[col_y], errors="ignore")
        if "insurance_type" in X_ref.columns:
            X_ref["insurance_type"] = X_ref["insurance_type"].fillna("Unknown")
        X_ref = pd.get_dummies(X_ref, drop_first=True)
        return X_ref.columns.tolist()

    return None


def safe_currency(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


def build_input_df(user_inputs: dict, feature_cols: list[str]) -> pd.DataFrame:
    """
    Convert user inputs dict -> DataFrame, then apply one-hot encoding and align to training columns.
    """
    df_in = pd.DataFrame([user_inputs])

    # Handle missing category
    if "insurance_type" in df_in.columns:
        df_in["insurance_type"] = df_in["insurance_type"].fillna("Unknown")

    df_ohe = pd.get_dummies(df_in, drop_first=True)
    df_ohe = df_ohe.reindex(columns=feature_cols, fill_value=0)
    return df_ohe


def suggest_categories_from_dataset(colname: str, fallback: list[str]) -> list[str]:
    """
    If a dataset csv exists, show real categories from it. Otherwise, use fallback.
    """
    if DATASET_PATH.exists():
        try:
            d = pd.read_csv(DATASET_PATH, usecols=[colname])
            vals = sorted([v for v in d[colname].dropna().unique().tolist()])
            return vals if vals else fallback
        except Exception:
            return fallback
    return fallback


# -----------------------
# Load model + columns
# -----------------------
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Make sure **healthcare_cost_best_gb_model.pkl** is in the same folder as streamlit_app.py."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
FEATURE_COLS = load_feature_columns()

if FEATURE_COLS is None:
    st.error(
        "Missing **feature_columns.json** (recommended) and no dataset CSV fallback found.\n\n"
        "Fix: export feature columns from your notebook:\n"
        "```python\n"
        "import json\n"
        "json.dump(X.columns.tolist(), open('feature_columns.json','w'))\n"
        "```"
    )
    st.stop()

# -----------------------
# Sidebar (clean + minimal)
# -----------------------
with st.sidebar:
    st.header("⚙️ Model Info")
    st.write("**Model:** Gradient Boosting Regressor (tuned)")
    st.caption("Enter patient details to estimate annual medical cost.")

# -----------------------
# Main: input form
# -----------------------
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("🧾 Patient Details")

    # Use dataset categories if available, else fallbacks
    gender_opts = suggest_categories_from_dataset("gender", ["female", "male"])
    smoker_opts = suggest_categories_from_dataset("smoker", ["no", "yes"])
    pal_opts = suggest_categories_from_dataset("physical_activity_level", ["low", "moderate", "high"])
    ins_opts = suggest_categories_from_dataset("insurance_type", ["private", "public", "employer", "Unknown"])
    city_opts = suggest_categories_from_dataset("city_type", ["urban", "suburban", "rural"])

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.1)
            daily_steps = st.number_input("Daily steps", min_value=0, max_value=60000, value=8000, step=100)

        with c2:
            gender = st.selectbox("Gender", options=gender_opts, index=0)
            smoker = st.selectbox("Smoker", options=smoker_opts, index=0)
            sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=16.0, value=7.0, step=0.1)

        with c3:
            physical_activity_level = st.selectbox("Physical activity level", options=pal_opts, index=0)
            stress_level = st.slider("Stress level", min_value=0, max_value=10, value=5, step=1)
            city_type = st.selectbox("City type", options=city_opts, index=0)

        st.markdown("#### 🩺 Health Conditions (0 = No, 1 = Yes)")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            diabetes = st.selectbox("Diabetes", options=[0, 1], index=0)
        with h2:
            hypertension = st.selectbox("Hypertension", options=[0, 1], index=0)
        with h3:
            heart_disease = st.selectbox("Heart disease", options=[0, 1], index=0)
        with h4:
            asthma = st.selectbox("Asthma", options=[0, 1], index=0)

        st.markdown("#### 🏥 Healthcare Utilisation")
        u1, u2, u3 = st.columns(3)
        with u1:
            doctor_visits_per_year = st.number_input(
                "Doctor visits per year", min_value=0, max_value=100, value=2, step=1
            )
        with u2:
            hospital_admissions = st.number_input(
                "Hospital admissions", min_value=0, max_value=50, value=0, step=1
            )
        with u3:
            medication_count = st.number_input(
                "Medication count", min_value=0, max_value=100, value=1, step=1
            )

        st.markdown("#### 🛡️ Insurance & Cost History")
        i1, i2, i3 = st.columns(3)
        with i1:
            insurance_type = st.selectbox("Insurance type", options=ins_opts, index=0)
        with i2:
            insurance_coverage_pct = st.slider(
                "Insurance coverage (%)", min_value=0, max_value=100, value=80, step=1
            )
        with i3:
            previous_year_cost = st.number_input(
                "Previous year cost", min_value=0, max_value=1_000_000, value=1200, step=100
            )

        submitted = st.form_submit_button("🔮 Predict annual medical cost")

# -----------------------
# Prediction + output
# -----------------------
with right:
    st.subheader("📈 Prediction")

    if submitted:
        user_inputs = {
            "age": int(age),
            "gender": str(gender),
            "bmi": float(bmi),
            "smoker": str(smoker),
            "diabetes": int(diabetes),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "asthma": int(asthma),
            "physical_activity_level": str(physical_activity_level),
            "daily_steps": int(daily_steps),
            "sleep_hours": float(sleep_hours),
            "stress_level": int(stress_level),
            "doctor_visits_per_year": int(doctor_visits_per_year),
            "hospital_admissions": int(hospital_admissions),
            "medication_count": int(medication_count),
            "insurance_type": str(insurance_type) if insurance_type else "Unknown",
            "insurance_coverage_pct": int(insurance_coverage_pct),
            "city_type": str(city_type),
            "previous_year_cost": int(previous_year_cost),
        }

        try:
            X_in = build_input_df(user_inputs, FEATURE_COLS)
            pred = float(model.predict(X_in)[0])

            st.success("Prediction generated successfully.")
            st.metric("Estimated Annual Medical Cost", safe_currency(pred))

            with st.expander("🧠 How to read this result"):
                st.write(
                    "This is the model’s estimated **annual medical cost** based on the inputs. "
                    "Higher healthcare utilisation (hospital admissions, medication count, doctor visits) "
                    "and higher previous-year cost typically increase the prediction."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("Fill in the patient details on the left, then click **Predict annual medical cost**.")

st.divider()
st.caption("Built with Streamlit • Regression model: Gradient Boosting Regressor")
