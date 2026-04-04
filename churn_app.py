import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn AI Dashboard", layout="wide")

st.markdown("""
<style>
body {background-color:#0f172a;}
h1 {text-align:center;color:#38bdf8;}
h2,h3 {color:#22c55e;}
.stButton>button {
    background:linear-gradient(90deg,#38bdf8,#22c55e);
    color:white;border-radius:10px;height:3em;font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "models", "rf_churn_model.joblib"))
    features = joblib.load(os.path.join(base, "models", "feature_columns.joblib"))
    return model, features

model, features = load_model()

# ---------------- FUNCTIONS ----------------
def retention_strategy(t, m, n, c):
    actions = []
    if c == "Month-to-month": actions.append("Offer long-term contract discounts.")
    if m > 80: actions.append("Provide pricing optimization or bundles.")
    if n < 3: actions.append("Upsell additional services.")
    if t < 12: actions.append("Send onboarding offers.")
    return actions or ["Customer stable. Offer loyalty rewards."]

def business_summary(pred, prob, t, m, n, c):
    if pred == 1:
        return f"""HIGH churn risk ({prob*100:.1f}%)
• Tenure: {t} months (low loyalty)
• Charges: ₹{m} (high)
• Services: {n} (low engagement)
• Contract: {c}

👉 Focus on pricing, bundling, and contract upgrade."""
    return f"""LOW churn risk ({prob*100:.1f}%)
• Stable tenure ({t} months)
• Good engagement ({n} services)

👉 Focus on loyalty rewards and upselling."""

# Feature importance
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance (%)": (model.feature_importances_ * 100).round(2)
}).sort_values("Importance (%)", ascending=False)

# ---------------- SESSION ----------------
st.session_state.setdefault("page", "input")

# ---------------- INPUT PAGE ----------------
if st.session_state.page == "input":

    st.title("📊 AI Customer Churn Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        t = st.number_input("Tenure", 0, 100, 12)
        m = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
        n = st.number_input("Services Used", 0, 10, 3)

    with col2:
        c = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        i = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        ts = st.selectbox("Tech Support", ["Yes", "No"])

    pb = st.selectbox("Paperless Billing", ["Yes", "No"])
    tc = st.number_input("Total Charges", 0.0, 20000.0, 1500.0)

    if st.button("🚀 Predict"):
        st.session_state.data = dict(t=t, m=m, tc=tc, n=n, c=c, i=i, ts=ts, pb=pb)
        st.session_state.page = "result"
        st.rerun()

# ---------------- RESULT PAGE ----------------
else:
    d = st.session_state.data

    df = pd.DataFrame({
        "tenure": [d["t"]],
        "MonthlyCharges": [d["m"]],
        "TotalCharges": [d["tc"]],
        "NumServices": [d["n"]],
        "Contract_Month-to-month": [d["c"] == "Month-to-month"],
        "Contract_One year": [d["c"] == "One year"],
        "Contract_Two year": [d["c"] == "Two year"],
        "InternetService_Fiber optic": [d["i"] == "Fiber optic"],
        "InternetService_DSL": [d["i"] == "DSL"],
        "TechSupport_Yes": [d["ts"] == "Yes"],
        "PaperlessBilling_Yes": [d["pb"] == "Yes"]
    }).astype(int)

    # Align features
    for col in features:
        if col not in df: df[col] = 0
    df = df[features]

    # Prediction
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.title("📊 Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob*100:.1f}%")
    col2.error("⚠️ High Risk" if pred else "✅ Stable")

    st.progress(int(prob * 100))

    st.subheader("🧠 Business Insight")
    st.info(business_summary(pred, prob, d["t"], d["m"], d["n"], d["c"]))

    if pred:
        st.subheader("💡 Recommended Actions")
        for a in retention_strategy(d["t"], d["m"], d["n"], d["c"]):
            st.write(f"• {a}")

    st.subheader("🔎 Top Influencing Features")
    st.dataframe(imp_df.head(5), use_container_width=True)

    if st.button("🔙 Back"):
        st.session_state.page = "input"
        st.rerun()