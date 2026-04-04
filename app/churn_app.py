import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn AI Dashboard", layout="wide")

st.markdown("""
<style>
body {background-color:#0f172a;}
.main {background:linear-gradient(135deg,#0f172a,#1e293b);}
h1 {text-align:center;font-size:42px;color:#38bdf8;}
h2,h3 {color:#22c55e;}
.stButton>button {
    background:linear-gradient(90deg,#38bdf8,#22c55e);
    color:white;border-radius:12px;height:3em;font-size:18px;font-weight:bold;
}
</style>
""", unsafe_allow_html=True)
# ----------------CACHING---------------
@st.cache_resource
def load_model():
    BASE = os.path.dirname(__file__)
    model = joblib.load(os.path.join(BASE, "../models/rf_churn_model.joblib"))
    features = joblib.load(os.path.join(BASE, "../models/feature_columns.joblib"))
    return model, features

model, features = load_model()
# ---------------- LOAD ----------------
BASE = os.path.dirname(os.path.abspath(_file_))

model_path = os.path.abspath(os.path.join(BASE, "..", "models", "rf_churn_model.joblib"))
feature_path = os.path.abspath(os.path.join(BASE, "..", "models", "feature_columns.joblib"))

model = joblib.load(model_path)
features = joblib.load(feature_path)

st.success("✅ App Loaded Successfully")

# ---------------- SESSION ----------------
st.session_state.setdefault("page", "input")

# ---------------- FUNCTIONS ----------------
def retention_strategy(t, m, n, c):
    s=[]
    if c=="Month-to-month": s.append("Offer long-term contract discounts.")
    if m>80: s.append("Provide pricing optimization or bundles.")
    if n<3: s.append("Upsell additional services.")
    if t<12: s.append("Send onboarding engagement offers.")
    return s or ["Customer stable. Offer loyalty rewards."]

def business_summary(p, prob, t, m, n, c):
    if p==1:
        return f"""HIGH churn risk ({prob*100:.1f}%)
• Tenure: {t} months (low loyalty)
• Charges: ₹{m} (high)
• Services: {n} (low engagement)
• Contract: {c}

Action: pricing + bundling + contract conversion."""
    return f"""LOW churn risk ({prob*100:.1f}%)
• Stable tenure ({t} months)
• Good engagement ({n} services)

Action: loyalty rewards & premium offers."""

# Feature Importance
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance (%)": (model.feature_importances_*100).round(2)
}).sort_values(by="Importance (%)", ascending=False)

# ---------------- PAGE: INPUT ----------------
if st.session_state.page=="input":
    st.title("📊 AI Customer Churn Dashboard")

    c1,c2=st.columns(2)
    with c1:
        t=st.number_input("Tenure",0,100,12)
        m=st.number_input("Monthly Charges",0.0,500.0,70.0)
        n=st.number_input("Services",0,10,3)
    with c2:
        c=st.selectbox("Contract",["Month-to-month","One year","Two year"])
        i=st.selectbox("Internet",["DSL","Fiber optic","No"])
        ts=st.selectbox("Tech Support",["Yes","No"])

    pb=st.selectbox("Paperless Billing",["Yes","No"])
    tc=st.number_input("Total Charges",0.0,20000.0,1500.0)

    if st.button("🚀 Predict"):
        st.session_state.input=dict(t=t,m=m,tc=tc,n=n,c=c,i=i,ts=ts,pb=pb)
        st.session_state.page="result"
        st.rerun()

# ---------------- PAGE: RESULT ----------------
else:
    d=st.session_state.input

    df=pd.DataFrame({
        'tenure':[d["t"]],
        'MonthlyCharges':[d["m"]],
        'TotalCharges':[d["tc"]],
        'NumServices':[d["n"]],
        'Contract_Month-to-month':[d["c"]=="Month-to-month"],
        'Contract_One year':[d["c"]=="One year"],
        'Contract_Two year':[d["c"]=="Two year"],
        'InternetService_Fiber optic':[d["i"]=="Fiber optic"],
        'InternetService_DSL':[d["i"]=="DSL"],
        'TechSupport_Yes':[d["ts"]=="Yes"],
        'PaperlessBilling_Yes':[d["pb"]=="Yes"]
    }).astype(int)

    for col in features:
        if col not in df: df[col]=0
    df=df[features]

    pred=model.predict(df)[0]
    prob=model.predict_proba(df)[0][1]

    st.title("📊 Prediction Result")

    c1,c2=st.columns(2)
    c1.metric("Churn Probability", f"{prob*100:.1f}%")
    c2.error("⚠️ High Risk" if pred else "✅ Stable")

    st.progress(int(prob*100))

    st.subheader("🧠 Business Insight")
    st.info(business_summary(pred,prob,d["t"],d["m"],d["n"],d["c"]))

    if pred:
        st.subheader("💡 Actions")
        for a in retention_strategy(d["t"],d["m"],d["n"],d["c"]):
            st.write(f"• {a}")

    st.subheader("🔎 Top Influencing Features")
    top_features = imp_df.head(5)
    st.dataframe(top_features, use_container_width=True)

    if st.button("🔙 Back"):
        st.session_state.page="input"
        st.rerun()
