from pathlib import Path
from html import escape
from textwrap import dedent

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]
STATE_DEFAULTS = {"page": "input", "data": None, "result": None}

FEATURE_EXPLANATIONS = {
    "MonthlyCharges": "Monthly price pressure can make a customer more likely to compare alternatives.",
    "tenure": "Shorter tenure often means lower loyalty and weaker attachment.",
    "TotalCharges": "Lifetime spend helps the model understand account value and maturity.",
    "NumServices": "Customers using more services are usually harder to lose.",
    "Contract_Month-to-month": "Month-to-month plans create less commitment and easier switching.",
    "Contract_One year": "Annual contracts usually reduce churn risk compared with monthly plans.",
    "Contract_Two year": "Two-year contracts usually signal the strongest retention.",
    "InternetService_Fiber optic": "Internet type can correlate with both pricing and satisfaction patterns.",
    "InternetService_DSL": "Service type helps the model distinguish customer usage segments.",
    "TechSupport_Yes": "Access to support often reduces frustration-driven churn.",
    "PaperlessBilling_Yes": "Billing behavior can indicate customer habits and digital engagement.",
}

HERO_CARDS = [
    ("Predict", "Measure churn probability", "Estimate how likely a customer is to leave based on key account signals."),
    ("Explain", "Turn model output into insight", "Translate model signals into language a business user can understand quickly."),
    ("Act", "Recommend retention moves", "Highlight useful actions such as pricing changes, bundles, and onboarding support."),
]

HELP_CARDS = [
    ("Fast path from input to decision", "The result screen now uses stored output instead of recalculating the model every rerun."),
    ("Clear screen separation", "Input and output are rendered in separate branches so the result page does not blend into the form page."),
    ("Graphs, not raw tables", "Top influencing features are shown as percentage bars so the output feels visual and easier to read."),
]

CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 25%),
        linear-gradient(135deg, #0f172a, #111827 55%, #1e293b);
    color: #e5eefb;
}
[data-testid="stHeader"] { background: transparent; }
h1 { color: #38bdf8; }
h2, h3 { color: #22c55e; }
.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    color: white;
    border-radius: 14px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 16px 35px rgba(56, 189, 248, 0.22);
}
.hero-shell, .info-card, .graph-card {
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.76);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.28);
}
.hero-shell { padding: 1.7rem; margin-bottom: 1.2rem; }
.hero-tag {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    color: #7dd3fc;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero-title {
    color: #f8fafc;
    font-size: 3.1rem;
    font-weight: 900;
    line-height: 0.95;
    margin: 0.8rem 0;
    max-width: 760px;
}
.hero-copy, .mini-copy, .intro-copy, .signal-copy, .graph-copy {
    color: #cbd5e1;
    line-height: 1.55;
    margin: 0;
}
.hero-copy { font-size: 1rem; line-height: 1.7; max-width: 760px; }
.story-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 1.15rem;
}
@media (max-width: 900px) {
    .story-grid {
        grid-template-columns: 1fr;
    }
}
.info-card { padding: 1rem; }
.mini-label, .section-label { font-weight: 700; text-transform: uppercase; }
.mini-label {
    color: #93c5fd;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
}
.mini-title, .signal-title, .graph-title { color: #f8fafc; font-weight: 800; }
.mini-title { font-size: 1.05rem; margin: 0.35rem 0; }
.mini-copy { font-size: 0.92rem; }
.section-label {
    color: #94a3b8;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    margin-bottom: 0.35rem;
}
.intro-copy { font-size: 0.96rem; margin-bottom: 1rem; line-height: 1.6; }
.signal-card {
    border-radius: 18px;
    padding: 1rem;
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.76));
    border: 1px solid rgba(148, 163, 184, 0.12);
    margin-bottom: 0.8rem;
}
.signal-title { font-size: 1rem; margin-bottom: 0.35rem; }
.signal-copy, .graph-copy { font-size: 0.92rem; }
.graph-card { padding: 1rem 1rem 0.3rem; margin-bottom: 0.85rem; }
.graph-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.45rem;
}
.graph-title { font-size: 0.98rem; }
.graph-value {
    color: #7dd3fc;
    font-size: 0.92rem;
    font-weight: 800;
}
.graph-track {
    width: 100%;
    height: 12px;
    background: rgba(51, 65, 85, 0.85);
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.graph-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #38bdf8 0%, #22c55e 100%);
}
.graph-copy { padding-bottom: 0.75rem; }
</style>
"""


st.set_page_config(page_title="Churn AI Dashboard", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading churn model...")
def load_model():
    return (
        joblib.load(MODEL_DIR / "rf_churn_model.joblib", mmap_mode="r"),
        tuple(joblib.load(MODEL_DIR / "feature_columns.joblib")),
    )


@st.cache_data
def get_feature_importance(feature_names, importances):
    return (
        pd.DataFrame({"Feature": list(feature_names), "Importance": pd.Series(importances).mul(100).round(2)})
        .sort_values("Importance", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )


def build_feature_frame(data, feature_names):
    record = {
        "tenure": data["t"],
        "MonthlyCharges": data["m"],
        "TotalCharges": data["tc"],
        "NumServices": data["n"],
        **{f"Contract_{name}": int(data["c"] == name) for name in CONTRACTS},
        **{f"InternetService_{name}": int(data["i"] == name) for name in INTERNET_SERVICES if name != "No"},
        "TechSupport_Yes": int(data["ts"] == "Yes"),
        "PaperlessBilling_Yes": int(data["pb"] == "Yes"),
    }
    return pd.DataFrame([record]).reindex(columns=feature_names, fill_value=0)


def retention_strategy(t, m, n, c):
    actions = [
        action
        for condition, action in (
            (c == "Month-to-month", "Offer long-term contract discounts."),
            (m > 80, "Provide pricing optimization or bundles."),
            (n < 3, "Upsell additional services."),
            (t < 12, "Send onboarding offers."),
        )
        if condition
    ]
    return actions or ["Customer stable. Offer loyalty rewards."]


def business_summary(pred, prob, t, m, n, c):
    if pred:
        return (
            f"HIGH churn risk ({prob*100:.1f}%)\n"
            f"- Tenure: {t} months (low loyalty)\n"
            f"- Charges: Rs. {m:.2f} (high)\n"
            f"- Services: {n} (low engagement)\n"
            f"- Contract: {c}\n\n"
            "Focus on pricing, bundling, and contract upgrade."
        )
    return (
        f"LOW churn risk ({prob*100:.1f}%)\n"
        f"- Stable tenure ({t} months)\n"
        f"- Good engagement ({n} services)\n\n"
        "Focus on loyalty rewards and upselling."
    )


def clean_html(markup):
    return dedent(markup).strip()


def cards_html(items, card_class, title_class, copy_class, label_class=None, wrap=True):
    blocks = []
    for item in items:
        label = f'<div class="{label_class}">{escape(str(item[0]))}</div>' if label_class else ""
        title = escape(str(item[1] if label_class else item[0]))
        copy = escape(str(item[2] if label_class else item[1]))
        blocks.append(f'<div class="{card_class}">{label}<div class="{title_class}">{title}</div><p class="{copy_class}">{copy}</p></div>')
    html = "".join(blocks)
    return f"<div>{html}</div>" if wrap else html


def render_feature_graphs(importance_rows):
    for row in importance_rows.itertuples(index=False):
        feature = row.Feature
        percent = float(row.Importance)
        explanation = escape(
            FEATURE_EXPLANATIONS.get(feature, "This feature contributes to the churn prediction signal.")
        )
        st.markdown(
            clean_html(
                f"""
                <div class="graph-card">
                    <div class="graph-head">
                        <div class="graph-title">{escape(str(feature))}</div>
                        <div class="graph-value">{percent:.2f}%</div>
                    </div>
                    <div class="graph-track">
                        <div class="graph-fill" style="width: {min(percent, 100):.2f}%"></div>
                    </div>
                    <p class="graph-copy">{explanation}</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


def calculate_result(data, model, features, importance_df):
    frame = build_feature_frame(data, features)
    pred = int(model.predict(frame)[0])
    prob = float(model.predict_proba(frame)[0][1])
    return {
        "pred": pred,
        "prob": prob,
        "summary": business_summary(pred, prob, data["t"], data["m"], data["n"], data["c"]),
        "actions": retention_strategy(data["t"], data["m"], data["n"], data["c"]),
        "importance": importance_df.to_dict("records"),
    }


def render_input_page():
    hero_cards = cards_html(HERO_CARDS, "info-card", "mini-title", "mini-copy", "mini-label", wrap=False)
    st.markdown(
        clean_html(
            f"""
            <div class="hero-shell">
                <div class="hero-tag">AI Churn Intelligence</div>
                <div class="hero-title">See which customers may leave before revenue walks out.</div>
                <p class="hero-copy">
                    This dashboard predicts customer churn from billing, contract, usage, and support signals,
                    then turns the prediction into business-friendly insight. Enter the customer profile, click
                    Analyze Risk once, and the result will open on a separate screen.
                </p>
                <div class="story-grid">{hero_cards}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown("#### Customer Profile")
        st.caption("The app only runs the prediction after you press the button.")
        with st.form("prediction_form", clear_on_submit=False):
            left, right = st.columns(2)
            with left:
                t = st.number_input("Tenure", 0, 100, 12)
                m = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
                n = st.number_input("Services Used", 0, 10, 3)
            with right:
                c = st.selectbox("Contract", CONTRACTS)
                i = st.selectbox("Internet Service", INTERNET_SERVICES)
                ts = st.selectbox("Tech Support", YES_NO)
            bottom_left, bottom_right = st.columns(2)
            with bottom_left:
                pb = st.selectbox("Paperless Billing", YES_NO)
            with bottom_right:
                tc = st.number_input("Total Charges", 0.0, 20000.0, 1500.0)
            submitted = st.form_submit_button("Analyze Risk")

        if submitted:
            data = {"t": t, "m": m, "tc": tc, "n": n, "c": c, "i": i, "ts": ts, "pb": pb}
            model, features = load_model()
            importance_df = get_feature_importance(features, tuple(model.feature_importances_))
            st.session_state.data = data
            st.session_state.result = calculate_result(data, model, features, importance_df)
            st.session_state.page = "result"
            st.rerun()

    with right_col:
        st.markdown("#### How It Helps")
        st.markdown(cards_html(HELP_CARDS, "signal-card", "signal-title", "signal-copy"), unsafe_allow_html=True)


def render_result_page():
    if not st.session_state.data or not st.session_state.result:
        st.session_state.page = "input"
        st.rerun()

    result = st.session_state.result
    data = st.session_state.data
    pred, prob = result["pred"], result["prob"]

    st.title("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob*100:.1f}%")
    col2.error("High Risk" if pred else "Stable")

    st.progress(int(prob * 100))
    st.subheader("Business Insight")
    st.info(result["summary"])

    if pred:
        st.subheader("Recommended Actions")
        for action in result["actions"]:
            st.write(f"- {action}")

    st.subheader("Top Influencing Features")
    render_feature_graphs(pd.DataFrame(result["importance"]))

    snap1, snap2, snap3 = st.columns(3)
    snap1.metric("Tenure", f'{data["t"]} months')
    snap2.metric("Monthly Charges", f'Rs. {data["m"]:.2f}')
    snap3.metric("Services", str(data["n"]))

    if st.button("Back"):
        st.session_state.page = "input"
        st.rerun()


for key, value in STATE_DEFAULTS.items():
    st.session_state.setdefault(key, value)

if st.session_state.page == "input":
    render_input_page()
else:
    render_result_page()
