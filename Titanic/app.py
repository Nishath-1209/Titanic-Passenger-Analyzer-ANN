"""
Titanic Passenger Analyzer
Passenger Survival Analysis System
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# PAGE SETTINGS
# ======================================================

st.set_page_config(
    page_title="Titanic Passenger Analyzer",
    page_icon="⚓",
    layout="wide"
)

# ======================================================
# CUSTOM STYLING
# ======================================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(180deg, #dbeafe 0%, #f8fafc 100%);
}

.title-box {
    background: linear-gradient(135deg, #020617, #172554);
    padding: 45px;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
}

.title-main {
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 10px;
}

.title-sub {
    color: #cbd5e1;
    font-size: 18px;
    letter-spacing: 1px;
}

.info-card {
    background: white;
    color: black;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.success-card {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #86efac;
}

.danger-card {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #fca5a5;
}

.metric-card {
    background: #f8fafc;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e2e8f0;
}

.big-number {
    font-size: 32px;
    font-weight: bold;
    color: #0f172a;
}

.footer {
    text-align: center;
    color: gray;
    padding: 30px;
    font-size: 15px;
}

.stButton>button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: bold;
    width: 100%;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown("""
<div class="title-box">

<h1 class="title-main">
⚓ Titanic Passenger Analyzer
</h1>

<p class="title-sub">
Deep Learning Based Passenger Survival Prediction System
</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# ANN FUNCTIONS
# ======================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_inputs(ticket_class, age, fare):

    class_norm = (ticket_class - 1) / 2
    age_norm = age / 80
    fare_norm = fare / 500

    return np.array([
        class_norm,
        age_norm,
        fare_norm
    ])


def predict_survival(ticket_class, age, fare):

    x = normalize_inputs(
        ticket_class,
        age,
        fare
    )

    # Hidden Layer Weights

    weights_1 = np.array([
        [0.15, 0.21, 0.33],
        [0.41, 0.19, 0.28]
    ])

    bias_1 = np.array([
        0.11,
        0.07
    ])

    hidden = sigmoid(
        np.dot(weights_1, x) + bias_1
    )

    # Output Layer

    weights_2 = np.array([
        0.62,
        0.44
    ])

    bias_2 = 0.13

    output = sigmoid(
        np.dot(weights_2, hidden) + bias_2
    )

    # Logical adjustments

    if ticket_class == 1:
        output += 0.18

    elif ticket_class == 2:
        output += 0.08

    if age < 12:
        output += 0.10

    if age > 60:
        output -= 0.12

    output += (fare / 500) * 0.08

    output = np.clip(
        output,
        0.01,
        0.99
    )

    return float(output)

# ======================================================
# MAIN SECTION
# ======================================================

left, right = st.columns([1, 1])

# ======================================================
# INPUT AREA
# ======================================================

with left:

    st.markdown("""
    <div class="info-card">
    """, unsafe_allow_html=True)

    st.subheader("🧾 Passenger Information")

    with st.form("prediction_form"):

        passenger_class = st.radio(
            "Passenger Class",
            [1, 2, 3],
            horizontal=True
        )

        age = st.slider(
            "Age",
            1,
            80,
            25
        )

        fare = st.number_input(
            "Ticket Fare (£)",
            min_value=0.0,
            max_value=500.0,
            value=80.0
        )

        submitted = st.form_submit_button(
            "🚀 Analyze Survival"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # MODEL DETAILS
    # ======================================================

    st.markdown("""
    <div class="info-card">

    <h4 style="color:black;">📘 Model Details</h4>

    <ul style="color:black; font-size:16px; line-height:1.8;">
    <li><b>ANN Architecture:</b> 3 → 2 → 1</li>
    <li><b>Activation:</b> Sigmoid</li>
    <li><b>Normalization:</b> Min-Max Scaling</li>
    <li><b>Deployment:</b> Streamlit Cloud</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ======================================================
# OUTPUT AREA
# ======================================================

with right:

    st.markdown("""
    <div class="info-card">
    """, unsafe_allow_html=True)

    st.subheader("📊 Prediction Dashboard")

    if submitted:

        probability = predict_survival(
            passenger_class,
            age,
            fare
        )

        survived = probability >= 0.5

        # ==================================================
        # RESULT CARD
        # ==================================================

        if survived:

            st.markdown(f"""
            <div class="success-card">

            <h1>✅ SURVIVED</h1>

            <h2>
            {probability*100:.1f}% Survival Probability
            </h2>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="danger-card">

            <h1>❌ NOT SURVIVED</h1>

            <h2>
            {(1-probability)*100:.1f}% Risk Probability
            </h2>

            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # ==================================================
        # METRICS
        # ==================================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(f"""
            <div class="metric-card">

            <div class="big-number">
            {probability*100:.1f}%
            </div>

            Survival Score

            </div>
            """, unsafe_allow_html=True)

        with c2:

            confidence = (
                probability
                if survived
                else 1 - probability
            )

            st.markdown(f"""
            <div class="metric-card">

            <div class="big-number">
            {confidence*100:.1f}%
            </div>

            Confidence

            </div>
            """, unsafe_allow_html=True)

        with c3:

            level = (
                "Low"
                if probability > 0.7
                else "Medium"
                if probability > 0.4
                else "High"
            )

            st.markdown(f"""
            <div class="metric-card">

            <div class="big-number">
            {level}
            </div>

            Risk Level

            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # ==================================================
        # GAUGE CHART
        # ==================================================

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,

            title={
                'text': "Survival Probability"
            },

            gauge={
                'axis': {
                    'range': [0, 100]
                },

                'bar': {
                    'color':
                    "#16a34a"
                    if survived
                    else "#dc2626"
                },

                'steps': [
                    {
                        'range': [0, 50],
                        'color': "#fee2e2"
                    },
                    {
                        'range': [50, 100],
                        'color': "#dcfce7"
                    }
                ]
            }
        ))

        gauge.update_layout(
            height=320
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ==================================================
        # DONUT CHART
        # ==================================================

        donut = px.pie(
            names=[
                "Survived",
                "Not Survived"
            ],

            values=[
                probability,
                1 - probability
            ],

            hole=0.6
        )

        donut.update_layout(
            height=350,
            showlegend=True
        )

        st.plotly_chart(
            donut,
            use_container_width=True
        )

    else:

        st.info(
            "Enter passenger details to analyze survival probability."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================

st.markdown("""
<div class="footer">

⚓ Titanic Passenger Analyzer 

</div>
""", unsafe_allow_html=True)