import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnSense",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ==============================
       GLOBAL
    ============================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(124, 58, 237, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(79, 70, 229, 0.13),
                transparent 28%
            ),
            #070910;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }


    /* ==============================
       STREAMLIT ELEMENTS
    ============================== */

    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }

    div[data-baseweb="select"] > div {
        background: #101827;
        border: 1px solid #1e293b;
        border-radius: 11px;
    }

    input {
        background: #101827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 11px !important;
        color: #f8fafc !important;
    }

    div[data-testid="stSlider"] {
        padding-top: 0.3rem;
    }


    /* ==============================
       BUTTON
    ============================== */

    .stButton > button {
        width: 100%;
        height: 3.25rem;
        border-radius: 13px;
        border: none;
        background: linear-gradient(
            135deg,
            #7c3aed,
            #6366f1
        );
        color: white;
        font-size: 0.95rem;
        font-weight: 700;
        box-shadow:
            0 12px 30px rgba(99, 102, 241, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 16px 40px rgba(99, 102, 241, 0.40);
    }


    /* ==============================
       PROGRESS
    ============================== */

    div[data-testid="stProgress"] > div > div {
        border-radius: 999px;
    }


    /* ==============================
       MOBILE
    ============================== */

    @media (max-width: 768px) {

        .hero-title {
            font-size: 2.5rem !important;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


model = load_model()


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div style="
        padding: 20px 0 35px 0;
    ">

        <div style="
            display:inline-block;
            padding:7px 13px;
            border-radius:999px;
            background:rgba(124,58,237,0.12);
            border:1px solid rgba(167,139,250,0.25);
            color:#c4b5fd;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.5px;
            text-transform:uppercase;
        ">
            ◈ MACHINE LEARNING · CUSTOMER INTELLIGENCE
        </div>

        <h1 style="
            margin:18px 0 10px 0;
            font-size:58px;
            line-height:1;
            letter-spacing:-3px;
            font-weight:800;
            color:#f8fafc;
        ">
            Know who might
            <span style="
                color:#8b5cf6;
            ">
                leave.
            </span>
        </h1>

        <p style="
            max-width:650px;
            margin:0;
            color:#94a3b8;
            font-size:16px;
            line-height:1.7;
        ">
            ChurnSense analyzes customer behavior and account
            information to estimate the likelihood of customer
            churn using machine learning.
        </p>

    </div>
    """
)


# ============================================================
# SECTION HEADERS
# ============================================================

left, right = st.columns(
    [1.55, 0.85],
    gap="large"
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

with left:

    st.html(
        """
        <div style="
            padding:20px 22px;
            margin-bottom:18px;
            border-radius:18px;
            background:rgba(15,23,42,0.62);
            border:1px solid rgba(148,163,184,0.10);
        ">

            <div style="
                color:#f8fafc;
                font-size:18px;
                font-weight:750;
            ">
                Customer profile
            </div>

            <div style="
                color:#64748b;
                font-size:13px;
                margin-top:5px;
            ">
                Demographics, account and subscribed services
            </div>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


# ============================================================
# ACCOUNT ECONOMICS
# ============================================================

with right:

    st.html(
        """
        <div style="
            padding:20px 22px;
            margin-bottom:18px;
            border-radius:18px;
            background:rgba(15,23,42,0.62);
            border:1px solid rgba(148,163,184,0.10);
        ">

            <div style="
                color:#f8fafc;
                font-size:18px;
                font-weight:750;
            ">
                Account economics
            </div>

            <div style="
                color:#64748b;
                font-size:13px;
                margin-top:5px;
            ">
                Customer spending indicators
            </div>

        </div>
        """
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.html(
        """
        <div style="
            padding:20px 22px;
            margin-bottom:15px;
            border-radius:18px;
            background:rgba(15,23,42,0.62);
            border:1px solid rgba(148,163,184,0.10);
        ">

            <div style="
                color:#f8fafc;
                font-size:18px;
                font-weight:750;
            ">
                Ready to analyze?
            </div>

            <div style="
                color:#64748b;
                font-size:13px;
                margin-top:5px;
                line-height:1.5;
            ">
                Run the trained Decision Tree model against
                this customer's profile.
            </div>

        </div>
        """
    )

    predict_button = st.button(
        "✦  Predict Churn Risk"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    customer_data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        }
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        customer_data
    )[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]

    probability_percent = probability * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:

        result_color = "#f59e0b"
        result_title = "High Churn Risk"
        result_icon = "⚠"
        result_text = (
            "This customer profile contains patterns associated "
            "with a higher likelihood of leaving."
        )

    else:

        result_color = "#34d399"
        result_title = "Low Churn Risk"
        result_icon = "✓"
        result_text = (
            "This customer profile currently shows a lower "
            "likelihood of leaving."
        )


    st.html(
        f"""
        <div style="
            padding:32px;
            text-align:center;
            border-radius:22px;
            background:
                linear-gradient(
                    145deg,
                    rgba(15,23,42,0.92),
                    rgba(15,23,42,0.62)
                );
            border:1px solid rgba(148,163,184,0.12);
            box-shadow:0 20px 60px rgba(0,0,0,0.20);
        ">

            <div style="
                color:#64748b;
                font-size:11px;
                font-weight:700;
                letter-spacing:2px;
                text-transform:uppercase;
            ">
                Prediction result
            </div>

            <div style="
                margin-top:10px;
                color:{result_color};
                font-size:36px;
                font-weight:800;
            ">
                {result_icon} {result_title}
            </div>

            <div style="
                max-width:600px;
                margin:10px auto 0 auto;
                color:#94a3b8;
                font-size:14px;
                line-height:1.6;
            ">
                {result_text}
            </div>

        </div>
        """
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # METRICS
    # ========================================================

    m1, m2, m3 = st.columns(3)

    with m1:

        st.html(
            f"""
            <div style="
                padding:20px;
                text-align:center;
                border-radius:17px;
                background:rgba(15,23,42,0.65);
                border:1px solid rgba(148,163,184,0.09);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:25px;
                    font-weight:800;
                ">
                    {probability_percent:.1f}%
                </div>

                <div style="
                    margin-top:5px;
                    color:#64748b;
                    font-size:11px;
                ">
                    CHURN PROBABILITY
                </div>

            </div>
            """
        )

    with m2:

        risk = "HIGH" if probability >= 0.5 else "LOW"

        st.html(
            f"""
            <div style="
                padding:20px;
                text-align:center;
                border-radius:17px;
                background:rgba(15,23,42,0.65);
                border:1px solid rgba(148,163,184,0.09);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:25px;
                    font-weight:800;
                ">
                    {risk}
                </div>

                <div style="
                    margin-top:5px;
                    color:#64748b;
                    font-size:11px;
                ">
                    RISK CLASSIFICATION
                </div>

            </div>
            """
        )

    with m3:

        st.html(
            """
            <div style="
                padding:20px;
                text-align:center;
                border-radius:17px;
                background:rgba(15,23,42,0.65);
                border:1px solid rgba(148,163,184,0.09);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:25px;
                    font-weight:800;
                ">
                    62.08%
                </div>

                <div style="
                    margin-top:5px;
                    color:#64748b;
                    font-size:11px;
                ">
                    MODEL F1-SCORE
                </div>

            </div>
            """
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.progress(
        probability,
        text=f"Estimated churn probability · {probability_percent:.1f}%"
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;
        margin-top:55px;
        color:#475569;
        font-size:11px;
        line-height:1.8;
    ">

        <strong style="color:#64748b;">
            ChurnSense
        </strong>

        · Customer Churn Prediction

        <br>

        Decision Tree · Scikit-Learn · Streamlit

    </div>
    """
)