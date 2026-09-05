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

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(124, 58, 237, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(99, 102, 241, 0.12),
                transparent 25%
            ),
            #070910;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
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


    /* ========================================================
       INPUTS
    ======================================================== */

    label {
        color: #94a3b8 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background: #0f1726;
        border: 1px solid #1e293b;
        border-radius: 10px;
    }

    div[data-baseweb="select"] span {
        color: #e2e8f0;
    }

    input {
        background: #0f1726 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }

    /* ========================================================
       BUTTON
    ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 3.2rem;
        border: 0;
        border-radius: 12px;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #6366f1
            );

        color: white;
        font-weight: 700;
        font-size: 0.92rem;

        box-shadow:
            0 12px 35px rgba(99, 102, 241, 0.25);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 16px 42px rgba(99, 102, 241, 0.38);
    }


    /* ========================================================
       SLIDER
    ======================================================== */

    div[data-testid="stSlider"] {
        padding-top: 0.25rem;
    }


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 768px) {

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

    return joblib.load(
        "models/best_model.pkl"
    )


model = load_model()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_F1 = 0.6208
MODEL_NAME = "Tuned Decision Tree"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_estimator():

    """
    Get the actual trained estimator.

    This also works if the saved object is a GridSearchCV
    object instead of a direct Pipeline.
    """

    if hasattr(model, "best_estimator_"):

        return model.best_estimator_

    return model


def get_feature_importance():

    """
    Extract feature importance from the trained Decision Tree.

    One-hot encoded categorical features are included because
    the model operates on the transformed feature space.
    """

    estimator = get_estimator()

    classifier = estimator.named_steps["classifier"]

    preprocessor = estimator.named_steps["preprocessor"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(5)
    )

    return importance_df


def clean_feature_name(name):

    """
    Make machine-generated feature names easier to read.
    """

    name = name.replace(
        "categorical__",
        ""
    )

    name = name.replace(
        "numerical__",
        ""
    )

    name = name.replace(
        "_",
        " "
    )

    return name.title()


# ============================================================
# HERO SECTION
# ============================================================

st.html(
    """
    <div style="
        padding:18px 0 35px 0;
    ">

        <div style="
            display:inline-flex;
            align-items:center;
            gap:8px;

            padding:7px 13px;

            border-radius:999px;

            background:
                rgba(124,58,237,0.10);

            border:
                1px solid rgba(167,139,250,0.22);

            color:#c4b5fd;

            font-size:10px;
            font-weight:700;
            letter-spacing:1.6px;
        ">

            <span style="
                width:6px;
                height:6px;
                border-radius:50%;
                background:#8b5cf6;
                display:inline-block;
            "></span>

            CUSTOMER INTELLIGENCE

        </div>


        <h1 style="
            margin:19px 0 10px 0;

            font-size:60px;
            line-height:1.02;

            letter-spacing:-3.5px;

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

            font-size:15px;

            line-height:1.7;
        ">

            ChurnSense uses machine learning to estimate
            customer churn risk from account behavior,
            service usage and billing information.

        </p>

    </div>
    """
)


# ============================================================
# INPUT SECTION
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

            margin-bottom:17px;

            border-radius:18px;

            background:
                rgba(15,23,42,0.60);

            border:
                1px solid rgba(148,163,184,0.09);
        ">

            <div style="
                color:#f8fafc;
                font-size:17px;
                font-weight:750;
            ">
                Customer profile
            </div>

            <div style="
                color:#64748b;
                font-size:12px;
                margin-top:5px;
            ">
                Account and subscribed services
            </div>

        </div>
        """
    )


    col1, col2 = st.columns(2)


    # ========================================================
    # LEFT INPUT COLUMN
    # ========================================================

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
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )


    # ========================================================
    # RIGHT INPUT COLUMN
    # ========================================================

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
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

            margin-bottom:17px;

            border-radius:18px;

            background:
                rgba(15,23,42,0.60);

            border:
                1px solid rgba(148,163,184,0.09);
        ">

            <div style="
                color:#f8fafc;
                font-size:17px;
                font-weight:750;
            ">
                Account economics
            </div>

            <div style="
                color:#64748b;
                font-size:12px;
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


    st.markdown(
        "<div style='height:20px'></div>",
        unsafe_allow_html=True
    )


    st.html(
        """
        <div style="
            padding:20px;

            border-radius:18px;

            background:
                linear-gradient(
                    145deg,
                    rgba(124,58,237,0.09),
                    rgba(15,23,42,0.65)
                );

            border:
                1px solid rgba(139,92,246,0.16);
        ">

            <div style="
                color:#f8fafc;
                font-size:16px;
                font-weight:750;
            ">
                Ready to analyze?
            </div>

            <div style="
                color:#64748b;
                font-size:12px;
                line-height:1.5;
                margin-top:6px;
                margin-bottom:17px;
            ">
                Run the trained machine learning model
                against this customer profile.
            </div>

        </div>
        """
    )


    predict_button = st.button(
        "✦  Predict Churn Risk"
    )


# ============================================================
# CREATE CUSTOMER DATA
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
    # RISK LEVEL
    # ========================================================

    if probability >= 0.66:

        risk_level = "HIGH"
        risk_color = "#f59e0b"
        risk_icon = "⚠"
        risk_description = (
            "This customer profile shows a strong "
            "likelihood of churn."
        )

    elif probability >= 0.33:

        risk_level = "MEDIUM"
        risk_color = "#a78bfa"
        risk_icon = "◐"
        risk_description = (
            "This customer profile shows moderate "
            "churn risk."
        )

    else:

        risk_level = "LOW"
        risk_color = "#34d399"
        risk_icon = "✓"
        risk_description = (
            "This customer profile currently shows "
            "a lower likelihood of churn."
        )


    # ========================================================
    # RESULT HERO
    # ========================================================

    st.markdown(
        "<div style='height:30px'></div>",
        unsafe_allow_html=True
    )


    # Convert probability to gauge angle

    gauge_angle = probability * 360


    st.html(
        f"""
        <div style="
            position:relative;

            overflow:hidden;

            padding:38px 30px;

            text-align:center;

            border-radius:24px;

            background:
                radial-gradient(
                    circle at 50% 100%,
                    rgba(124,58,237,0.12),
                    transparent 55%
                ),
                rgba(15,23,42,0.72);

            border:
                1px solid rgba(148,163,184,0.10);

            box-shadow:
                0 25px 70px rgba(0,0,0,0.18);
        ">

            <div style="
                color:#64748b;
                font-size:10px;
                font-weight:700;
                letter-spacing:2.2px;
                text-transform:uppercase;
            ">
                Prediction result
            </div>


            <div style="
                width:190px;
                height:190px;

                margin:25px auto 20px auto;

                border-radius:50%;

                display:flex;
                align-items:center;
                justify-content:center;

                background:
                    conic-gradient(
                        {risk_color}
                        0deg {gauge_angle}deg,
                        #1e293b {gauge_angle}deg 360deg
                    );

                position:relative;
            ">

                <div style="
                    width:155px;
                    height:155px;

                    border-radius:50%;

                    background:#0b1020;

                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                ">

                    <div style="
                        color:#f8fafc;
                        font-size:39px;
                        font-weight:850;
                        letter-spacing:-2px;
                    ">
                        {probability_percent:.1f}%
                    </div>

                    <div style="
                        color:#64748b;
                        font-size:9px;
                        font-weight:700;
                        letter-spacing:1.5px;
                        text-transform:uppercase;
                    ">
                        estimated risk
                    </div>

                </div>

            </div>


            <div style="
                color:{risk_color};
                font-size:28px;
                font-weight:800;
                letter-spacing:-1px;
            ">
                {risk_icon} {risk_level} CHURN RISK
            </div>


            <div style="
                max-width:620px;
                margin:9px auto 0 auto;

                color:#94a3b8;
                font-size:13px;
                line-height:1.6;
            ">
                {risk_description}
            </div>

        </div>
        """
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    st.markdown(
        "<div style='height:18px'></div>",
        unsafe_allow_html=True
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        st.html(
            f"""
            <div style="
                padding:22px;
                text-align:center;

                border-radius:17px;

                background:
                    rgba(15,23,42,0.62);

                border:
                    1px solid rgba(148,163,184,0.08);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:24px;
                    font-weight:800;
                ">
                    {probability_percent:.1f}%
                </div>

                <div style="
                    margin-top:5px;
                    color:#64748b;
                    font-size:9px;
                    font-weight:700;
                    letter-spacing:1.2px;
                ">
                    CHURN PROBABILITY
                </div>

            </div>
            """
        )


    with m2:

        st.html(
            f"""
            <div style="
                padding:22px;
                text-align:center;

                border-radius:17px;

                background:
                    rgba(15,23,42,0.62);

                border:
                    1px solid rgba(148,163,184,0.08);
            ">

                <div style="
                    color:{risk_color};
                    font-size:24px;
                    font-weight:800;
                ">
                    {risk_level}
                </div>

                <div style="
                    margin-top:5px;
                    color:#64748b;
                    font-size:9px;
                    font-weight:700;
                    letter-spacing:1.2px;
                ">
                    RISK CLASSIFICATION
                </div>

            </div>
            """
        )


    with m3:

        st.html(
            f"""
            <div style="
                padding:22px;
                text-align:center;

                border-radius:17px;

                background:
                    rgba(15,23,42,0.62);

                border:
                    1px solid rgba(148,163,184,0.08);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:24px;
                    font-weight:800;
                ">
                    {MODEL_F1 * 100:.2f}%
                </div>

                <div style="
                    margin-top:5px;
                    color:#64748b;
                    font-size:9px;
                    font-weight:700;
                    letter-spacing:1.2px;
                ">
                    MODEL F1-SCORE
                </div>

            </div>
            """
        )


    # ========================================================
    # LOWER INFORMATION AREA
    # ========================================================

    st.markdown(
        "<div style='height:30px'></div>",
        unsafe_allow_html=True
    )


    info_left, info_right = st.columns(
        [1.2, 0.8],
        gap="large"
    )


    # ========================================================
    # CUSTOMER SNAPSHOT
    # ========================================================

    with info_left:

        st.html(
            """
            <div style="
                padding:22px;

                border-radius:18px;

                background:
                    rgba(15,23,42,0.60);

                border:
                    1px solid rgba(148,163,184,0.09);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:16px;
                    font-weight:750;
                ">
                    Customer snapshot
                </div>

                <div style="
                    color:#64748b;
                    font-size:11px;
                    margin-top:4px;
                    margin-bottom:18px;
                ">
                    Profile used for this prediction
                </div>

            </div>
            """
        )


        snapshot = [
            ("Contract", contract),
            ("Tenure", f"{tenure} months"),
            ("Internet", internet_service),
            ("Monthly charges", f"${monthly_charges:,.0f}"),
            ("Total charges", f"${total_charges:,.0f}"),
            ("Payment", payment_method),
        ]


        for index in range(0, len(snapshot), 2):

            c1, c2 = st.columns(2)


            key1, value1 = snapshot[index]


            with c1:

                st.html(
                    f"""
                    <div style="
                        padding:13px 15px;
                        margin-bottom:9px;

                        border-radius:12px;

                        background:#0b1220;

                        border:
                            1px solid #172033;
                    ">

                        <div style="
                            color:#64748b;
                            font-size:9px;
                            text-transform:uppercase;
                            letter-spacing:1px;
                        ">
                            {key1}
                        </div>

                        <div style="
                            color:#e2e8f0;
                            font-size:13px;
                            font-weight:650;
                            margin-top:4px;
                        ">
                            {value1}
                        </div>

                    </div>
                    """
                )


            if index + 1 < len(snapshot):

                key2, value2 = snapshot[index + 1]

                with c2:

                    st.html(
                        f"""
                        <div style="
                            padding:13px 15px;
                            margin-bottom:9px;

                            border-radius:12px;

                            background:#0b1220;

                            border:
                                1px solid #172033;
                        ">

                            <div style="
                                color:#64748b;
                                font-size:9px;
                                text-transform:uppercase;
                                letter-spacing:1px;
                            ">
                                {key2}
                            </div>

                            <div style="
                                color:#e2e8f0;
                                font-size:13px;
                                font-weight:650;
                                margin-top:4px;
                            ">
                                {value2}
                            </div>

                        </div>
                        """
                    )


    # ========================================================
    # MODEL SIGNALS
    # ========================================================

    with info_right:

        st.html(
            """
            <div style="
                padding:22px;

                border-radius:18px;

                background:
                    rgba(15,23,42,0.60);

                border:
                    1px solid rgba(148,163,184,0.09);
            ">

                <div style="
                    color:#f8fafc;
                    font-size:16px;
                    font-weight:750;
                ">
                    Model signals
                </div>

                <div style="
                    color:#64748b;
                    font-size:11px;
                    margin-top:4px;
                    margin-bottom:18px;
                ">
                    Most influential learned features
                </div>

            </div>
            """
        )


        try:

            importance_df = get_feature_importance()


            for _, row in importance_df.iterrows():

                feature = clean_feature_name(
                    row["feature"]
                )

                importance = row["importance"]


                st.html(
                    f"""
                    <div style="
                        margin-bottom:13px;
                    ">

                        <div style="
                            display:flex;
                            justify-content:space-between;

                            margin-bottom:5px;

                            color:#cbd5e1;

                            font-size:11px;
                            font-weight:600;
                        ">

                            <span>
                                {feature}
                            </span>

                            <span style="
                                color:#64748b;
                            ">
                                {importance:.3f}
                            </span>

                        </div>


                        <div style="
                            height:5px;

                            border-radius:999px;

                            background:#172033;
                        ">

                            <div style="
                                width:{min(importance * 1000, 100):.1f}%;

                                height:100%;

                                border-radius:999px;

                                background:
                                    linear-gradient(
                                        90deg,
                                        #7c3aed,
                                        #8b5cf6
                                    );
                            ">
                            </div>

                        </div>

                    </div>
                    """
                )


        except Exception:

            st.caption(
                "Feature importance unavailable."
            )


    # ========================================================
    # MODEL FOOTER
    # ========================================================

    st.markdown(
        "<div style='height:20px'></div>",
        unsafe_allow_html=True
    )


    st.html(
        f"""
        <div style="
            padding:18px 22px;

            border-radius:15px;

            background:
                rgba(124,58,237,0.055);

            border:
                1px solid rgba(124,58,237,0.10);

            display:flex;
            justify-content:space-between;
            align-items:center;
        ">

            <div>

                <div style="
                    color:#cbd5e1;
                    font-size:12px;
                    font-weight:700;
                ">
                    {MODEL_NAME}
                </div>

                <div style="
                    color:#64748b;
                    font-size:10px;
                    margin-top:3px;
                ">
                    Hyperparameter tuned · 5-fold cross-validation
                </div>

            </div>


            <div style="
                color:#a78bfa;
                font-size:12px;
                font-weight:700;
            ">
                F1 · {MODEL_F1 * 100:.2f}%
            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;

        margin-top:55px;

        color:#334155;

        font-size:10px;

        line-height:1.8;
    ">

        <strong style="color:#475569;">
            ChurnSense
        </strong>

        · Customer Churn Prediction

        <br>

        Python · Scikit-Learn · Streamlit

    </div>
    """
)