import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Churn Prediction")

st.markdown(
    "Fill customer details below to predict whether the customer is likely to churn."
)

# -----------------------------------
# Load Model
# -----------------------------------

model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# -----------------------------------
# Input Form
# -----------------------------------

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
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
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1500.0
)

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame({

        "gender":[gender],

        "SeniorCitizen":[senior],

        "Partner":[partner],

        "Dependents":[dependents],

        "tenure":[tenure],

        "PhoneService":[phone],

        "MultipleLines":[multiple],

        "InternetService":[internet],

        "OnlineSecurity":[security],

        "OnlineBackup":[backup],

        "DeviceProtection":[protection],

        "TechSupport":[support],

        "StreamingTV":[tv],

        "StreamingMovies":[movies],

        "Contract":[contract],

        "PaperlessBilling":[paperless],

        "PaymentMethod":[payment],

        "MonthlyCharges":[monthly],

        "TotalCharges":[total]

    })

    # Encode categorical variables

    categorical_cols = input_data.select_dtypes(include="object").columns

    for col in categorical_cols:

        encoder = encoders[col]

        input_data[col] = encoder.transform(input_data[col])

    # Arrange columns

    input_data = input_data[feature_columns]

    # Prediction

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ Customer is likely to CHURN")

    else:

        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.markdown("---")

    st.subheader("Recommendation")

    if probability > 0.70:

        st.warning(
            """
High Risk Customer

Recommended Actions:

- Offer Discount
- Provide Loyalty Benefits
- Contact Customer Support
- Personalized Retention Offer
"""
        )

    elif probability > 0.40:

        st.info(
            """
Medium Risk Customer

Recommended Actions:

- Email Campaign
- Upsell Better Plan
- Customer Satisfaction Survey
"""
        )

    else:

        st.success(
            """
Low Risk Customer

Customer is Loyal.

Maintain Current Service.
"""
        )

st.markdown("---")

st.caption("Developed using Streamlit | Scikit-Learn | Python")