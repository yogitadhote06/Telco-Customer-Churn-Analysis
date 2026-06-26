import streamlit as st
import pandas as pd

st.title("📊 Dashboard")

df = pd.read_csv("data/Telco-Customer-Churn.csv")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", len(df))

col2.metric("Features", df.shape[1])

col3.metric(
    "Churn Rate",
    f"{(df['Churn'].value_counts(normalize=True)['Yes']*100):.2f}%"
)

col4.metric(
    "Average Monthly Charges",
    f"${df['MonthlyCharges'].mean():.2f}"
)

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)