import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Exploratory Data Analysis")

df = pd.read_csv("data/Telco-Customer-Churn.csv")

fig = px.pie(
    df,
    names="Churn",
    title="Customer Churn Distribution"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    df,
    x="Contract",
    color="Churn",
    barmode="group"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(
    df,
    x="tenure",
    y="MonthlyCharges",
    color="Churn"
)

st.plotly_chart(fig, use_container_width=True)