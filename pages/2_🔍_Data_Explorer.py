import streamlit as st
import pandas as pd

st.title("🔍 Data Explorer")

df = pd.read_csv("data/Telco-Customer-Churn.csv")

st.subheader("Dataset")

st.dataframe(df)

st.subheader("Dataset Shape")

st.write(df.shape)

st.subheader("Columns")

st.write(df.columns.tolist())

st.subheader("Data Types")

st.write(df.dtypes)

st.subheader("Missing Values")

st.write(df.isnull().sum())

st.subheader("Statistical Summary")

st.write(df.describe())