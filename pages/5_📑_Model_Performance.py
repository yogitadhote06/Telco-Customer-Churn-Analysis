import streamlit as st
import pandas as pd

st.title("📑 Model Performance")

results = pd.read_csv("models/model_comparison.csv")

st.dataframe(results, use_container_width=True)

st.bar_chart(results.set_index("Model")["Accuracy"])