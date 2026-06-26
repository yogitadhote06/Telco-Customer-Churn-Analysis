#import streamlit as st

#st.title("Telco Customer Churn Analysis")

#st.success("Environment setup completed successfully!")

import streamlit as st

st.set_page_config(
    page_title="Telco Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Customer Churn Analysis & Prediction")

st.markdown("---")

st.write("""
Welcome to the **Telco Customer Churn Analysis & Prediction System**.

This project combines:

- 📊 Exploratory Data Analysis (EDA)
- 📈 Interactive Visualizations
- 🤖 Machine Learning Prediction
- 📑 Model Performance Comparison

Developed using:

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Plotly
""")

st.markdown("---")

st.subheader("Project Workflow")

st.image(
    "https://miro.medium.com/v2/resize:fit:1400/1*8eA6rQJQ0dS4zOQb6l6f9A.png",
    use_container_width=True
)

st.success("Use the sidebar to navigate through the project.")