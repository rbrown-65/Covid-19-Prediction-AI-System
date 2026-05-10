import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="COVID-19 Prediction AI System",
    layout="wide"
)

st.title("COVID-19 Prediction AI System")
st.subheader("COVIDCARE Data, Machine Learning, Network Analysis, and Gemini Explanation")

st.write(
    """
    This app summarizes a COVID-19 prediction workflow using COVIDCARE survey data,
    DEMI knowledgebase processing, machine learning models, and an optional Gemini-based
    explanation component.
    """
)

st.markdown("## Project Objective")
st.write(
    """
    The objective was to develop and evaluate a workflow for predicting PCR-confirmed
    COVID-19 status from survey-based variables.
    """
)

st.markdown("## Dataset Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Raw participants", "822")
col2.metric("Modeling participants", "559")
col3.metric("Predictor variables", "444")

st.write(
    """
    After filtering for available PCR outcome values, the analytic dataset included
    501 PCR-negative cases and 58 PCR-positive cases.
    """
)

st.markdown("## Model Results")

model_results = pd.DataFrame({
    "Model": ["XGBoost", "Logistic Regression", "LASSO Logistic Regression"],
    "AUC": [0.948, 0.877, 0.868],
    "Accuracy": [0.936, 0.957, 0.950],
    "McFadden R²": [0.565, -0.178, -0.241],
    "% Variation Explained": ["56.5%", "-17.8%", "-24.1%"]
})

st.dataframe(model_results, use_container_width=True)

st.success("Best model based on AUC: XGBoost")

st.markdown("## Top Direct Predictors")

direct_predictors = pd.DataFrame({
    "Predictor": [
        "30158-Symtpom_Neuro-7",
        "30766-pinkblue_confirm",
        "31386-covid_results-1",
        "31386-covid_results-2",
        "30183-states_specify-42",
        "30183-states_specify-33",
        "32137-vaccine_avail",
        "32138-vaccine_factors-13",
        "30158-Symtpom_Neuro-8",
        "30141-covid_tst_symptoms-3"
    ],
    "Coefficient": [1.060, 0.648, 0.544, -0.382, 0.364, 0.357, 0.351, 0.326, 0.316, 0.311]
})

st.dataframe(direct_predictors, use_container_width=True)

st.write(
    """
    Positive coefficients indicate variables associated with higher predicted probability
    of PCR positivity. Negative coefficients indicate lower predicted probability.
    These are predictive associations, not causal effects.
    """
)

st.markdown("## Network Analysis")

st.write(
    """
    The Markov blanket parent-modeling step did not identify usable parent models in this run.
    Therefore, the final network represents direct predictors pointing to PCR-confirmed
    COVID-19 status rather than a full causal network.
    """
)

if os.path.exists("covid_network_clean.png"):
    st.image("covid_network_clean.png", caption="Direct Predictor Network for PCR-Confirmed COVID-19")
else:
    st.info("Network image is created when the notebook is run.")

st.markdown("## Gemini / Fallback Explanation")

if os.path.exists("llm_model_explanation.txt"):
    with open("llm_model_explanation.txt", "r", encoding="utf-8") as file:
        explanation = file.read()
    st.text_area("Explanation Output", explanation, height=300)
else:
    st.write(
        """
        The notebook includes an optional Gemini explanation component. If no API key is available,
        it generates a fallback automated explanation instead.
        """
    )

st.markdown("## Repository")

st.write("GitHub repository:")
st.code("https://github.com/rbrown-65/Covid-19-Prediction-AI-System")
