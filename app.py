import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="COVID-19 At-Home Screening AI System",
    layout="wide"
)

st.title("COVID-19 At-Home Screening AI System")
st.subheader("COVIDCARE Data, Machine Learning, Network Analysis, and Gemini Explanation")

st.write(
    """
    This app summarizes and demonstrates a COVID-19 at-home screening workflow using
    COVIDCARE survey data, DEMI knowledgebase processing, machine learning models,
    and an optional Gemini-based explanation component.
    """
)

st.info(
    """
    This project is intended for screening support and education. It does not replace
    PCR testing, clinically accepted diagnostic testing, or medical judgment.
    """
)

st.markdown("## Project Objective")

st.write(
    """
    The objective was to develop and evaluate a workflow for estimating the probability
    of PCR-confirmed COVID-19 using information that could reasonably be available at home
    before a clinic or emergency room visit.
    """
)

st.markdown("## Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Raw participants", "822")
col2.metric("Modeling participants", "559")
col3.metric("Home-available predictors", "394")

st.write(
    """
    After filtering for available PCR outcome values, the analytic dataset included
    501 PCR-negative cases and 58 PCR-positive cases. The updated workflow excludes
    likely clinic, laboratory, PCR-result, and post-diagnosis variables so the prediction
    better reflects information available before a clinical visit.
    """
)

st.markdown("## Model Results")

model_results = pd.DataFrame({
    "Model": [
        "XGBoost",
        "Logistic Regression",
        "LASSO Logistic Regression"
    ],
    "AUC": [
        0.933,
        0.857,
        0.856
    ],
    "Accuracy": [
        0.936,
        0.929,
        0.929
    ],
    "McFadden R²": [
        0.473,
        -0.698,
        -0.858
    ],
    "% Variation Explained": [
        "47.3%",
        "-69.8%",
        "-85.8%"
    ]
})

st.dataframe(model_results, use_container_width=True)

st.success(
    """
    Best model based on AUC: XGBoost. After restricting predictors to home-available
    information, XGBoost still showed the strongest discrimination with an AUC of 0.933.
    """
)

st.markdown("## Top Direct Predictors")

direct_predictors = pd.DataFrame({
    "Predictor": [
        "30158-Symtpom_Neuro-7",
        "30766-pinkblue_confirm",
        "32359-blue_nopink_confirm_2",
        "30183-states_specify-33",
        "30153-Symptoms-2",
        "32137-vaccine_avail",
        "32138-vaccine_factors-17",
        "30183-states_specify-42",
        "30158-Symtpom_Neuro-8",
        "31396-join_hear-6"
    ],
    "Coefficient": [
        1.212,
        0.925,
        -0.494,
        0.428,
        0.422,
        0.421,
        0.421,
        0.415,
        0.405,
        0.396
    ]
})

st.dataframe(direct_predictors, use_container_width=True)

st.write(
    """
    Positive coefficients indicate variables associated with higher predicted probability
    of PCR positivity. Negative coefficients indicate lower predicted probability.
    These are predictive associations, not causal effects.
    """
)

st.markdown("## At-Home COVID-19 Screening Example")

st.write(
    """
    The final notebook section converts the trained XGBoost model into a simple
    at-home screening-support workflow. The example below uses information that could
    reasonably be known before a clinic or emergency room visit, such as symptoms
    and at-home test confirmation fields.
    """
)

st.markdown("### Example At-Home Inputs")

home_inputs = pd.DataFrame({
    "Input Variable": [
        "30158-Symtpom_Neuro-7",
        "30158-Symtpom_Neuro-8",
        "30141-covid_tst_symptoms-3",
        "30766-pinkblue_confirm"
    ],
    "Value": [
        1,
        1,
        1,
        1
    ],
    "Interpretation": [
        "Neurological symptom field present",
        "Neurological symptom field present",
        "COVID-like symptom field present",
        "At-home test confirmation field present"
    ]
})

st.dataframe(home_inputs, use_container_width=True)

st.markdown("### Screening Result")

screen_col1, screen_col2, screen_col3 = st.columns(3)

screen_col1.metric("Predicted PCR-positive probability", "0.9225")
screen_col2.metric("Risk category", "High")
screen_col3.metric("Best model", "XGBoost")

st.warning(
    """
    Recommendation: High screening risk. Seek confirmatory testing or clinical guidance,
    especially if symptoms worsen.
    """
)

st.info(
    """
    This is a screening-support estimate, not a definitive diagnosis. PCR or another
    clinically accepted test remains the diagnostic standard.
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
    st.image(
        "covid_network_clean.png",
        caption="Direct Predictor Network for PCR-Confirmed COVID-19"
    )
else:
    st.info("Network image is created when the notebook is run.")

st.markdown("## Gemini / Fallback Explanation")

st.write(
    """
    The notebook includes an optional Gemini explanation component. When a valid Gemini API key
    is available, Gemini explains the at-home screening result. If no valid API key is available,
    the notebook creates a fallback automated explanation instead.
    """
)

st.write(
    """
    The explanation component summarizes the predicted probability, risk category,
    home-available inputs, and why the output should be interpreted as screening support
    rather than a definitive diagnosis.
    """
)

if os.path.exists("llm_model_explanation.txt"):
    with open("llm_model_explanation.txt", "r", encoding="utf-8") as file:
        explanation = file.read()

    st.text_area("Explanation Output", explanation, height=300)
else:
    fallback_preview = """
At-Home COVID-19 Screening Explanation

The screening system used the XGBoost model because it had the strongest AUC during model evaluation.
For the example at-home patient scenario, the predicted probability of PCR-positive COVID-19 was 0.923.
This was classified as High risk.

This result should be interpreted as screening support, not a definitive diagnosis. A higher predicted
probability suggests that confirmatory testing or clinical guidance may be appropriate, especially if
symptoms worsen or exposure risk is high. PCR or another clinically accepted test remains the diagnostic standard.
"""

    st.text_area("Fallback Explanation Preview", fallback_preview, height=260)

st.markdown("## Repository")

st.write("Full reproducible notebook and project files are available in the GitHub repository:")

st.code("https://github.com/rbrown-65/Covid-19-Prediction-AI-System")

st.markdown("## Deployment Note")

st.write(
    """
    This Streamlit app is a deployed project demonstration. The full model-building workflow
    is contained in the Jupyter notebook and can be executed from the GitHub repository using
    GitHub Codespaces.
    """
)
