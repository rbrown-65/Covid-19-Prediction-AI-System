import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="COVID-19 At-Home Screening AI System",
    layout="wide"
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_resource
def load_model_artifacts():
    model = joblib.load("covid_home_screening_model.pkl")

    with open("model_feature_columns.json", "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    with open("home_input_mapping.json", "r", encoding="utf-8") as file:
        input_mapping = json.load(file)

    return model, feature_columns, input_mapping


def assign_risk_category(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    else:
        return "High"


def build_case_row(user_inputs, feature_columns, input_mapping):
    case_row = pd.DataFrame(0, index=[0], columns=feature_columns)

    for label, selected in user_inputs.items():
        variable = input_mapping.get(label)

        if selected and variable in case_row.columns:
            case_row.loc[0, variable] = 1

    # Recreate interaction terms if they exist in the model features
    for col in case_row.columns:
        if "__X__" in col:
            parts = col.split("__X__")

            if all(part in case_row.columns for part in parts):
                product_value = 1

                for part in parts:
                    product_value = product_value * case_row.loc[0, part]

                case_row.loc[0, col] = product_value

    return case_row


def make_recommendation(risk_category):
    if risk_category == "Low":
        return "Low screening risk. Continue monitoring symptoms and follow public health guidance."
    elif risk_category == "Moderate":
        return "Moderate screening risk. Consider confirmatory testing and monitor symptoms closely."
    else:
        return "High screening risk. Seek confirmatory testing or clinical guidance, especially if symptoms worsen."


# =========================================================
# PAGE HEADER
# =========================================================

st.title("COVID-19 At-Home Screening AI System")
st.subheader("COVIDCARE Data, Machine Learning, Network Analysis, and Gemini Explanation")

st.write(
    """
    This app demonstrates an at-home COVID-19 screening-support workflow using
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

# =========================================================
# PROJECT OBJECTIVE
# =========================================================

st.markdown("## Project Objective")

st.write(
    """
    The objective was to develop and evaluate a workflow for estimating the probability
    of PCR-confirmed COVID-19 using information that could reasonably be available at home
    before a clinic or emergency room visit.
    """
)

# =========================================================
# DATASET SUMMARY
# =========================================================

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

# =========================================================
# MODEL RESULTS
# =========================================================

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

# =========================================================
# INTERACTIVE AT-HOME SCREENING SYSTEM
# =========================================================

st.markdown("## Interactive At-Home COVID-19 Screening")

st.write(
    """
    Select the home-available findings below. The app loads the trained XGBoost model
    from the notebook and estimates the probability of PCR-confirmed COVID-19.
    """
)

model_files_exist = (
    os.path.exists("covid_home_screening_model.pkl")
    and os.path.exists("model_feature_columns.json")
    and os.path.exists("home_input_mapping.json")
)

if not model_files_exist:
    st.error(
        """
        Model artifact files are missing. The notebook must be run through Part Q2
        to create `covid_home_screening_model.pkl`, `model_feature_columns.json`,
        and `home_input_mapping.json`.
        """
    )

else:
    model, feature_columns, input_mapping = load_model_artifacts()

    st.markdown("### At-Home Inputs")

    user_inputs = {}

    for label in input_mapping.keys():
        user_inputs[label] = st.checkbox(label, value=False)

    if st.button("Estimate COVID-19 Risk"):
        case_row = build_case_row(user_inputs, feature_columns, input_mapping)

        probability = model.predict_proba(case_row)[0, 1]
        risk_category = assign_risk_category(probability)
        recommendation = make_recommendation(risk_category)

        st.markdown("### Screening Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Predicted PCR-positive probability",
            f"{probability:.4f}"
        )

        result_col2.metric(
            "Risk category",
            risk_category
        )

        result_col3.metric(
            "Model used",
            "XGBoost"
        )

        if risk_category == "Low":
            st.success(recommendation)
        elif risk_category == "Moderate":
            st.warning(recommendation)
        else:
            st.error(recommendation)

        st.info(
            """
            This is a screening-support estimate, not a definitive diagnosis.
            PCR or another clinically accepted test remains the diagnostic standard.
            """
        )

        selected_inputs = [
            label for label, selected in user_inputs.items()
            if selected
        ]

        if selected_inputs:
            st.markdown("### Selected Inputs")
            st.write(selected_inputs)
        else:
            st.markdown("### Selected Inputs")
            st.write("No at-home findings were selected.")

        explanation = f"""
At-Home COVID-19 Screening Explanation

The trained XGBoost model estimated a PCR-positive COVID-19 probability of {probability:.3f}.
This result was classified as {risk_category} risk.

The selected at-home inputs were:
{", ".join(selected_inputs) if selected_inputs else "None selected"}

This output should be interpreted as screening support, not a definitive diagnosis.
A higher predicted probability suggests that confirmatory testing or clinical guidance may be appropriate,
especially if symptoms worsen or exposure risk is high. PCR or another clinically accepted test remains
the diagnostic standard.
"""

        st.text_area("Plain-Language Explanation", explanation, height=260)

# =========================================================
# TOP DIRECT PREDICTORS
# =========================================================

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

# =========================================================
# NETWORK ANALYSIS
# =========================================================

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

# =========================================================
# GEMINI / FALLBACK EXPLANATION
# =========================================================

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

    st.text_area("Notebook Explanation Output", explanation, height=300)
else:
    fallback_preview = """
At-Home COVID-19 Screening Explanation

The screening system used the XGBoost model because it had the strongest AUC during model evaluation.
The result should be interpreted as screening support, not a definitive diagnosis.

A higher predicted probability suggests that confirmatory testing or clinical guidance may be appropriate,
especially if symptoms worsen or exposure risk is high. PCR or another clinically accepted test remains
the diagnostic standard.
"""

    st.text_area("Fallback Explanation Preview", fallback_preview, height=260)

# =========================================================
# REPOSITORY AND DEPLOYMENT NOTE
# =========================================================

st.markdown("## Repository")

st.write("Full reproducible notebook and project files are available in the GitHub repository:")

st.code("https://github.com/rbrown-65/Covid-19-Prediction-AI-System")

st.markdown("## Deployment Note")

st.write(
    """
    This Streamlit app is a deployed interactive project demonstration. The full
    model-building workflow is contained in the Jupyter notebook and can be executed
    from the GitHub repository using GitHub Codespaces.
    """
)