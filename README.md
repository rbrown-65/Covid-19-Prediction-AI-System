# Covid-19-Prediction-AI-System

## Using COVIDCARE Data, Machine Learning, Network Analysis, and Gemini LLM Explanation

**Course:** HI:823 - Causal Analysis & Comparative Effectiveness Spring 2026
**Instructor:** Abdul Hafeez
**Project:** COVID-19 Home Diagnosis AI System  
**Author:** Rebekah Brown 
**School:** George Mason University 

---

## What This Project Does

This project builds a COVID-19 prediction workflow using patient-level COVIDCARE survey data and a DEMI knowledgebase.

The notebook:

1. Loads patient-level COVIDCARE data, the DEMI knowledgebase, and the survey dictionary.
2. Assigns variables into temporal tiers, from demographics to PCR-confirmed COVID-19 outcome.
3. Calculates pairwise associations between variables using odds ratios, log odds ratios, phi coefficients, support, and conditional probabilities.
4. Trains Logistic Regression, LASSO Logistic Regression, and XGBoost/Gradient Boosting models to predict PCR-confirmed COVID-19.
5. Identifies direct predictors of PCR positivity using the LASSO model.
6. Attempts to build parent models for Markov blanket/network interpretation.
7. Creates a directed network diagram showing predictors leading to PCR test positivity.
8. Generates CPT-style tables for Netica when parent models are available.
9. Uses an optional Gemini LLM component to summarize the model results in plain English.

The LLM does **not** train the model or change the prediction. It is used only to explain the results.

---

## Files in This Repository

| File | Description |
|---|---|
| `Covid_Prediction.ipynb` | Main notebook for the full COVID prediction workflow |
| `COVIDCARE_FORSUBMISSION_MIT_CLEANED_Phase_II_2021-12-03.csv` | Raw patient-level COVIDCARE data |
| `COVIDCARE_DEMI_knowledgebase_v4.csv` | DEMI knowledgebase used for pairwise variable relationships |
| `COVIDCARE_survey_dictionary_v2_ForSubmission_MIT_Phase_II_2021-12-26.csv` | Survey data dictionary |
| `merged.csv` | Additional merged knowledgebase/data file included in the repository |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project description and run instructions |

---

## Output Files Created by the Notebook

Running the notebook may create the following output files:

| Output File | Description |
|---|---|
| `pairwise_associations.csv` | Pairwise association results from the knowledgebase |
| `pairwise_frequencies.csv` | Co-occurrence frequency table |
| `model_results.csv` | Model performance results |
| `direct_predictors_pcr.csv` | LASSO-selected direct predictors of PCR positivity |
| `markov_blanket_predictors.csv` | Parent-predictor summary if Markov parent models are found |
| `markov_variation_explained.csv` | Percent variation explained by parent models, if available |
| `covid_network_edges.csv` | Edge list for the COVID predictor network |
| `covid_network_clean.png` | Network visualization |
| `llm_model_explanation.txt` | Gemini or fallback explanation of model results |

---

## How to Run

### Option 1 — GitHub Codespaces

1. Click the green **Code** button on this repository.
2. Select **Codespaces**.
3. Click **Create codespace on main**.
4. Open `Covid_Prediction.ipynb`.
5. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

6. Run the notebook cells from top to bottom.

The notebook can run without a Gemini API key because it includes a fallback automated explanation.

---

### Option 2 — Run Locally

To run this project on a local computer, complete the following steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/rbrown-65/Covid-19-Prediction-AI-System.git
   ```

2. Move into the project folder:

   ```bash
   cd Covid-19-Prediction-AI-System
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook:

   ```bash
   jupyter notebook Covid_Prediction.ipynb
   ```

5. Run the notebook cells from top to bottom.

The Gemini API key is **not required** to run the notebook. If no API key is provided, the notebook will use the fallback automated explanation in Part R.

---

## Model Results

After running the notebook, model performance results are saved in:

```text
model_results.csv
