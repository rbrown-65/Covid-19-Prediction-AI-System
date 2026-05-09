# COVID-19 Prediction AI System

## Using COVIDCARE Data, Machine Learning, Network Analysis, and Gemini LLM Explanation

**Course:** Comparative Effectiveness  
**Project:** COVID-19 Home Diagnosis AI System  
**Author:** Rebekah Brown  

---

## What This Project Does

This project builds a COVID-19 prediction workflow using patient-level COVIDCARE survey data and a DEMI knowledgebase.

The notebook:

1. Loads patient-level COVIDCARE data, the DEMI knowledgebase, and the survey dictionary.
2. Assigns variables into temporal tiers, from demographics to PCR-confirmed COVID-19 outcome.
3. Calculates pairwise associations between variables using odds ratios, log odds ratios, phi coefficients, support, and conditional probabilities.
4. Trains Logistic Regression, LASSO Logistic Regression, and XGBoost models to predict PCR-confirmed COVID-19.
5. Identifies direct predictors of PCR positivity using the LASSO model.
6. Attempts to build parent models for Markov blanket/network interpretation.
7. Creates a directed network diagram showing predictors leading to PCR test positivity.
8. Creates CPT-style tables for Netica when parent models are available.
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
| `merged.csv` | Additional merged data file included in the repository |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project description and run instructions |

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

## Optional Gemini LLM Setup

The notebook includes an optional Gemini LLM explanation component in **Part R**.

If a Gemini API key is available, Part R uses Gemini to generate a plain-English explanation of the model results. If no API key is available, the notebook still runs and uses a fallback automated explanation.

To use Gemini, set an environment variable before running Part R:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

or:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Do **not** paste an API key directly into the notebook or commit it to GitHub.

---

## Dataset Summary

The raw COVIDCARE dataset contained:

```text
822 patients
472 raw variables
```

The DEMI knowledgebase contained:

```text
70,983 knowledgebase rows
11 columns
```

The survey dictionary contained:

```text
472 variables
10 columns
```

After removing records with missing PCR outcome values, the modeling dataset contained:

```text
559 patients
445 modeling columns
444 predictor variables
```

PCR outcome distribution:

| PCR Test Result | Count |
|---|---:|
| Negative | 501 |
| Positive | 58 |

The dataset is imbalanced, with many more PCR-negative cases than PCR-positive cases. Because of this imbalance, AUC was emphasized more than accuracy when selecting the best model.

---

## Temporal Tier Structure

Variables were assigned into five temporal tiers. This was done to organize predictors according to when they occur relative to PCR confirmation.

| Tier | Variable Type | Count |
|---|---|---:|
| Tier 0 | Birth and demographic variables | 27 |
| Tier 1 | Vaccination-related variables | 50 |
| Tier 2 | Symptoms, exposures, and illness-period variables | 377 |
| Tier 3 | At-home test-related variables | 17 |
| Tier 4 | PCR lab confirmation outcome | 1 |

The PCR test result was treated as the final outcome. Predictors were restricted to variables that occur before PCR confirmation.

---

## Pairwise Association Analysis

The notebook calculates pairwise relationships between variables using the DEMI knowledgebase.

For each pair, the notebook creates 2x2 table counts:

| Count | Meaning |
|---|---|
| `n11` | Concept and target both present |
| `n10` | Concept present, target absent |
| `n01` | Concept absent, target present |
| `n00` | Concept and target both absent |

The notebook then calculates:

- Odds ratio
- Log odds ratio
- Phi coefficient
- Support
- Probability of target given concept
- Probability of target given no concept

The results are saved as:

```text
pairwise_associations.csv
pairwise_frequencies.csv
```

---

## Machine Learning Models

The notebook compares three model types:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline interpretable classification model |
| LASSO Logistic Regression | Feature selection and direct predictor identification |
| XGBoost | Nonlinear machine learning model for prediction |

Model performance was evaluated using:

- AUC
- Accuracy
- McFadden pseudo R²
- Percent variation explained

The best model was selected based on AUC.

---

## Model Results

The notebook compared Logistic Regression, LASSO Logistic Regression, and XGBoost for predicting PCR-confirmed COVID-19.

| Model | AUC | Accuracy | McFadden R² | % Variation Explained |
|---|---:|---:|---:|---:|
| XGBoost | 0.948 | 93.6% | 0.565 | 56.5% |
| Logistic Regression | 0.877 | 95.7% | -0.178 | -17.8% |
| LASSO Logistic Regression | 0.868 | 95.0% | -0.241 | -24.1% |

The best-performing model based on AUC was **XGBoost**.

Although Logistic Regression had the highest accuracy, XGBoost had the strongest overall discrimination based on AUC and the best McFadden R². This matters because the dataset is imbalanced, with many more PCR-negative cases than PCR-positive cases, so accuracy alone can be misleading.

The baseline example COVID-19 probability with all predictors set to zero was:

```text
0.3808
```

Model results are saved as:

```text
model_results.csv
```

---

## Direct Predictors of PCR Positivity

The LASSO model was used to identify direct predictors of PCR test positivity.

The strongest direct predictor was:

```text
30158-Symtpom_Neuro-7
```

with a LASSO coefficient of:

```text
1.060
```

Top direct predictors included:

| Predictor | Coefficient |
|---|---:|
| `30158-Symtpom_Neuro-7` | 1.060 |
| `30766-pinkblue_confirm` | 0.648 |
| `31386-covid_results-1` | 0.544 |
| `31386-covid_results-2` | -0.382 |
| `30183-states_specify-42` | 0.364 |
| `30183-states_specify-33` | 0.357 |
| `32137-vaccine_avail` | 0.351 |
| `32138-vaccine_factors-13` | 0.326 |
| `30158-Symtpom_Neuro-8` | 0.316 |
| `30141-covid_tst_symptoms-3` | 0.311 |

Positive coefficients indicate variables associated with higher predicted probability of PCR positivity. Negative coefficients indicate variables associated with lower predicted probability.

These predictors should be interpreted as **predictive associations**, not proof of causation.

The direct predictor table is saved as:

```text
direct_predictors_pcr.csv
```

---

## Network Analysis

The notebook attempted to build parent models for the Markov blanket step by regressing each direct PCR predictor on earlier-tier variables.

In this run:

```text
Number of direct predictors with parent models: 0
```

Because no usable parent models were found, the final network diagram shows the top direct predictors pointing directly to the PCR-confirmed COVID-19 outcome.

The network contains:

```text
16 nodes
15 edges
```

The graph was saved as:

```text
covid_network_clean.png
```

The edge list was saved as:

```text
covid_network_edges.csv
```

This network should be interpreted as a **predictor network**, not a full causal network.

---

## CPT Tables for Netica

The notebook includes code to create CPT-style tables for Netica when parent models are available.

In this run, no Markov parent models were available, so no CPT table was created.

This section remains in the notebook so the workflow can create CPT tables if parent models are found in a future run.

---

## Gemini LLM Explanation

The notebook includes a Gemini LLM explanation component in Part R.

The LLM uses:

- Model results
- Top direct predictors
- Network summary
- Markov parent-model results when available

The LLM then generates a plain-English explanation of the model results.

The LLM does **not** train the model and does **not** change the predictions.

In this run, Gemini successfully generated an explanation and saved it as:

```text
llm_model_explanation.txt
```

If no Gemini API key is available, the notebook still runs and creates a fallback automated explanation instead.

---

## Interpretation

The XGBoost model performed best overall based on AUC, with an AUC of 0.948. This suggests the model was strong at ranking PCR-positive cases higher than PCR-negative cases.

The dataset was imbalanced, with 501 PCR-negative cases and 58 PCR-positive cases after filtering. Because of this imbalance, accuracy alone was not used to choose the best model. XGBoost had slightly lower accuracy than Logistic Regression, but it had much better AUC and McFadden R².

The LASSO model was used for feature selection. It identified neurological symptoms, at-home test confirmation variables, COVID result variables, vaccine-related variables, and location-related variables among the strongest predictors.

These results should be interpreted as predictive, not causal. The model identifies patterns associated with PCR positivity, but it does not prove that any predictor causes COVID-19 infection.

---

## Output Files Created by the Notebook

Running the notebook creates the following output files:

| Output File | Description |
|---|---|
| `pairwise_associations.csv` | Pairwise association measures from the knowledgebase |
| `pairwise_frequencies.csv` | Co-occurrence frequency table |
| `model_results.csv` | Model performance results |
| `direct_predictors_pcr.csv` | LASSO-selected direct predictors of PCR positivity |
| `markov_blanket_predictors.csv` | Markov blanket parent-predictor summary, if parent models are found |
| `markov_variation_explained.csv` | Percent variation explained by parent models, if available |
| `covid_network_edges.csv` | Edge list for the predictor network |
| `covid_network_clean.png` | Network visualization |
| `llm_model_explanation.txt` | Gemini or fallback explanation of model results |

In this run, no Markov parent models were found, so the Markov blanket and CPT sections were skipped or limited.

---

## Requirements

The project uses:

```text
pandas
numpy
matplotlib
networkx
scikit-learn
xgboost
google-genai
ipywidgets
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Limitations

This project should be interpreted as a predictive modeling workflow, not proof of causality.

Potential limitations include:

- Missing data
- Small number of PCR-positive cases
- Imbalanced outcome distribution
- Survey coding choices
- Possible overfitting from interaction terms
- Limited generalizability to new populations
- Changing COVID-19 variants, testing behavior, and vaccination patterns over time

The model can identify patterns associated with PCR-confirmed COVID-19, but it does not prove that any symptom, exposure, demographic factor, or vaccine-related variable causes COVID-19 infection.

---

## Data Source

COVIDCARE Phase II Survey data  
MIT, December 2021  

Dataset dimensions used in this project:

```text
822 participants
472 raw variables
70,983 DEMI knowledgebase rows
```

