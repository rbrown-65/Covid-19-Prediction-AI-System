# COVID-19 Prediction AI System

## Using COVIDCARE Data, Machine Learning, Network Analysis, and Gemini LLM Explanation

**Course:** HI 823: Causal Analysis & Comparative Effectiveness  
**School:** George Mason University  
**Instructor:** Dr. Abdul Hafeez  
<sub>This project was prepared using course materials by Dr. Farrokh Alemi for the Comparative Effectiveness course at http://openonlinecourses.com/causalanalysis/ and starter code provided by TA Chandana Reddy Gajjala. ChatGPT was used for debugging, code organization, and repository setup support.</sub>  
**Project:** COVID-19 Home Diagnosis AI System  
**Author:** Rebekah Brown  

---

## Objective

The objective of this project was to develop and evaluate a COVID-19 prediction workflow using COVIDCARE survey data, DEMI knowledgebase-derived pairwise association processing, machine learning classifiers, and a Gemini-based explanation component.

The final workflow was designed as an at-home screening-support system that estimates the probability of PCR-confirmed COVID-19 using information available before a clinic or emergency room visit.

---

## What This Project Does

This project builds a COVID-19 prediction workflow using patient-level COVIDCARE survey data and a DEMI knowledgebase.

The notebook:

1. Loads patient-level COVIDCARE data, the DEMI knowledgebase, and the survey dictionary.
2. Assigns variables into temporal tiers, from demographics to PCR-confirmed COVID-19 outcome.
3. Calculates pairwise associations between variables using odds ratios, log odds ratios, phi coefficients, support, and conditional probabilities.
4. Restricts modeling predictors to home-available information expected to be known before clinic or emergency room evaluation.
5. Trains Logistic Regression, LASSO Logistic Regression, and XGBoost models to predict PCR-confirmed COVID-19.
6. Identifies direct predictors of PCR positivity using the LASSO model.
7. Attempts to build parent models for Markov blanket/network interpretation.
8. Creates a directed network diagram showing predictors leading to PCR test positivity.
9. Converts the trained model into an example at-home COVID-19 screening system.
10. Uses an optional Gemini LLM component to explain the screening result in plain English.

The LLM does **not** train the model or change the prediction. It is used only to explain the results.

---

## Files in This Repository

| File | Description |
|---|---|
| `Covid_Prediction.ipynb` | Main notebook for the full COVID prediction and at-home screening workflow |
| `COVIDCARE_FORSUBMISSION_MIT_CLEANED_Phase_II_2021-12-03.csv` | Raw patient-level COVIDCARE data |
| `COVIDCARE_DEMI_knowledgebase_v4.csv` | DEMI knowledgebase used for pairwise variable relationships |
| `COVIDCARE_survey_dictionary_v2_ForSubmission_MIT_Phase_II_2021-12-26.csv` | Survey data dictionary |
| `app.py` | Streamlit deployment app |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Prevents temporary files and generated notebook outputs from being committed |
| `README.md` | Project description and run instructions |
| `covid_network_clean.png` | Network image used by the Streamlit app, if included in the repository |

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

If a Gemini API key is available, Part R uses Gemini to generate a plain-English explanation of the at-home screening result. If no API key is available, the notebook still runs and uses a fallback automated explanation.

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
394 home-available predictor variables
```

The updated model excludes likely clinic, laboratory, PCR-result, and post-diagnosis variables so that the prediction workflow better reflects information available at home before a clinic or emergency room visit.

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

The notebook compared Logistic Regression, LASSO Logistic Regression, and XGBoost for predicting PCR-confirmed COVID-19 using home-available predictors.

| Model | AUC | Accuracy | McFadden R² | % Variation Explained |
|---|---:|---:|---:|---:|
| XGBoost | 0.933 | 93.6% | 0.473 | 47.3% |
| Logistic Regression | 0.857 | 92.9% | -0.698 | -69.8% |
| LASSO Logistic Regression | 0.856 | 92.9% | -0.858 | -85.8% |

The best-performing model based on AUC was **XGBoost**. After restricting predictors to home-available information, XGBoost still showed the strongest discrimination, with an AUC of 0.933.

This matters because the dataset is imbalanced, with many more PCR-negative cases than PCR-positive cases, so accuracy alone can be misleading.

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
1.212
```

Top direct predictors included:

| Predictor | Coefficient |
|---|---:|
| `30158-Symtpom_Neuro-7` | 1.212 |
| `30766-pinkblue_confirm` | 0.925 |
| `32359-blue_nopink_confirm_2` | -0.494 |
| `30183-states_specify-33` | 0.428 |
| `30153-Symptoms-2` | 0.422 |
| `32137-vaccine_avail` | 0.421 |
| `32138-vaccine_factors-17` | 0.421 |
| `30183-states_specify-42` | 0.415 |
| `30158-Symtpom_Neuro-8` | 0.405 |
| `31396-join_hear-6` | 0.396 |

Positive coefficients indicate variables associated with higher predicted probability of PCR positivity. Negative coefficients indicate variables associated with lower predicted probability.

These predictors should be interpreted as **predictive associations**, not proof of causation.

The direct predictor table is saved as:

```text
direct_predictors_pcr.csv
```

---

## At-Home COVID-19 Screening System

The final notebook section converts the trained model into a simple at-home screening-support workflow. The system uses home-available inputs, such as symptoms and at-home test confirmation fields, to estimate the probability of PCR-confirmed COVID-19.

For the example patient scenario in the notebook, the system produced:

```text
Predicted probability of PCR-positive COVID-19: 0.9225
Risk category: High
Recommendation: High screening risk. Seek confirmatory testing or clinical guidance, especially if symptoms worsen.
```

This output should be interpreted as screening support, not a definitive diagnosis. PCR or another clinically accepted test remains the diagnostic standard.

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

The LLM explains the at-home screening result, including:

- The predicted probability of PCR-positive COVID-19
- The risk category
- The home-available inputs used in the example
- Why the output is screening support rather than a definitive diagnosis
- Why confirmatory testing or clinical guidance may still be needed

If no Gemini API key is available, the notebook creates a fallback automated explanation instead. The LLM does **not** train the model, change predictors, alter predictions, or evaluate performance.

---

## Interpretation

The XGBoost model performed best overall based on AUC, with an AUC of 0.933 after restricting predictors to home-available information. This suggests the model was strong at ranking PCR-positive cases higher than PCR-negative cases.

The dataset was imbalanced, with 501 PCR-negative cases and 58 PCR-positive cases after filtering. Because of this imbalance, AUC was used as the main model selection metric. XGBoost had slightly higher accuracy than the Logistic Regression and LASSO models and had much better AUC and McFadden R².

The LASSO model was used for feature selection. It identified neurological symptoms, at-home test confirmation variables, symptom variables, vaccine-related variables, and location-related variables among the strongest predictors.

These results should be interpreted as predictive, not causal. The model identifies patterns associated with PCR positivity, but it does not prove that any predictor causes COVID-19 infection.

---

## Repository Status

The main notebook is designed to run from top to bottom using the files included in this repository. Generated outputs are recreated when the notebook is executed. The Gemini LLM component is optional and will use a fallback explanation if no valid API key is available.

---

## Output Files Created by the Notebook

Running the notebook creates the following output files. These files are generated when the notebook is run and may not be stored directly in the repository.

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
google-generativeai
ipywidgets
streamlit
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

---

## References

1. World Health Organization. Coronavirus disease (COVID-19). https://www.who.int/health-topics/coronavirus#tab=tab_1. Accessed May 10, 2026.
2. Menni C, Valdes AM, Freidin MB, et al. Real-time tracking of self-reported symptoms to predict potential COVID-19. Nat Med. 2020;26(7):1037–1040. doi:10.1038/s41591-020-0916-2
3. Canas LS, Sudre CH, Capdevila Pujol J, et al. Early detection of COVID-19 in the UK using self-reported symptoms: a large-scale, prospective, epidemiological surveillance study. Lancet Digit Health. 2021;3(9):e587–e598. doi:10.1016/S2589-7500(21)00131-X
4. Zoabi Y, Deri-Rozov S, Shomron N. Machine learning-based prediction of COVID-19 diagnosis based on symptoms. npj Digit Med. 2021;4:3. doi:10.1038/s41746-020-00372-6
5. Langer T, Favarato M, Giudici R, et al. Development of machine learning models to predict RT-PCR results for severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) in patients with influenza-like symptoms using only basic clinical data. Scand J Trauma Resusc Emerg Med. 2020;28:113. doi:10.1186/s13049-020-00808-8
6. Shakeel SM, Kumar NS, Madalli PP, Srinivasaiah R, Swamy DR. COVID-19 prediction models: a systematic literature review. Osong Public Health Res Perspect. 2021;12(4):215–229. doi:10.24171/j.phrp.2021.0100
7. Alemi F. DEMI: Directed expectation-maximization for inference—a causal AI algorithm [course materials]. Comparative Effectiveness, George Mason University; 2026.

---

## Deployment

A Streamlit version of this project is deployed here:

https://rbrown-65-covid-19-prediction-ai-system-app-v1qsd3.streamlit.app/

The deployed app demonstrates the at-home COVID-19 screening workflow. It summarizes the dataset, model results, direct predictors, network interpretation, and example screening output from the final XGBoost model.

The app shows an example at-home patient scenario with home-available inputs, a predicted probability of PCR-positive COVID-19, a risk category, and a screening recommendation. The output is intended for screening support only and does not replace PCR testing or clinical judgment.

The full reproducible notebook is available in this repository and can be executed using GitHub Codespaces.
