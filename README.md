# Diabetes Risk Prediction from NFHS-5 Data

An exploratory and machine-learning project for identifying diabetes risk among respondents in an NFHS-5-derived dataset. The notebook combines data quality checks, feature engineering, imbalance-aware model training, held-out evaluation, and model explainability.

> This is a research and educational project, not a clinical decision tool. Its predictions should not be used for diagnosis, screening, or treatment without independent clinical validation.

## Project snapshot

- **136,136 respondents** and **95 columns** in the checked-in CSV
- **1,227 diabetes-positive records** and **134,909 negative records**
- Diabetes prevalence: **0.9013%**
- **36 states** represented in the data
- **19 final model features** after cleaning and feature engineering
- Held-out split: **108,648 training rows** and **27,163 test rows**
- Six model families compared with up to three imbalance strategies

The very low positive-class prevalence is central to the project. A model can reach approximately 99% accuracy by predicting almost everyone as non-diabetic while missing nearly every positive case, so recall, precision, F1, ROC-AUC, and log loss are reported alongside accuracy.

## Workflow

The notebook `Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb` implements the following pipeline:

1. Load the NFHS-5-derived CSV and inspect shape, columns, target balance, missingness, and selected value ranges.
2. Keep demographic, socioeconomic, anthropometric, lifestyle, hypertension, state, and diabetes fields. Child/pregnancy-specific variables, redundant household assets, possible post-diagnosis comorbidities, and low-relevance fields are excluded from the modeling subset.
3. Remove implausible records using height, weight, and age-at-marriage checks. Derive BMI and retain values in the notebook's safety range of 12-60.
4. Derive Asian-WHO BMI categories, age groups, and `Years_Married`. The binned BMI and age features are used for analysis and then removed from the final model matrix; continuous `BMI` and `Years_Married` remain.
5. Split 80/20 using a stratification key made from `State` and `Diabetes`. The two state-by-outcome groups with fewer than two observations are merged into a `rare_group` bucket so the split can run.
6. Remove `State` from the model features after the split. Scaling is applied to logistic regression and SVM models.
7. Compare unmodified training, SMOTE oversampling, and class-weighted training. SMOTE is inside the `imblearn` pipeline so synthetic samples are created only within training folds.
8. Tune hyperparameters with five-fold `GridSearchCV` using ROC-AUC scoring.
9. Evaluate every fitted model on the untouched test set and generate ROC, precision-recall, heatmap, SHAP, and logistic-regression odds-ratio outputs.

### Final model features

```text
Res_Age, Married_age, ResidenceType_Urban, Religion_Muslim,
Religion_Other, Religion_Sikh, Ethnicity_No caste / tribe,
Ethnicity_Tribe, Edu_level, Wealth_Idx_Lb, HealthInsurance,
Smoke, Smoke_atHome, Tobacco, Betel_Leaf, Alcohol, Hypertension,
BMI, Years_Married
```

## Models and imbalance strategies

The notebook evaluates:

- Logistic Regression
- RBF SVM
- Random Forest
- Gaussian Naive Bayes
- AdaBoost
- XGBoost

The three strategies are:

- **Unmodified**: train on the original class distribution.
- **SMOTE**: oversample the minority class inside the cross-validation pipeline.
- **Class weight**: use `class_weight="balanced"` for Logistic Regression, SVM, and Random Forest.

The SVM is capped at **15,000 training rows** to keep its grid search tractable; other models use all **108,648** training rows. The notebook evaluates approximately **730 cross-validation fits** across the six model families and their supported strategies.

## Checked-in results

The metrics below come from `Outputs/trained_models_data.pkl` and are measured on the held-out test set.

| Selection criterion | Model and strategy | Result | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Highest ROC-AUC | SVM (RBF) + class weight | **0.7446** | 0.0213 | 0.6157 | 0.7446 |
| Highest recall | XGBoost + SMOTE | **0.9339** | 0.0089 | 0.9339 | 0.5247 |
| Highest F1 | Naive Bayes, unmodified | **0.0861** | 0.0482 | 0.4050 | 0.7289 |
| Highest accuracy | Logistic Regression, unmodified | **0.9911** | 0.0000 | 0.0000 | 0.7184 |

These results show the main trade-off in the project: improving sensitivity can produce extremely low precision under a prevalence below 1%. The current artifacts do not establish a single clinically preferable model or decision threshold. The full model-by-strategy comparison is available in the [results heatmap](Outputs/Graphical%20analysis/results_heatmap.png).

## Repository layout

```text
.
├── Data/
│   └── Data.csv
├── Notebooks/
│   └── Diabetes_Risk_Prediction_NFHS5.ipynb
├── Outputs/
│   ├── trained_models_data.pkl
│   ├── lr_odds_ratios.csv
│   └── Graphical analysis/
│       ├── ML pipeline diagram.png
│       ├── Initial_corr.png
│       ├── categorical_analysis.png
│       ├── age_bmi_prevalence.png
│       ├── class_distribution_per_approach.png
│       ├── ROC Curves.png
│       ├── PR Curves.png
│       ├── results_heatmap.png
│       ├── shap_random_forest_none.png
│       └── shap_xg_boost_none.png
├── Report/
│   └── Project_report.pdf
└── README.md
```

### Important files

- [`Data/Data.csv`](Data/Data.csv) - checked-in dataset used for the project.
- [`Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb`](Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb) - end-to-end analysis and training notebook.
- [`Outputs/trained_models_data.pkl`](Outputs/trained_models_data.pkl) - serialized dictionary containing `results`, fitted `model_store` pipelines, and `roc_store` curve data for 15 model/strategy combinations.
- [`Outputs/lr_odds_ratios.csv`](Outputs/lr_odds_ratios.csv) - logistic-regression coefficients, odds ratios, and confidence-limit columns.
- [`Report/Project_report.pdf`](Report/Project_report.pdf) - project report.
- [`Outputs/Graphical analysis/ML pipeline diagram.png`](Outputs/Graphical%20analysis/ML%20pipeline%20diagram.png) - visual overview of the pipeline.

## Running the notebook

### 1. Create an environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost shap joblib jupyter
```

### 2. Open the notebook

Run these commands from the project root:

```powershell
jupyter notebook
```

Then open `Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb` and run the cells in order.


## Inspecting the trained artifact

The pickle contains fitted pipelines that expect the **19 already-engineered model features** listed above. It is not a standalone prediction service and does not include a raw-data preprocessing API.

```python
import joblib
import pandas as pd

artifact = joblib.load("Outputs/trained_models_data.pkl")
results = pd.DataFrame(artifact["results"])
print(results.sort_values("ROC-AUC", ascending=False).head())
print(artifact["model_store"].keys())
```

## Visual outputs

![Pipeline diagram](Outputs/Graphical%20analysis/ML%20pipeline%20diagram.png)

Additional plots include [ROC curves](Outputs/Graphical%20analysis/ROC%20Curves.png), [precision-recall curves](Outputs/Graphical%20analysis/PR%20Curves.png), [SHAP explanations for Random Forest](Outputs/Graphical%20analysis/shap_random_forest_none.png), and [SHAP explanations for XGBoost](Outputs/Graphical%20analysis/shap_xg_boost_none.png).

## Limitations

- The data are observational; model associations must not be interpreted as causal effects.
- The positive class is severely under-represented, and the reported precision values can be very low even when recall is high.
- No independent external validation, probability calibration study, threshold-selection protocol, confidence intervals, or clinical utility analysis is included in the current notebook.
- The notebook does not expose a production inference API and does not document survey-weighted estimation or deployment monitoring.
- SHAP explains model behavior, not medical causality.

## License and data use
Added a proper MIT license for the project.
