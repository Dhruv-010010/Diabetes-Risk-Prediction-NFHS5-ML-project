<div align="center">

# 🩺 Diabetes Risk Prediction from NFHS-5 Data

### An imbalance-aware ML pipeline for screening Type-II diabetes risk across 36 Indian states, with SHAP-based interpretability

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EA4C2E?style=for-the-badge" alt="XGBoost" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=for-the-badge" alt="SHAP" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

</div>

> [!IMPORTANT]
> This is a **research and educational project, not a clinical decision tool**. Predictions should not be used for diagnosis, screening, or treatment without independent clinical validation.

## ✨ Overview

This project screens for Type-II diabetes risk using demographic, socioeconomic, anthropometric, lifestyle, and hypertension fields derived from an NFHS-5 (National Family Health Survey) dataset. The notebook covers the full pipeline: data-quality checks, feature engineering, imbalance-aware model training across six model families, held-out evaluation, and SHAP-based explainability.

Because diabetes prevalence in the data is under 1%, the project treats class imbalance as a first-class problem rather than an afterthought — comparing unmodified training, SMOTE, and class-weighted learning side by side.

## 📸 Project snapshot

| Metric | Value |
| --- | --- |
| Respondents | **136,136** |
| Columns in checked-in CSV | **95** |
| Diabetes-positive records | **1,227** |
| Diabetes-negative records | **134,909** |
| Diabetes prevalence | **0.9013%** |
| States represented | **36** |
| Final model features | **19** |
| Train / test split | **108,648** / **27,163** rows |
| Model families compared | **6**, with up to 3 imbalance strategies each |
| Cross-validation fits | **~730** |

> [!NOTE]
> A model can reach ~99% accuracy by predicting almost everyone as non-diabetic while missing nearly every positive case. That's why recall, precision, F1, ROC-AUC, and log loss are all reported alongside accuracy — accuracy alone is misleading here.

## 🧭 Workflow

The notebook [`Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb`](Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb) runs the following pipeline:

```text
NFHS-5 CSV → clean & filter → feature engineer → stratified 80/20 split
           → scale (LR / SVM) → resample (SMOTE / class-weight) → GridSearchCV (5-fold)
           → evaluate on held-out test set → SHAP + odds-ratio explainability
```

1. **Load & inspect** — shape, columns, target balance, missingness, value ranges.
2. **Filter fields** — keep demographic, socioeconomic, anthropometric, lifestyle, hypertension, state, and diabetes fields; drop child/pregnancy-specific variables, redundant household assets, likely post-diagnosis comorbidities, and low-relevance fields.
3. **Clean records** — remove implausible rows via height, weight, and age-at-marriage checks; derive BMI and keep values in the safety range **12–60**.
4. **Engineer features** — derive Asian-WHO BMI categories, age groups, and `Years_Married`. Binned BMI/age are used for analysis only; continuous `BMI` and `Years_Married` go into the final model matrix.
5. **Split** — 80/20 stratified on a `State` × `Diabetes` key. State–outcome groups with fewer than 2 observations are merged into a `rare_group` bucket so the split can run.
6. **Prepare features** — drop `State` after the split; scale features for logistic regression and SVM.
7. **Handle imbalance** — compare unmodified training, SMOTE (inside an `imblearn` pipeline, so synthetic samples never leak out of training folds), and class-weighted training.
8. **Tune** — 5-fold `GridSearchCV` with ROC-AUC scoring.
9. **Evaluate & explain** — score every fitted model on the untouched test set; generate ROC, precision-recall, heatmap, SHAP, and logistic-regression odds-ratio outputs.

### Final model features

```text
Res_Age, Married_age, ResidenceType_Urban, Religion_Muslim,
Religion_Other, Religion_Sikh, Ethnicity_No caste / tribe,
Ethnicity_Tribe, Edu_level, Wealth_Idx_Lb, HealthInsurance,
Smoke, Smoke_atHome, Tobacco, Betel_Leaf, Alcohol, Hypertension,
BMI, Years_Married
```

## 🤖 Models & imbalance strategies

| Models compared | Imbalance strategies |
| --- | --- |
| Logistic Regression | **Unmodified** — original class distribution |
| RBF SVM | **SMOTE** — oversampling inside the CV pipeline |
| Random Forest | **Class weight** — `class_weight="balanced"` (LR, SVM, RF) |
| Gaussian Naive Bayes | |
| AdaBoost | |
| XGBoost | |

> [!NOTE]
> The SVM grid search is capped at **15,000** training rows to stay tractable; every other model trains on the full **108,648** rows.

## 📊 Results

Metrics below come from `Outputs/trained_models_data.pkl`, measured on the held-out test set.

| Selection criterion | Model & strategy | Result | Precision | Recall | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Highest ROC-AUC | SVM (RBF) + class weight | **0.7446** | 0.0213 | 0.6157 | 0.7446 |
| Highest recall | XGBoost + SMOTE | **0.9339** | 0.0089 | 0.9339 | 0.5247 |
| Highest F1 | Naive Bayes, unmodified | **0.0861** | 0.0482 | 0.4050 | 0.7289 |
| Highest accuracy | Logistic Regression, unmodified | **0.9911** | 0.0000 | 0.0000 | 0.7184 |

These rows summarize the core trade-off in the project: pushing sensitivity up tends to crater precision at sub-1% prevalence. The current artifacts don't pick a single "best" model or decision threshold — that choice depends on the intended use case.

<p align="center">
  <img src="Outputs/Graphical%20analysis/results_heatmap.png" alt="Model comparison heatmap" width="820" />
</p>

## 🗂️ Repository structure

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

### 📌 Important files

| File | What it is |
| --- | --- |
| [`Data/Data.csv`](Data/Data.csv) | Checked-in dataset used for the project |
| [`Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb`](Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb) | End-to-end analysis and training notebook |
| [`Outputs/trained_models_data.pkl`](Outputs/trained_models_data.pkl) | Serialized `results`, fitted `model_store` pipelines, and `roc_store` curve data for 15 model/strategy combinations |
| [`Outputs/lr_odds_ratios.csv`](Outputs/lr_odds_ratios.csv) | Logistic-regression coefficients, odds ratios, and confidence-limit columns |
| [`Report/Project_report.pdf`](Report/Project_report.pdf) | Full project report |
| [`Outputs/Graphical analysis/ML pipeline diagram.png`](Outputs/Graphical%20analysis/ML%20pipeline%20diagram.png) | Visual overview of the pipeline |

## 🚀 Running the notebook

### 1. Create an environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost shap joblib jupyter
```

### 2. Launch Jupyter 📓

Run from the project root:

```powershell
jupyter notebook
```

Then open `Notebooks/Diabetes_Risk_Prediction_NFHS5.ipynb` and run the cells in order.

## 🔍 Inspecting the trained artifact

The pickle contains fitted pipelines that expect the **19 already-engineered model features** listed above — it's not a standalone prediction service and doesn't include a raw-data preprocessing API.

```python
import joblib
import pandas as pd

artifact = joblib.load("Outputs/trained_models_data.pkl")
results = pd.DataFrame(artifact["results"])
print(results.sort_values("ROC-AUC", ascending=False).head())
print(artifact["model_store"].keys())
```

## 🖼️ Visual outputs

<p align="center">
  <img src="Outputs/Graphical%20analysis/ML%20pipeline%20diagram.png" alt="Pipeline diagram" width="820" />
</p>





## 🧯 Troubleshooting

| Issue | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate the virtual environment, then re-run the `pip install` command above. |
| Notebook can't find `Data.csv` | Launch Jupyter from the project root so relative paths in the notebook resolve correctly. |
| `joblib.load` fails or keys look wrong | Confirm `Outputs/trained_models_data.pkl` was pulled with Git LFS / downloaded in full, not truncated. |
| Plots don't render inline | Make sure `matplotlib` and `seaborn` installed cleanly, and restart the notebook kernel. |

## ⚠️ Limitations

- The data are observational; model associations must not be interpreted as causal effects.
- The positive class is severely under-represented, and reported precision can be very low even when recall is high.
- No independent external validation, probability calibration study, threshold-selection protocol, confidence intervals, or clinical utility analysis is included in the current notebook.
- The notebook doesn't expose a production inference API and doesn't document survey-weighted estimation or deployment monitoring.
- SHAP explains model behavior, not medical causality.

## 📜 License and data use

This project is released under the **MIT License** — see `LICENSE` for details. The underlying NFHS-5 data should be used in accordance with its original source's terms of use.

## 🙌 Acknowledgements

- National Family Health Survey (NFHS-5)
- scikit-learn, XGBoost, imbalanced-learn, SHAP, pandas, and Jupyter