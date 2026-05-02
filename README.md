# Credit Card Fraud Detection

A binary classification project on severely imbalanced data — detecting fraudulent credit card transactions among 284,807 European cardholder transactions from September 2013.

The central challenge is **extreme class imbalance**: fraud accounts for only **0.17%** of all transactions, making accuracy a misleading metric and requiring dedicated imbalance-handling strategies.

## Dataset

> **Note:** The dataset file (`creditcard.csv`) exceeds 100 MB and cannot be uploaded to GitHub. Download it from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `Classic Datasets/` folder before running the notebook.

| Property | Value |
|---|---|
| Source | `Classic Datasets/creditcard.csv` |
| Rows | 284,807 |
| Features | 30 (V1–V28, Time, Amount) |
| Target | `Class` — 0: Legitimate · 1: Fraud |
| Missing values | None |
| Memory | 67.4 MB |

**Class distribution:**

| Class | Count | Share |
|---|---|---|
| Legitimate (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |

**Features:** `V1`–`V28` are PCA components applied to the original raw data to protect cardholder privacy — direct interpretation is not possible. `Time` is seconds since the first transaction; `Amount` is the transaction value in euros.

## Why Accuracy Fails Here

A naive model that predicts "Legitimate" for every transaction achieves **99.83% accuracy** while detecting zero fraud. The metrics that matter:

| Metric | Formula | Priority |
|---|---|---|
| **Recall** | TP / (TP + FN) | Critical — missing a fraud is costly |
| **Precision** | TP / (TP + FP) | Important — too many false alarms causes friction |
| **F1-score** | 2 × P × R / (P + R) | Primary comparison metric |
| **ROC-AUC** | Area under ROC curve | Overall discrimination ability |

## Workflow

1. **Configuration & Imports**
2. **Data Loading & Inspection** — structure, missing values, class imbalance quantification
3. **Naive Model** — demonstrates the accuracy trap with an unweighted Decision Tree
4. **Imbalance Handling** — theory and application of three strategies
5. **EDA** — class distribution, amount/time distributions, PCA feature separation, correlation heatmap
6. **Model Comparison** — 6 models evaluated on the real (non-synthetic) test set

## Imbalance Handling Strategies

| Strategy | Method | Trade-off |
|---|---|---|
| **Undersampling** | `RandomUnderSampler` — reduces majority class | Fast but discards ~283,800 legitimate observations |
| **Oversampling** | `SMOTE` — generates synthetic minority samples by interpolation between neighbors | Preserves all data; reduces overfitting vs plain duplication |
| **Combined** | `SMOTEENN` / `SMOTETomek` — SMOTE + decision boundary cleanup | Better boundaries; higher compute cost |
| **Class weighting** | `class_weight="balanced"` / `scale_pos_weight` — modifies loss function | No data modification; parameter to calibrate |

**Key practice:** Resampling is applied **inside a Pipeline** on the training set only — never on the test set — to prevent data leakage that would artificially inflate metrics.

## Preprocessing

- `RobustScaler` applied to `Time` and `Amount` — robust to the outliers present in transaction amounts
- `V1`–`V28` already scaled from PCA, no further scaling needed
- Train/test split: 80/20, stratified on `Class`

## Models Compared

| Model | Imbalance Strategy |
|---|---|
| Dummy (baseline) | Always predicts majority class |
| Logistic Regression | `class_weight="balanced"` |
| Decision Tree | `class_weight="balanced"`, `max_depth=8` |
| Random Forest | `class_weight="balanced"`, 100 trees |
| XGBoost | `scale_pos_weight = n_negative / n_positive ≈ 578` |
| LightGBM | `class_weight="balanced"` |

## Key Findings from EDA

- **V14, V17, V12, V10** show the clearest separation between fraud and legitimate transactions
- Fraudulent transactions tend to cluster at **lower amounts** than legitimate ones
- Fraud is distributed more **uniformly across time**, while legitimate transactions follow a clear daily pattern (two peaks)
- Mean difference analysis (Fraud − Legitimate) identifies V14, V17, V12, V4 as the most discriminating PCA components

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imbalanced-learn
```
