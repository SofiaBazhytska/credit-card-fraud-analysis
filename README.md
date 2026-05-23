# Financial Fraud Detection and Analysis using Machine Learning

This repository contains the source code of an experimental study aimed at solving the problem of classifying financial transactions under conditions of extreme Class Imbalance.

## 📌 About the Project
The main goal of this research is to compare the effectiveness of different data balancing strategies and their impact on the ability of machine learning algorithms to detect fraudulent operations. Special emphasis is placed on analyzing False Positives and evaluating models using `Precision`, `Recall`, `F2-score`, and `PR-AUC` metrics.

## Tech Stack
* **Programming Language:** Python
* **Core Libraries:** `scikit-learn`, `xgboost`, `imbalanced-learn` (`imblearn`), `pandas`, `numpy`
* **Validation:** 10-fold cross-validation (Repeated Stratified K-Fold) with strict Data Leakage prevention using `Pipeline`.

## Experimental Scenarios
The study involves testing three algorithms (**Logistic Regression**, **Random Forest**, **XGBoost**) across five different feature space configurations:
1. **Baseline** (Standard scenario without intervention).
2. **Tomek Links** (Targeted undersampling to clean class boundaries).
3. **SMOTE** (Synthetic generation of the minority class).
4. **Cost-Sensitive Learning** (Algorithmic class weighting).
5. **SMOTE + Tomek Links** (Hybrid approach: generation + cleaning).

## How to Run
The code is fully ready for execution in a Jupyter Notebook or Google Colab environment.
