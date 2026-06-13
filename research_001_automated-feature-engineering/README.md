# Research 001: Automated Feature Engineering in ML Pipelines

## Overview

This research investigates methods for automatically discovering compact, information-preserving, and non-redundant feature representations from raw structured/tabular data, addressing one of the primary bottlenecks in machine learning workflows.

## Problem Statement

Manual feature engineering remains a critical and time-consuming component of ML pipelines. The key challenge is: **Can machine learning systems autonomously discover optimal feature representations that generalize across model families?**

## Key Questions

1. Are there universal, model-agnostic features that persist across different algorithm families?
2. How can we identify and prioritize highly informative, non-redundant features while reducing overfitting?
3. Can we combine statistical methods, model-based importance metrics, and dimensionality reduction to extract robust features?

## Research Plan

1. **Data Understanding** — Describe dataset characteristics, shape, and distributions
2. **Data Cleaning** — Handle missing values, outliers, and inconsistencies
3. **Exploratory Data Analysis (EDA)** — Visualizations, correlation analysis, feature interactions
4. **Feature Engineering & Selection** — Combine:
   - Statistical methods (correlation, information gain)
   - Model-based importance (permutation, SHAP, coefficients)
   - Dimensionality reduction (PCA, autoencoders)
5. **Model Building** — Train with selected features, apply cross-validation and hyperparameter tuning
6. **Evaluation & Interpretation** — Compare feature selection approaches and model performance

## Datasets

- **Abalone** — Regression task for predicting abalone age from physical measurements

## Resources

- [Research Introduction](research_intro.md) — Detailed problem exploration and considerations
- [Introduction](introduction.md) — Initial research notes

---

*Part of the Adaptive ML Systems research framework*
