# 🥭 Mango FPO AI Decision Support System

> **AI/ML-based market-price forecasting and decision-support prototype for Mango Farmer Producer Organizations (FPOs)**

The **Mango FPO AI Decision Support System** is an academic/research prototype designed to support market-oriented decision-making for Mango Farmer Producer Organizations.

The system processes historical and current Mango market-price observations, performs data cleaning and feature engineering, trains machine learning models, predicts the **next observed market-reporting price**, and converts the prediction into transparent revenue scenarios based on FPO-entered quantity, transport cost, and storage cost.

---

## 📌 Project Overview

Mango Farmer Producer Organizations operate in an environment where market prices can vary across:

- Markets
- States
- Districts
- Mango varieties
- Reporting dates
- Market conditions

A practical decision-support system can help an FPO understand the available market information and estimate potential revenue under different assumptions.

This project develops a prototype that combines:

- Market-data processing
- Exploratory data analysis
- Time-aware feature engineering
- Machine learning regression
- Price prediction
- Revenue scenario calculation
- Market scenario analysis
- Interactive visualization

The current implementation focuses primarily on **Mango market-price forecasting and FPO revenue scenario analysis**.

---

# 🎯 Problem Statement

Mango FPOs may need to answer questions such as:

1. What is the latest observed Mango market price?
2. What price does the model estimate for the next observed market report?
3. How much has the predicted price changed?
4. What revenue could be generated for a given Mango quantity?
5. How does transport cost affect estimated revenue?
6. How does storage cost affect estimated revenue?
7. How do recent market-level scenarios compare?

Traditional market information may provide historical observations, but it does not directly convert those observations into a simple decision-support interface.

This project attempts to bridge that gap using machine learning and transparent scenario calculations.

---

# 💡 Proposed Solution

The system follows a complete data-to-decision pipeline:

```text
Mango Market Data
       │
       ▼
Data Extraction
       │
       ▼
Data Cleaning & Validation
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Machine Learning Models
       │
       ├───────────────┐
       │               │
       ▼               ▼
Random Forest       XGBoost
       │               │
       └───────┬───────┘
               ▼
       Model Evaluation
               │
               ▼
       Price Prediction
               │
               ▼
    FPO Decision Support
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
   Quantity Transport Storage
       │       │        │
       └───────┼────────┘
               ▼
       Revenue Scenarios
               │
               ▼
      Market Comparison
               │
               ▼
      Streamlit Dashboard