#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
For this Assignemnet I have used Jupyter notebook 

Data_analysis.py

Description:
    - This script analyzes an investment dataset to:
        1. Find the ideal combination of Market Cap, Type, and Risk for maximum 3-Year Return (%)
        2. Determine the optimal Sharpe Ratio to maximize 1-Year Return (%)
    - Includes visualizations and machine learning modeling.

Usage:
    - Placing the dataset (XLSX) in the same folder and name it "dataset.xlsx"
    - Run: python investment_analysis.py
Outputs:
    - Prints analysis summary to console
    - Saves plots:
        - 'top_3yr_combos.png'
        - 'sharpe_vs_1yrreturn.png'
"""

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# STEP 1: LOAD DATASET
# -----------------------------
# Read CSV
df = pd.read_excel("Dataset.xlsx")

# Clean column names (remove spaces, special chars, standardize)
df.columns = df.columns.str.strip().str.replace(" ", "").str.replace("-", "").str.replace("_", "")
print("Cleaned Column Names:", df.columns.tolist())

# Preview first rows
print("\nSample Data:")
print(df.head())

# -----------------------------
# STEP 2: DATA CLEANING
# -----------------------------
# Convert % columns to numeric (if stored as strings like "12.5%")
percent_cols = [c for c in df.columns if 'Return' in c or 'Sharpe' in c]
for col in percent_cols:
    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify key columns (adjust if your dataset uses different names)
market_col = 'MarketCap'
type_col = 'Type'
risk_col = 'Risk'
return_3yr_col = [c for c in df.columns if '3YrReturn' in c][0]
return_1yr_col = [c for c in df.columns if '1YrReturn' in c][0]
sharpe_col = [c for c in df.columns if 'Sharpe' in c][0]

# Drop rows with missing essential values
df = df.dropna(subset=[market_col, type_col, risk_col, return_3yr_col, return_1yr_col, sharpe_col])

# Standardize text columns
for col in [market_col, type_col, risk_col]:
    df[col] = df[col].str.title().str.strip()

# -----------------------------
# STEP 3: PROBLEM 1 — BEST COMBINATION FOR 3-YR RETURN
# -----------------------------
# Group by MarketCap, Type, Risk and calculate mean 3YrReturn
grouped = df.groupby([market_col, type_col, risk_col])[return_3yr_col].mean().reset_index()
grouped = grouped.sort_values(by=return_3yr_col, ascending=False)

best_combo = grouped.iloc[0]
print("\n  Ideal Combination for Highest 3-Year Return:")
print(best_combo)

# Visualization: Top 10 combinations
top10 = grouped.head(10)
plt.figure(figsize=(10,6))
sns.barplot(
    y=top10.apply(lambda x: f"{x[market_col]} | {x[type_col]} | {x[risk_col]}", axis=1),
    x=top10[return_3yr_col],
    palette="Blues_r"
)
plt.xlabel("Average 3-Year Return (%)")
plt.ylabel("MarketCap | Type | Risk")
plt.title("Top 10 Combinations by 3-Year Return")
plt.tight_layout()
plt.savefig("top_3yr_combos.png", dpi=150)
plt.close()

# -----------------------------
# STEP 4: PROBLEM 2 — OPTIMAL SHARPE RATIO FOR 1-YR RETURN
# -----------------------------
# Features: include Sharpe Ratio and categorical variables
X = df[[sharpe_col, market_col, type_col, risk_col]]
y = df[return_1yr_col]

# Preprocessing pipelines
numeric_features = [sharpe_col]
categorical_features = [market_col, type_col, risk_col]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Random Forest Regressor pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=200))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
print(f"\n1-Year Return Prediction - Test R2: {r2_score(y_test, y_pred):.4f}, MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# -----------------------------
# Find optimal Sharpe Ratio
# -----------------------------
# Create baseline (median numeric, mode categorical)
baseline = {}
baseline[sharpe_col] = 0  # placeholder, will vary
for col in categorical_features:
    baseline[col] = X[col].mode()[0]

# Generate a Sharpe ratio grid
sharpe_min, sharpe_max = X[sharpe_col].min(), X[sharpe_col].max()
sharpe_grid = np.linspace(sharpe_min, sharpe_max, 200)
pred_returns = []

for s in sharpe_grid:
    row = baseline.copy()
    row[sharpe_col] = s
    row_df = pd.DataFrame([row])
    pred = model.predict(row_df)[0]
    pred_returns.append(pred)

pred_returns = np.array(pred_returns)
best_idx = pred_returns.argmax()
optimal_sharpe = sharpe_grid[best_idx]
optimal_return = pred_returns[best_idx]

# Visualization
plt.figure(figsize=(10,6))
plt.plot(sharpe_grid, pred_returns, label='Predicted 1YrReturn')
plt.axvline(optimal_sharpe, color='r', linestyle='--', label=f'Optimal Sharpe ≈ {optimal_sharpe:.3f}')
plt.xlabel("Sharpe Ratio")
plt.ylabel("Predicted 1-Year Return (%)")
plt.title("Predicted 1-Year Return vs Sharpe Ratio")
plt.legend()
plt.tight_layout()
plt.savefig("sharpe_vs_1yrreturn.png", dpi=150)
plt.close()

print(f"\n  Optimal Sharpe Ratio to maximize 1-Year Return: {optimal_sharpe:.4f}")
print(f"Predicted 1-Year Return at optimal Sharpe: {optimal_return:.4f}%")

# -----------------------------
# ANALYSIS SUMMARY
# -----------------------------
print("\n  Analysis Summary:")
print(f"- Best MarketCap|Type|Risk combination (3YrReturn): {best_combo[market_col]} | {best_combo[type_col]} | {best_combo[risk_col]}")
print(f"- Predicted 3YrReturn: {best_combo[return_3yr_col]:.2f}%")
print(f"- Optimal Sharpe Ratio to maximize 1YrReturn: {optimal_sharpe:.4f}")
print(f"- Predicted 1YrReturn at optimal Sharpe: {optimal_return:.2f}%")


# In[ ]:




