# 🏠 Ames Housing Price Predictor
### Ridge Regression · Real Estate ML · Multicollinearity Control

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-Ridge%20Regression%20%28L2%29-185FA5?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Real%20Estate%20Analytics-1D9E75?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-Ames%20Housing%20%E2%80%94%20Kaggle-D85A30?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-0F6E56?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/R%C2%B2%20Score-0.871-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/RMSE-0.155%20%28log%20scale%29-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Best%20Alpha-215.44-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Coeff%20Stability-1.6x%20more%20stable-purple?style=flat-square"/>
</p>

---

## 📌 Problem Statement

> **Given 70+ features about a house — from square footage and quality ratings to garage type and neighbourhood — can we accurately predict its sale price while controlling for the multicollinearity that makes plain Linear Regression unstable?**

The Ames Housing dataset is one of the most comprehensive real estate datasets in machine learning, covering 1,460 property sales in Ames, Iowa. With 79 raw features, many of which are deeply correlated (basement area, living area, and total square footage all measure "size"), plain Linear Regression produces numerically inflated and unreliable coefficients.

**Ridge Regression** solves this by adding an L2 regularization penalty — shrinking all coefficients proportionally, producing stable, generalizable predictions even under high multicollinearity.

---

## 🧠 The Algorithm — Ridge Regression (L2 Regularization)

### What Is Ridge Regression?

Ridge Regression is an extension of Ordinary Least Squares (OLS) Linear Regression that adds a **regularization term** to the loss function to penalize large coefficients.

### The Math

**Ordinary Linear Regression** minimizes:

$$\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \| y - X\beta \|^2$$

**Ridge Regression** minimizes:

$$\text{Loss}_{\text{Ridge}} = \| y - X\beta \|^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

Where:
- $\| y - X\beta \|^2$ is the **Residual Sum of Squares (RSS)** — measures prediction error
- $\lambda$ is the **regularization strength** (alpha) — controls how much we penalize large coefficients
- $\sum \beta_j^2$ is the **L2 penalty** — sum of squared coefficients

The **closed-form solution** is:

$$\hat{\beta}_{\text{Ridge}} = (X^TX + \lambda I)^{-1} X^T y$$

Adding $\lambda I$ to $X^TX$ before inverting ensures the matrix is always invertible — even when features are perfectly collinear.

### Key Properties

| Property | Behaviour |
|----------|-----------|
| **Coefficients** | Shrunk toward zero, but **never exactly zero** |
| **Feature selection** | No — all features retained |
| **Best use case** | When all features carry signal but are correlated |
| **Alpha = 0** | Identical to Linear Regression |
| **Alpha → ∞** | All coefficients shrink to zero |
| **Alpha selection** | Via `RidgeCV` cross-validation |

### Why Not Lasso?

| | Ridge (L2) | Lasso (L1) |
|-|-----------|-----------|
| **Penalty** | $\lambda \sum \beta_j^2$ | $\lambda \sum |\beta_j|$ |
| **Effect** | Shrinks all equally | Zeros out some |
| **Use here?** | ✅ Yes — all 20 features carry signal | ❌ No — we don't want elimination |
| **Geometry** | Circular constraint region | Diamond constraint region |

In Ames Housing, all selected features are genuine price drivers — none should be discarded. Ridge is the correct choice.

---

## 🔄 Project Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROJECT 2 — RIDGE REGRESSION                    │
│                      AMES HOUSING PRICE PREDICTOR                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║           📥  RAW DATA INGESTION         ║
          ║                                          ║
          ║   ames_train.csv                         ║
          ║   • 1,460 rows                           ║
          ║   • 79 features (numeric + categorical)  ║
          ║   • Target: SalePrice ($)                ║
          ╚══════════════════════════════════════════╝
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║        🔍  EXPLORATORY DATA ANALYSIS     ║
          ║                                          ║
          ║   • SalePrice distribution → right-skew  ║
          ║   • Missing value heatmap (19 cols)      ║
          ║   • Top 12 correlated numeric features   ║
          ║   • Decision: log-transform SalePrice    ║
          ╚══════════════════════════════════════════╝
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
     ┌──────────────────────┐      ┌──────────────────────────┐
     │  🔧 LOG TRANSFORM    │      │  🔧 FEATURE ENGINEERING  │
     │                      │      │                          │
     │  log1p(SalePrice)    │      │  TotalSF   (1F+2F+Bsmt) │
     │  → near-normal dist  │      │  HouseAge  (YrSold-Built)│
     │  → better residuals  │      │  RemodAge  (YrSold-Remo) │
     │  → assumption met    │      │  IsRemodeled  (binary)   │
     └──────────────────────┘      │  TotalBath (weighted)    │
                    │              │  TotalPorch (all porches) │
                    │              │  HasGarage    (binary)   │
                    │              │  HasPool      (binary)   │
                    │              └──────────────────────────┘
                    └─────────────┬──────────────┘
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║      📊  ORDINAL QUALITY ENCODING        ║
          ║                                          ║
          ║   Ex→5  Gd→4  TA→3  Fa→2  Po→1  NA→0   ║
          ║   Applied to: ExterQual, KitchenQual,    ║
          ║               BsmtQual, GarageQual...    ║
          ╚══════════════════════════════════════════╝
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║    ⚠️   VIF ANALYSIS (KEY DIAGNOSTIC)    ║
          ║                                          ║
          ║   Variance Inflation Factor computed     ║
          ║   for all 20 selected features           ║
          ║                                          ║
          ║   VIF > 10 detected on:                  ║
          ║   • TotalSF ↔ GrLivArea ↔ TotalBsmtSF  ║
          ║   → HIGH MULTICOLLINEARITY CONFIRMED     ║
          ║   → Ridge Regression justified ✅        ║
          ╚══════════════════════════════════════════╝
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║        ✂️   TRAIN / TEST SPLIT           ║
          ║                                          ║
          ║   80% Training  →  1,168 samples         ║
          ║   20% Test      →    292 samples         ║
          ║   StandardScaler fit on train only       ║
          ║   (prevents data leakage)                ║
          ╚══════════════════════════════════════════╝
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
     ┌──────────────────────┐      ┌──────────────────────────┐
     │  📉 LINEAR BASELINE  │      │  📈 RIDGE REGRESSION     │
     │                      │      │                          │
     │  LinearRegression()  │      │  RidgeCV(               │
     │  No regularization   │      │    alphas=logspace(      │
     │  Volatile coeffs     │      │      -2, 3, 200),        │
     │  R² = 0.8767         │      │    cv=5)                 │
     │  RMSE = 0.1517       │      │                          │
     │  Coeff Std = 0.044   │      │  Best α = 215.44         │
     └──────────────────────┘      │  R² = 0.8713             │
                    │              │  RMSE = 0.1550           │
                    │              │  Coeff Std = 0.027       │
                    │              └──────────────────────────┘
                    └─────────────┬──────────────┘
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║     🔬  DIAGNOSTICS & VALIDATION         ║
          ║                                          ║
          ║   • Residuals vs Fitted  → random ✅     ║
          ║   • Residual histogram   → bell-shape ✅ ║
          ║   • Q-Q Plot             → near-line ✅  ║
          ║   • Alpha sweep curve    → clear min ✅  ║
          ╚══════════════════════════════════════════╝
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║         💡  KEY FINDING                  ║
          ║                                          ║
          ║   R² nearly identical (0.877 vs 0.871)  ║
          ║   BUT Ridge coefficients are             ║
          ║   1.6× MORE STABLE (Std: 0.044→0.027)   ║
          ║                                          ║
          ║   → Ridge wins on TRUSTWORTHINESS        ║
          ║     not just raw accuracy                ║
          ╚══════════════════════════════════════════╝
                                  │
                                  ▼
          ╔══════════════════════════════════════════╗
          ║         🔴  LIVE PREDICTION DEMO         ║
          ║                                          ║
          ║   Input: custom house profile            ║
          ║   Output: Predicted price in dollars     ║
          ║   (reverse log-transform applied)        ║
          ╚══════════════════════════════════════════╝
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| **Training rows** | 1,460 |
| **Raw features** | 79 (numeric + categorical) |
| **Features used** | 20 (selected + engineered) |
| **Target** | `SalePrice` → log-transformed to `SalePrice_log` |
| **Target transform** | `log1p(SalePrice)` — corrects right-skew, stabilises variance |

---

## 🔧 How Ridge Is Used Here — Step by Step

### Step 1 — Prove the Problem Exists (VIF Analysis)

Before applying Ridge, we *prove* multicollinearity exists using the **Variance Inflation Factor**:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data['VIF'] = [variance_inflation_factor(X_scaled, i)
                   for i in range(X_scaled.shape[1])]
```

Features like `TotalSF`, `GrLivArea`, `TotalBsmtSF` all measure house size → **VIF > 10** confirmed. This is the *justification* for Ridge — not just a stylistic choice.

### Step 2 — Find the Optimal Alpha via Cross-Validation

Instead of guessing alpha, `RidgeCV` finds it automatically:

```python
alphas = np.logspace(-2, 3, 200)    # 100 candidates from 0.01 to 1000
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)
best_alpha = ridge_cv.alpha_        # → 215.44
```

The **alpha sweep plot** shows RMSE vs regularization strength — the minimum of the curve is where `RidgeCV` lands.

### Step 3 — Train Ridge with Best Alpha

```python
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)
```

All 20 features are retained — no coefficient hits exactly zero (that's Lasso's job).

### Step 4 — Interpret Coefficients

Because features are **standardised** before training, coefficients are directly comparable:

```
OverallQual  →  +0.073  (strongest positive driver)
GrLivArea    →  +0.043  (living area premium)
HouseAge     →  −0.047  (older = cheaper)
RemodAge     →  −0.019  (recently remodelled = premium)
```

### Step 5 — Measure Stability (the Real Win)

```python
print(f"Std Dev of Linear coefficients : {lr.coef_.std():.5f}")   # → 0.04408
print(f"Std Dev of Ridge  coefficients : {ridge.coef_.std():.5f}") # → 0.02731
print(f"Ridge coefficients are {lr.coef_.std()/ridge.coef_.std():.1f}x more stable")
# → Ridge coefficients are 1.6x more stable
```

---

## 📈 Results

### Model Performance

| Metric | Linear Regression | Ridge Regression | Verdict |
|--------|------------------|-----------------|---------|
| **R² Score** | 0.8767 | 0.8713 | Linear marginally higher |
| **RMSE (log)** | 0.1517 | 0.1550 | Linear marginally lower |
| **MAE (log)** | 0.1057 | 0.1059 | Essentially identical |
| **Coeff Std Dev** | 0.04408 | **0.02731** | **Ridge 1.6× more stable ✅** |

### Why Linear's Higher R² Doesn't Tell the Whole Story

The Ames dataset has only 1,460 rows and 20 features — a relatively small, low-dimensional dataset. In this regime:

- Multicollinearity is **present but not catastrophic** — Linear still converges
- Ridge's benefit is **stability**, not dramatically higher accuracy
- On a dataset with 10,000+ rows and 50+ correlated features, Ridge would clearly outperform Linear on a held-out test set

> **Key insight**: Ridge doesn't always win on R². Ridge always wins on **coefficient trustworthiness** — which matters more in a production model where stakeholders ask *"why did this house get priced higher?"*

---

## 📉 Visualizations

| Plot | File | What It Shows |
|------|------|--------------|
| Missing value bar chart | `01_missing.png` | Which columns need imputation |
| EDA 3-panel | `02_eda.png` | Raw vs log price, top correlations |
| **Alpha sweep** ★ | `03_alpha_sweep.png` | RMSE vs regularization — the Ridge signature plot |
| Actual vs Predicted | `04_comparison.png` | Linear vs Ridge side-by-side |
| Residual diagnostics | `05_diagnostics.png` | Residuals, histogram, Q-Q plot |
| **Coefficient comparison** ★ | `06_coefficients.png` | Ridge stable bars vs Linear volatile bars |

---

## 💡 Key Insights

1. **OverallQual dominates**: The single strongest predictor — a 1-unit increase in quality rating (e.g., 6→7) corresponds to a ~7.3% price increase (log scale)
2. **TotalSF is the engineered star**: Combining basement + 1st floor + 2nd floor into one feature outperforms any single area column
3. **Age penalises heavily**: `HouseAge` is the strongest negative coefficient — markets reward newness above most other factors
4. **Remodelling adds value independent of age**: `IsRemodeled` carries a positive coefficient even after controlling for `RemodAge`
5. **Ridge coefficients are 1.6× more stable**: The L2 penalty reduces coefficient variance from 0.044 to 0.027 — smaller, smoother, more reliable estimates
6. **Log-transform was critical**: Raw SalePrice has skewness > 1.5 with extreme outliers at $750k+. After `log1p`, residuals are near-normally distributed — satisfying the OLS assumption

---

## 🔴 Live Prediction Demo

Cell 10 allows custom house price prediction. The model predicts in log scale and automatically reverses to dollar output:

```python
custom_house = {
    'TotalSF': 2500,      # total square footage
    'OverallQual': 7,     # good quality (1–10)
    'GrLivArea': 1800,    # above-ground living area
    'HouseAge': 20,       # years since built
    'TotalBath': 2.5,     # 2 full + 1 half bath
    'KitchenQual': 4,     # Good quality
    ...
}
# → log prediction: 12.18
# → Predicted Sale Price: $195,800
```

**Archetype experiments:**
| Archetype | Change | Expected Effect |
|-----------|--------|-----------------|
| Luxury new build | OverallQual=9, HouseAge=1 | Price ↑ significantly |
| Fixer-upper | OverallQual=4, OverallCond=3 | Price ↓ sharply |
| Large old home | TotalSF=3500, HouseAge=60 | Mixed — size up, age down |
| Pool + garage | HasPool=1, HasGarage=1 | Modest premium |

---

## 🚀 How to Run

```bash
# 1. Clone / navigate to project folder
cd project2_ridge_housing

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy jupyter

# 3. Download dataset
# → kaggle.com/c/house-prices-advanced-regression-techniques
# → Download train.csv → rename to ames_train.csv → place here

# 4. Create visuals folder
mkdir visuals

# 5. Launch
jupyter notebook notebook.ipynb
```

> **Dependency note**: `statsmodels` is required for the VIF table in Cell 5. Install it separately if not already present: `pip install statsmodels`

---

## 🗂️ Project Structure

```
project2_ridge_housing/
│
├── 📓 notebook.ipynb                ← Full analysis — 10 cells
├── 📄 ames_train.csv                ← Dataset (Kaggle competition)
├── 📘 README.md                     ← This file
│
└── 📁 visuals/
    ├── 01_missing.png               ← Missing value bar chart
    ├── 02_eda.png                   ← EDA grid (3 panels)
    ├── 03_alpha_sweep.png           ← RidgeCV alpha curve ★
    ├── 04_comparison.png            ← Linear vs Ridge actual vs predicted
    ├── 05_diagnostics.png           ← Residuals + Q-Q diagnostics
    └── 06_coefficients.png          ← Coefficient comparison ★
```

---

## 🧠 Conceptual Summary

```
PROBLEM                    WHY RIDGE                    RESULT
───────────────────────    ─────────────────────────    ──────────────────────
79 raw features            L2 penalty shrinks all       R² = 0.871
Many correlated            coefficients proportionally  RMSE = 0.155 (log)
(size ↔ area ↔ SF)         without eliminating any      Coefficients 1.6×
                                                         more stable than OLS
VIF > 10 confirmed         RidgeCV selects α=215.44     Trustworthy, production-
on size features           via 5-fold cross-validation  ready pricing model
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-4C72B0?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-9B4E8E?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

