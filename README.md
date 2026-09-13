# 🧠 Predicting Cognitive Performance Through Modifiable Lifestyle Factors

**Using Interpretable Regression Models**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

> SDG 3 — Good Health & Well-Being · SDG 4 — Quality Education

A machine learning project that quantifies how modifiable daily habits — sleep, stress, exercise, caffeine, screen time, diet, reaction time, and memory — drive an individual's **Cognitive_Score**, and turns that into interpretable, data-driven wellness recommendations.

---

## 📌 Table of Contents

- [Background](#-background)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Modeling](#-modeling)
- [Results](#-results)
- [Effect of PCA](#-effect-of-pca)
- [Key Findings & Business Insights](#-key-findings--business-insights)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Background

Lifestyle choices significantly influence cognitive performance, but they are often evaluated using generic, one-size-fits-all assumptions. Organizations such as corporate wellness providers or digital health apps need a way to **accurately determine the optimal lifestyle profile** required to maximize an individual's cognitive output — balancing peak mental performance (focus, learning, decision-making) with long-term mental health risk management.

This project uses machine learning to identify the key lifestyle drivers behind `Cognitive_Score` and enable targeted, data-driven wellness interventions instead of generic advice.

## ❓ Problem Statement

> How can we construct an interpretable and reliable regression model to predict an individual's `Cognitive_Score`, and — more importantly — quantify the weighted influence of each lifestyle factor (sleep, stress, caffeine intake, exercise, etc.) on cognitive outcomes, so that wellness programs can design the most effective interventions?

**Analytical goal:** minimize prediction error (MSE / MAE) while maximizing the interpretability of feature importance across lifestyle factors, using regression models capable of capturing non-linear relationships (e.g. Random Forest, XGBoost, Stacking Ensembles).

## 🎯 Project Objectives

1. Quantify the impact of modifiable lifestyle factors on `Cognitive_Score`.
2. Build an accurate regression model for `Cognitive_Score` prediction.
3. Identify the optimal lifestyle profile for maximizing cognitive performance.
4. Generate personalized, data-driven recommendations for productivity and mental health optimization.

## 📊 Dataset

| | |
|---|---|
| **Dimensions** | 84,000 rows × 11 columns |
| **Target variable** | `Cognitive_Score` (continuous) |
| **Numerical features** | `Sleep_Duration`, `Stress_Level`, `Daily_Screen_Time`, `Caffeine_Intake`, `Reaction_Time`, `Memory_Test_Score` |
| **Categorical features** | `Gender`, `Diet_Type`, `Exercise_Frequency` |
| **Identifier** | `User_ID` (dropped before modeling) |

The dataset contains individual daily lifestyle data (sleep duration, stress level, exercise frequency, caffeine intake, and diet type) along with the corresponding `Cognitive_Score`, used as the prediction target.

## 🔄 Workflow

```
Data Understanding
        │
Handle Missing Value ──► Drop Duplicates
        │
Feature Selection ──► Checking Correlation ──► Checking Skewness & Outliers
        │
Target Definition ──► Feature Scaling ──► Feature Encoding
        │
Train / Test Split
        │
Modeling ──► Hyperparameter Tuning ──► PCA Comparison ──► Evaluation
```

**Data cleaning steps:**
- Missing values in `Sleep_Duration`, `Stress_Level`, and `Memory_Test_Score` were imputed with the median.
- Duplicate rows were dropped.
- `User_ID` was removed as a non-predictive identifier.

## 🔍 Exploratory Data Analysis

- **Univariate:** All numerical features are non-normally distributed (D'Agostino normality test, p ≈ 0.0 for every column); no significant outliers were detected via the IQR method.
- **Bivariate (numeric vs. target):** Spearman correlation shows `Reaction_Time` (strong negative) and `Memory_Test_Score` (moderate positive) as the strongest correlates of `Cognitive_Score`; `Stress_Level` and `Daily_Screen_Time` show weak/insignificant relationships.
- **Bivariate (categorical vs. target):** One-way ANOVA shows `Exercise_Frequency` is a statistically significant predictor (p < 0.05), while `Gender` and `Diet_Type` are **not** significant and were dropped from the feature set.
- **Multivariate:** A Spearman correlation heatmap and pairplot confirm `Reaction_Time` and `Memory_Test_Score` as the dominant numerical drivers of the target.

## 🛠 Feature Engineering

- **Encoding:** `LabelEncoder` applied to remaining categorical column(s) (`Exercise_Frequency`).
- **Scaling:** `RobustScaler` applied to all features (robust to the lifestyle data's outliers/skew).
- **Feature filtering:** `Gender` and `Diet_Type` removed after statistical testing showed no significant relationship with the target.
- **Split:** 80% train / 20% test (`random_state = 0`).

## 🤖 Modeling

Baseline models trained on the fully engineered (non-PCA) feature set:

| Model | Role |
|---|---|
| **Linear Regression** | Interpretable baseline |
| **Decision Tree Regressor** | Non-linear baseline |
| **Random Forest Regressor** | Stable, robust ensemble — preserves data integrity and avoids the information loss seen with PCA |
| **XGBoost Regressor** | High accuracy on raw features via sensitivity to strong signals like `Memory_Test_Score` and `Reaction_Time` |
| **Voting Regressor** | Ensemble of the above four base models |
| **Stacking Regressor** | Ensemble (base learners + `Ridge` meta-model) — maximizes prediction by capturing non-linear relationships (Stress, Memory) while overcoming PCA's feature loss |

Random Forest and XGBoost were further optimized with `RandomizedSearchCV` (hyperparameter tuning), then combined into a **tuned Stacking Regressor**.

## 📈 Results

**Baseline models:**

| Model | MSE | R² | MAE |
|---|---|---|---|
| Linear Regression | 41.4865 | 0.9219 | 5.8675 |
| Decision Tree | 32.7207 | 0.9384 | 4.2970 |
| Voting Regressor | 11.3753 | 0.9786 | 2.4255 |
| Random Forest Regressor | 11.0236 | 0.9793 | 2.3416 |
| XGBoost Regressor | 8.9626 | 0.9831 | 1.9972 |
| **Stacking Regressor** | 7.9292 | **0.9851** | 1.7757 |

The **Stacking Regressor** achieves the best overall performance — highest R² and lowest error — confirming that combining multiple models improves both accuracy and reliability over any single model.

## 🧪 Effect of PCA

To test whether dimensionality reduction could simplify the model without hurting performance, PCA (95% explained variance) was applied before re-training the tuned Random Forest, XGBoost, and Stacking models, then compared against their non-PCA counterparts.

**Finding:** PCA **drastically reduced** predictive performance across all models. This proves that every raw lifestyle feature — even ones with seemingly weak individual correlation — carries unique, essential information that should not be compressed or discarded.

**Conclusion:** the optimal predictive strategy is the **Stacking Regressor (Ensemble)** or the **Tuned XGBoost Regressor**, both trained **without PCA**, as they achieve the highest R² scores.

## 💡 Key Findings & Business Insights

- **Reaction_Time** and **Memory_Test_Score** are the primary determinants of `Cognitive_Score`.
- **Exercise_Frequency** is the only categorical feature that is statistically significant — validating its role in intervention programs.
- **Gender** and **Diet_Type** are statistically insignificant and should be excluded from final models and recommendations, for model parsimony and efficient resource allocation.
- **PCA hurts performance** here — all raw features must be retained.
- **Actionable recommendation:** wellness interventions should prioritize **improving reaction time**, **improving memory function**, and **promoting regular exercise** to most effectively and data-drivenly maximize mental performance.

## 🧰 Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** SciPy (`normaltest`, `spearmanr`, `f_oneway`)
- **Modeling:** scikit-learn (Linear Regression, Decision Tree, Random Forest, Voting/Stacking Regressor, PCA, RobustScaler, LabelEncoder, RandomizedSearchCV), XGBoost
- **Environment:** Jupyter / Google Colab Notebook

## 📁 Project Structure

```
.
├── Cognitive_Performance.ipynb   # Full analysis: EDA → feature engineering → modeling → evaluation
├── Poster_ML_AOL.pdf             # Project poster summary
├── cognitive_performance1.csv    # Dataset (not included — see Getting Started)
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost jupyter

# Place cognitive_performance1.csv in the project root, then run
jupyter notebook Cognitive_Performance.ipynb
```

## 👥 Contributors

Data Science Students, Bina Nusantara University

| NIM | Name |
|---|---|
| 2802545655 | Nazhifa Kirana Mulia Nugraha |
| 2802504876 | Michael Yeremia |
| 2802545466 | Samuel Christopher |
| 2802524120 | Valentino Kurniawan |

## 📄 License

This project is released under the [MIT License](LICENSE).
