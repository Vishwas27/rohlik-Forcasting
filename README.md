# 📦 Rohlik Order Forecasting Project

This project involves accurate order forecasting for an e-grocery service. The goal is to improve workforce allocation, delivery logistics, inventory management, and supply chain efficiency by predicting the number of orders with high accuracy.

## 📝 Problem Overview

Accurate order forecasting is essential for e-grocery services like Rohlik. This project aims to build a forecasting model that predicts the number of orders for each given ID in the dataset. The predictions are evaluated based on the Mean Absolute Percentage Error (MAPE).

## ⚙️ Approach to Solution

### 📅 Data Preprocessing

- **Date-Based Features**: Extracted year, month, day, day of the week, week of the year, day of the year, quarter, and flags for month/quarter start and end. 
- **Sine & Cosine Transformations**: Applied to cyclic features like month and day to preserve the cyclical nature of time.
- **Handling Missing Values**: Filled missing values in columns like `holiday_name` with 'None' and used `OneHotEncoder` for encoding categorical variables.
- **Label Encoding**: Applied label encoding to categorical columns such as `warehouse`.
- **Holiday Effects**: Engineered features like the day before and after holidays to capture their effects on orders.

### 🎯 Modeling Strategy

The model employs an **ensemble and stacking** approach to leverage the strengths of different algorithms:

- **Base Models**:
  - LightGBM 🌿
  - XGBoost 🚀
  - CatBoost 🐱
  - RandomForest 🌳
  - Logistic Regression 🔑
  - AdaBoost 🚧
  - Decision Tree 🌴
  - Gradient Boosting 📈

Each model is trained using **10-fold cross-validation**, and the predictions from these base models are stacked to feed into a meta-model for final predictions.

### 🏗️ Meta Model

The meta-model is a **LightGBM** regressor, which is trained on the stacked outputs from the base models. It uses fine-tuned hyperparameters for the best performance.

### 🧪 Cross-Validation & Stacking

- 10-fold cross-validation was applied to ensure robustness.
- The predictions from each fold were combined to form the stacking dataset.
- The final predictions were made by the meta-model based on the stacked dataset.

### 🏆 Model Evaluation

The model's performance was evaluated using the **Mean Absolute Percentage Error (MAPE)**. The simple average ensemble model achieved a MAPE of **0.0318**, indicating a high level of accuracy in predicting the number of orders.

### 📊 Final Submission

The final predictions are stored in a `submission.csv` file containing the `ID` and predicted `ORDERS`.

## 📚 Libraries Used

- `pandas` 🐼
- `numpy` 🔢
- `matplotlib` 📊
- `sklearn` ⚙️
- `lightgbm` 🌿
- `xgboost` 🚀
- `catboost` 🐱

## 🚀 How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
