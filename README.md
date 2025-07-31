# 🧠 Parkinson's Disease Prediction using Machine Learning

This project develops a machine learning pipeline to predict whether a person has Parkinson’s Disease based on voice and health-related features. It includes data preprocessing, visualization, model training, evaluation, and prediction capabilities.

## 📁 Dataset

The dataset used contains biomedical voice measurements from patients with and without Parkinson’s Disease. It includes over 750 features such as `PPE`, `RPDE`, `DFA`, signal jitter metrics, and wavelet transform statistics.

## 🎯 Project Goals

### 1. 📦 Importing Libraries and Dataset
- **Libraries Used**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn` for ML and preprocessing
  - `xgboost` for gradient boosting
  - `imblearn` for SMOTE (handling class imbalance)

- **Steps**:
  - Load dataset using `pandas`
  - Display structure and check for missing values

### 2. ⚙️ Data Preprocessing
- Handled missing values
- Scaled features using `StandardScaler`
- Encoded categorical variables (e.g., gender)
- Applied **SMOTE** to resolve class imbalance
- Performed **train-test split (80-20)**

### 3. 📊 Exploratory Data Analysis (EDA)
- Plotted **histograms** and **boxplots** for key voice features
- Created **correlation heatmaps** to identify important predictors
- Visualized feature trends across `Healthy` vs `Parkinson’s` patients using **violin plots**

### 4. 🤖 Model Training and Selection
Trained and evaluated the following models:
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Support Vector Machine (SVM)
- ✅ XGBoost

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

Bar plots were used to visually compare performance metrics.

### 5. 🧪 Model Evaluation and Prediction
- Selected the best model based on **F1-score**
- Plotted **ROC-AUC curves** for model performance
- Implemented manual prediction functionality using:
  ```python
  predict_parkinson(input_values, model, scaler)

**✅ Conclusion**
The pipeline successfully predicts Parkinson’s Disease based on voice features.

Provides early detection support for healthcare diagnostics.

**Future improvements may involve:**

Deep learning models

Feature reduction via PCA

Model deployment via Streamlit or Flask
