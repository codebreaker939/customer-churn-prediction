# Customer Churn Prediction Using Machine Learning

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. The project covers data analysis, preprocessing, model training, cross-validation, hyperparameter tuning, model comparison, and deployment through an interactive Streamlit application.

---

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are likely to leave allows businesses to take proactive retention measures.

This project builds a binary classification system to predict whether a customer will:

- **Stay (`No`)**
- **Churn (`Yes`)**

The project compares **Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree** models and selects the best-performing model using **F1-score**.

---

## 🎯 Objectives

- Analyze customer churn patterns using Exploratory Data Analysis (EDA)
- Clean and preprocess the dataset
- Encode categorical features using One-Hot Encoding
- Train multiple classification models
- Evaluate models using Accuracy, Precision, Recall, and F1-score
- Perform 5-fold cross-validation
- Tune model hyperparameters using GridSearchCV
- Select and save the best-performing model
- Build an interactive Streamlit application for real-time predictions

---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

**Dataset size:** 7,043 customers × 21 features

**Source:** [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains information about customer demographics, services, contracts, tenure, billing, payment methods, and churn status.

> The dataset CSV is excluded from the Git repository through `.gitignore`.

---

## 🔬 Project Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning & Preprocessing
   ↓
Train/Test Split
   ↓
Baseline Models
   ↓
5-Fold Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Model Saving
   ↓
Streamlit Application



📈 Exploratory Data Analysis

The initial analysis identified several important churn patterns.

Churn Distribution
Customer Status    Count    Percentage
---------------------------------------
No Churn           5,174      73.46%
Churn              1,869      26.54%

Key Findings

Contract Type

Contract           Churn Rate
-----------------------------
Month-to-month       42.71%
One year             11.27%
Two year              2.83%

Customers with month-to-month contracts showed significantly higher churn.

Internet Service

Internet Service   Churn Rate
-----------------------------
DSL                   18.96%
Fiber optic           41.89%
No internet            7.40%

Other observations

Average monthly charges were higher for churned customers: $74.44 vs $61.27
Average tenure was much lower for churned customers: 17.98 vs 37.57 months

These findings helped establish the business patterns that the machine learning models would learn from.

🧹 Data Preprocessing

The following preprocessing steps were performed:

Converted TotalCharges from string to numeric
Identified 11 missing TotalCharges values
The missing values corresponded to customers with tenure = 0, so they were replaced with 0
Removed customerID because it is an identifier rather than a predictive feature
Encoded Churn as:
No → 0
Yes → 1
Applied One-Hot Encoding to categorical features
Used a ColumnTransformer and Scikit-learn pipelines
Performed an 80/20 stratified train-test split
🤖 Models Evaluated

Three classification algorithms were trained:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Baseline Results

Model                  Accuracy   Precision   Recall   F1-Score
----------------------------------------------------------------
Logistic Regression      80.55%      65.72%    55.88%     60.40%
KNN                      76.37%      55.27%    57.49%     56.36%
Decision Tree            72.89%      48.97%    50.80%     49.87%


Logistic Regression was the strongest baseline model.

🔄 Cross-Validation & Hyperparameter Tuning

To obtain a more reliable estimate of model performance, 5-fold cross-validation was performed.

Hyperparameters were then optimized using GridSearchCV with F1-score as the scoring metric.

The models were tuned independently and evaluated on the untouched test set.

Final Tuned Results

Model                     Accuracy   Precision   Recall   F1-Score
-----------------------------------------------------------------
Tuned Logistic Regression   74.17%      50.87%    78.61%     61.76%
Tuned KNN                   78.14%      59.07%    57.49%     58.27%
Tuned Decision Tree         74.59%      51.40%    78.34%     62.08%


🏆 Final Model
Selected Model: Tuned Decision Tree

Metric             Result
--------------------------
Accuracy           74.59%
Precision          51.40%
Recall             78.34%
F1-Score           62.08%
5-Fold CV F1       62.04%


The Decision Tree was selected because it achieved the highest F1-score among the three tuned models.

Best parameters:

max_depth = 3
min_samples_split = 2
min_samples_leaf = 1
class_weight = balanced
Final Performance
Metric	Result
Accuracy	74.59%
Precision	51.40%
Recall	78.34%
F1-Score	62.08%
5-Fold CV F1	62.04%
Why F1-score?

The dataset contains more non-churn customers than churn customers. Therefore, accuracy alone is not sufficient.

F1-score provides a balance between precision and recall, while recall is particularly important for identifying customers who are actually at risk of leaving.

The final model achieved 78.34% recall, identifying a large proportion of actual churners.

📊 Final Confusion Matrix
                 Predicted
                 No      Yes

Actual No       758      277
Actual Yes       81      293

The model correctly identified 293 out of 374 actual churners, while missing 81 churners.

💾 Model Saving

The final trained pipeline was saved using Joblib:

models/best_model.pkl

This allows the trained model and preprocessing pipeline to be reused without retraining every time the application starts.

🖥️ Streamlit Application

An interactive Streamlit application was developed on top of the trained model.

Users can enter customer information such as:

Tenure
Contract type
Internet service
Security and support services
Payment method
Monthly charges
Total charges
Other customer/service details

The application provides:

Churn prediction
Churn probability
Risk classification
Customer profile summary
Model information
Important model signals
Example Predictions

A high-risk customer profile with short tenure, month-to-month contract, fiber optic internet, limited support services, and high monthly charges produced approximately:

87.1% churn probability

A low-risk profile with long tenure, a two-year contract, additional services, automatic payment, and lower monthly charges produced approximately:

6.4% churn probability

🗂️ Project Structure
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│
├── src/
│   ├── exploration.py
│   ├── preprocessing.py
│   └── train.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

🛠️ Tech Stack
Python
Pandas & NumPy — Data processing
Matplotlib & Seaborn — Visualization
Scikit-learn — Machine Learning
Joblib — Model serialization
Streamlit — Interactive application
Git & GitHub — Version control


▶️ Installation & Usage

Clone the repository:

git clone <https://github.com/codebreaker939/customer-churn-prediction>
cd customer-churn-prediction

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

To reproduce the ML workflow:

python src/exploration.py
python src/preprocessing.py
python src/train.py
⚠️ Limitations
The final F1-score is moderate at 62.08%
The model produces false positives, which may lead to unnecessary retention efforts
The model is trained on a specific telecom dataset and may not generalize directly to other businesses
Additional behavioral, transactional, and customer-support data could improve performance


🚀 Future Improvements

Potential future improvements include:

Random Forest, XGBoost, and Gradient Boosting
Advanced feature engineering
Probability calibration
SHAP/LIME-based model explainability
More extensive hyperparameter optimization
Cloud deployment
Real-time customer data integration


✅ Final Conclusion

This project successfully implements a complete machine learning workflow for customer churn prediction.

After comparing three classification algorithms, performing cross-validation, and tuning their hyperparameters, the Tuned Decision Tree was selected as the final model.

With 74.59% accuracy, 78.34% recall, and 62.08% F1-score, the model provides a useful baseline for identifying customers at risk of churn.

The trained model was then integrated into a Streamlit application, turning the machine learning pipeline into an interactive prediction system.


👨‍💻 Author
Aniket Rai