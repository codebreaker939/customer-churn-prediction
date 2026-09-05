import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# Remove customer ID
df = df.drop("customerID", axis=1)


# Encode target variable
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]


# Identify categorical and numerical features
categorical_columns = X.select_dtypes(
    include="str"
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude="str"
).columns.tolist()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# Create Logistic Regression pipeline
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# Create KNN pipeline
knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=5)
        )
    ]
)


# Create Decision Tree pipeline
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42)
        )
    ]
)


# Train Logistic Regression
logistic_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")


# Train KNN
knn_model.fit(X_train, y_train)

print("KNN model trained successfully.")


# Train Decision Tree
decision_tree_model.fit(X_train, y_train)

print("Decision Tree model trained successfully.")


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Make Logistic Regression predictions
logistic_pred = logistic_model.predict(X_test)


# Make KNN predictions
knn_pred = knn_model.predict(X_test)


# Make Decision Tree predictions
decision_tree_pred = decision_tree_model.predict(X_test)


# Create function to evaluate models
def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    print(f"\n{name} Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")


# Evaluate Logistic Regression
evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred
)


# Evaluate KNN
evaluate_model(
    "KNN",
    y_test,
    knn_pred
)


# Evaluate Decision Tree
evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_pred
)


# Perform 5-fold cross-validation

logistic_cv = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)

knn_cv = cross_val_score(
    knn_model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)

decision_tree_cv = cross_val_score(
    decision_tree_model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)


print("\n5-Fold Cross-Validation F1 Scores:")

print("\nLogistic Regression:")
print(logistic_cv)
print("Mean F1:", logistic_cv.mean())

print("\nKNN:")
print(knn_cv)
print("Mean F1:", knn_cv.mean())

print("\nDecision Tree:")
print(decision_tree_cv)
print("Mean F1:", decision_tree_cv.mean())


# Define hyperparameters for Logistic Regression
logistic_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [None, "balanced"]
}


# Define hyperparameters for KNN
knn_param_grid = {
    "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "classifier__weights": ["uniform", "distance"]
}


# Define hyperparameters for Decision Tree
decision_tree_param_grid = {
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__class_weight": [None, "balanced"]
}


# Create GridSearchCV for Logistic Regression
logistic_grid = GridSearchCV(
    estimator=logistic_model,
    param_grid=logistic_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


# Create GridSearchCV for KNN
knn_grid = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


# Create GridSearchCV for Decision Tree
decision_tree_grid = GridSearchCV(
    estimator=decision_tree_model,
    param_grid=decision_tree_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


# Perform hyperparameter tuning
print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING")
print("=" * 50)


print("\nTuning Logistic Regression...")

logistic_grid.fit(
    X_train,
    y_train
)


print("Tuning KNN...")

knn_grid.fit(
    X_train,
    y_train
)


print("Tuning Decision Tree...")

decision_tree_grid.fit(
    X_train,
    y_train
)


# Display best parameters
print("\n" + "=" * 50)
print("BEST PARAMETERS")
print("=" * 50)


print("\nLogistic Regression:")
print(logistic_grid.best_params_)
print("Best CV F1:", logistic_grid.best_score_)


print("\nKNN:")
print(knn_grid.best_params_)
print("Best CV F1:", knn_grid.best_score_)


print("\nDecision Tree:")
print(decision_tree_grid.best_params_)
print("Best CV F1:", decision_tree_grid.best_score_)


# Make predictions using tuned models
tuned_logistic_pred = logistic_grid.predict(X_test)

tuned_knn_pred = knn_grid.predict(X_test)

tuned_tree_pred = decision_tree_grid.predict(X_test)


# Evaluate tuned Logistic Regression
evaluate_model(
    "Tuned Logistic Regression",
    y_test,
    tuned_logistic_pred
)


# Evaluate tuned KNN
evaluate_model(
    "Tuned KNN",
    y_test,
    tuned_knn_pred
)


# Evaluate tuned Decision Tree
evaluate_model(
    "Tuned Decision Tree",
    y_test,
    tuned_tree_pred
)


# Create final model comparison table
results = pd.DataFrame({

    "Model": [
        "Tuned Logistic Regression",
        "Tuned KNN",
        "Tuned Decision Tree"
    ],

    "Accuracy": [
        accuracy_score(y_test, tuned_logistic_pred),
        accuracy_score(y_test, tuned_knn_pred),
        accuracy_score(y_test, tuned_tree_pred)
    ],

    "Precision": [
        precision_score(y_test, tuned_logistic_pred),
        precision_score(y_test, tuned_knn_pred),
        precision_score(y_test, tuned_tree_pred)
    ],

    "Recall": [
        recall_score(y_test, tuned_logistic_pred),
        recall_score(y_test, tuned_knn_pred),
        recall_score(y_test, tuned_tree_pred)
    ],

    "F1-Score": [
        f1_score(y_test, tuned_logistic_pred),
        f1_score(y_test, tuned_knn_pred),
        f1_score(y_test, tuned_tree_pred)
    ]
})


# Display final comparison
print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print(
    results.to_string(index=False)
)


# Select best model based on F1-score
best_model_name = results.loc[
    results["F1-Score"].idxmax(),
    "Model"
]


print("\nBest model based on F1-Score:")
print(best_model_name)


# Calculate confusion matrix for the best model
best_predictions = {
    "Tuned Logistic Regression": tuned_logistic_pred,
    "Tuned KNN": tuned_knn_pred,
    "Tuned Decision Tree": tuned_tree_pred
}

best_pred = best_predictions[best_model_name]

cm = confusion_matrix(
    y_test,
    best_pred
)


# Display confusion matrix
print("\nConfusion Matrix for Best Model:")
print(cm)


# Visualize confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title(f"{best_model_name} Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()