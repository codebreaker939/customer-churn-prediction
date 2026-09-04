import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original dataset shape:")
print(df.shape)

print("\nOriginal data types:")
print(df.dtypes)


# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

print("\nTotalCharges data type after conversion:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())


# Inspect rows with missing TotalCharges
missing_total_charges = df[df["TotalCharges"].isnull()]

print("\nROWS WITH MISSING TOTAL CHARGES:")
print(
    missing_total_charges[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\nTENURE OF MISSING TOTAL CHARGES:")
print(missing_total_charges["tenure"].tolist())

print("\nNUMBER OF MISSING TOTAL CHARGES:")
print(len(missing_total_charges))


# Handle missing TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing TotalCharges after handling:")
print(df["TotalCharges"].isnull().sum())


# Remove customer ID
df = df.drop("customerID", axis=1)

print("\nDataset shape after removing customerID:")
print(df.shape)

print("\nRemaining columns:")
print(df.columns.tolist())


# Encode target variable
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn values after encoding:")
print(df["Churn"].value_counts())

print("\nChurn data type:")
print(df["Churn"].dtype)


# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# Identify categorical and numerical features
categorical_columns = X.select_dtypes(include="str").columns.tolist()
numerical_columns = X.select_dtypes(exclude="str").columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)


# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Fit preprocessing only on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Transform testing data using the fitted preprocessor
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded training feature shape:")
print(X_train_encoded.shape)

print("\nEncoded testing feature shape:")
print(X_test_encoded.shape)


# Final verification
print("\nFINAL VERIFICATION")
print("-------------------")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Encoded training features:", X_train_encoded.shape[1])
print("Encoded testing features:", X_test_encoded.shape[1])

print("\nMissing values in target:")
print("Training:", y_train.isnull().sum())
print("Testing:", y_test.isnull().sum())

print("\nPreprocessing completed successfully.")