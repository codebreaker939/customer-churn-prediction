import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Identify categorical and numerical features
categorical_columns = df.select_dtypes(include="str").columns.tolist()
numerical_columns = df.select_dtypes(exclude="str").columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\nIs Churn in X?")
print("Churn" in X.columns)

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

# Fit and transform the features
X_encoded = preprocessor.fit_transform(X)

print("\nEncoded feature shape:")
print(X_encoded.shape)

# Final preprocessing verification

print("\nFINAL VERIFICATION")
print("-------------------")

print("Original feature shape:", X.shape)
print("Encoded feature shape:", X_encoded.shape)
print("Target shape:", y.shape)

print("\nMissing values in target:")
print(y.isnull().sum())

print("\nTarget distribution:")
print(y.value_counts())

print("\nPreprocessing completed successfully.")