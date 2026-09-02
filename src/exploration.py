import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Display dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display data types
print("\nDATA TYPES:")
print(df.dtypes)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# Check for blank values in TotalCharges
print("\nBLANK TOTAL CHARGES:")
print((df["TotalCharges"].str.strip() == "").sum())

# Unique values in Churn
print("\nCHURN VALUES:")
print(df["Churn"].value_counts())

# Churn percentage
print("\nCHURN PERCENTAGE:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Contract distribution
print("\nCONTRACT DISTRIBUTION:")
print(df["Contract"].value_counts())

# Internet service distribution
print("\nINTERNET SERVICE DISTRIBUTION:")
print(df["InternetService"].value_counts())

# Churn rate by contract type
print("\nCHURN RATE BY CONTRACT:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Churn rate by internet service
print("\nCHURN RATE BY INTERNET SERVICE:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Average monthly charges by churn status
print("\nAVERAGE MONTHLY CHARGES BY CHURN:")
print(
    df.groupby("Churn")["MonthlyCharges"].mean()
)

# Average tenure by churn status
print("\nAVERAGE TENURE BY CHURN:")
print(
    df.groupby("Churn")["tenure"].mean()
)


# Churn distribution
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# Churn rate by contract type
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

contract_churn["Yes"].plot(
    kind="bar",
    figsize=(7, 4)
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Churn rate by internet service
internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

internet_churn["Yes"].plot(
    kind="bar",
    figsize=(7, 4)
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Monthly charges by churn status
plt.figure(figsize=(7, 4))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# Tenure by churn status
plt.figure(figsize=(7, 4))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")

plt.tight_layout()
plt.show()