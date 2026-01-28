import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. LOAD DATA
df = pd.read_csv("Accident.csv")

# Keep a copy of the raw data for comparison
df_raw = df.copy()

# 2. BEFORE-CLEANING SUMMARY
print("=== BEFORE CLEANING ===")
print(f"Rows: {len(df_raw)}")
print(f"Duplicate rows: {df_raw.duplicated().sum()}")
print("\nMissing values per column:")
print(df_raw.isna().sum())

if "Driver Age" in df_raw.columns:
    print("\n'Driver Age' stats (raw):")
    print(df_raw["Driver Age"].describe())

# 3. DATA CLEANING
# Remove duplicates
df = df.drop_duplicates()

# ---- Date processing ----
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Hour"] = df["Date"].dt.hour

# ---- Clean driver age ----
# Keep ages between 16 and 90, set others to NaN
df["Driver Age Clean"] = df["Driver Age"].where(
    (df["Driver Age"] >= 16) & (df["Driver Age"] <= 90)
)

# Fill missing/invalid ages with median of valid ages
median_age = df["Driver Age Clean"].median()
df["Driver Age Clean"] = df["Driver Age Clean"].fillna(median_age)

# ---- MERGED target variable ----
# Injury_or_Death = 1 if any injury or death occurred, else 0
df["Injury_or_Death"] = (
    (df["Simple Injuries"] > 0) |
    (df["Medium Injuries"] > 0) |
    (df["Severe Injuries"] > 0) |
    (df["Death"] > 0)
).astype(int)

# ---- Drop leakage / replaced columns ----
drop_cols = [
    "Simple Injuries", "Medium Injuries", "Severe Injuries",
    "Death", "Date", "Driver Age"
]
df = df.drop(columns=drop_cols)

# 4. AFTER-CLEANING SUMMARY
print("\n=== AFTER CLEANING ===")
print(f"Rows: {len(df)}")
print("\nMissing values per column:")
print(df.isna().sum())

print("\nTarget 'Injury_or_Death' distribution:")
print(df["Injury_or_Death"].value_counts(normalize=True).rename("proportion"))

# 5. SAVE CLEANED DATA
df.to_csv("Accident_cleaned.csv", index=False)
print("\nCleaned data saved to 'Accident_cleaned.csv'")

# 6. FEATURES / TARGET
X = df.drop(columns=["Injury_or_Death"])
y = df["Injury_or_Death"]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# 7. PREPROCESSOR
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 8. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== DATA SPLIT SHAPES ===")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

# 9. (OPTIONAL) FIT PREPROCESSOR
# X_train_trans = preprocess.fit_transform(X_train)
# X_test_trans = preprocess.transform(X_test)
# print("\nTransformed X_train shape:", X_train_trans.shape)
# print("Transformed X_test shape:", X_test_trans.shape)
