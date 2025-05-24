import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# === 1. Load Data ===
RAW_PATH = "../data/raw/accepted_2007_to_2018Q4.csv"
df = pd.read_csv(RAW_PATH, low_memory=False)

# === 2. Filter Relevant Loan Statuses ===
df['loan_status'] = df['loan_status'].map({
    'Fully Paid': 0,
    'Charged Off': 1
})
df = df[df['loan_status'].isin([0, 1])]

# === 3. Select Features and Target ===
features = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'purpose', 'dti']
target = 'loan_status'

df = df[features + [target]].dropna()

# === 4. One-hot Encode Categorical Features ===
df = pd.get_dummies(df, columns=['term', 'purpose'], drop_first=True)

# === 5. Normalize Numeric Features ===
scaler = StandardScaler()
numeric_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === 6. Split X and y ===
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 7. Save to CSV or Pickle (optional) ===
os.makedirs("../data/processed", exist_ok=True)
X_train.to_csv("../data/processed/X_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

print("✅ Preprocessing complete. Data saved in /data/processed/")
