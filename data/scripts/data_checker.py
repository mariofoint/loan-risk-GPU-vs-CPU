import pandas as pd

# Load the raw dataset
csv_path = "./raw/accepted_2007_to_2018Q4.csv"

try:
    df = pd.read_csv(csv_path)
    print("CSV loaded.")
except FileNotFoundError:
    print(f"File not found at {csv_path}")
    exit()

# Map loan status to binary and filter
df['loan_status'] = df['loan_status'].map({
    'Fully Paid': 0,
    'Charged Off': 1
})
df = df[df['loan_status'].isin([0, 1])]

# Show distribution
print("Loan Status Distribution:")
print(df['loan_status'].value_counts())
print(df['loan_status'].value_counts(normalize=True) * 100)
