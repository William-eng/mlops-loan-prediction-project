import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample loan data
n_samples = 614

data = {
    'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Married': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
    'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
    'ApplicantIncome': np.random.randint(150, 15000, n_samples),
    'CoapplicantIncome': np.random.randint(0, 8000, n_samples),
    'LoanAmount': np.random.randint(10, 500, n_samples),
    'Loan_Amount_Term': np.random.choice([360, 180, 120, 240, 300, 480], n_samples),
    'Credit_History': np.random.choice([0.0, 1.0], n_samples, p=[0.15, 0.85]),
    'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples)
}

df = pd.DataFrame(data)

# Create target variable with some logic
# Higher income + good credit history = more likely to be approved
df['Loan_Status'] = 'N'
conditions = (
    (df['ApplicantIncome'] > 4000) & 
    (df['Credit_History'] == 1.0) & 
    (df['LoanAmount'] < 200)
)
df.loc[conditions, 'Loan_Status'] = 'Y'

# Add some randomness
random_approvals = np.random.choice(df.index, size=int(len(df) * 0.2), replace=False)
df.loc[random_approvals, 'Loan_Status'] = np.random.choice(['Y', 'N'], len(random_approvals))

# Add some missing values (realistic scenario)
missing_indices = np.random.choice(df.index, size=int(len(df) * 0.1), replace=False)
df.loc[missing_indices[:20], 'Gender'] = np.nan
df.loc[missing_indices[20:40], 'LoanAmount'] = np.nan
df.loc[missing_indices[40:60], 'Credit_History'] = np.nan

# Split into train and test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Save
train_df.to_csv('data/raw/train.csv', index=False)
test_df.to_csv('data/raw/test.csv', index=False)

print(f"Created train.csv with {len(train_df)} rows")
print(f"Created test.csv with {len(test_df)} rows")
print(f"\nTarget distribution in train:")
print(train_df['Loan_Status'].value_counts())